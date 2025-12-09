import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import glob
import time

# --- CONFIGURATION ---
# (Matches your "Small" Model exactly)
CONFIG = {
    "vocab_size": 32000,
    "dim": 768,               # Small Model
    "n_heads": 12,
    "head_dim": 64,
    "n_recurrent_loops": 8,   # 8 Loops
    "max_seq_len": 512,
    "window_size": 128,
    "soft_cap": 30.0,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 4e-4
}

# HARDWARE LIMITS
VRAM_LIMIT_GB = 10.0
CHECKPOINT_DIR = "checkpoints_test"  # New folder for this test run
MAX_STEPS = 5000                     # Stop exactly at 5000
TEST_EVERY_STEPS = 500               # Run validation every 500 steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(
        (VRAM_LIMIT_GB * 1024**3) / torch.cuda.get_device_properties(0).total_memory
    )
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- ARCHITECTURE (THE LIQUID RECURRENT SAVANT) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm_x * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class LiquidGatedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.head_dim = config["head_dim"]
        self.soft_cap = config["soft_cap"]
        self.q_proj = nn.Linear(config["dim"], config["dim"], bias=False)
        self.k_proj = nn.Linear(config["dim"], config["dim"], bias=False)
        self.v_proj = nn.Linear(config["dim"], config["dim"], bias=False)
        self.o_proj = nn.Linear(config["dim"], config["dim"], bias=False)
        self.ln = RMSNorm(config["dim"])
        self.liquid_gate = nn.Linear(config["dim"], config["dim"])

    def forward(self, x, mask):
        B, T, C = x.size()
        residual = x
        x = self.ln(x)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
        logits = logits + mask.to(x.dtype)
        attn_probs = F.softmax(logits, dim=-1)
        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        
        alpha = torch.sigmoid(self.liquid_gate(residual))
        return (1 - alpha) * residual + (alpha * out)

class LiquidRecurrentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = LiquidGatedAttention(config)
        self.ln_ffn = RMSNorm(config["dim"])
        hidden_dim = int(2 * (4 * config["dim"]) / 3)
        self.ffn = nn.Sequential(
            nn.Linear(config["dim"], hidden_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(hidden_dim, config["dim"], bias=False)
        )
        self.liquid_gate_ffn = nn.Linear(config["dim"], config["dim"])

    def forward(self, x, mask):
        x = self.attn(x, mask)
        residual = x
        x_norm = self.ln_ffn(x)
        ffn_out = self.ffn(x_norm)
        alpha = torch.sigmoid(self.liquid_gate_ffn(residual))
        return (1 - alpha) * residual + (alpha * ffn_out)

class UniversalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config["vocab_size"], config["dim"])
        self.recurrent_block = LiquidRecurrentBlock(config)
        self.ln_f = RMSNorm(config["dim"])
        self.head = nn.Linear(config["dim"], config["vocab_size"], bias=False)
        self.register_buffer("global_mask", None)
        self.register_buffer("local_mask", None)

    def _get_masks(self, T, device):
        if self.global_mask is None or self.global_mask.size(0) != T:
            self.global_mask = torch.triu(torch.ones(T, T, device=device) * float('-inf'), diagonal=1)
            local = torch.triu(torch.ones(T, T, device=device) * float('-inf'), diagonal=1)
            for i in range(T):
                local[i, :max(0, i - self.config["window_size"])] = float('-inf')
            self.local_mask = local
        return self.global_mask, self.local_mask

    def forward(self, input_ids):
        B, T = input_ids.size()
        x = self.embed(input_ids)
        global_mask, local_mask = self._get_masks(T, x.device)
        for i in range(self.config["n_recurrent_loops"]):
            current_mask = global_mask if (i + 1) % 4 == 0 else local_mask
            if self.training:
                x = torch.utils.checkpoint.checkpoint(self.recurrent_block, x, current_mask, use_reentrant=False)
            else:
                x = self.recurrent_block(x, current_mask)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["max_seq_len"]:]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- DATA & TESTING ---
def get_dataloaders(tokenizer):
    print("📚 Loading TinyStories (Train & Validation)...")
    
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encodings = tokenizer(
            texts, truncation=True, max_length=CONFIG["max_seq_len"], 
            padding="max_length", return_tensors="pt"
        )
        return encodings.input_ids

    # 1. Train Set (Streaming)
    ds_train = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    train_loader = DataLoader(ds_train, batch_size=CONFIG["batch_size"], collate_fn=collate_fn)

    # 2. Validation Set (Streaming)
    ds_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_loader = DataLoader(ds_val, batch_size=CONFIG["batch_size"], collate_fn=collate_fn)

    return train_loader, val_loader

# --- MAIN ---
def main():
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    model = UniversalTransformer(CONFIG).to(device).to(torch.bfloat16)
    
    # Optimizer Check
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG["lr"])
        print("✅ Using 8-bit AdamW")
    except:
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
        print("⚠️ Using Standard AdamW")

    # Resume Logic
    start_step = 0
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/step_*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"🔄 Resuming from: {latest}")
        ckpt = torch.load(latest)
        sd = ckpt['model_state_dict']
        # Clean masks
        for k in ["global_mask", "local_mask", "recurrent_block.attn.mask"]: 
            if k in sd: del sd[k]
        model.load_state_dict(sd, strict=False)
        try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except: pass
        start_step = ckpt['step']
    else:
        print("🆕 Starting 3000 Step Test Run")

    train_loader, val_loader = get_dataloaders(tokenizer)
    model.train()
    iter_train = iter(train_loader)
    
    print(f"\n🚀 Running until Step {MAX_STEPS}...")

    for step in range(start_step, MAX_STEPS + 1):
        try:
            batch = next(iter_train).to(device)
        except StopIteration:
            iter_train = iter(train_loader)
            batch = next(iter_train).to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch)
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, CONFIG["vocab_size"]),
                batch[..., 1:].contiguous().view(-1)
            )

        if torch.isnan(loss):
            print("❌ NaN detected. Skipping.")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | Train Loss: {loss.item():.4f}")

        # --- TESTING PHASE (Every 500 Steps) ---
        if step > 0 and step % TEST_EVERY_STEPS == 0:
            print("\n--- 🔍 STARTING TESTING PHASE ---")
            model.eval()
            val_loss_accum = 0
            val_batches = 20
            iter_val = iter(val_loader)
            
            with torch.no_grad():
                for _ in range(val_batches):
                    try:
                        v_batch = next(iter_val).to(device)
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            v_logits = model(v_batch)
                            v_loss = F.cross_entropy(
                                v_logits[..., :-1, :].contiguous().view(-1, CONFIG["vocab_size"]),
                                v_batch[..., 1:].contiguous().view(-1)
                            )
                            val_loss_accum += v_loss.item()
                    except StopIteration:
                        break
            
            avg_val_loss = val_loss_accum / val_batches
            print(f"📊 Validation Loss: {avg_val_loss:.4f} (Train Loss: {loss.item():.4f})")
            
            # Smart Analysis
            if avg_val_loss < 1.0:
                print("✅ RESULT: Model understands the data perfectly.")
            elif avg_val_loss > loss.item() + 0.5:
                print("⚠️ RESULT: Overfitting detected! (Memorization)")
            else:
                print("ℹ️ RESULT: Healthy learning curve.")

            # Text Generation Test
            print("📝 Generating Sample:")
            prompt = "Once upon a time,"
            inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
            out = model.generate(inp, max_new_tokens=30)
            print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}\n")
            print("-----------------------------------")
            
            model.train()
            
            # Save
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{CHECKPOINT_DIR}/step_{step}.pt")

    print("\n🏁 TEST RUN COMPLETE.")

if __name__ == "__main__":
    main()