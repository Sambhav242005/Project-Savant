import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import os
import math
import time
import glob
import logging
import sys

# --- HARDWARE & SAFETY ---
VRAM_LIMIT_GB = 10.0
CHECKPOINT_DIR = "checkpoints_savant"
CHECKPOINT_EVERY_STEPS = 500  
VALIDATION_EVERY_STEPS = 500   
MAX_STEPS = 80000             

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(
        (VRAM_LIMIT_GB * 1024**3) / torch.cuda.get_device_properties(0).total_memory
    )

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- LOGGING SETUP ---
log_file = os.path.join(CHECKPOINT_DIR, "training_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

if torch.cuda.is_available():
    logger.info(f"🔒 VRAM Hard Limit set to: {VRAM_LIMIT_GB} GB")

# --- THE SAVANT CONFIG (FIXED) ---
CONFIG = {
    "vocab_size": 32000,
    "dim": 1024,
    "n_heads": 16,
    "head_dim": 64,
    "n_recurrent_loops": 12,
    "max_seq_len": 1024,
    "window_size": 256,
    "soft_cap": 50.0,
    "dropout": 0.1,
    
    # --- CRITICAL FIXES ---
    "batch_size": 12,         # Safe size
    "lr": 2e-5,               # LOW LR (Fine-tuning mode)
    "accumulation": 16        # Added missing key (Effective Batch ~160)
}

# --- ARCHITECTURE CLASSES ---
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
        
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
        if mask is not None:
            logits = logits + mask.to(logits.device).to(logits.dtype)
            
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
        if self.global_mask is None or self.global_mask.size(0) != T or self.global_mask.device != device:
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
    def generate(self, idx, max_new_tokens, temperature=0.7):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["max_seq_len"]:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == 3: # EOS (SEP for Albert)
                 break
        return idx

from datasets import load_dataset, interleave_datasets

# --- DATA LOADER ---
def get_dataloaders(tokenizer):
    logger.info("📚 Loading Savant Datasets...")
    if not os.path.exists("dataset/savant_dataset_train.jsonl"):
        logger.error("❌ CRITICAL ERROR: 'dataset/savant_dataset_train.jsonl' not found.")
        logger.error("💡 Please run: python prepare_dataset.py")
        sys.exit(1)

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encodings = tokenizer(texts, truncation=True, max_length=CONFIG["max_seq_len"], padding="max_length", return_tensors="pt")
        return encodings.input_ids

    # 1. Load Main Dataset (The "Bulk")
    ds_main = load_dataset("json", data_files="dataset/savant_dataset_train.jsonl", split="train")
    
    # 2. Load Genius Dataset (The "Cram School")
    if os.path.exists("dataset/genius_dataset.jsonl"):
        logger.info("🧠 Found 'dataset/genius_dataset.jsonl' - Injecting High-IQ Logic...")
        ds_genius = load_dataset("json", data_files="dataset/genius_dataset.jsonl", split="train")
        
        # 3. Mix/Interleave
        # Probabilities: 80% Main, 20% Genius (Massive oversampling of Genius since it's small)
        ds_train = interleave_datasets([ds_main, ds_genius], probabilities=[0.8, 0.2], seed=42)
    else:
        logger.warning("⚠️ 'dataset/genius_dataset.jsonl' not found. Training on bulk only.")
        ds_train = ds_main

    # 4. Create Loader
    train_loader = DataLoader(ds_train, batch_size=CONFIG["batch_size"], collate_fn=collate_fn, shuffle=False, num_workers=0, pin_memory=True) # Shuffle False because interleave already shuffles

    ds_val = load_dataset("json", data_files="dataset/savant_dataset_val.jsonl", split="train")
    val_loader = DataLoader(ds_val, batch_size=CONFIG["batch_size"], collate_fn=collate_fn, shuffle=False, pin_memory=True)

    return train_loader, val_loader

# --- MAIN LOOP ---
def main():
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    model = UniversalTransformer(CONFIG).to(device).to(torch.bfloat16)
    
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG["lr"])
        logger.info("✅ Using 8-bit AdamW")
    except:
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
        logger.warning("⚠️ Using Standard AdamW")

    # --- SCHEDULER SETUP ---
    # Convert total steps to scheduler steps (since we step once per accumulation cycle)
    # However, standard practice is to step scheduler every optimizer step.
    # We step optimizer every 'accumulation' micro-batches.
    total_steps = MAX_STEPS 
    warmup_steps = int(0.05 * total_steps) # 5% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    logger.info(f"📅 Scheduler set: Cosine with {warmup_steps} warmup steps")

    start_step = 0
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/step_*.pt"), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    if checkpoints:
        latest = checkpoints[-1]
        logger.info(f"🔄 Resuming from: {latest}")
        ckpt = torch.load(latest)
        sd = ckpt['model_state_dict']
        for k in ["global_mask", "local_mask", "recurrent_block.attn.mask"]: 
            if k in sd: del sd[k]
        model.load_state_dict(sd, strict=False)
        try: 
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            logger.info("✅ Optimizer state loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load optimizer state: {e}")
            pass
        
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            logger.info("✅ Scheduler state loaded successfully.")
        except:
             logger.warning("⚠️ No scheduler state found in checkpoint (starting fresh).")
             pass
        start_step = ckpt['step']
    else:
        logger.info("🆕 Starting SAVANT PRODUCTION RUN")

    train_loader, val_loader = get_dataloaders(tokenizer)
    model.train()
    
    def cycle(dl):
        while True:
            for b in dl: yield b
    iter_train = cycle(train_loader)
    
    last_val_time = time.time()
    
    logger.info(f"\n🚀 Running until Step {MAX_STEPS}... (Press Ctrl+C to PAUSE)")
    
    try:
        optimizer.zero_grad()
        accumulated_loss = 0
        
        for step in range(start_step, MAX_STEPS + 1):
            
            # 1. ACCUMULATION PHASE
            for _ in range(CONFIG["accumulation"]):
                try:
                    batch = next(iter_train).to(device)
                except StopIteration:
                    iter_train = cycle(train_loader)
                    batch = next(iter_train).to(device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(batch)
                    loss = F.cross_entropy(
                        logits[..., :-1, :].contiguous().view(-1, CONFIG["vocab_size"]),
                        batch[..., 1:].contiguous().view(-1),
                        ignore_index=0
                    )
                    loss = loss / CONFIG["accumulation"]

                if torch.isnan(loss):
                    logger.error("❌ NaN detected. Skipping micro-batch.")
                    continue

                loss.backward()
                accumulated_loss += loss.item()

            # 2. UPDATE PHASE
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            current_lr = scheduler.get_last_lr()[0]
            if step % 10 == 0:
                logger.info(f"Step {step} | Train Loss: {accumulated_loss:.4f} | LR: {current_lr:.2e}")
            
            accumulated_loss = 0

            # 3. VALIDATION PHASE (GPU)
            if step > 0 and step % VALIDATION_EVERY_STEPS == 0:
                logger.info("\n--- 🔍 VALIDATING ON GPU ---")
                current_time = time.time()
                time_diff_min = (current_time - last_val_time) / 60.0
                last_val_time = current_time 
                logger.info(f"⏱️ Time since last validation: {time_diff_min:.2f} minutes")

                model.eval()
                val_loss_accum = 0
                val_batches = 10 
                iter_val = iter(val_loader)
                
                logger.info(f"⏳ Running GPU Validation ({val_batches} batches)...")
                with torch.no_grad():
                    for _ in range(val_batches):
                        try:
                            v_batch = next(iter_val).to(device)
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                v_logits = model(v_batch)
                                v_loss = F.cross_entropy(
                                    v_logits[..., :-1, :].contiguous().view(-1, CONFIG["vocab_size"]),
                                    v_batch[..., 1:].contiguous().view(-1),
                                    ignore_index=0
                                )
                                val_loss_accum += v_loss.item()
                        except StopIteration:
                            break
                
                avg_val_loss = val_loss_accum / val_batches
                logger.info(f"📉 Validation Loss: {avg_val_loss:.4f}")
                
                logger.info("📝 Testing Logic (GPU Inference - CoT):")
                prompt = "Question: If X = 5 and Y = 10, what is X + Y?\nAnswer: Let's think step by step:"
                inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
                out = model.generate(inp, max_new_tokens=300, temperature=0.6)
                logger.info(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}\n")
                
                model.train()

                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, f"{CHECKPOINT_DIR}/step_{step}.pt")
                
                all_ckpt = sorted(glob.glob(f"{CHECKPOINT_DIR}/step_*.pt"))
                if len(all_ckpt) > 3: os.remove(all_ckpt[0])

    except KeyboardInterrupt:
        logger.info("\n\n🛑 PAUSE/STOP SIGNAL RECEIVED (Ctrl+C)")
        logger.info("💾 Saving emergency checkpoint...")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, f"{CHECKPOINT_DIR}/step_{step}.pt")
        logger.info(f"✅ Saved to {CHECKPOINT_DIR}/step_{step}.pt")
        sys.exit(0)

    logger.info("\n🏁 SAVANT TRAINING COMPLETE.")

if __name__ == "__main__":
    main()