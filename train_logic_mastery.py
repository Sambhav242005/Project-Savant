import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import re
import random

# --- CONFIGURATION ---
CONFIG = {
    "vocab_size": 32000,
    "real_vocab_size": 30000,
    "dim": 1024,
    "n_heads": 16,
    "head_dim": 64,
    "n_recurrent_loops": 12,
    "max_seq_len": 1024,
    "window_size": 256,
    "soft_cap": 50.0,
    "dropout": 0.0, 
    "batch_size": 1,
    "lr": 2e-4,          # <--- AGGRESSIVE START (Was 1e-4)
    "accumulation": 8    # <--- NEW: Simulate Batch Size 8
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ARCHITECTURE (Same as Production) ---
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
        if mask is not None: logits = logits + mask.to(logits.dtype)
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
            if self.training and x.device.type == 'cuda':
                x = torch.utils.checkpoint.checkpoint(self.recurrent_block, x, current_mask, use_reentrant=False)
            else:
                x = self.recurrent_block(x, current_mask)
        return self.head(self.ln_f(x))

# --- IMPROVED MASTERY LOGIC ---
def extract_number(text):
    # Strip everything except numbers
    text = text.lower().strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums: return nums[-1]
    return None

def train_on_concept(model, tokenizer, optimizer, question, solution, target_answer):
    print(f"\n🎓 CONCEPT: {question}")
    print(f"   Target: {target_answer}")
    
    attempts = 0
    max_attempts = 100 
    
    train_text = f"Question: {question}\nAnswer: {solution}"
    prompt = f"Question: {question}\nAnswer:"
    
    train_ids = tokenizer.encode(train_text, return_tensors="pt").to(DEVICE)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    while attempts < max_attempts:
        attempts += 1
        
        # 1. TEST (Generation)
        # We need to see if it learned it.
        model.eval()
        gen_ids = prompt_ids.clone()
        generated_tokens = [] 
        
        for _ in range(30): 
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(gen_ids[:, -CONFIG["max_seq_len"]:])
                
                next_logits = logits[:, -1, :]
                 # Repetition Penalty
                for t in set(generated_tokens):
                    next_logits[:, t] /= 1.2
                
                next_logits[:, 0] = float('-inf') 
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                if next_token.item() == tokenizer.eos_token_id: break
                if "\n" in tokenizer.decode(next_token[0]): break 
        
        full_output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        answer_part = full_output[len(prompt):].strip()
        model_ans = extract_number(answer_part)
        
        # Check correctness
        if model_ans == target_answer:
            print(f"✅ MASTERED in {attempts} steps! Output: {model_ans}")
            return True

        # 2. TRAIN (With Accumulation)
        model.train()
        
        # Loop 8 times to simulate a larger batch
        accumulated_loss = 0
        optimizer.zero_grad()
        
        for _ in range(CONFIG["accumulation"]):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(train_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = train_ids[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, CONFIG["vocab_size"]), 
                    shift_labels.view(-1),
                    ignore_index=0
                )
                loss = loss / CONFIG["accumulation"] # Normalize
            
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # PRINT THE LOSS so you know if it's High or Low
        print(f"   Step {attempts}: Loss {accumulated_loss:.4f} | Got '{model_ans}'", end="\r")

    print(f"\n❌ Failed to master.")
    return False

# --- MAIN ---
def main():
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    
    import glob
    
    # Priority: mastery_latest.pt -> step_*.pt -> Random
    mastery_checkpoint = "checkpoints_savant/mastery_latest.pt"
    
    if os.path.exists(mastery_checkpoint):
        print(f"🧠 Loading MASTERY checkpoint: {mastery_checkpoint}")
        sd_path = mastery_checkpoint
    else:
        checkpoints = sorted(glob.glob("checkpoints_savant/step_*.pt"), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"🧠 Loading LATEST STEP checkpoint: {latest}")
            sd_path = latest
        else:
            print("⚠️ No checkpoints found. Initializing random weights.")
            sd_path = None

    model = UniversalTransformer(CONFIG).to(DEVICE).to(torch.bfloat16)
    
    if sd_path:
        ckpt = torch.load(sd_path, map_location=DEVICE)
        sd = ckpt['model_state_dict']
        for k in ["global_mask", "local_mask", "recurrent_block.attn.mask"]: 
            if k in sd: del sd[k]
        model.load_state_dict(sd, strict=False)

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG["lr"])
    except:
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    # Curriculum
    curriculum = [
        ("1 + 1", "2", "2"),
        ("2 + 2", "4", "4"),
        ("5 + 5", "10", "10"),
        ("10 - 5", "5", "5"),
        ("3 * 3", "9", "9"),
        ("If X = 5, what is X + 2?", "7", "7"),
    ]

    print("\n🚀 STARTING MASTERY LOOP (With Repetition Penalty)")
    for q, soln, ans in curriculum:
        train_on_concept(model, tokenizer, optimizer, q, soln, ans)
        
        # Save
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': 999999
        }, f"checkpoints_savant/mastery_latest.pt")

if __name__ == "__main__":
    main()