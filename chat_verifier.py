import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import glob
import os
import math

# --- 1. CONFIGURATION ---
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
    "batch_size": 1
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. ARCHITECTURE (Must match training) ---
# [PASTE YOUR ARCHITECTURE CLASSES HERE: RMSNorm, SwiGLU, LiquidGatedAttention, LiquidRecurrentBlock, UniversalTransformer]
# (I am omitting them to save space, but you MUST copy them from train_savant_production.py)

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
            x = self.recurrent_block(x, current_mask)
        return self.head(self.ln_f(x))

# --- 3. LOADER ---
def load_latest_model():
    checkpoints = sorted(glob.glob("checkpoints_savant/step_*.pt"), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    if not checkpoints: return None, None
    latest = checkpoints[-1]
    print(f"🧠 Loading Brain: {latest}")
    model = UniversalTransformer(CONFIG).to(DEVICE).to(torch.bfloat16)
    try:
        checkpoint = torch.load(latest, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        for k in ["global_mask", "local_mask", "recurrent_block.attn.mask"]: 
            if k in state_dict: del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except: return None

# --- 4. VERIFIER GENERATION LOGIC ---
@torch.no_grad()
def generate_step(model, tokenizer, input_ids, max_new=50, temp=0.6):
    for _ in range(max_new):
        cond_ids = input_ids[:, -CONFIG["max_seq_len"]:]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(cond_ids)
        
        next_token_logits = logits[:, -1, :]
        next_token_logits[:, CONFIG["real_vocab_size"]:] = float('-inf') # Ban Ghosts
        for ban_id in [0, 1, 2, 3]: next_token_logits[:, ban_id] = float('-inf') # Ban Specials

        probs = F.softmax(next_token_logits / temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Stop at newline to control the flow
        # if next_token.item() == 13: break 

    return input_ids

def solve_with_verification(model, tokenizer, question):
    print(f"\n📝 Question: {question}")
    
    # STEP 1: DRAFTING
    print("🤖 Thinking (Drafting)... ", end="", flush=True)
    prompt = f"Question: {question}\nAnswer: Let's think step by step:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate Draft
    input_ids = generate_step(model, tokenizer, input_ids, max_new=60, temp=0.7)
    draft_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"\n   [Draft]: {draft_text.split('Answer:')[-1].strip()}")

    # STEP 2: VERIFICATION TRIGGER
    # We force the model to critique itself
    print("🕵️ Verifying... ", end="", flush=True)
    critic_trigger = "\nWait, let me double check that calculation. The correct answer is:"
    critic_ids = tokenizer.encode(critic_trigger, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    input_ids = torch.cat([input_ids, critic_ids], dim=1)

    # STEP 3: FINAL ANSWER
    input_ids = generate_step(model, tokenizer, input_ids, max_new=20, temp=0.4) # Lower temp for precision
    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # Extract just the last part
    final_answer = final_text.split("The correct answer is:")[-1].strip()
    print(f"\n✅ [Final Logic]: {final_answer}")

# --- MAIN ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    model = load_latest_model()
    
    if model:
        print("--- LOGIC VERIFIER MODE ---")
        while True:
            q = input("\nInput Math Question: ")
            if q.lower() in ["exit", "quit"]: break
            solve_with_verification(model, tokenizer, q)