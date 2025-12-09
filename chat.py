import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import glob
import os
import math

# --- 1. CONFIGURATION (MUST MATCH YOUR TRAINING RUN) ---
CONFIG = {
    "vocab_size": 32000,
    "real_vocab_size": 30000, # Added missing key for ALBERT tokenizer limit
    "dim": 1024,              # Production Size
    "n_heads": 16,
    "head_dim": 64,
    "n_recurrent_loops": 12,  # Production Depth
    "max_seq_len": 1024,
    "window_size": 256,
    "soft_cap": 50.0,
    "dropout": 0.0,           # Disabled for inference
    "batch_size": 1
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. ARCHITECTURE (COPY OF PRODUCTION CLASSES) ---
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
        # Handle mask device mismatch during inference
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

# --- 3. GENERATION LOGIC (TUNED) ---
def load_latest_model():
    # Find latest checkpoint in production folder
    checkpoints = sorted(glob.glob("checkpoints_savant/step_*.pt"), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    if not checkpoints:
        print("❌ No checkpoints found in checkpoints_savant/")
        return None, None
    
    latest = checkpoints[-1]
    print(f"🧠 Loading Brain: {latest}")
    
    model = UniversalTransformer(CONFIG).to(DEVICE).to(torch.bfloat16)
    
    try:
        checkpoint = torch.load(latest, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        
        # Clean ephemeral keys
        for k in ["global_mask", "local_mask", "recurrent_block.attn.mask"]: 
            if k in state_dict: del state_dict[k]
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, checkpoint['step']
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return None, None

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.6, repetition_penalty=1.0):
    # Prepare Input
    # Manual construction to avoid trailing [SEP]
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.cls_token_id] + input_ids # Add [CLS] manually
    input_ids = torch.tensor([input_ids]).to(DEVICE)
    
    print(f"\n📝 Input IDs: {input_ids.tolist()}")
    print("🤖 Stream: ", end="", flush=True)
    
    # Store generated tokens to check for repetition
    generated_tokens = []

    for _ in range(max_new_tokens):
        cond_ids = input_ids[:, -CONFIG["max_seq_len"]:]
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(cond_ids)
        
        next_token_logits = logits[:, -1, :]
        
        # --- FIX 1: BAN GHOST TOKENS (Indices > 30000) ---
        next_token_logits[:, CONFIG["real_vocab_size"]:] = float('-inf')

        # --- FIX 2: BAN SPECIAL TOKENS (The Silence Killers) ---
        for ban_id in [0, 1]: # Allow CLS (2) and SEP (3)
            next_token_logits[:, ban_id] = float('-inf')

        # --- FIX 3: REPETITION PENALTY (Relaxed to 1.0 default) ---
        if repetition_penalty != 1.0:
            for token in set(generated_tokens[-10:]): # Look back 10 tokens
                 next_token_logits[:, token] /= repetition_penalty

        # Temperature
        next_token_logits = next_token_logits / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_tokens.append(next_token.item())
        
        token_id = next_token.item()
        word = tokenizer.decode([token_id], skip_special_tokens=True)
        
        if not word:
            pass # Skip empty (special) tokens
        else:
            print(word, end="", flush=True)
            
        if token_id == 3: # SEP/EOS
            break

    print("\n✅ Done!")
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# --- 4. INTERACTIVE LOOP ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    model, step = load_latest_model()
    
    if model:
        print(f"--- SAVANT INTERFACE (Step {step}) ---")
        print("Type 'exit' to quit.")
        print("Try these prompts to test IQ:")
        print("1. 'Once upon a time,' (Grammar Test)")
        print("2. 'Question: If X = 2, what is X + 2? Answer:' (Logic Test)")
        print("3. 'def add(a, b):' (Code Test)")
        print("---------------------------------------")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]: break
            
            # Format input like training data to give it a hint
            if "def" in user_input:
                formatted = f"Programming Task:\nWrite a python function.\n\nSolution:\n{user_input}"
            else:
                 formatted = f"Question: {user_input}\nAnswer:"
                
            generate(model, tokenizer, formatted)