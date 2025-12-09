from datasets import load_dataset
import os

# --- CONFIG ---
# We only want 5,000 high-quality logic examples
TARGET_SIZE = 5000 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "dataset"), exist_ok=True)

print("🧠 Creating 'Genius Baby' Dataset...")

# 1. Download Orca-Math (Pure Logic)
ds_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train")

# 2. Format it
def format_math(example):
    return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}

# 3. Take the 'Golden Subset'
# We shuffle first to get a random mix, then take 5k
ds_math = ds_math.shuffle(seed=42).select(range(TARGET_SIZE))
ds_math = ds_math.map(format_math).select_columns(["text"])

# 4. Save
output_path = os.path.join(BASE_DIR, "dataset", "genius_dataset.jsonl")
ds_math.to_json(output_path)

print(f"✅ Created {output_path} with {len(ds_math)} logic samples.")
print("This is your 'Cram School' dataset.")
