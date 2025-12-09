from datasets import load_dataset, concatenate_datasets
import json
import os

# --- CONFIGURATION ---
TARGET_SIZE = 1_000_000 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"🚀 Starting Data Preparation in: {BASE_DIR}")
os.makedirs(os.path.join(BASE_DIR, "dataset"), exist_ok=True)

# 1. DOWNLOAD COSMOPEDIA (The Textbooks)
print("📚 Downloading Cosmopedia (Textbooks)...")
try:
    ds_textbook = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train[:20%]", trust_remote_code=True)
    ds_textbook = ds_textbook.select_columns(["text"])
    ds_textbook.to_json(os.path.join(BASE_DIR, "dataset", "source_textbooks.jsonl"))
except Exception as e:
    print(f"⚠️ Textbook download skipped or failed: {e}")

# 2. DOWNLOAD MATH (The Logic) - FIXED TYPO
print("🧮 Downloading Orca-Math (Logic)...")
try:
    ds_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    
    def format_math(example):
        return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}
    
    ds_math = ds_math.map(format_math).select_columns(["text"])
    ds_math.to_json(os.path.join(BASE_DIR, "dataset", "source_math.jsonl"))
    print("✅ Orca-Math Downloaded.")

except Exception as e:
    print(f"⚠️ Orca-Math failed: {e}")
    print("🔄 Trying Backup: MetaMathQA...")
    ds_math = load_dataset("meta-math/MetaMathQA", split="train[:100000]")
    def format_meta(example):
        return {"text": f"Question: {example['query']}\nAnswer: {example['response']}"}
    ds_math = ds_math.map(format_meta).select_columns(["text"])
    ds_math.to_json(os.path.join(BASE_DIR, "dataset", "source_math.jsonl"))

# 3. DOWNLOAD CODE (The Syntax)
print("💻 Downloading TinyCodes...")
try:
    ds_code = load_dataset("nampdn-ai/tiny-codes", split="train[:50000]")
    def format_code(example):
        return {"text": f"Programming Task:\n{example['prompt']}\n\nSolution:\n{example['response']}"}
    ds_code = ds_code.map(format_code).select_columns(["text"])
    ds_code.to_json(os.path.join(BASE_DIR, "dataset", "source_code.jsonl"))
    print("✅ TinyCodes Downloaded.")
except Exception as e:
    print(f"⚠️ TinyCodes failed: {e}")
    print("🔄 Trying Backup: Python Code Instructions...")
    try:
        ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        def format_code_backup(example):
            return {"text": f"Programming Task:\n{example['instruction']}\n\nSolution:\n{example['output']}"}
        ds_code = ds_code.map(format_code_backup).select_columns(["text"])
        ds_code.to_json(os.path.join(BASE_DIR, "dataset", "source_code.jsonl"))
        print("✅ Backup Code Dataset Downloaded.")
    except Exception as e2:
        print(f"❌ Backup Code Dataset failed too: {e2}")

# 4. MIX THEM
print("🌪️ Mixing Datasets...")
data_files = []
if os.path.exists(os.path.join(BASE_DIR, "dataset", "source_textbooks.jsonl")):
    data_files.append(load_dataset("json", data_files=os.path.join(BASE_DIR, "dataset", "source_textbooks.jsonl"), split="train"))
if os.path.exists(os.path.join(BASE_DIR, "dataset", "source_math.jsonl")):
    data_files.append(load_dataset("json", data_files=os.path.join(BASE_DIR, "dataset", "source_math.jsonl"), split="train"))
if os.path.exists(os.path.join(BASE_DIR, "dataset", "source_code.jsonl")):
    data_files.append(load_dataset("json", data_files=os.path.join(BASE_DIR, "dataset", "source_code.jsonl"), split="train"))

if not data_files:
    print("❌ No datasets found! Exiting.")
    exit(1)

# Calculate limits based on available datasets
datasets_to_concat = []
total_files = len(data_files)
print(f"ℹ️ Found {total_files} datasets to mix.")

for ds in data_files:
    # Take a portion to not exceed TARGET_SIZE too much if we have huge datasets
    # For now, let's just take up to TARGET_SIZE // total_files from each to be safe and balanced
    limit = TARGET_SIZE // total_files
    actual_limit = min(limit, len(ds))
    datasets_to_concat.append(ds.take(actual_limit))

final_dataset = concatenate_datasets(datasets_to_concat)

# Shuffle
final_dataset = final_dataset.shuffle(seed=42)

# 5. SPLIT & SAVE (TRAIN/VAL)
print("✂️ Splitting into Train (95%) and Validation (5%)...")
split_dataset = final_dataset.train_test_split(test_size=0.05)
train_ds = split_dataset['train']
val_ds = split_dataset['test']

train_path = os.path.join(BASE_DIR, "dataset", "savant_dataset_train.jsonl")
val_path = os.path.join(BASE_DIR, "dataset", "savant_dataset_val.jsonl")

print(f"💾 Saving Train ({len(train_ds)}) to '{train_path}'...")
train_ds.to_json(train_path)

print(f"💾 Saving Validation ({len(val_ds)}) to '{val_path}'...")
val_ds.to_json(val_path)

print("✅ Done! Point your training script to these files.")