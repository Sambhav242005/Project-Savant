# Training Guide

This document details the training limits, safety mechanisms, and operational workflows defined in `train_main.py`.

## Core Hyperparameters

| Parameter | Value | Reason |
| :--- | :--- | :--- |
| **Batch Size** | 12 | Tuned for 10GB VRAM stability. |
| **Accumulation** | 16 | Effective Batch Size = $12 \times 16 = 192$. Allows large-batch stability on small hardware. |
| **Learning Rate** | 2e-5 | Conservative rate for fine-tuning/stability. |
| **Max Steps** | 80,000 | Total training duration. |
| **Warmup** | 5% | 4,000 steps to prevent early divergence. |

## Hardware & Safety

### VRAM Management

- **Hard Limit**: 10.0 GB (`VRAM_LIMIT_GB`).
- The script sets `torch.cuda.set_per_process_memory_fraction` to strictly enforce this, preventing system freezes allowing OS changes to remain responsive.

### Stability Mechanisms

- **NaN Detection**: If `loss` becomes `NaN` (Not a Number) during a micro-batch, that specific batch is **skipped** entirely to prevent polluting the weights.
- **Gradient Clipping**: Norms are clipped at `1.0` to prevent exploding gradients.
- **Soft Capping**: (Mentioned in Architecture) Logits are capped at 50.0 to stabilize attention.

## Workflow

### 1. The Loop

The training loop runs continuously until `MAX_STEPS` or manual interruption.

- **Data**: Iterator cycles infinitely over the dataset.
- **Precision**: Uses `bfloat16` for everything (weights and computation) via `amp.autocast`.

### 2. Validation (Every 500 Steps)

Every 500 steps, the model pauses training to:

1. **Calculate Loss**: Runs 10 batches on the validation set (`savant_dataset_val.jsonl`).
2. **Logic Test (CoT)**: Generates a sample response to a logic prompt ("If X=5 and Y=10...") to visually confirm reasoning capabilities.
3. **Logs**: Prints time, loss, and the generated output to the console and `training_log.txt`.

### 3. Checkpointing

- **Routine**: Saves `step_X.pt` every 500 steps (integrated with validation).
- **Rotation**: Keeps only the **3 most recent** checkpoints to save disk space.
- **Emergency Save**: Captures `Ctrl+C` (KeyboardInterrupt) to save state immediately before exiting, ensuring no progress is lost.

## Resuming

The script automatically detects existing `step_*.pt` files in `checkpoints_savant/`.

- It loads the **latest** checkpoint (by step number).
- Restores Model Weights, Optimizer State, and Scheduler State.
- If the optimizer/scheduler fail to load (e.g., changed architecture), it proceeds with a fresh optimizer but keeps the weights.
