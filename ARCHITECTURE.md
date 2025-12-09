# Model Architecture

This document describes the design and choices behind the custom "Savant" Language Model.

## Overview

The model is a **Universal Transformer** (Recurrent Transformer) designed for efficiency and improved reasoning capabilities through parameter sharing and adaptive computation.

## Core Components

### 1. Universal Transformer (Recurrent)

Unlike standard Transformers that have $N$ distinct layers, this model uses a single **Recurrent Block** that is applied iteratively (looped) for `n_recurrent_loops` (12) times.

- **Benefit**: Significantly reduces parameter count (weights are tied) while maintaining deep computation depth.
- **Analogy**: It's like "thinking" about the input 12 times using the same brain cells, rather than passing it through 12 different brains.

### 2. Liquid Gating

Integrating concepts from Liquid Neural Networks, the model uses a gating mechanism to control information flow.

- **Mechanism**: `alpha = sigmoid(Gate(x))`
- **Update Rule**: `Output = (1 - alpha) * Old + alpha * New`
- **Effect**: Allows the model to adaptively preserve or update its state at each recurrent step, potentially stabilizing training in deep recurrent networks.

### 3. Attention Mechanism

- **Soft Capping**: Logits are tanh-capped (limit: 50.0) before Softmax to prevent instability and gradient explosion.
- **Dynamic Masking**:
  - **Global Mask**: Full attention context.
  - **Local Mask**: Constrained to a window (size 256).
  - **Strategy**: Cycles between Local and Global masks (Global every 4th step) to balance efficiency with long-range dependency modeling.

### 4. Feed-Forward Network (FFN)

- **SwiGLU**: Uses Swish-Gated Linear Units for better performance than standard ReLU/GELU.
- **RMSNorm**: Root Mean Square Layer Normalization is used for stability.

## Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Vocab Size** | 32,000 | Uses ALBERT tokenizer |
| **Dim** | 1024 | Hidden state dimension |
| **Heads** | 16 | Number of attention heads |
| **Head Dim** | 64 | Dimension per head |
| **Loops** | 12 | Number of recurrent steps |
| **Window** | 256 | Local attention window size |

## Training Strategy

### Optimization

- **Optimizer**: AdamW (8-bit via `bitsandbytes` if available for memory efficiency).
- **Precision**: `bfloat16` Mixed Precision.
- **Scheduler**: Cosine Decay with 5% Warmup.
- **Gradient Accumulation**: Accumulates gradients over 16 micro-batches to simulate a larger effective batch size.

### Data Mixing (The "Savant" Strategy)

The model uses a specific data mixing strategy to balance general knowledge with logic reasoning:

1. **Savant Dataset (Training)**: The bulk corpus (Textbooks, Code, etc.).
2. **Genius Dataset (Logic)**: A high-quality subset of pure logic/math problems.
3. **Interleaving**: The loader mixes these with a ratio of **80% Savant / 20% Genius**. This over-sampling of logic is intended to boost reasoning skills ("Cram School" effect).
