# Models Project

This repository contains scripts for training and using a custom language model, along with dataset preparation tools.

## Installation

### 1. Install PyTorch with CUDA

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the command for your system. Common example for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 2. Install Core Dependencies

```bash
pip install transformers datasets accelerate bitsandbytes
```

> [!NOTE]
> `bitsandbytes` is required for 8-bit optimizer support found in `train_main.py`.
> Windows users might need a patched version. Try:
>
> ```bash
> pip install bitsandbytes-windows
> ```

## Usage

### Training

To start the main training loop:

```bash
python train_main.py
```

### Chatting

To interact with the trained model:

```bash
python chat.py
```

Or for the verifier chat:

```bash
python chat_verifier.py
```

### Dataset Preparation

To prepare and mix the datasets:

```bash
python prepare_dataset.py
```

## Architecture

The model is a **Universal (Recurrent) Transformer** designed for efficiency and reasoning density.

- **Looping**: Reuses the same "brain" (layers) 12 times per token.
- **Liquid Gating**: Uses adaptive gating to stabilize deep recurrence (inspired by Liquid Neural Networks).
- **Data Mixing**: Interleaves general knowledge ("Textbooks") with high-intensity logic ("Orca-Math") to boost reasoning.

For full technical details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## File Descriptions

- **`train_main.py`**: The primary script for training. See [Training Guide](TRAINING.md).
- **`ARCHITECTURE.md`**: Technical details of the model.
- **`TRAINING.md`**: Guide to hyperparameters and training workflow.
- **`chat.py`**: A CLI interface for chatting with the trained model.
- **`chat_verifier.py`**: A specialized chat interface, likely for verification or testing specific model behaviors.
- **`prepare_dataset.py`**: Utility script to download, process, and mix datasets (e.g., Savant, Genius, etc.) into `jsonl` format.
- **`train_logic_mastery.py`**: Contains advanced training logic, possibly for "mastery" learning or specific curriculum modifications.
- **`create_genius_dataset.py`**: Dedicated script for creating the "genius" dataset.
- **`train.py`**: Alternative or legacy training script.
- **`.gitignore`**: Specifies files to be ignored by git (e.g., large datasets, checkpoints, temporary files).

## Data

The project expects or generates large `.jsonl` dataset files (e.g., `savant_dataset_train.jsonl`) which are ignored by git.
