# Small Transformer Language Model

Minimal decoder-only Transformer language model training on TinyStories.

Core Transformer pieces are implemented from scratch using PyTorch tensor operations and `nn.Module` wiring, rather than using high-level built-ins like `nn.Transformer`, `nn.MultiheadAttention`, `nn.Linear`, `nn.Embedding`, `nn.CrossEntropyLoss`, or `torch.optim.AdamW`.

## What This Project Includes

- BPE tokenizer and training script (`tokenizer/`)
- Transformer model + optimizer/loss/schedule (`model.py`)
- Training loop with memmap dataset caching and checkpointing (`train.py`)
- Autoregressive decoding with temperature and top-p sampling (`decode.py`)

This repo includes a pre-trained tokenizer JSON at:

- `outputs/tokenizer/tinystories_tokenizer.json`

So you can run training/decoding immediately without retraining the tokenizer.
If you want a tokenizer specialized for a different corpus, you can retrain it with `tokenizer/train_tokenizer.py`.

## Architecture Overview

Data flow per token sequence:

1. Token IDs -> `Embedding`
2. Repeated `TransformerBlock` (pre-norm style)
3. Final `RMSNorm`
4. Output projection (`lmhead`) -> logits over vocabulary

Each `TransformerBlock` contains:

- `RMSNorm` before attention
- Causal multi-head self-attention with RoPE (`MHA_with_RoPE`)
- Residual connection
- `RMSNorm` before FFN
- SwiGLU feed-forward network (`FFN_swiglu`)
- Residual connection

Core components implemented in `model.py`:

- `Linear`
- `Embedding`
- `RMSNorm`
- `RoPE`
- `MHA_with_RoPE`
- `FFN_swiglu`
- `TransformerLM`
- `CrossEntropy`
- `AdamW`
- `learning_rate_schedule` (warmup + cosine decay)
- `gradient_clipping`
- `save_checkpoint` / `load_checkpoint`

## Project Structure

- `model.py`: model and training primitives
- `train.py`: tokenization cache build + training
- `decode.py`: checkpoint loading + generation
- `tokenizer/bpe_tokenizer.py`: tokenizer wrapper
- `tokenizer/train_tokenizer.py`: tokenizer training entrypoint

## Environment

Tested on Apple Silicon (`mps`) and CPU.

Dependencies:

- `torch`
- `numpy`
- `tokenizers`

Example setup:

```bash
conda create -n llm python=3.12 -y
conda activate llm
pip install torch numpy tokenizers
```

## How To Run

### 1. Download TinyStories Data

Create the `data/` folder and download train/valid splits:

```bash
mkdir -p data
cd data
curl -L -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..
```

If you prefer `wget`, those commands work too.

### 2. Train Tokenizer (once)

```bash
python tokenizer/train_tokenizer.py
```

Creates:

- `outputs/tokenizer/tinystories_tokenizer.json`

If you are using the tokenizer JSON included in this repo, this step is optional.

### 3. Train Model

```bash
python train.py
```

By default, this project is configured for TinyStories.
You can train on your own text dataset by changing dataset paths in `train.py` (and optionally retraining tokenizer).

Default train config (`train.py`):

- `BATCH_SIZE=32`
- `CONTEXT_LENGTH=256`
- `NUM_LAYERS=4`
- `D_MODEL=256`
- `NUM_HEADS=8`
- `D_FF=1024`
- `MAX_STEPS=5000`

Token budget:

- `BATCH_SIZE * MAX_STEPS * CONTEXT_LENGTH`
- `32 * 5000 * 256 = 40,960,000` tokens

Notes:

- First run tokenizes dataset into memmap files.
- Later runs reuse cached tokenized files.
- Training prints device, parameter count, losses, and wallclock minutes.

### 4. Resume Training (optional)

Set in `train.py`:

```python
RESUME_PATH = "outputs/checkpoints/ckpt_step_XXXX.pt"
```

Then rerun `python train.py`.

### 5. Generate Text

```bash
python decode.py
```

`decode.py` loads:

- tokenizer from `outputs/tokenizer/tinystories_tokenizer.json`
- checkpoint from `outputs/checkpoints/ckpt_final.pt`

Decoding supports:

- context-window truncation (`last context_length tokens`)
- temperature scaling
- top-p (nucleus) sampling
- max generated token limit

## Checkpoints and Fresh Runs

- Checkpoints are stored in `outputs/checkpoints/`.
- If `RESUME_PATH=None` and old checkpoints exist, `train.py` archives them to:
  - `outputs/checkpoints_archive/run_YYYYMMDD_HHMMSS/`

This lets you start a fresh run without losing older models.

## Results Snapshot

From a recent TinyStories run on MPS:

- ~40.96M tokens processed
- Validation loss reached about `2.00` at step 5000

Text quality is generally coherent and improves with longer training and decoding parameter tuning.

