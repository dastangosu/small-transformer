import os
import time
import shutil
import numpy as np
import torch

from tokenizer import BPETokenizer
from model import (
    TransformerLM,
    CrossEntropy,
    AdamW,
    learning_rate_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(ROOT, "outputs", "tokenizer", "tinystories_tokenizer.json")
TRAIN_TEXT_PATH = os.path.join(ROOT, "data", "TinyStoriesV2-GPT4-train.txt")
VALID_TEXT_PATH = os.path.join(ROOT, "data", "TinyStoriesV2-GPT4-valid.txt")
TOKENIZED_DIR = os.path.join(ROOT, "outputs", "tokenized")

TRAIN_BIN = os.path.join(TOKENIZED_DIR, "tinystories_train.bin")
TRAIN_LEN = os.path.join(TOKENIZED_DIR, "tinystories_train.len")
VALID_BIN = os.path.join(TOKENIZED_DIR, "tinystories_valid.bin")
VALID_LEN = os.path.join(TOKENIZED_DIR, "tinystories_valid.len")

CHECKPOINT_DIR = os.path.join(ROOT, "outputs", "checkpoints")
CHECKPOINT_ARCHIVE_DIR = os.path.join(ROOT, "outputs", "checkpoints_archive")

# hyperparameters
BATCH_SIZE = 32
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
ROPE_THETA = 10000.0
ATTENTION_IMPL = "triton"  # edit this: "naive" | "triton"

MAX_STEPS = 5000
EVAL_EVERY = 100
EVAL_STEPS = 20
CHECKPOINT_EVERY = 250

MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 200
CLIP_NORM = 1.0
WEIGHT_DECAY = 0.1
SEED = 42

# set to a checkpoint path to resume, or keep None to train from scratch.
RESUME_PATH = None

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _token_dtype(vocab_size: int):
    return np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32


def _count_tokens(text_path: str, tokenizer: BPETokenizer, eot_ids):
    total = 0
    line_count = 0
    print(f"Counting tokens: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            total += len(tokenizer.encode(line)) + len(eot_ids)
            if line_count % 100_000 == 0:
                print(f"  counted lines={line_count:,} | tokens~{total:,}")
    print(f"Done counting: lines={line_count:,} | total_tokens={total:,}")
    return total


def _write_tokens_to_memmap(text_path: str, bin_path: str, len_path: str, tokenizer: BPETokenizer):
    eot_ids = tokenizer.encode("<|endoftext|>")
    total = _count_tokens(text_path, tokenizer, eot_ids)
    dtype = _token_dtype(tokenizer.vocab_size)

    print(f"Writing token memmap: {bin_path}")
    arr = np.memmap(bin_path, dtype=dtype, mode="w+", shape=(total,))
    ptr = 0
    line_count = 0
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            ids = tokenizer.encode(line)
            if ids:
                n = len(ids)
                arr[ptr : ptr + n] = np.asarray(ids, dtype=dtype)
                ptr += n
            if eot_ids:
                n = len(eot_ids)
                arr[ptr : ptr + n] = np.asarray(eot_ids, dtype=dtype)
                ptr += n
            if line_count % 100_000 == 0:
                pct = 100.0 * ptr / total if total > 0 else 100.0
                print(f"  written lines={line_count:,} | tokens={ptr:,}/{total:,} ({pct:.1f}%)")
    arr.flush()
    print(f"Done writing tokens: {ptr:,}")

    with open(len_path, "w", encoding="utf-8") as f:
        f.write(str(total))


def _load_or_build_tokens(text_path: str, bin_path: str, len_path: str, tokenizer: BPETokenizer):
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    if not (os.path.exists(bin_path) and os.path.exists(len_path)):
        print(f"Tokenizing: {text_path}")
        _write_tokens_to_memmap(text_path, bin_path, len_path, tokenizer)
    else:
        print(f"Using cached tokenized data: {bin_path}")

    with open(len_path, "r", encoding="utf-8") as f:
        length = int(f.read().strip())

    dtype = _token_dtype(tokenizer.vocab_size)
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=(length,))


def get_batch(token_ids, batch_size: int, context_length: int, device: str):
    max_start = len(token_ids) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    x = np.stack([token_ids[i : i + context_length] for i in starts]).astype(np.int64)
    y = np.stack([token_ids[i + 1 : i + context_length + 1] for i in starts]).astype(np.int64)

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss(model, token_ids, theta: float):
    model.eval()
    losses = []
    for _ in range(EVAL_STEPS):
        x, y = get_batch(token_ids, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        logits = model(x, theta)
        loss = CrossEntropy(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # If starting a fresh run, archive old checkpoints to avoid overwriting
    if RESUME_PATH is None and os.path.isdir(CHECKPOINT_DIR):
        existing = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
        if existing:
            stamp = time.strftime("run_%Y%m%d_%H%M%S")
            archive_path = os.path.join(CHECKPOINT_ARCHIVE_DIR, stamp)
            os.makedirs(CHECKPOINT_ARCHIVE_DIR, exist_ok=True)
            shutil.move(CHECKPOINT_DIR, archive_path)
            print(f"Archived old checkpoints to: {archive_path}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_time = time.time()

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    train_ids = _load_or_build_tokens(TRAIN_TEXT_PATH, TRAIN_BIN, TRAIN_LEN, tokenizer)
    valid_ids = _load_or_build_tokens(VALID_TEXT_PATH, VALID_BIN, VALID_LEN, tokenizer)

    if len(train_ids) <= CONTEXT_LENGTH + 1 or len(valid_ids) <= CONTEXT_LENGTH + 1:
        raise ValueError("Tokenized dataset is smaller than context_length + 1")

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        attention_impl=ATTENTION_IMPL,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    start_step = 0

    if RESUME_PATH is not None:
        start_step = load_checkpoint(RESUME_PATH, model, optimizer) + 1
        print(f"Resumed from: {RESUME_PATH} (next step: {start_step})")

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Attention impl: {ATTENTION_IMPL}")
    print(f"Params: total={total_params:,} | trainable={trainable_params:,}")
    print(f"Train tokens: {len(train_ids):,} | Validation tokens: {len(valid_ids):,}")

    for step in range(start_step, MAX_STEPS):
        lr = learning_rate_schedule(step, MAX_LR, MIN_LR, WARMUP_STEPS, MAX_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)

        logits = model(x, ROPE_THETA)
        loss = CrossEntropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), CLIP_NORM)
        optimizer.step()

        if step % EVAL_EVERY == 0:
            train_loss = estimate_loss(model, train_ids, ROPE_THETA)
            valid_loss = estimate_loss(model, valid_ids, ROPE_THETA)
            elapsed_min = (time.time() - start_time) / 60.0
            print(
                f"step={step:5d} | time_min={elapsed_min:7.2f} | lr={lr:.6f} | "
                f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f}"
            )

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_step_{step}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            elapsed_min = (time.time() - start_time) / 60.0
            print(f"Saved checkpoint: {ckpt_path} | time_min={elapsed_min:7.2f}")

    final_train_loss = estimate_loss(model, train_ids, ROPE_THETA)
    final_valid_loss = estimate_loss(model, valid_ids, ROPE_THETA)
    elapsed_min = (time.time() - start_time) / 60.0
    final_lr = learning_rate_schedule(MAX_STEPS - 1, MAX_LR, MIN_LR, WARMUP_STEPS, MAX_STEPS)
    print(
        f"step={MAX_STEPS:5d} | time_min={elapsed_min:7.2f} | lr={final_lr:.6f} | "
        f"train_loss={final_train_loss:.4f} | valid_loss={final_valid_loss:.4f}"
    )

    final_ckpt = os.path.join(CHECKPOINT_DIR, "ckpt_final.pt")
    save_checkpoint(model, optimizer, MAX_STEPS, final_ckpt)
    elapsed_min = (time.time() - start_time) / 60.0
    print(f"Saved final checkpoint: {final_ckpt} | time_min={elapsed_min:7.2f}")


if __name__ == "__main__":
    main()
