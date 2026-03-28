import os
from tokenizer import BPETokenizer

# -------- PATH SETUP --------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT_DIR, "data", "TinyStoriesV2-GPT4-train.txt")
OUTPUT_PATH = os.path.join(ROOT_DIR, "outputs", "tokenizer", "tinystories_tokenizer.json")

VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Training tokenizer...")
    print(f"Data path: {DATA_PATH}")

    tokenizer = BPETokenizer.train(
        files=[DATA_PATH],
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.save(OUTPUT_PATH)

    print("Done!")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()