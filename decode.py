import os
import torch

from tokenizer import BPETokenizer
from model import TransformerLM, load_checkpoint, softmax


ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(ROOT, "outputs", "tokenizer", "tinystories_tokenizer.json")
CHECKPOINT_PATH = os.path.join(ROOT, "outputs", "checkpoints", "ckpt_final.pt")

# must match the model that produced CHECKPOINT_PATH!!
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
ROPE_THETA = 10000.0
ATTENTION_IMPL = "naive"  # edit this: "naive" | "triton"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)

    keep = cdf <= top_p

    # include the first token where cumulative probability crosses top_p.
    cutoff = keep.sum(dim=-1, keepdim=True)
    cutoff = torch.clamp(cutoff, max=probs.shape[-1] - 1)
    keep.scatter_(-1, cutoff, True)
    keep[..., 0] = True  # always keep at least one token

    filtered = sorted_probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)

    sampled_in_sorted = torch.multinomial(filtered, num_samples=1)
    sampled_token = torch.gather(sorted_idx, -1, sampled_in_sorted)
    return sampled_token.squeeze(-1)


@torch.no_grad()
def decode(
    model: TransformerLM,
    prompt_ids,
    theta: float,
    max_new_tokens: int,
    eot_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.context_length :]
        logits = model(x_cond, theta)[:, -1, :]  # (1, vocab)

        if temperature != 1.0:
            logits = logits / temperature

        probs = softmax(logits, dim=-1)

        if top_p < 1.0:
            next_token = sample_top_p(probs, top_p).unsqueeze(0)  # (1, 1)
        else:
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        x = torch.cat([x, next_token], dim=1)

        if next_token.item() == eot_token_id:
            break

    return x[0].tolist()


def build_model(vocab_size: int) -> TransformerLM:
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        attention_impl=ATTENTION_IMPL,
    ).to(DEVICE)


def main():
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    model = build_model(tokenizer.vocab_size)
    print(f"Device: {DEVICE} | Attention impl: {ATTENTION_IMPL}")
    load_checkpoint(CHECKPOINT_PATH, model)
    model.eval()

    eot_ids = tokenizer.encode("<|endoftext|>")
    if len(eot_ids) != 1:
        raise ValueError("Expected <|endoftext|> to map to exactly one token")
    eot_token_id = eot_ids[0]

    prompt = "friends"
    prompt_ids = tokenizer.encode(prompt)

    out_ids = decode(
        model=model,
        prompt_ids=prompt_ids,
        theta=ROPE_THETA,
        max_new_tokens=200,
        eot_token_id=eot_token_id,
        temperature=0.9,
        top_p=0.9,
    )

    print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
