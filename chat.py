import os

import torch
import torch.nn.functional as F

from hgt_model import HierarchicalGravityTransformer


def build_causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)


def encode(text, stoi, fallback_token):
    return [stoi.get(ch, fallback_token) for ch in text]


def decode(tokens, itos):
    return "".join(itos[i] for i in tokens)


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, device, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        mask = build_causal_mask(idx_cond.size(1), device)
        logits = model(idx_cond, mask=mask)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -1e9

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


def main():
    checkpoint_path = os.path.join("checkpoints", "shakespeare.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    stoi = vocab["stoi"]
    itos = vocab["itos"]
    vocab_size = config["vocab_size"]

    model = HierarchicalGravityTransformer(
        num_tokens=vocab_size,
        hidden_dim=config["hidden_dim"],
        coord_dim=config["coord_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    block_size = config["max_seq_len"]
    fallback_token = stoi.get(" ", 0)

    print("Loaded checkpoint. Type /quit to exit.")
    while True:
        prompt = input("prompt> ")
        if prompt.strip() in {"/quit", "/exit"}:
            break

        if prompt == "":
            prompt = "\n"

        encoded = encode(prompt, stoi, fallback_token)
        idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        out = generate(
            model,
            idx,
            max_new_tokens=400,
            block_size=block_size,
            device=device,
            temperature=0.9,
            top_k=40,
        )
        completion = decode(out[0].tolist(), itos)
        print(completion)


if __name__ == "__main__":
    main()
