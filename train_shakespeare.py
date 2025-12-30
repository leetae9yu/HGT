import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from hgt_model import HierarchicalGravityTransformer
from prepare_data import ensure_data


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def build_causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)


def get_batch(data, block_size, batch_size, device):
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, mask, criterion, eval_iters):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x, mask=mask)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


DEFAULT_CONFIG = {
    "data_path": os.path.join("data", "input.txt"),
    "checkpoint_path": os.path.join("checkpoints", "shakespeare.pt"),
    "resume": False,
    "batch_size": 64,
    "block_size": 256,
    "max_steps": 5000,
    "eval_interval": 100,
    "eval_iters": 50,
    "learning_rate": 3e-4,
    "grad_clip": 1.0,
    "hidden_dim": 256,
    "coord_dim": 32,
    "num_layers": 6,
    "num_heads": 8,
    "mlp_dim": 1024,
    "dropout": 0.1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train HGT on TinyShakespeare.")
    parser.add_argument("--data-path", default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--checkpoint-path", default=DEFAULT_CONFIG["checkpoint_path"])
    parser.add_argument("--resume", action="store_true", default=DEFAULT_CONFIG["resume"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--block-size", type=int, default=DEFAULT_CONFIG["block_size"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_CONFIG["eval_interval"])
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_CONFIG["eval_iters"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--coord-dim", type=int, default=DEFAULT_CONFIG["coord_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--num-heads", type=int, default=DEFAULT_CONFIG["num_heads"])
    parser.add_argument("--mlp-dim", type=int, default=DEFAULT_CONFIG["mlp_dim"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = ensure_data(args.data_path)
    checkpoint_path = args.checkpoint_path
    resume = args.resume

    batch_size = args.batch_size
    block_size = args.block_size
    max_steps = args.max_steps
    eval_interval = args.eval_interval
    eval_iters = args.eval_iters
    learning_rate = args.learning_rate
    grad_clip = args.grad_clip

    hidden_dim = args.hidden_dim
    coord_dim = args.coord_dim
    num_layers = args.num_layers
    num_heads = args.num_heads
    mlp_dim = args.mlp_dim
    dropout = args.dropout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = read_text(data_path)
    _, stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    data = encode(text, stoi)

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    model = HierarchicalGravityTransformer(
        num_tokens=vocab_size,
        hidden_dim=hidden_dim,
        coord_dim=coord_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        max_seq_len=block_size,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    mask = build_causal_mask(block_size, device)

    start_step = 0
    best_val = float("inf")

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint.get("iter", 0)
        best_val = checkpoint.get("best_val", best_val)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    t0 = time.time()
    for step in range(start_step, max_steps):
        x, y = get_batch(train_data, block_size, batch_size, device)
        logits = model(x, mask=mask)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if (step + 1) % eval_interval == 0:
            train_loss = estimate_loss(
                model,
                train_data,
                block_size,
                batch_size,
                device,
                mask,
                criterion,
                eval_iters,
            )
            val_loss = estimate_loss(
                model,
                val_data,
                block_size,
                batch_size,
                device,
                mask,
                criterion,
                eval_iters,
            )
            elapsed = time.time() - t0
            print(
                f"step {step + 1}/{max_steps} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "iter": step + 1,
                        "best_val": best_val,
                        "config": {
                            "hidden_dim": hidden_dim,
                            "coord_dim": coord_dim,
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                            "mlp_dim": mlp_dim,
                            "max_seq_len": block_size,
                            "dropout": dropout,
                            "vocab_size": vocab_size,
                        },
                        "vocab": {"stoi": stoi, "itos": itos},
                    },
                    checkpoint_path,
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iter": max_steps,
            "best_val": best_val,
            "config": {
                "hidden_dim": hidden_dim,
                "coord_dim": coord_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "mlp_dim": mlp_dim,
                "max_seq_len": block_size,
                "dropout": dropout,
                "vocab_size": vocab_size,
            },
            "vocab": {"stoi": stoi, "itos": itos},
        },
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
