import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from hgt_model import HierarchicalGravityTransformer

def train(epochs):
    # 1. Data Preparation (Toy Character-level LM)
    text = "hello world! this is a hierarchical gravity transformer proof of concept."
    chars = sorted(list(set(text)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    data = [char_to_ix[ch] for ch in text]
    x = torch.tensor(data[:-1]).unsqueeze(0) # [1, L-1]
    y = torch.tensor(data[1:]).unsqueeze(0)  # [1, L-1]
    
    # 2. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = 128
    coord_dim = 16
    num_layers = 4
    num_heads = 8
    mlp_dim = 256
    log_interval = 100
    generate_len = 200
    
    model = HierarchicalGravityTransformer(
        num_tokens=vocab_size,
        hidden_dim=hidden_dim,
        coord_dim=coord_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        max_seq_len=128
    )
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    x = x.to(device)
    y = y.to(device)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Create Causal Mask [B, 1, L, L]
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)
        
        logits, stats = model(x, mask=mask, return_stats=True) # [1, L-1, vocab_size]
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % log_interval == 0:
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                f"gamma: {stats['gamma_mean'].item():.4f}, "
                f"dist: {stats['dist_mean'].item():.4f}, "
                f"energy: {stats['energy_mean'].item():.4f}, "
                f"entropy: {stats['entropy_mean'].item():.4f}"
            )
            
    # 4. Inference
    model.eval()
    with torch.no_grad():
        max_seq_len = model.coord_emb.num_embeddings
        test_input = x[:, :5] # "hello"
        generated = test_input.tolist()[0]
        for _ in range(generate_len):
            inp = torch.tensor(generated[-max_seq_len:], device=device).unsqueeze(0)
            seq_len = inp.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
            logits = model(inp, mask=mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token)
            
        print("Generated text:", "".join([ix_to_char[i] for i in generated]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HGT on toy data.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="Number of training epochs.",
    )
    args = parser.parse_args()
    train(args.epochs)
