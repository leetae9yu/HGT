import torch
import torch.nn as nn
import torch.optim as optim
from hgt_model import HierarchicalGravityTransformer

def train():
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
    hidden_dim = 128
    coord_dim = 16
    num_layers = 4
    num_heads = 8
    mlp_dim = 256
    
    model = HierarchicalGravityTransformer(
        num_tokens=vocab_size,
        hidden_dim=hidden_dim,
        coord_dim=coord_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        max_seq_len=128
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(x) # [1, L-1, vocab_size]
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
            
    # 4. Inference
    model.eval()
    with torch.no_grad():
        test_input = x[:, :5] # "hello"
        generated = test_input.tolist()[0]
        for _ in range(20):
            inp = torch.tensor(generated).unsqueeze(0)
            logits = model(inp)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token)
            
        print("Generated text:", "".join([ix_to_char[i] for i in generated]))

if __name__ == "__main__":
    train()
