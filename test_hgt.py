import torch
import pytest
from hgt_model import GravityAttention, HGTBlock, HierarchicalGravityTransformer

def test_gravity_attention_forward():
    B, L, D, Z, H = 2, 16, 64, 8, 4
    model = GravityAttention(hidden_dim=D, coord_dim=Z, num_heads=H)
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    
    h_new, z_new = model(h, z)
    
    assert h_new.shape == (B, L, D)
    assert z_new.shape == (B, L, Z)

def test_hgt_block_forward():
    B, L, D, Z, H, M = 2, 16, 64, 8, 4, 128
    block = HGTBlock(hidden_dim=D, coord_dim=Z, num_heads=H, mlp_dim=M)
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    
    h_new, z_new = block(h, z)
    
    assert h_new.shape == (B, L, D)
    assert z_new.shape == (B, L, Z)

def test_hgt_model_forward():
    num_tokens = 100
    model = HierarchicalGravityTransformer(
        num_tokens=num_tokens, 
        hidden_dim=32, 
        coord_dim=8, 
        num_layers=2, 
        num_heads=4, 
        mlp_dim=64
    )
    
    x = torch.randint(0, num_tokens, (2, 10))
    logits = model(x)
    
    assert logits.shape == (2, 10, num_tokens)

def test_gravity_attention_masking():
    B, L, D, Z, H = 1, 4, 16, 4, 2
    model = GravityAttention(hidden_dim=D, coord_dim=Z, num_heads=H, dropout=0.0)
    model.eval()
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    
    # Causal mask to prevent attending to future tokens
    mask = torch.tril(torch.ones(L, L)).unsqueeze(0).unsqueeze(0)
    
    h_new, z_new, attn_weights = model(h, z, mask=mask, return_attn=True)
    future_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    assert attn_weights[..., future_mask].max().item() <= 1e-6
    assert h_new.shape == (B, L, D)

def test_repulsion_loss_stability():
    from train_shakespeare import compute_repulsion_loss
    # Case: Identical coordinates (r=0)
    B, L, Z = 2, 4, 8
    z = torch.zeros(B, L, Z)
    loss = compute_repulsion_loss(z, min_dist=1e-3)
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss > 0

def test_hgt_return_signature():
    num_tokens = 50
    model = HierarchicalGravityTransformer(
        num_tokens=num_tokens, 
        hidden_dim=16, 
        coord_dim=4, 
        num_layers=1, 
        num_heads=2, 
        mlp_dim=32
    )
    x = torch.randint(0, num_tokens, (1, 5))
    
    # Standard return
    logits = model(x)
    assert isinstance(logits, torch.Tensor)
    
    # Return last coords and mass
    logits, z, m = model(x, return_last_coords=True)
    assert logits.shape == (1, 5, num_tokens)
    assert z.shape == (1, 5, 4)
    assert m.shape == (1, 5, 1)
    assert (m >= 0).all()

if __name__ == "__main__":
    pytest.main([__file__])
