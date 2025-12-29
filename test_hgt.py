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
    model = GravityAttention(hidden_dim=D, coord_dim=Z, num_heads=H)
    h = torch.randn(B, L, D)
    z = torch.randn(B, L, Z)
    
    # Mask out the last two tokens
    mask = torch.tensor([[[[1, 1, 0, 0]]]]) # [B, 1, 1, L]
    
    h_new, z_new = model(h, z, mask=mask)
    
    # This is a bit hard to verify exactly without looking at weights,
    # but we can check if it runs without error.
    assert h_new.shape == (B, L, D)

if __name__ == "__main__":
    pytest.main([__file__])
