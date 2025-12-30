import torch
import torch.nn as nn
import torch.nn.functional as F

class GravityAttention(nn.Module):
    """
    Hierarchical Gravity Transformer (HGT) Gravity Attention.
    """
    def __init__(
        self, 
        hidden_dim: int, 
        coord_dim: int, 
        num_heads: int, 
        head_coord_dim: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.num_heads = num_heads
        self.head_coord_dim = head_coord_dim
        
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.coord_proj_attn = nn.Linear(coord_dim, num_heads * head_coord_dim)
        self.coord_proj_next = nn.Linear(hidden_dim, coord_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.gamma = nn.Parameter(torch.zeros(1, num_heads, 1, 1)) # Initialize with 0 -> Softplus(0) = log(2)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, coordinates, mask=None, return_stats=False):
        batch_size, seq_len, _ = hidden_states.size()

        # Value Projection
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.transpose(1, 2)

        # Coordinate Projection for Attention
        z_heads = self.coord_proj_attn(coordinates)
        z_heads = z_heads.view(batch_size, seq_len, self.num_heads, self.head_coord_dim)
        z_heads = z_heads.transpose(1, 2)

        # Vectorized Distance Calculation
        z_sq = torch.sum(z_heads ** 2, dim=-1, keepdim=True)
        squared_dist = z_sq + z_sq.transpose(-1, -2) - 2 * torch.matmul(z_heads, z_heads.transpose(-1, -2))
        squared_dist = torch.clamp(squared_dist, min=0.0)

        # Calculate Gravity Score
        gamma = self.softplus(self.gamma) 
        attn_scores = -gamma * squared_dist

        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

        # Weighted Sum
        attn_weights = F.softmax(attn_scores, dim=-1)
        if mask is not None:
            attn_weights = attn_weights * mask
            denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            attn_weights = attn_weights / denom

        stats = None
        if return_stats:
            eps = 1e-9
            entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
            stats = {
                "gamma_mean": gamma.mean(),
                "dist_mean": squared_dist.mean(),
                "energy_mean": (gamma * squared_dist).mean(),
                "entropy_mean": entropy.mean(),
            }

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        # Finalize Outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        updated_hidden = self.out_proj(attn_output)

        # Coordinate Evolution
        updated_coords = self.coord_proj_next(hidden_states)

        if return_stats:
            return updated_hidden, updated_coords, stats
        return updated_hidden, updated_coords

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class HGTBlock(nn.Module):
    def __init__(self, hidden_dim, coord_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = GravityAttention(hidden_dim, coord_dim, num_heads, dropout=dropout)
        self.ffn = FeedForward(hidden_dim, mlp_dim, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Coordinate Evolution Norm (optional, but good for stability)
        self.coord_norm = nn.LayerNorm(coord_dim)

    def forward(self, h, z, mask=None, return_stats=False):
        # Attention + Residual
        if return_stats:
            h_attn, z_next, stats = self.attn(self.norm1(h), z, mask=mask, return_stats=True)
        else:
            h_attn, z_next = self.attn(self.norm1(h), z, mask=mask)
        h = h + h_attn
        
        # FFN + Residual
        h = h + self.ffn(self.norm2(h))
        
        # Coordinate Update (Residual + Norm)
        z = self.coord_norm(z + z_next)
        
        if return_stats:
            return h, z, stats
        return h, z

class HierarchicalGravityTransformer(nn.Module):
    def __init__(
        self, 
        num_tokens, 
        hidden_dim, 
        coord_dim, 
        num_layers, 
        num_heads, 
        mlp_dim, 
        max_seq_len=512, 
        dropout=0.1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, hidden_dim)
        # Coordinate embedding - can be learned or based on absolute positions
        self.coord_emb = nn.Embedding(max_seq_len, coord_dim)
        
        self.layers = nn.ModuleList([
            HGTBlock(hidden_dim, coord_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x, mask=None, return_stats=False):
        b, l = x.size()
        device = x.device
        
        # Initial states
        h = self.token_emb(x)
        
        # Initial coordinates based on position
        pos = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
        z = self.coord_emb(pos)
        
        stats_list = []
        for layer in self.layers:
            if return_stats:
                h, z, layer_stats = layer(h, z, mask=mask, return_stats=True)
                stats_list.append(layer_stats)
            else:
                h, z = layer(h, z, mask=mask)
            
        h = self.norm(h)
        logits = self.head(h)
        if return_stats:
            stack = {
                key: torch.stack([s[key] for s in stats_list]).mean()
                for key in stats_list[0].keys()
            }
            return logits, stack
        return logits

if __name__ == "__main__":
    # Test Full Model
    model = HierarchicalGravityTransformer(
        num_tokens=100, 
        hidden_dim=64, 
        coord_dim=16, 
        num_layers=4, 
        num_heads=8, 
        mlp_dim=256
    )
    
    x = torch.randint(0, 100, (2, 20))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print("HGT Model Forward Pass Successful!")
