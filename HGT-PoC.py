import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GravityAttention(nn.Module):
    """
    Hierarchical Gravity Transformer (HGT)를 위한 Gravity Attention 모듈.
    
    기존의 Dot-product Attention을 거리 기반의 중력 커널(Distance-based Gravity Kernel)로 대체합니다.
    입력으로 Hidden States와 Latent Coordinates를 동시에 받습니다.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        coord_dim: int, 
        num_heads: int, 
        head_coord_dim: int = 16,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim (int): 입력 특징 벡터의 차원 (d_model)
            coord_dim (int): 입력 잠재 좌표의 차원 (z_dim)
            num_heads (int): 멀티 헤드 개수
            head_coord_dim (int): 각 헤드 내부에서 투영될 좌표의 차원 (기본값: 16)
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.num_heads = num_heads
        self.head_coord_dim = head_coord_dim
        
        # Value 벡터의 차원 (일반적으로 hidden_dim / num_heads)
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # --- Projections ---
        # 1. Value Projection (특징 벡터 h -> Value v)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 2. Coordinate Projection for Attention (좌표 z -> Head별 좌표 z_head)
        # 각 헤드가 서로 다른 '관점'에서 거리를 잴 수 있도록 좌표를 투영합니다.
        self.coord_proj_attn = nn.Linear(coord_dim, num_heads * head_coord_dim)
        
        # 3. Coordinate Evolution (다음 레이어로 전달할 좌표 업데이트)
        # HGT의 계층적 특성을 위해 좌표 자체도 진화시킵니다.
        self.coord_proj_next = nn.Linear(coord_dim, coord_dim)

        # 4. Output Projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # --- Gravity Parameters ---
        # Learnable Temperature/Gravity constant (gamma)
        # 각 헤드마다 독립적인 중력 상수를 가집니다.
        # 초기값은 학습 안정을 위해 적절한 스케일로 설정합니다.
        self.gamma = nn.Parameter(torch.randn(1, num_heads, 1, 1))
        self.softplus = nn.Softplus() # gamma가 항상 양수가 되도록 강제

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, coordinates, mask=None):
        """
        Args:
            hidden_states (Tensor): [Batch, SeqLen, HiddenDim] - 특징 벡터 h
            coordinates (Tensor): [Batch, SeqLen, CoordDim] - 잠재 좌표 z
            mask (Tensor, optional): [Batch, 1, 1, SeqLen] or [Batch, 1, SeqLen, SeqLen]
            
        Returns:
            updated_hidden (Tensor): [Batch, SeqLen, HiddenDim]
            updated_coords (Tensor): [Batch, SeqLen, CoordDim]
        """
        batch_size, seq_len, _ = hidden_states.size()

        # ----------------------------------------------------------------------
        # 1. Prepare Streams
        # ----------------------------------------------------------------------
        
        # (A) Value Projection: [B, L, H, D_v] -> [B, H, L, D_v]
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.transpose(1, 2)

        # (B) Coordinate Projection for Attention: [B, L, H * D_z_head]
        z_heads = self.coord_proj_attn(coordinates)
        # Reshape to [B, H, L, D_z_head]
        z_heads = z_heads.view(batch_size, seq_len, self.num_heads, self.head_coord_dim)
        z_heads = z_heads.transpose(1, 2)

        # ----------------------------------------------------------------------
        # 2. Vectorized Distance Calculation (Gravity Kernel)
        # ----------------------------------------------------------------------
        # 목표: 모든 i, j 쌍에 대해 ||z_i - z_j||^2 계산
        # 수식: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        # 이 방식은 torch.cdist보다 메모리 효율적일 수 있으며, sqrt(0) 역전파 문제를 방지합니다.
        
        # z_sq: [B, H, L, 1] - 각 토큰 좌표의 제곱합
        z_sq = torch.sum(z_heads ** 2, dim=-1, keepdim=True)
        
        # squared_dist: [B, H, L, L]
        # (L, 1) + (1, L) - 2 * (L, D) @ (D, L)
        squared_dist = z_sq + z_sq.transpose(-1, -2) - 2 * torch.matmul(z_heads, z_heads.transpose(-1, -2))
        
        # 수치적 안정성을 위해 음수 값(부동소수점 오차)을 0으로 클램핑
        squared_dist = torch.clamp(squared_dist, min=0.0)

        # ----------------------------------------------------------------------
        # 3. Calculate Gravity Score
        # ----------------------------------------------------------------------
        # w_{ij} = exp(-gamma * dist^2)
        # Softplus를 사용하여 gamma가 항상 양수임을 보장 (중력은 인력이므로)
        gamma = self.softplus(self.gamma) 
        
        # Log-space calculation for stability:
        # Attention Score = -gamma * squared_dist
        attn_scores = -gamma * squared_dist

        if mask is not None:
            # 마스킹된 위치에 매우 작은 값을 더해 Softmax 후 0이 되도록 함
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # ----------------------------------------------------------------------
        # 4. Weighted Sum (Attention)
        # ----------------------------------------------------------------------
        # Softmax를 통해 확률 분포로 변환 (일반적인 어텐션 메커니즘 유지)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted Sum: [B, H, L, L] @ [B, H, L, D_v] -> [B, H, L, D_v]
        attn_output = torch.matmul(attn_weights, value)

        # ----------------------------------------------------------------------
        # 5. Finalize Outputs
        # ----------------------------------------------------------------------
        # (A) Hidden States Update
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        updated_hidden = self.out_proj(attn_output)

        # (B) Coordinate Evolution
        # 다음 레이어를 위해 좌표를 투영 (Residual connection은 외부 블록에서 처리 권장)
        updated_coords = self.coord_proj_next(coordinates)

        return updated_hidden, updated_coords

# --- Example Usage ---
if __name__ == "__main__":
    # Hyperparameters
    B, L, D, Z_dim, H = 2, 10, 64, 4, 8
    
    # Model
    model = GravityAttention(hidden_dim=D, coord_dim=Z_dim, num_heads=H)
    
    # Dummy Inputs
    h = torch.randn(B, L, D) # Hidden states
    z = torch.randn(B, L, Z_dim) # Latent coordinates
    
    # Forward
    h_new, z_new = model(h, z)
    
    print(f"Input Hidden: {h.shape}, Input Coords: {z.shape}")
    print(f"Output Hidden: {h_new.shape}, Output Coords: {z_new.shape}")
    print("Gravity Attention Forward Pass Successful!")
