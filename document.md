# Hierarchical Gravity Transformer (HGT): Technical Reference & Implementation Guide

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Project:** HGT (Hierarchical Gravity Transformer)

---

## Table of Contents

### Part I: Theoretical Foundations
1.  **Introduction**
    *   1.1 The Limitations of Dot-Product Attention
    *   1.2 The Gravity Metaphor: Information as Mass and Distance
    *   1.3 Hierarchical Latent Spaces
2.  **Mathematical Formulation**
    *   2.1 Latent Coordinates ($z$) & Evolution
    *   2.2 Mass Embeddings ($m$)
    *   2.3 Gravity Attention Kernel ($1/r^\alpha$)
    *   2.4 Physics-Informed Regularization (Repulsion)

### Part II: Architecture & Implementation
3.  **Model Overview (`hgt_model.py`)**
    *   3.1 High-Level Architecture Diagram
    *   3.2 The HGT Block Structure
    *   3.3 Embedding Layers: Token, Mass, and Coordinates
4.  **Deep Dive: Gravity Attention Mechanism**
    *   4.1 The `GravityAttention` Class
    *   4.2 Coordinate Projection & Distance Calculation
    *   4.3 Computing Gravity Scores (Attention Weights)
    *   4.4 Coordinate Evolution Logic
5.  **Training Dynamics (`train_shakespeare.py`)**
    *   5.1 The Objective Function: Cross-Entropy + Repulsion
    *   5.2 Implementing `compute_repulsion_loss`
    *   5.3 Numerical Stability & Gradient Clipping
    *   5.4 Checkpoint Management (`_best` vs `_last`)
    *   5.5 Visualization Hooks
    *   5.6 Probe Word Tracking

### Part III: Data & Input Processing
6.  **Data Pipeline (`prepare_data.py`)**
    *   6.1 Tokenization Strategy (Character-level vs BPE)
    *   6.2 Vocabulary Construction
    *   6.3 Batch Generation & Causal Masking

### Part IV: Inference & Analysis
7.  **Generation & Chat (`chat.py`)**
    *   7.1 Autoregressive Generation Loop
    *   7.2 Temperature & Top-k Sampling
    *   7.3 Legacy Compatibility Layer
8.  **Visualization & Interpretability (`visualize_evolution.py`)**
    *   8.1 Semantic Clustering in Gravity Space
    *   8.2 Probe Words & Trajectory Tracking
    *   8.3 TensorBoard Integration

### Part V: Appendix
*   A. Hyperparameter Reference
*   B. Directory Structure
*   C. Troubleshooting & Common Runtime Errors

---

# Part I: Theoretical Foundations

## 1. Introduction
Transformers excel at sequence modeling, but the dot-product kernel conflates similarity and scale, making it difficult to impose geometric structure on tokens. HGT replaces $QK^T$ with a distance-aware interaction where tokens occupy positions in a latent gravity space. Tokens exert forces on each other based on learned mass and distance, encouraging semantically related tokens to cluster while preventing collapse through an explicit repulsion prior.

### 1.1 The Limitations of Dot-Product Attention
- Dot products reward magnitude; tokens with large norms dominate even if directions misalign.
- There is no explicit notion of distance, so interpretability of spatial relations is weak.
- Numerical stability hinges on softmax temperature scaling rather than physically meaningful constraints.

### 1.2 The Gravity Metaphor: Information as Mass and Distance
We model each token as a particle with learned mass $m$ and coordinates $z \in \mathbb{R}^{d_z}$. Attention is driven by an energy term that scales with both mass and separation. This framing creates a smooth inductive bias: nearby, massive tokens influence more; distant tokens attenuate.

### 1.3 Hierarchical Latent Spaces
Coordinates are propagated layer by layer, mirroring how semantics refine through depth. Each block updates both hidden states and coordinates, letting geometry co-evolve with meaning.

## 2. Mathematical Formulation
### 2.1 Latent Coordinates ($z$) & Evolution
Each layer outputs updated coordinates $z^{(\ell+1)} = \mathcal{E}(h^{(\ell)}, z^{(\ell)})$ via a learned projection (`coord_proj_next`). Residual updates plus layer norm keep coordinate scales stable.

### 2.2 Mass Embeddings ($m$)
`mass_emb` maps tokens to scalar logits that pass through Softplus to ensure $m > 0$. At inference, legacy checkpoints without this head are initialized to a neutral mass of 1.0.

### 2.3 Gravity Attention Kernel
Instead of the standard attention mechanism:
$$ \text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

HGT utilizes a distance-based gravity kernel:
$$ \text{Energy}_{ij} = - \gamma \cdot \frac{m_i m_j}{|z_i - z_j|^\alpha} $$
In practice (`GravityAttention.forward`):
- Coordinates are projected per head to $z_h \in \mathbb{R}^{d_{coord}^{head}}$.
- Pairwise squared distances are computed with a vectorized expansion:
  $$ \|z_i - z_j\|^2 = \|z_i\|^2 + \|z_j\|^2 - 2 z_i^\top z_j $$
- A learnable, positive $\gamma$ (Softplus of a parameter) scales the negative distance to produce attention scores.
- Softmax over the causal mask yields weights that are renormalized to guard against mask-induced zero rows.

### 2.4 Physics-Informed Regularization (Repulsion)
To prevent particles from collapsing, a repulsion loss enforces finite separation:
$$ E_{rep} = \frac{1}{|P|} \sum_{i<j} \frac{m_i m_j}{\max(r_{ij}, \text{min\_dist})^\alpha} $$
Implementation notes:
- `torch.cdist` computes $r_{ij}$ in batch.
- Distances are clamped with `min_dist` to avoid NaNs at $r \to 0$.
- $m_i m_j$ is broadcast over pairs; pairs are masked with an upper-triangular mask to avoid self-interaction and double counting.
- The default $\alpha=2$ recovers a Coulomb-like potential; $\alpha=1$ is also supported.

---

# Part II: Architecture & Implementation

## 3. Model Overview
The `HierarchicalGravityTransformer` deviates from standard GPT architectures by maintaining two parallel streams of data:
1.  **Hidden States ($h$)**: The semantic content (standard transformer flow).
2.  **Latent Coordinates ($z$)**: The geometric positions of tokens in the "semantic universe".

### 3.1 High-Level Architecture Diagram
Input tokens -> Token Embedding -> HGT Blocks (each updates $h$ and $z$) -> LayerNorm -> LM Head. Coordinates are initialized from positional embeddings; masses from token IDs.

### 3.2 The HGT Block Structure
Each block contains:
- LayerNorm on hidden states.
- Gravity Attention (value projection + coordinate projection).
- Residual update on $h$.
- Feed-forward network with residual.
- Coordinate update through `coord_proj_next` and layer norm on $z$.

### 3.3 Embeddings
- **Token Embedding** (`token_emb`): Maps indices to $d_{model}$.
- **Mass Embedding** (`mass_emb`): Produces scalar logits; `Softplus` ensures $m>0$ and keeps gradients stable near zero.
- **Coordinate Embedding** (`coord_emb`): Provides initial $z_0$ from absolute positions up to `max_seq_len`.
- Outputs: `forward(..., return_last_coords=True)` returns `(logits, z, m)` for downstream visualization and repulsion.

## 4. Deep Dive: Gravity Attention Mechanism
### 4.1 The `GravityAttention` Class
Projects hidden states to values and coordinates to per-head coordinate frames. Maintains a learnable $\gamma$ shared across heads, constrained to be positive via Softplus.

### 4.2 Coordinate Projection & Distance Calculation
Vectorized Euclidean distance avoids explicit loops:
```python
z_sq = (z_heads ** 2).sum(-1, keepdim=True)
squared_dist = z_sq + z_sq.transpose(-1, -2) - 2 * z_heads @ z_heads.transpose(-1, -2)
squared_dist = squared_dist.clamp_min(0.0)
```
This costs $O(B \cdot H \cdot L^2 \cdot d_{coord}^{head})$ but is fully fused and GPU friendly.

### 4.3 Computing Gravity Scores (Attention Weights)
```python
gamma = softplus(self.gamma)         # ensure gamma > 0
attn_scores = -gamma * squared_dist  # negative distance => attractive
attn_scores = attn_scores.masked_fill(~mask, -1e9)  # causal
attn_weights = softmax(attn_scores, dim=-1)
attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True).clamp_min(1e-9)
```
Renormalization after masking prevents rows of zeros. Dropout is applied to weights before the value projection is aggregated.

### 4.4 Coordinate Evolution Logic
Coordinates are updated via `coord_proj_next(hidden_states)`, then added residually and normalized:
```python
z_next = coord_proj_next(h_norm)
z = coord_norm(z + z_next)
```
This allows coordinates to follow the semantic drift of hidden states while staying bounded.

## 5. Training Dynamics (`train_shakespeare.py`)
### 5.1 The Objective Function: Cross-Entropy + Repulsion
The total loss is `task_loss + lambda_repulsion * rep_loss`, where `task_loss` is standard next-token cross entropy and `rep_loss` enforces geometric dispersion.

### 5.2 Implementing `compute_repulsion_loss`
- Accepts batched coordinates `z` (`[B, L, C]`) and optional masses `m` (`[B, L, 1]` or `[B, L]`).
- Ensures batched input by unsqueezing 2D tensors.
- Uses `torch.cdist` for pairwise $r_{ij}$, clamps with `min_dist`, raises to $\alpha$, multiplies by $m_i m_j$, and averages unique pairs (`torch.triu` mask).
- Returns zero if no pairs exist (short sequences).

### 5.3 Numerical Stability & Gradient Clipping
- Distances are clamped from below; $\alpha$ defaults to 2.0.
- Gradients are clipped to `grad_clip` (default 1.0) before optimizer steps.
- Softplus on masses and $\gamma$ ensures positivity without hard constraints.

### 5.4 Checkpoint Management (`_best` vs `_last`)
- Training tracks the best validation loss; the best model is saved to `{checkpoint_path}_best.pt`.
- The final state after all steps is saved to `{checkpoint_path}_last.pt`.
- Resume order prioritizes `_last.pt`, then `_best.pt`, then the legacy base path for backward compatibility.
- Checkpoint payload includes model/optimizer state, iteration, best validation score, config, and vocab.

### 5.5 Visualization Hooks
- TensorBoard (`SummaryWriter`) logs `loss/train`, `loss/val`, and `loss/repulsion` every `eval_interval`.
- Embedding Projector: every `vis_interval`, latent coordinates `z` are flattened and logged with character metadata for interactive inspection.

### 5.6 Probe Word Tracking
`PROBE_WORDS` defines semantic groups (royalty, family, antonyms). During visualization intervals, matching spans in the batch are recorded with their coordinates and masses and appended to `checkpoints/gravity_evolution.pkl` for downstream animations.

---

# Part III: Data & Input Processing

## 6. Data Pipeline (`prepare_data.py`)
- Source: TinyShakespeare (`input.txt`) downloaded on demand; existing files are reused.
- Vocabulary: character-level; `build_vocab` sorts unique chars to form `stoi`/`itos`.
- Splits: 90/10 train/validation by contiguous slicing for stable evaluation.
- Batching: `get_batch` draws random windows of length `block_size`, creating paired inputs/targets shifted by one token. Causal masks are built once per sequence length.

---

# Part IV: Inference & Analysis

## 7. Generation & Chat (`chat.py`)
### 7.1 Autoregressive Generation Loop
Greedy temperature-scaled sampling with optional top-k filtering:
- Input is cropped to `block_size` to honor positional limits.
- Causal mask is rebuilt per step.
- Samples one token at a time and appends to the running context.

### 7.2 Temperature & Top-k Sampling
Temperature scales logits before softmax; top-k zeroes out all but the largest $k$ logits to sharpen distributions.

### 7.3 Legacy Compatibility Layer
- `strict=False` loads tolerate missing `mass_emb` weights or extra legacy keys.
- Legacy coordinate projection: detects checkpoints where `coord_proj_next` expects `[coord_dim, coord_dim]` and swaps in compatible layers before loading.
- Missing mass embeddings: initializes weights so that `Softplus(mass_emb)` yields mass ~1.0.
- Fallback token: space (`" "`) if unseen characters appear at inference time.

## 8. Visualization & Interpretability (`visualize_evolution.py`)
### 8.1 Semantic Clustering in Gravity Space
Uses the saved probe history to study how specific word groups move through training. PCA is fit once on all recorded coordinates to preserve temporal consistency.

### 8.2 Probe Words & Trajectory Tracking
`record_probe_snapshot` scans each batch for probe spans, storing:
- group and word
- character-level positions
- coordinates and masses
- training step
These snapshots accumulate in `gravity_evolution.pkl`.

### 8.3 TensorBoard Integration
- Scalars: train loss, validation loss, and repulsion loss.
- Embeddings: latent coordinates with token metadata, enabling the Projector tab for 2D/3D exploration.

---

# Part V: Appendix

## A. Hyperparameter Reference
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Dimension of the semantic hidden state |
| `coord_dim` | 32 | Dimension of the latent coordinate space |
| `lambda_repulsion` | 0.05 | Weight of the physics-based regularization term |
| `alpha` | 2.0 | Repulsion exponent (1.0 for linear decay) |
| `min_dist` | 1e-3 | Distance clamp to avoid $r \to 0$ blow-up |
| `block_size` | 256 | Context length for training and inference |
| `eval_interval` | 100 | Steps between validation checks |
| `vis_interval` | `eval_interval` | Steps between TensorBoard embeddings and probe logging |
| `grad_clip` | 1.0 | Global norm clip to stabilize updates |

## B. Directory Structure
- `train_shakespeare.py`: training loop, logging, checkpoints, probe recording.
- `hgt_model.py`: model definition (embeddings, blocks, gravity attention).
- `chat.py`: inference CLI with legacy checkpoint support.
- `prepare_data.py`: data download and validation.
- `visualize_evolution.py`: PCA projection and animation of probe trajectories.
- `test_hgt.py`: unit tests for attention, blocks, model signatures, and repulsion stability.

## C. Troubleshooting & Common Runtime Errors
- **Sequence too long**: Inputs longer than `max_seq_len` are truncated in inference; in training, lengths must not exceed `block_size`.
- **NaNs in repulsion**: Ensure `min_dist` is positive; current implementation clamps distances before exponentiation and handles zero-pair batches.
- **Checkpoint mismatch**: If loading older checkpoints, rely on the built-in compatibility layer (coord projection swap, neutral mass init, `strict=False`).
- **Empty probe logs**: Probe words are uppercase; ensure vocab contains the needed characters and `vis_interval` is set.
