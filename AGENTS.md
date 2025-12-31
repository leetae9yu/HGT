# Agents & Personas

This document defines the autonomous agents and personas operating within the HGT (Hierarchical Gravity Transformer) project.

## Active Agents

### 1. **Codex** (Codebase Investigator & Reviewer)
- **Role**: Responsible for deep code analysis, architectural review, and ensuring alignment with research goals.
- **Capabilities**:
  - Static Analysis
  - Logic Verification
  - Research Context Alignment
- **Status**: Active

### 2. **Gemini** (Lead Developer & Orchestrator)
- **Role**: Handles implementation, refactoring, and execution of tasks.
- **Capabilities**:
  - File Manipulation
  - Shell Execution
  - Test Generation
- **Status**: Active

---

## 🎯 Current Mission: Physics-Informed Regularization (Refinement & Stabilization)

**Objective**: Enhance the stability of the repulsion regularization and improve checkpoint management to prevent data loss.

### Detailed Assignments

#### **Phase 6: Logic Refinement & Stability (Mass-Based Repulsion)**

**[Codex]**
1.  **Refine Checkpoint Logic (`train_shakespeare.py`)**:
    -   **Goal**: Prevent "best model" overwrites by the final save.
    -   **Action**:
        -   Modify the saving block.
        -   Save best validation models to `"{checkpoint_path}_best.pt"`.
        -   Save the final model state to `"{checkpoint_path}_last.pt"`.
        -   Ensure `resume` logic prioritizes loading `_last.pt` if available, or falls back gracefully.

2.  **Stabilize & Enhance Repulsion Loss (`train_shakespeare.py`)**:
    -   **Goal**: Prevent numerical instability (NaN/Inf) at $r \to 0$ and introduce mass-based physics.
    -   **Action**:
        -   Update `compute_repulsion_loss(z, mass=None, alpha=2.0, min_dist=1e-3)`:
        -   **Numerical Stability**: Clamp distances: `dists = torch.clamp(dists, min=min_dist)`. Use $r^\alpha$ (default $\alpha=2$ for Coulomb-like, or $\alpha=1$) in the denominator.
        -   **Mass-Based**: Allow optional `mass` tensor. If `None`, assume $m=1$. Formula: $E = \sum_{i \neq j} \frac{m_i m_j}{r_{ij}^\alpha}$.
        -   **Efficiency**: Use `torch.cdist` and broadcasting for $m_i m_j$. Apply `torch.triu` or mask to avoid double counting and self-interaction efficiently.

#### **Phase 7: Testing & Validation**

**[Gemini]**
1.  **Unit Testing (`test_hgt.py`)**:
    -   **Repulsion Stability**: Create a test case `test_repulsion_loss_stability` passing identical coordinates ($r=0$) to ensure no NaNs are produced and the loss is clamped/bounded.
    -   **Return Signature**: Verify `HierarchicalGravityTransformer.forward(..., return_last_coords=True)` returns a valid `(logits, z)` tuple.    

2.  **Checkpoint Verification**:
    -   **Test**: Create a temporary test (or modify `test_hgt.py`) to run a dummy training loop for 2 steps, trigger a "best" save, then a "final" save, and assert that two distinct files (`_best.pt`, `_last.pt`) exist and are loadable.

    - **Sanity Check**:
    -   Run a short training session (`train_shakespeare.py --max-steps 50`) to verify loss scaling and logging output format.

#### **Phase 8: Real-time Visualization & Monitoring**

**[Gemini]**
1.  **Setup Environment**:
    -   Ensure `tensorboard` is available (usually included with PyTorch/Colab).
    -   Remove `matplotlib` from visualization requirements if not needed for other plots.

2.  **Integrate TensorBoard (`train_shakespeare.py`)**:
    -   **Goal**: Monitor training metrics and visualize high-dimensional latent space evolution using the TensorBoard Projector.
    -   **Action**:
        -   Import `from torch.utils.tensorboard import SummaryWriter`.
        -   Initialize `writer = SummaryWriter(log_dir="runs/hgt_experiment")`.
        -   **Scalar Logging**: Log `train_loss`, `val_loss`, and `rep_loss` every `eval_interval`.
        -   **Embedding Visualization**:
            -   In the training loop (every `vis_interval`):
            -   Extract a batch of latent coordinates `z` (shape `[B, T, C]`) and tokens.
            -   Flatten `z` to `[B*T, C]` and tokens to `[B*T]`.
            -   Use `writer.add_embedding(mat=z, metadata=tokens_text, global_step=step)` to visualize the gravity space.
            -   *Note*: This enables the "Projector" tab in TensorBoard for interactive 3D analysis.


---

## Agent Logs

### [Gemini] Plan Update
**Date**: 2025-12-31
**Action**: Updated plan to include Checkpoint Safety and Mass-Based Repulsion Stability.
**Rationale**: Previous naive $1/r$ repulsion risks numerical explosion. Separate checkpoint files ensure the best model is preserved during long training runs.
