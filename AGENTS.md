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

## 🎯 Current Mission: Physics-Informed Regularization (Repulsion & Determinism)

**Objective**: Mitigate mode collapse and overfitting by replacing stochastic noise (Langevin Dynamics) with deterministic repulsive forces (Pauli Exclusion Principle).

**Context**:
- **Problem**: The HGT model exhibits signs of mode collapse (over-clustering) and overfitting (gap > 0.2) in the latent space.
- **Solution**: Remove the current random noise injection and implement a repulsive potential energy term to force particles to maintain distance, creating a more structured manifold.

### Detailed Assignments

#### **Phase 4: Code Refactoring & Logic Implementation**

**[Codex]**
1.  **Modify `hgt_model.py` (Remove Noise & Expose Coordinates)**:
    -   **Remove Noise Logic**:
        -   Remove `noise_scale` from `HGTBlock.__init__` and `HierarchicalGravityTransformer.__init__`.
        -   In `HGTBlock.forward`, remove `torch.randn_like` logic. Ensure coordinate update is deterministic: `z = self.coord_norm(z + z_next)`.
    -   **Expose Final Coordinates**:
        -   Update `HierarchicalGravityTransformer.forward` to accept `return_last_coords=False`.
        -   Return `(logits, z)` tuple when this flag is `True`.

2.  **Modify `train_shakespeare.py` (Add Repulsion Loss)**:
    -   **Implement Loss Function**:
        -   Create `compute_repulsion_loss(z)`: Calculate pairwise Euclidean distances (`torch.cdist`), apply identity mask, and compute potential energy ($E \propto 1/r$).
    -   **Update Training Loop**:
        -   Define `lambda_repulsion = 0.05` in `main()`.
        -   Call model with `return_last_coords=True`.
        -   Compute `rep_loss` and add to total `loss`: `loss = task_loss + lambda_repulsion * rep_loss`.
    -   **Update Evaluation**:
        -   Adjust `estimate_loss` to handle the new return tuple signature.

#### **Phase 5: Validation & Tuning**

**[Gemini]**
1.  **Execution**: Run the updated training script.
2.  **Monitoring**:
    -   Check if `Repulsion Loss` starts at a reasonable magnitude (not exploding).
    -   Verify that the Train/Val loss gap reduces compared to the previous noisy run.
3.  **Visual Check**: (Optional) Generate a 3D plot to confirm particles are spreading out.

---

## Agent Logs

### [Gemini] Plan Update
**Date**: 2025-12-30
**Action**: Initiating physics-informed regularization strategy.
**Rationale**: Previous noise-based approach led to unstable convergence. Switching to deterministic repulsion to enforce manifold structure directly.