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

## Agent Logs

### [Codex] Code Review Report
**Date**: 2025-12-29
**Target**: Full Codebase (`hgt_model.py`, `train.py`)
**Context**: Hierarchical Gravity Transformer Research

#### Summary
The implementation generally aligns with the architectural goals defined in `GEMINI.md`. The **Gravity Attention** mechanism is correctly implemented using a Gaussian kernel structure (squared Euclidean distance in the exponent). Coordinate evolution follows a residual linear path, which is valid.

#### Key Findings

1.  **Critical Bug - Missing Causal Mask (`train.py`)**:
    *   **Issue**: The training loop in `train.py` was feeding the full sequence to the model without a causal (triangular) mask. Since `GravityAttention` computes pairwise distances for all tokens, the model could "see" future tokens (leakage), making autoregressive training invalid.
    *   **Severity**: **High**. The model would fail to generalize.
    *   **Action Taken**: **Fixed**. A lower-triangular mask was added to the training loop in `train.py`.

2.  **Architecture Verification**:
    *   `GravityAttention`: Correctly implements $Attention(Q, K, V) 
propto 
exp(-
 gamma ||z_i - z_j||^2)$.
    *   `Gamma Initialization`: Initialized to 0 (resulting in $
approx 0.69$ after Softplus), which is a safe starting point.
    *   `Coordinate Evolution`: Implemented as $z_{l+1} = 
text{Norm}(z_l + 
text{Linear}(z_l))$, preserving the geometric structure while allowing adaptation.

3.  **Recommendations**:
    *   **Positional Encoding**: Currently, absolute positions are mapped to initial coordinates. Future experiments could explore relative initialization or learnable initial coordinates without position dependence to fully leverage the "gravity" concept.
    *   **Benchmarking**: As noted in the Roadmap, performance benchmarking against standard Dot-Product Attention is the next logical step.

#### Status
**Review Passed**. Critical issues resolved. Codebase is ready for the next phase of research.
