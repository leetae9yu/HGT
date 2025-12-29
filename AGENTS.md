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

## 🎯 Current Mission: TinyShakespeare Integration & Automation

**Objective**: Establish a robust, automated training pipeline for the HGT model using the **TinyShakespeare** dataset.

**Context - TinyShakespeare Dataset**:
- **Description**: A concatenation of Shakespeare's plays, commonly used for character-level language modeling benchmarks.
- **Source**: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- **Specs**: ~1.1MB file size, approx. 40,000 lines.
- **Expected Vocabulary**: ~65 unique characters (alphanumeric + punctuation).
- **Goal**: The HGT model should learn to generate Shakespeare-like text by training on this data at the character level.

### Detailed Assignments

#### **Phase 1: Data Acquisition & Inspection**

**[Gemini]**
1.  **Implement `prepare_data.py`**:
    -   Create a script to download `input.txt` from the source URL.
    -   Save it to a local `data/` directory (create if not exists).
    -   Print basic success messages.

**[Codex]**
1.  **Dataset Analysis**:
    -   Once downloaded, analyze `data/input.txt`.
    -   **Task**: Calculate and report:
        -   Total file size (bytes).
        -   Total character count.
        -   Unique vocabulary size (ensure it matches expectation of ~65).
        -   Top 5 most frequent characters.
    -   **Output**: Confirm data integrity before training begins.

#### **Phase 2: Pipeline Integration & Configuration**

**[Gemini]**
1.  **Update `train_shakespeare.py`**:
    -   Import/Call `prepare_data.py` to ensure data exists before training.
    -   **Refactor**: Move hardcoded hyperparameters (e.g., `batch_size`, `block_size`, `learning_rate`) into a clearly defined configuration block or use `argparse`.
    -   **Dynamic Vocab**: Ensure the model's `num_tokens` is set dynamically based on the actual file's vocabulary, not a hardcoded number.

**[Codex]**
1.  **Hyperparameter Review**:
    -   Based on the **Dataset Analysis**, review the configuration in `train_shakespeare.py`.
    -   **Check**: Is `max_seq_len` (block size) appropriate for the average line length of Shakespeare? (Usually ~256 is good).
    -   **Check**: Are `hidden_dim` and `num_layers` sufficient for the dataset complexity without overfitting immediately?

#### **Phase 3: Execution & Verification**

**[Gemini]**
1.  **Run Training**: Execute the pipeline.
2.  **Monitor**: Observe the first 100 iterations.
3.  **Artifact Check**: Ensure `checkpoints/shakespeare.pt` is created.

---

## Agent Logs

### [Gemini] Plan Update
**Date**: 2025-12-29
**Action**: Detailed planning for TinyShakespeare integration.
**Rationale**: Added specific analysis steps for Codex to ensure the model configuration matches the physical properties of the downloaded dataset (vocabulary, length).

### [Codex] Code Review Report
**Date**: 2025-12-29
**Target**: Full Codebase (`hgt_model.py`, `train.py`)
**Context**: Hierarchical Gravity Transformer Research

#### Summary
The implementation generally aligns with the architectural goals defined in `GEMINI.md`. The **Gravity Attention** mechanism is correctly implemented using a Gaussian kernel structure.

#### Key Findings
1.  **Resolved**: Critical Bug - Missing Causal Mask in `train.py`. Fixed and verified.
2.  **Verified**: Gravity Attention mechanism $\propto \exp(-\gamma ||z_i - z_j||^2)$ is correct.
3.  **Status**: Ready for data integration.