# Hierarchical Gravity Transformer (HGT) Project

This project aims to implement and experiment with the Hierarchical Gravity Transformer, a novel transformer architecture that replaces dot-product attention with distance-based gravity kernels in a latent coordinate space.

## Architecture Overview

- **Gravity Attention**: Uses Euclidean distance between latent coordinates to determine attention scores.
- **Latent Coordinates**: Each token has an associated position in a latent space which evolves through layers.
- **Hierarchical Structure**: Potential for multi-scale representation by manipulating the latent space.

## Roadmap

- [x] Initial Proof of Concept for Gravity Attention (`HGT-PoC.py`)
- [x] Complete HGT Block implementation (`hgt_model.py`)
- [x] Full Model architecture (`hgt_model.py`)
- [x] Testing and Validation (`test_hgt.py`)
- [x] Simple Training Demo (`train.py`)
- [ ] Performance Benchmarking
