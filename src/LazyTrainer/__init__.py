"""LazyTrainer package
This package provides a lightweight, memory-efficient alternative training pipeline
for Graph Neural Networks using PyTorch Geometric graphs stored on disk.

It re-uses the existing models and utilities in Clean_Code/GNN_Training but
introduces a streaming/lazy dataset so that graphs are read from disk only as
needed.
"""
