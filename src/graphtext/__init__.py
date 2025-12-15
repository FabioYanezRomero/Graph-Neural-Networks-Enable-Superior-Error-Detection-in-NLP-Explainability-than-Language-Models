"""
GraphText: Modular orchestration for text-to-graph GNN workflows.

This lightweight package wraps the existing implementation (now migrated under `src/`)
and exposes a clean, extensible interface and CLI for:
- LLM fine-tuning
- Graph generation (multiple methodologies)
- Embedding extraction onto graphs
- Conversion to PyTorch Geometric
- GNN training
- Explainability

Extension points are provided via simple registries to add new graph builders,
embedders, and explainers without touching the core pipeline.
"""

__all__ = [
    "registry",
]
