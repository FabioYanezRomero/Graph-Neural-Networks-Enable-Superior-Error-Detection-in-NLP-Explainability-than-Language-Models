# Graph Neural Networks Enable Superior Error Detection in NLP Explainability than Language Models

Official code repository for the paper *"Graph Neural Networks Enable Superior Error Detection in NLP Explainability than Language Models"*.

## Overview

This repository implements a modular pipeline for comparing explainability signatures between discrete graph-based (GNN) and continuous token-based (LLM) architectures for error detection in NLP text classification.

**Key Finding**: GNN-based explainers achieve 99.7–100.0% error detection accuracy compared to 88.1–89.6% for LLM-based explainers.

## Paper-to-Code Mapping

| Paper Section | Description | Code Location |
|---------------|-------------|---------------|
| 3.1 Text-to-Graph Conversion | Constituency, Dependency, Window, Skip-gram graphs | `src/graph_builders/` |
| 3.2 LLM Fine-tuning & Embeddings | BERT fine-tuning and node embedding extraction | `src/finetuning/`, `src/embeddings/` |
| 3.3 GNN Training | GCN-based surrogates trained via LLM-as-teacher | `src/gnn_training/` |
| 3.4 Post-hoc Explainability | SubgraphX, GraphSVX (GNN), TokenSHAP (LLM) | `src/explain/gnn/`, `src/explain/llm/` |
| 3.5 4-Dimension Evaluation | AUC, Fidelity, Consistency, Progression | `src/Analytics/` |
| 3.6 Logistic Regression | Error signal analysis | `src/use_case/`, `src/Insights/` |

## Requirements

- Docker & Docker Compose v2
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

## Quick Start: Reproduce All Experiments

### 1. Build All Containers

```bash
# Build all Docker services (first time setup)
make build

# Or build without cache for fresh install
make build-no-cache
```

### 2. Start Containers

```bash
# Start all services
make up

# Or start specific container
docker compose up -d app
```

### 3. Run Full Pipeline

```bash
# Reproduce all paper experiments (Sections 3.1-3.6)
make reproduce
```

This runs the complete pipeline:
1. **Fine-tune LLM** (Section 3.2) - BERT classification on SST-2 & AG News
2. **Build Graphs** (Section 3.1) - 4 graph types: constituency, syntactic, skipgrams, window
3. **Generate Embeddings** (Section 3.2) - Extract node embeddings from fine-tuned LLM
4. **Train GNNs** (Section 3.3) - 2-layer GCN surrogates
5. **Run Explainability** (Section 3.4) - SubgraphX, GraphSVX, TokenSHAP
6. **Run Analytics** (Section 3.5-3.6) - 4-dimension evaluation & logistic regression

## Step-by-Step Execution

Run individual pipeline steps:

```bash
make step-1-finetune      # Fine-tune LLM
make step-2-graphs        # Build graph representations
make step-3-embeddings    # Generate node embeddings
make step-4-train         # Train GNN models
make step-5-explain       # Run explainability (uses subgraphx/graphsvx/tokenshap containers)
make step-6-analytics     # Run 4-dimension evaluation
```

Each step script supports options:

```bash
# View options
./scripts/01_finetune_llm.sh --help

# Dry run (print commands without executing)
./scripts/02_build_graphs.sh --dry-run

# Filter datasets
./scripts/04_train_gnns.sh --datasets sst2 --graph-types constituency,syntactic
```

## Docker Architecture

| Container | Purpose | GPU | Used In |
|-----------|---------|-----|---------|
| `app` | Main training environment | ✓ | Steps 1-4, 6 |
| `subgraphx` | SubgraphX explainer (tree graphs) | ✓ | Step 5 |
| `graphsvx` | GraphSVX explainer (non-tree graphs) | ✓ | Step 5 |
| `tokenshap` | TokenSHAP explainer (LLM baseline) | ✓ | Step 5 |

**Why separate containers?** Each explainer has conflicting dependencies. Isolated containers ensure reproducibility.

### Container Shell Access

```bash
make subgraphx-shell      # Open shell in SubgraphX container
make graphsvx-shell       # Open shell in GraphSVX container  
make tokenshap-shell      # Open shell in TokenSHAP container
```

### Build Individual Containers

```bash
make build-app            # Main environment
make build-subgraphx      # SubgraphX explainer
make build-graphsvx       # GraphSVX explainer
make build-tokenshap      # TokenSHAP explainer
```

## Repository Structure

```
├── scripts/                 # Numbered pipeline scripts
│   ├── 01_finetune_llm.sh
│   ├── 02_build_graphs.sh
│   ├── 03_generate_embeddings.sh
│   ├── 04_train_gnns.sh
│   ├── 05_run_explainers.sh
│   └── 06_run_analytics.sh
├── src/
│   ├── finetuning/          # LLM fine-tuning (BERT)
│   ├── embeddings/          # Node embedding extraction
│   ├── graph_builders/      # Text-to-graph conversion
│   │   ├── constituency.py  # Constituency trees
│   │   ├── syntactic.py     # Dependency trees
│   │   ├── skipgrams.py     # Skip-gram graphs
│   │   └── window.py        # Window-based graphs
│   ├── convert/             # NetworkX → PyTorch Geometric
│   ├── gnn_training/        # GNN training pipeline
│   ├── explain/             # Explainability modules
│   │   ├── gnn/subgraphx/   # For tree-structured graphs
│   │   ├── gnn/graphsvx/    # For non-hierarchical graphs
│   │   └── llm/             # TokenSHAP (LLM baseline)
│   ├── Analytics/           # 4-Dimension Evaluation
│   │   ├── auc/             # Dimension 1
│   │   ├── fidelity/        # Dimension 2
│   │   ├── consistency/     # Dimension 3
│   │   └── progression/     # Dimension 4
│   └── Insights/            # Metrics extraction
├── tests/                   # Pytest test suite
├── configs/                 # Pipeline configurations
├── docker/                  # Dockerfiles
└── outputs/                 # Generated outputs (gitignored)
```

## Testing

```bash
# Run all tests
make test

# Or run directly
pytest tests/ -v

# Fast tests only (skip slow integration tests)
pytest tests/ -m "not slow"

# Run specific test file
pytest tests/test_02_graph_builders.py -v
```

## Datasets

- **SST-2**: Binary sentiment analysis (Socher et al., 2013)
- **AG News**: 4-class topic classification (Zhang et al., 2015)

## Explainability Methods

| Method | Architecture | Graph Types | Paper Section |
|--------|-------------|-------------|---------------|
| SubgraphX | GNN | Constituency, Syntactic (trees) | 3.4.1 |
| GraphSVX | GNN | Window, Skip-gram (non-trees) | 3.4.1 |
| TokenSHAP | LLM | Tokens | 3.4.1 |

## 4-Dimension Evaluation Framework

The evaluation framework (Section 3.5) assesses explainability across four dimensions:

1. **AUC Discriminative Capacity**: Measures Insertion/Deletion AUC separation
2. **Behavioral Faithfulness**: Quantifies necessity (M⁻) and sufficiency (M⁺)
3. **Consistency Across Outcomes**: Evaluates margin preservation under perturbation
4. **Feature Ranking Stability**: Analyzes importance concentration in top-k features

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Check container can access GPU
docker compose exec app nvidia-smi
```

### Container Build Fails

```bash
# Clean rebuild
make clean
make build-no-cache
```

### Out of Memory

Reduce batch size in scripts:
```bash
./scripts/01_finetune_llm.sh --batch_size 8
./scripts/04_train_gnns.sh --batch-size 16
```

## Citation

```bibtex
@article{yanez2025gnn,
  title={Graph Neural Networks Enable Superior Error Detection in NLP Explainability than Language Models},
  author={Yáñez-Romero, Fabio and Montoya, Andrés and Suárez, Armando and Gutiérrez, Yoan and Mitkov, Ruslan},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) for details.
