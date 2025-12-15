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

## Repository Structure

```
src/
├── finetuning/          # LLM fine-tuning (BERT)
├── embeddings/          # Node embedding extraction
├── graph_builders/      # Text-to-graph conversion
│   ├── constituency.py  # Constituency trees
│   ├── syntactic.py     # Dependency trees
│   ├── skipgrams.py     # Skip-gram graphs
│   └── window.py        # Window-based graphs
├── convert/             # NetworkX → PyTorch Geometric
├── gnn_training/        # GNN training pipeline
├── explain/             # Explainability modules
│   ├── gnn/
│   │   ├── subgraphx/   # For tree-structured graphs
│   │   └── graphsvx/    # For non-hierarchical graphs
│   └── llm/
│       └── tokenshap/   # LLM baseline (TokenSHAP)
├── Analytics/           # 4-Dimension Evaluation Framework
│   ├── auc/             # Dimension 1: AUC Discriminative Capacity
│   ├── fidelity/        # Dimension 2: Behavioral Faithfulness
│   ├── consistency/     # Dimension 3: Consistency Across Outcomes
│   └── progression/     # Dimension 4: Feature Ranking Stability
├── Insights/            # Metrics extraction and integration
└── use_case/            # Logistic regression error analysis
```

## Quick Start

### 1. Fine-tune LLM
```bash
python -m src.graphtext.cli finetune --dataset_name stanfordnlp/sst2 --output_dir outputs/llm
```

### 2. Build Graphs
```bash
# Syntactic (dependency) trees
python -m src.graphtext.cli build-graphs --graph_type syntactic --dataset stanfordnlp/sst2 --output_dir outputs/graphs

# Other graph types: constituency, skipgrams, window
```

### 3. Generate Embeddings
```bash
python -m src.graphtext.cli embed --graph_type syntactic --dataset_name stanfordnlp/sst2 --split validation --output_dir outputs/embeddings
```

### 4. Convert to PyG Format
```bash
python -m src.graphtext.cli to-pyg --label_source llm --hf_dataset_name stanfordnlp/sst2 --graph_type syntactic
```

### 5. Train GNN
```bash
python -m src.graphtext.cli train --train_data_dir <pyg_train_dir> --val_data_dir <pyg_val_dir>
```

### 6. Run Explainability
```bash
# Auto-selects SubgraphX for tree graphs, GraphSVX for non-tree graphs
python -m src.graphtext.cli explain --dataset stanfordnlp/sst2 --graph_type syntactic --split validation --method auto
```

## Datasets

- **AG News**: 4-class topic classification (Zhang et al., 2015)
- **SST-2**: Binary sentiment analysis (Socher et al., 2013)

## Explainability Methods

| Method | Architecture | Graph Types | Paper Section |
|--------|-------------|-------------|---------------|
| SubgraphX | GNN | Constituency, Syntactic (trees) | 3.4.1 |
| GraphSVX | GNN | Window, Skip-gram (non-trees) | 3.4.1 |
| TokenSHAP | LLM | Tokens | 3.4.1 |

## 4-Dimension Evaluation Framework

The evaluation framework (Section 3.5) assesses explainability across four dimensions:

1. **AUC Discriminative Capacity**: Measures Insertion/Deletion AUC separation between correct and incorrect predictions
2. **Behavioral Faithfulness**: Quantifies necessity (M⁻) and sufficiency (M⁺) through fidelity metrics
3. **Consistency Across Outcomes**: Evaluates margin preservation under perturbation
4. **Feature Ranking Stability**: Analyzes importance concentration in top-k features

## Docker Services

- `app`: Main CUDA-enabled development environment
- `subgraphx`: SubgraphX explainer container
- `graphsvx`: GraphSVX explainer container
- `tokenshap`: TokenSHAP explainer container

```bash
docker compose up -d subgraphx
make subgraphx-shell
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
