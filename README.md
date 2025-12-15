# Graph Neural Networks Enable Superior Error Detection in NLP Explainability than Language Models

Official code repository for the paper *"Graph Neural Networks Enable Superior Error Detection in NLP Explainability than Language Models"*.

---

## 🎯 Key Contributions

This research demonstrates that **graph-based explainability methods systematically outperform language model-based approaches** for detecting classification errors in NLP systems.

### Main Findings

| Metric | GNN-based (SubgraphX/GraphSVX) | LLM-based (TokenSHAP) |
|--------|-------------------------------|----------------------|
| **Error Detection Accuracy** | 99.7–100.0% | 88.1–89.6% |
| **AUC Separation** | Clear discrimination | Overlapping distributions |
| **Fidelity Patterns** | Consistent quadrant placement | Inconsistent patterns |

### Why GNNs Outperform LLMs for Explainability

1. **Discrete vs. Continuous**: GNNs operate on discrete graph structures, producing cleaner feature attribution signals
2. **Structural Awareness**: Graph representations preserve linguistic relationships (syntax, constituency) that inform explanations
3. **Compression Benefits**: The knowledge distillation from LLM → GNN acts as a regularizer, producing more robust predictions
4. **Subgraph Semantics**: GNN explainers identify meaningful substructures rather than individual tokens

---

## 📊 4-Dimension Evaluation Framework

The evaluation framework (Section 3.5) provides a comprehensive assessment of explainability quality across four orthogonal dimensions:

### Dimension 1: AUC Discriminative Capacity

**Purpose**: Measures how well the explainer distinguishes between correct and incorrect predictions through insertion/deletion curves.

**Metrics**:
- **Deletion AUC**: Area under curve when progressively removing important features
- **Insertion AUC**: Area under curve when progressively adding important features

**Key Insight**: Correct predictions show higher deletion AUC (removing important features hurts more) and lower insertion AUC (less important features already present). GNN explainers achieve **clear separation** between correct/incorrect, while LLM explainers show **overlapping distributions**.

**Formula**:
```
Separability = √(SD_correct² + SD_incorrect²)
```

### Dimension 2: Behavioral Faithfulness (Fidelity)

**Purpose**: Quantifies whether the identified features are truly necessary and sufficient for the prediction.

**Metrics**:
- **M⁺ (Sufficiency)**: Does masking to only the important features maintain the prediction?
- **M⁻ (Necessity)**: Does masking out the important features change the prediction?

**Quadrant Analysis**:
| Quadrant | M⁺ | M⁻ | Interpretation |
|----------|----|----|----------------|
| Q1: Sufficient & Necessary | >0 | >0 | Ideal explanations |
| Q2: Sufficient & Redundant | >0 | ≤0 | Features work but aren't unique |
| Q3: Insufficient & Necessary | ≤0 | >0 | Missing key features |
| Q4: Insufficient & Redundant | ≤0 | ≤0 | Poor explanations |

**Key Insight**: GNN explainers consistently place correct predictions in Q1 (ideal) and incorrect predictions in Q3/Q4, enabling easy error detection.

**Asymmetry Index**:
```
A = (M⁻ - M⁺) / (|M⁻| + |M⁺|)
```

### Dimension 3: Consistency Across Outcomes

**Purpose**: Evaluates whether explanations maintain prediction margins under perturbation.

**Metrics**:
- **Origin Margin**: Original prediction confidence gap
- **Masked Margin**: Margin when keeping only top-k features
- **Maskout Margin**: Margin when removing top-k features

**Key Insight**: For correct predictions, masked margin should be close to origin (features are sufficient), while maskout margin should be small (features are necessary). GNNs show **consistent margin preservation patterns** distinguishing correct from incorrect.

### Dimension 4: Feature Ranking Stability (Progression)

**Purpose**: Analyzes how importance is distributed across features and whether top features alone drive the prediction.

**Metrics**:
- **Maskout Progression**: Confidence drop as features are progressively removed
- **Sufficiency Progression**: Confidence increase as features are progressively added
- **Concentration Ratio**: Importance mass in top-k vs. remaining features

**Key Insight**: GNN explainers produce **steeper progression curves**, indicating more concentrated and meaningful feature rankings.

---

## 🔬 Logistic Regression Error Detection

Section 3.6 demonstrates the practical application: using explainability metrics as features for automatic error detection.

### Feature Vector Construction

For each prediction, we extract a feature vector from the 4 dimensions:

```python
features = [
    # Dimension 1: AUC
    deletion_auc, insertion_auc,
    
    # Dimension 2: Fidelity
    fidelity_plus, fidelity_minus, asymmetry_index,
    
    # Dimension 3: Consistency
    origin_margin, masked_margin, maskout_margin,
    margin_preservation_ratio,
    
    # Dimension 4: Progression
    maskout_drop_k1, maskout_drop_k2, maskout_drop_k3,
    sufficiency_gain_k1, sufficiency_gain_k2, sufficiency_gain_k3,
    concentration_ratio
]
```

### Binary Classification

```
y = 1 if prediction is INCORRECT (error)
y = 0 if prediction is CORRECT
```

### Results

| Explainer | Dataset | Accuracy | AUC-ROC |
|-----------|---------|----------|---------|
| SubgraphX (constituency) | SST-2 | 100.0% | 1.000 |
| GraphSVX (skipgrams) | SST-2 | 99.7% | 0.998 |
| TokenSHAP | SST-2 | 88.1% | 0.912 |
| SubgraphX (syntactic) | AG News | 100.0% | 1.000 |
| GraphSVX (window) | AG News | 99.8% | 0.999 |
| TokenSHAP | AG News | 89.6% | 0.923 |

### Coefficient Analysis

The logistic regression coefficients reveal which dimensions are most predictive:

- **Dimension 2 (Fidelity)**: Highest absolute coefficients (~40% contribution)
- **Dimension 4 (Progression)**: Second highest (~30% contribution)
- **Dimension 1 (AUC)**: Moderate (~20% contribution)
- **Dimension 3 (Consistency)**: Supporting role (~10% contribution)

---

## 📈 Interactive Visualizations

The `Images/` directory contains interactive HTML visualizations for each evaluation dimension:

```
Images/
├── AUC Discriminative Capacity/
│   ├── sst-2_connected_scatter_deletion.html
│   ├── sst-2_connected_scatter_insertion.html
│   ├── ag-news_connected_scatter_deletion.html
│   └── ag-news_connected_scatter_insertion.html
├── Fidelity/
│   ├── fidelity_quadrants_stanfordnlp_sst2.html
│   ├── fidelity_quadrants_setfit_ag_news.html
│   ├── fidelity_asymmetry_*.html
│   └── fidelity_quadrant_distribution_*.html
├── Consistency Across Outcomes/
│   └── [8 interactive plots]
└── Feature Ranking Stability/
    └── [2 interactive plots]
```

**Open these files in a browser** to explore the data interactively with hover tooltips, zoom, and filtering.

---

## Paper-to-Code Mapping

| Paper Section | Description | Code Location |
|---------------|-------------|---------------|
| 3.1 Text-to-Graph Conversion | Constituency, Dependency, Window, Skip-gram graphs | `src/graph_builders/` |
| 3.2 LLM Fine-tuning & Embeddings | BERT fine-tuning and node embedding extraction | `src/finetuning/`, `src/embeddings/` |
| 3.3 GNN Training | GCN-based surrogates trained via LLM-as-teacher | `src/gnn_training/` |
| 3.4 Post-hoc Explainability | SubgraphX, GraphSVX (GNN), TokenSHAP (LLM) | `src/explain/gnn/`, `src/explain/llm/` |
| 3.5 4-Dimension Evaluation | AUC, Fidelity, Consistency, Progression | `src/Analytics/` |
| 3.6 Logistic Regression | Error signal analysis | `src/use_case/`, `src/Insights/` |

---

## 🚀 Quick Start: Reproduce All Experiments

### Requirements

- Docker & Docker Compose v2
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### 1. Build All Containers

```bash
make build
```

### 2. Start Containers

```bash
make up
```

### 3. Run Full Pipeline

```bash
make reproduce
```

This runs the complete pipeline:
1. **Fine-tune LLM** (Section 3.2) - BERT classification on SST-2 & AG News
2. **Build Graphs** (Section 3.1) - 4 graph types: constituency, syntactic, skipgrams, window
3. **Generate Embeddings** (Section 3.2) - Extract node embeddings from fine-tuned LLM
4. **Train GNNs** (Section 3.3) - 2-layer GCN surrogates
5. **Run Explainability** (Section 3.4) - SubgraphX, GraphSVX, TokenSHAP
6. **Run Analytics** (Section 3.5-3.6) - 4-dimension evaluation & logistic regression

---

## Step-by-Step Execution

Run individual pipeline steps:

```bash
make step-1-finetune      # Fine-tune LLM
make step-2-graphs        # Build graph representations
make step-3-embeddings    # Generate node embeddings
make step-4-train         # Train GNN models
make step-5-explain       # Run explainability
make step-6-analytics     # Run 4-dimension evaluation
```

Each step script supports options:

```bash
./scripts/01_finetune_llm.sh --help
./scripts/02_build_graphs.sh --dry-run
./scripts/04_train_gnns.sh --datasets sst2 --graph-types constituency,syntactic
```

---

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

---

## Repository Structure

```
├── Images/                  # Interactive HTML visualizations
├── scripts/                 # Numbered pipeline scripts (01-06)
├── src/
│   ├── finetuning/          # LLM fine-tuning (BERT)
│   ├── embeddings/          # Node embedding extraction
│   ├── graph_builders/      # Text-to-graph conversion
│   ├── convert/             # NetworkX → PyTorch Geometric
│   ├── gnn_training/        # GNN training pipeline
│   ├── explain/             # Explainability modules
│   │   ├── gnn/subgraphx/   # For tree-structured graphs
│   │   ├── gnn/graphsvx/    # For non-hierarchical graphs
│   │   └── llm/             # TokenSHAP (LLM baseline)
│   ├── Analytics/           # 4-Dimension Evaluation
│   └── Insights/            # Metrics extraction
├── tests/                   # Pytest test suite (85 tests)
├── configs/                 # Pipeline configurations
└── outputs/                 # Generated outputs (gitignored)
```

---

## Testing

```bash
# Run all tests inside container
docker compose exec -w /app app pytest tests/ -v

# Fast tests only
docker compose exec -w /app app pytest tests/ -m "not slow"
```

---

## Datasets

- **SST-2**: Binary sentiment analysis (Socher et al., 2013)
- **AG News**: 4-class topic classification (Zhang et al., 2015)

---

## Explainability Methods

| Method | Architecture | Graph Types | Paper Section |
|--------|-------------|-------------|---------------|
| SubgraphX | GNN | Constituency, Syntactic (trees) | 3.4.1 |
| GraphSVX | GNN | Window, Skip-gram (non-trees) | 3.4.1 |
| TokenSHAP | LLM | Tokens | 3.4.1 |

---

## Troubleshooting

### GPU Not Detected

```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
docker compose exec app nvidia-smi
```

### Container Build Fails

```bash
make clean
make build-no-cache
```

### Out of Memory

```bash
./scripts/01_finetune_llm.sh --batch_size 8
./scripts/04_train_gnns.sh --batch-size 16
```

---

## Citation

```bibtex
@article{yanez2025gnn,
  title={Graph Neural Networks Enable Superior Error Detection in NLP Explainability than Language Models},
  author={Yáñez-Romero, Fabio and Montoya, Andrés and Suárez, Armando and Gutiérrez, Yoan and Mitkov, Ruslan},
  year={2025}
}
```

---

## License

See [LICENSE](LICENSE) for details.
