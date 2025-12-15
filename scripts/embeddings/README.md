# Embedding Generation for PyTorch Geometric

This directory contains scripts to generate embeddings and PyTorch Geometric (PyG) format graphs from the existing graph structures in your repository.

## Overview

The scripts generate embeddings using:
- **Finetuned models** from `models.env` according to the specific dataset
- **Full-sentence forwards** through the checkpointed language model reconstructed directly from the stored graphs
- **Last layer, token-averaged embeddings** aligned with each word node in the graph
- **CLS embeddings** for special tokens (constituency tree labels)
- **CLS node embeddings** for knowledge graphs, using each node's full textual description
- **PyTorch Geometric format** for direct GNN training

## Scripts

### 1. `generate_embeddings_pyg.sh`
Main script for generating embeddings and PyG graphs for a specific dataset and graph type.

```bash
# Generate embeddings for SST-2 constituency graphs
./generate_embeddings_pyg.sh \
    --dataset_name stanfordnlp/sst2 \
    --graph_type constituency \
    --subsets train,validation,test

# Generate embeddings for AG News syntactic graphs with custom settings
./generate_embeddings_pyg.sh \
    --dataset_name SetFit/ag_news \
    --graph_type syntactic \
    --batch_size 64 \
    --device cuda:1
```

**Options:**
- `--dataset_name`: Dataset name (stanfordnlp/sst2, SetFit/ag_news)
- `--graph_type`: Graph type (constituency, syntactic, window, ngrams, skipgrams, knowledge)
- `--subsets`: Comma-separated subsets (train,validation,test)
- `--batch_size`: Processing batch size (default: 128)
- `--device`: Device to use (default: cuda:0)
- `--output_base`: Base output directory (default: outputs)
- `--model_name`: Override model name
- `--force`: Overwrite existing embeddings
- `--dry_run`: Show what would be done without executing

Subsets are auto-detected from the folders already present in `outputs/graphs/{dataset_name}`, so the command works fully offline once graphs exist.

### 2. `generate_all_embeddings_pyg.sh`
Batch script that processes ALL datasets and graph types at once.

```bash
# Process all datasets and graph types
./generate_all_embeddings_pyg.sh

# Process specific datasets only
./generate_all_embeddings_pyg.sh --datasets stanfordnlp/sst2

# Process in parallel with custom settings
./generate_all_embeddings_pyg.sh \
    --parallel \
    --max_jobs 4 \
    --batch_size 256 \
    --force
```

**Options:**
- `--datasets`: Comma-separated datasets (default: all available)
- `--graph_types`: Comma-separated graph types (default: all available)
- `--parallel`: Run in parallel
- `--max_jobs`: Max parallel jobs (default: 4)
- `--force`: Overwrite existing embeddings
- `--dry_run`: Show what would be done without executing

### 3. `validate_embeddings_pyg.py`
Validation script to check that generated embeddings and PyG files are correct.

```bash
# Validate specific combination
python3 validate_embeddings_pyg.py \
    --dataset_name stanfordnlp/sst2 \
    --graph_type constituency \
    --subset validation

# Validate all combinations
python3 validate_embeddings_pyg.py --all
```

**Options:**
- `--dataset_name`: Dataset name
- `--graph_type`: Graph type
- `--subset`: Dataset subset
- `--all`: Validate all available combinations

### 4. `test_pipeline.sh`
Quick test script to validate the entire pipeline.

```bash
# Run pipeline test
./test_pipeline.sh
```

## Output Structure

The scripts generate embeddings in the following structure:

```
outputs/
├── embeddings/
│   └── {dataset_name}/
│       └── {subset}/
│           └── {graph_type}/
│               ├── 00000.pkl
│               ├── 00001.pkl
│               └── ...
└── pyg_graphs/
    └── {dataset_name}/
        └── {subset}/
            └── {graph_type}/
                ├── 00000.pt
                ├── 00001.pt
                └── ...
```

## Features

### Finetuned Model Integration
- Automatically loads finetuned models from `models.env`
- Uses dataset-specific checkpoints:
  - SST-2: `SST2_CHECKPOINT`
  - AG News: `AGNEWS_CHECKPOINT`
- Falls back to base model weights if checkpoint not found

### Special Token Handling
- **Constituency trees**: Special labels (NP, VP, etc.) use CLS embeddings from the last layer
- **Syntactic trees**: Word nodes use last layer embeddings, syntactic labels use CLS
- **Other graphs**: Appropriate embedding strategy based on node type

### Word Position Embeddings
- Reconstructs each dataset sentence from the stored graph tokens so every sample runs through the checkpointed model
- Uses the last hidden layer of the finetuned language model
- Properly handles subword tokenization with mean aggregation back to the original word span order

### Label Attachments
- Predicted labels from the finetuned checkpoint (`predictions.json`) are injected into the resulting PyG objects
- Falls back to the original dataset labels when predictions are not available
- Each PyG graph carries `data_index`, `true_label`, and `sentence` metadata for downstream auditing

### PyTorch Geometric Format
- Converts NetworkX graphs to PyG Data objects
- Preserves node labels and embeddings
- Maintains edge structure and labels
- Ready for direct GNN training

## Usage Examples

### Quick Start
```bash
# 1. Test the pipeline
./test_pipeline.sh

# 2. Generate all embeddings
./generate_all_embeddings_pyg.sh --force

# 3. Validate results
python3 validate_embeddings_pyg.py --all
```

### Advanced Usage
```bash
# Generate embeddings for specific dataset/graph combination
./generate_embeddings_pyg.sh \
    --dataset_name stanfordnlp/sst2 \
    --graph_type constituency \
    --subsets train,test \
    --batch_size 256 \
    --device cuda:0 \
    --force

# Process in parallel for speed
./generate_all_embeddings_pyg.sh \
    --parallel \
    --max_jobs 8 \
    --batch_size 128 \
    --force
```

### Integration with GNN Training
```python
import torch
from torch_geometric.data import DataLoader

# Load generated PyG graphs
pyg_graphs = torch.load('outputs/pyg_graphs/stanfordnlp/sst2/train/constituency/constituency_train_batch_0000_pyg_graphs.pt')

# Create data loader for GNN training
loader = DataLoader(pyg_graphs, batch_size=32, shuffle=True)

# Train your GNN
for batch in loader:
    # batch.x: node embeddings [num_nodes, embedding_dim]
    # batch.edge_index: edge indices [2, num_edges]
    # batch.node_labels: node labels (optional)
    pass
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   ./generate_embeddings_pyg.sh --batch_size 64 --device cpu
   ```

2. **Missing graph files**
   ```bash
   # Check that graphs exist
   ls outputs/graphs/stanfordnlp/sst2/train/constituency/
   ```

3. **Model loading issues**
   ```bash
   # Check models.env file
   cat scripts/models.env
   ```

4. **Validation failures**
   ```bash
   # Run validation on specific combination
   python3 validate_embeddings_pyg.py \
       --dataset_name stanfordnlp/sst2 \
       --graph_type constituency \
       --subset validation
   ```

### Performance Tips

- Use `--parallel` with `--max_jobs` for faster processing on multi-GPU systems
- Adjust `--batch_size` based on available GPU memory
- Use `--dry_run` first to see what will be processed
- Process one dataset/graph_type at a time for better monitoring

## Requirements

- Python packages: torch, transformers, datasets, networkx, torch-geometric
- Bash shell
- Sufficient disk space for embeddings (typically 2-5x the size of original graphs)
- GPU recommended for faster processing

## Integration

The generated embeddings and PyG graphs are ready to use with:
- PyTorch Geometric GNN models
- Graph Neural Network training pipelines
- Graph classification and regression tasks
- Any framework that accepts PyG Data objects
