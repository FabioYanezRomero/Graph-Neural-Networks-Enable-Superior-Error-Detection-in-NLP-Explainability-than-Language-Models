#!/usr/bin/env bash
# ==============================================================================
# Step 4: Train GNNs (Section 3.3)
# Container: app
# Usage: ./scripts/04_train_gnns.sh [--datasets sst2,ag_news] [--graph-types all]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
DATASETS_FILTER=""
GRAPH_TYPES_FILTER=""
PYG_BASE="outputs/pyg_graphs"
MODEL_BASE="outputs/gnn_models"
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.001
PATIENCE=5
DRY_RUN=false

show_help() {
    cat <<'USAGE'
Usage: 04_train_gnns.sh [options]

Train GNN surrogates using LLM-as-teacher paradigm (Paper Section 3.3)

Trains a 2-layer GCN for each dataset/graph-type combination.

Options:
  --datasets LIST      Comma-separated filter (default: all found)
  --graph-types LIST   Comma-separated filter (default: all found)
  --pyg-dir DIR        PyG graphs input (default: outputs/pyg_graphs)
  --model-dir DIR      Model output directory (default: outputs/gnn_models)
  --epochs N           Training epochs (default: 20)
  --batch-size N       Batch size (default: 32)
  --learning-rate LR   Learning rate (default: 0.001)
  --patience N         Early stopping patience (default: 5)
  --dry-run            Print commands without executing
  --help               Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS_FILTER="$2"; shift 2 ;;
        --graph-types) GRAPH_TYPES_FILTER="$2"; shift 2 ;;
        --pyg-dir) PYG_BASE="$2"; shift 2 ;;
        --model-dir) MODEL_BASE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help) show_help ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

# Build command for train_all_gnns.sh (which has comprehensive logic)
CMD="bash scripts/gnn/train_all_gnns.sh \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --patience $PATIENCE \
    --pyg_base $PYG_BASE \
    --model_base $MODEL_BASE"

[[ -n "$DATASETS_FILTER" ]] && CMD="$CMD --datasets $DATASETS_FILTER"
[[ -n "$GRAPH_TYPES_FILTER" ]] && CMD="$CMD --graph_types $GRAPH_TYPES_FILTER"
[[ "$DRY_RUN" == true ]] && CMD="$CMD --dry_run"

echo "=========================================="
echo "Step 4: Training GNNs"
echo "  PyG Base: $PYG_BASE"
echo "  Model Output: $MODEL_BASE"
echo "  Epochs: $EPOCHS"
echo "=========================================="

eval $CMD
