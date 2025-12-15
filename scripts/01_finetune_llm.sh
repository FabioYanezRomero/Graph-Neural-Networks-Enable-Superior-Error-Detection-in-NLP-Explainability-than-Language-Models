#!/usr/bin/env bash
# ==============================================================================
# Step 1: Fine-tune LLM (Section 3.2)
# Container: app
# Usage: ./scripts/01_finetune_llm.sh [--dataset sst2|ag_news] [--dry-run]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
DATASET="sst2"
MODEL_NAME="google-bert/bert-base-uncased"
NUM_EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=1e-6
OUTPUT_DIR="outputs/llm"
DRY_RUN=false

show_help() {
    cat <<'USAGE'
Usage: 01_finetune_llm.sh [options]

Fine-tune BERT for text classification (Paper Section 3.2)

Options:
  --dataset NAME       Dataset: sst2, ag_news (default: sst2)
  --model_name NAME    HuggingFace model (default: google-bert/bert-base-uncased)
  --epochs N           Training epochs (default: 5)
  --batch_size N       Batch size (default: 16)
  --learning_rate LR   Learning rate (default: 1e-6)
  --output_dir DIR     Output directory (default: outputs/llm)
  --dry-run            Print command without executing
  --help               Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help) show_help ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Map dataset name to HuggingFace identifier
case "$DATASET" in
    sst2) HF_DATASET="stanfordnlp/sst2" ;;
    ag_news) HF_DATASET="SetFit/ag_news" ;;
    *) HF_DATASET="$DATASET" ;;
esac

cd "$REPO_ROOT"

CMD="python -m src.finetuning \
    --dataset_name $HF_DATASET \
    --model_name $MODEL_NAME \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir $OUTPUT_DIR"

echo "=========================================="
echo "Step 1: Fine-tuning LLM"
echo "  Dataset: $HF_DATASET"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $NUM_EPOCHS"
echo "=========================================="

if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] $CMD"
else
    eval $CMD
fi
