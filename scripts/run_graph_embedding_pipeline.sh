#!/bin/bash
# Usage: ./run_graph_embedding_pipeline.sh --graph_type GRAPH_TYPE --dataset_name DATASET --split SPLIT --tree_dir TREE_DIR --output_dir OUTPUT_DIR [--model_name MODEL_NAME] [--device DEVICE]

# Default values
GRAPH_TYPE="syntactic"
MODEL_NAME="bert-base-uncased"
WEIGHTS_PATH=""
DEVICE="cuda"
DATASET_NAME="stanfordnlp/sst2"
SPLIT="validation"
TREE_DIR="outputs/graphs/${DATASET_NAME}/${SPLIT}/${GRAPH_TYPE}"
OUTPUT_DIR="outputs/embeddings/${DATASET_NAME}/${SPLIT}/${GRAPH_TYPE}"
BATCH_SIZE=128

# Optional central model settings
if [ -f "scripts/models.env" ]; then
  # shellcheck disable=SC1091
  source scripts/models.env
fi

# Prefer explicit fine-tuned route if provided by env
if [[ -n "${FINETUNED_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${FINETUNED_MODEL_NAME}"
elif [[ -n "${GRAPHTEXT_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${GRAPHTEXT_MODEL_NAME}"
fi

# Allow global weights path via env
if [[ -n "${GRAPHTEXT_WEIGHTS_PATH:-}" ]]; then
  WEIGHTS_PATH="${GRAPHTEXT_WEIGHTS_PATH}"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --graph_type)
            GRAPH_TYPE="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --tree_dir)
            TREE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --weights_path)
            WEIGHTS_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --graph_type GRAPH_TYPE --dataset_name DATASET --split SPLIT --tree_dir TREE_DIR --output_dir OUTPUT_DIR [--model_name MODEL_NAME] [--device DEVICE]"
            exit 0
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$GRAPH_TYPE" || -z "$DATASET_NAME" || -z "$SPLIT" || -z "$TREE_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 --graph_type GRAPH_TYPE --dataset_name DATASET --split SPLIT --tree_dir TREE_DIR --output_dir OUTPUT_DIR [--model_name MODEL_NAME] [--device DEVICE]"
    exit 1
fi

# If no explicit weights provided, pick dataset-specific default if available
if [[ -z "$WEIGHTS_PATH" ]]; then
  case "$DATASET_NAME" in
    stanfordnlp/sst2)
      WEIGHTS_PATH="/app/outputs/finetuned_llms/stanfordnlp/sst2/sst2_2025-06-04_14-52-49/model_epoch_2.pt"
      ;;
    SetFit/ag_news|setfit/ag_news)
      WEIGHTS_PATH="/app/outputs/finetuned_llms/setfit/ag_news/model_epoch_4.pt"
      ;;
  esac
fi

# Validate that the weights path exists and is a file; warn if not
if [[ -n "$WEIGHTS_PATH" && ! -f "$WEIGHTS_PATH" ]]; then
  echo "[warn] Expected checkpoint not found at $WEIGHTS_PATH; proceeding with base model weights." >&2
fi

# Run the embedding pipeline
python3 -m src.embeddings.generate \
    --graph_type "$GRAPH_TYPE" \
    --dataset_name "$DATASET_NAME" \
    --split "$SPLIT" \
    --tree_dir "$TREE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --weights_path "$WEIGHTS_PATH" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
