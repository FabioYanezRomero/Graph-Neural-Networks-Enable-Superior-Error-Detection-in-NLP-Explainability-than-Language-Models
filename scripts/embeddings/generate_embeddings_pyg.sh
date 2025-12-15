#!/usr/bin/env bash
set -euo pipefail

# Script to generate embeddings and PyTorch Geometric format graphs from existing graph structures
# Uses finetuned models from models.env according to the specific dataset

DATASETS=("stanfordnlp/sst2" "SetFit/ag_news")
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ensure REPO_ROOT is set correctly
if [[ ! -d "$REPO_ROOT/outputs" ]]; then
    echo "[error] Repository root not found correctly. REPO_ROOT=$REPO_ROOT"
    exit 1
fi

# Source models.env if it exists
if [ -f "$REPO_ROOT/scripts/models.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/models.env"
fi

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Generate embeddings and PyTorch Geometric format from existing graph structures."
    echo ""
    echo "Options:"
    echo "  --dataset_name NAME      Dataset name (e.g., stanfordnlp/sst2, SetFit/ag_news)"
    echo "  --graph_type TYPE        Graph type (constituency, syntactic, window, ngrams, skipgrams, knowledge)"
    echo "  --subsets SUBSETS        Comma-separated subsets (train,val,test) [default: all available]"
    echo "  --batch_size N           Processing batch size (default: 128)"
    echo "  --device DEVICE          Device to use (default: cuda:0)"
    echo "  --output_base DIR        Base output directory (default: outputs)"
    echo "  --model_name NAME        Override model name (default: auto-detect from finetuned)"
    echo "  --force                  Overwrite existing embeddings"
    echo "  --dry_run                Show what would be done without executing"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dataset_name stanfordnlp/sst2 --graph_type constituency"
    echo "  $0 --dataset_name SetFit/ag_news --graph_type syntactic --subsets train,test"
    echo ""
    exit 1
}

# Parse command line arguments
dataset_name=""
graph_type=""
subsets=""
batch_size="$BATCH_SIZE"
device="$DEVICE"
output_base="$OUT_BASE"
model_name=""
force=false
dry_run=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            dataset_name="$2"
            shift 2
            ;;
        --graph_type)
            graph_type="$2"
            shift 2
            ;;
        --subsets)
            subsets="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --output_base)
            output_base="$2"
            shift 2
            ;;
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --force)
            force=true
            shift
            ;;
        --dry_run)
            dry_run=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Interpret "all" as request for every available subset
if [[ "$subsets" =~ ^[Aa][Ll][Ll]$ ]]; then
    subsets=""
fi

# Validate required arguments
if [ -z "$dataset_name" ] || [ -z "$graph_type" ]; then
    echo "Error: --dataset_name and --graph_type are required"
    show_help
fi

# Auto-fallback to CPU if CUDA is not available
if [[ "$device" == cuda:* || "$device" == cuda ]]; then
  if ! python3 - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
  then
    echo "[warn] torch.cuda.is_available()==False; falling back to CPU." >&2
    device=cpu
  fi
fi

# Function to get available subsets dynamically
get_subsets() {
    local dataset=$1
    local graph_root="$REPO_ROOT/outputs/graphs/$dataset"
    if [[ -d "$graph_root" ]]; then
        local subsets
        subsets=$(find "$graph_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
        if [[ -n "$subsets" ]]; then
            echo "$subsets" | tr '\n' ' '
            return
        fi
    fi

    python3 - <<PY
from datasets import get_dataset_split_names
import sys
try:
    splits = get_dataset_split_names("$dataset")
    print(" ".join(splits))
except Exception:
    if "$dataset" == "stanfordnlp/sst2":
        print("train validation test")
    elif "$dataset" == "SetFit/ag_news":
        print("train test")
    else:
        print("train test")
PY
}

# Function to get checkpoint path for dataset
get_checkpoint_path() {
    local dataset=$1
    case "$dataset" in
      stanfordnlp/sst2)
        if [[ -n "${SST2_CHECKPOINT:-}" ]]; then
          echo "$SST2_CHECKPOINT"
        fi
        ;;
      SetFit/ag_news)
        if [[ -n "${AGNEWS_CHECKPOINT:-}" ]]; then
          echo "$AGNEWS_CHECKPOINT"
        fi
        ;;
    esac
}

# Function to determine graph directory pattern
get_graph_dir_pattern() {
    local graph_type=$1
    case "$graph_type" in
        constituency)
            echo "constituency"
            ;;
        syntactic)
            echo "syntactic"
            ;;
        window)
            echo "window.word.k1"
            ;;
        ngrams)
            echo "ngrams.word.n2"
            ;;
        skipgrams)
            echo "skipgrams.word.k1.n2"
            ;;
        knowledge)
            echo "knowledge"
            ;;
        *)
            echo "$graph_type"
            ;;
    esac
}

# Function to get model name from checkpoint config
get_model_from_checkpoint() {
    local checkpoint_path=$1
    if [[ -z "$checkpoint_path" || ! -f "$checkpoint_path" ]]; then
        echo "bert-base-uncased"  # fallback
        return
    fi

    local config_model
    config_model=$(python3 - <<PY
import json
import os
import sys
config_path = os.path.join(os.path.dirname("$checkpoint_path"), 'config.json')
if not os.path.isfile(config_path):
    print("bert-base-uncased")
    sys.exit(0)
try:
    with open(config_path) as f:
        data = json.load(f)
except Exception:
    print("bert-base-uncased")
    sys.exit(0)
for key in ("model_name", "model_name_or_path", "pretrained_model_name"):
    val = data.get(key)
    if isinstance(val, str) and val.strip():
        print(val.strip())
        sys.exit(0)
# If no model found, print fallback
print("bert-base-uncased")
PY
    )

    if [[ -n "$config_model" ]]; then
        echo "$config_model"
    else
        echo "bert-base-uncased"  # fallback
    fi
}

# Main processing function
process_dataset() {
    local dataset=$1
    local graph_type=$2
    local subset_list=$3

    echo "Processing dataset: $dataset, graph_type: $graph_type"

    # Get checkpoint path
    local checkpoint_path
    checkpoint_path=$(get_checkpoint_path "$dataset")

    # Get model name
    local model_name_for_dataset="$model_name"
    if [[ -z "$model_name_for_dataset" ]]; then
        if [[ -n "$checkpoint_path" && -f "$checkpoint_path" ]]; then
            model_name_for_dataset=$(get_model_from_checkpoint "$checkpoint_path")
            echo "[info] Using model from checkpoint config: $model_name_for_dataset"
        else
            model_name_for_dataset="bert-base-uncased"
            echo "[warn] No checkpoint found, using base model: $model_name_for_dataset"
        fi
    fi

    # Extract epoch from checkpoint path for label loading
    local checkpoint_epoch
    if [[ -n "$checkpoint_path" ]]; then
        checkpoint_epoch=$(basename "$checkpoint_path" | sed 's/model_epoch_\([0-9]\+\)\.pt/\1/')
    else
        checkpoint_epoch=0
    fi

    # Get graph directory pattern
    local graph_dir_pattern
    graph_dir_pattern=$(get_graph_dir_pattern "$graph_type")

    # Process each subset
    for subset in $subset_list; do
        echo "Processing subset: $subset"

        # Define paths
        local graph_dir="$REPO_ROOT/outputs/graphs/$dataset/$subset/$graph_dir_pattern"
        local embedding_dir="$output_base/embeddings/$dataset/$subset/$graph_type"
        local pyg_dir="$output_base/pyg_graphs/$dataset/$subset/$graph_type"

        # Check if graph directory exists
        if [[ ! -d "$graph_dir" ]]; then
            echo "[warn] Graph directory not found: $graph_dir"
            continue
        fi

        # Check if embeddings already exist
        if [[ -d "$embedding_dir" && "$force" != true ]]; then
            echo "[info] Embeddings already exist for $dataset/$subset/$graph_type. Use --force to overwrite."
            continue
        fi

        # Create output directories
        mkdir -p "$embedding_dir"
        mkdir -p "$pyg_dir"

        # Create predictions file path
        local predictions_file="${checkpoint_path%/*}/predictions.json"

        # Build command using efficient generation
        local cmd="cd $REPO_ROOT && python3 -m src.embeddings.generate_efficient \\
            --graph_type $graph_type \\
            --dataset_name $dataset \\
            --split $subset \\
            --tree_dir $graph_dir \\
            --output_dir $embedding_dir \\
            --pyg_output_dir $pyg_dir \\
            --model_name $model_name_for_dataset \\
            --device $device \\
            --batch_size $batch_size"

        # Add checkpoint and predictions if available
        if [[ -n "$checkpoint_path" && -f "$checkpoint_path" ]]; then
            cmd="$cmd --weights_path $checkpoint_path"
        fi

        if [[ -n "$predictions_file" && -f "$predictions_file" ]]; then
            cmd="$cmd --predictions_file $predictions_file --epoch $checkpoint_epoch"
        fi

        echo "[info] Running embedding generation command:"
        echo "$cmd"

        if [[ "$dry_run" != true ]]; then
            eval "$cmd"
            echo "[info] Efficient embedding generation and PyG conversion with labels completed for $dataset/$subset/$graph_type"
        fi
    done
}

# If no specific subsets provided, get available ones
if [[ -z "$subsets" ]]; then
    available_subsets=$(get_subsets "$dataset_name")
    echo "Available subsets for $dataset_name: $available_subsets"
else
    # Convert comma-separated to space-separated
    available_subsets=$(echo "$subsets" | tr ',' ' ')
fi

# Process the dataset
process_dataset "$dataset_name" "$graph_type" "$available_subsets"

echo "All processing completed!"
