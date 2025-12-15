#!/usr/bin/env bash
# ==============================================================================
# Step 3: Generate Embeddings (Section 3.2)
# Container: app
# Usage: ./scripts/03_generate_embeddings.sh [--dataset sst2|ag_news]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
DATASETS="sst2,ag_news"
GRAPH_TYPES="constituency,syntactic,skipgrams,window"
GRAPHS_BASE="outputs/graphs"
OUTPUT_BASE="outputs/pyg_graphs"
DRY_RUN=false

show_help() {
    cat <<'USAGE'
Usage: 03_generate_embeddings.sh [options]

Generate node embeddings from fine-tuned LLM and convert to PyG format (Paper Section 3.2)

Options:
  --datasets LIST      Comma-separated: sst2,ag_news (default: both)
  --graph-types LIST   Comma-separated graph types (default: all four)
  --graphs-dir DIR     Input graphs directory (default: outputs/graphs)
  --output-dir DIR     Output PyG directory (default: outputs/pyg_graphs)
  --dry-run            Print commands without executing
  --help               Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS="$2"; shift 2 ;;
        --graph-types) GRAPH_TYPES="$2"; shift 2 ;;
        --graphs-dir) GRAPHS_BASE="$2"; shift 2 ;;
        --output-dir) OUTPUT_BASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help) show_help ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

IFS=',' read -ra DATASET_ARR <<< "$DATASETS"
IFS=',' read -ra GRAPH_ARR <<< "$GRAPH_TYPES"

for dataset in "${DATASET_ARR[@]}"; do
    case "$dataset" in
        sst2) HF_DATASET="stanfordnlp/sst2"; SPLITS="train validation" ;;
        ag_news) HF_DATASET="SetFit/ag_news"; SPLITS="train test" ;;
        *) HF_DATASET="$dataset"; SPLITS="train validation" ;;
    esac

    for graph_type in "${GRAPH_ARR[@]}"; do
        for split in $SPLITS; do
            echo "=========================================="
            echo "Embedding $dataset / $graph_type / $split"
            echo "=========================================="

            CMD="python -m src.graphtext.cli embed \
                --graph_type $graph_type \
                --dataset_name $HF_DATASET \
                --split $split \
                --output_dir $OUTPUT_BASE/$dataset/$split/$graph_type"

            if [[ "$DRY_RUN" == true ]]; then
                echo "[dry-run] $CMD"
            else
                eval $CMD || echo "[warn] Failed: $dataset/$graph_type/$split"
            fi
        done
    done
done

echo ""
echo "Embedding generation complete. PyG graphs in: $OUTPUT_BASE/"
