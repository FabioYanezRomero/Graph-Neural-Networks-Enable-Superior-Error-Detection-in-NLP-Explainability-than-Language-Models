#!/usr/bin/env bash
# ==============================================================================
# Step 2: Build Graphs (Section 3.1)
# Container: app
# Usage: ./scripts/02_build_graphs.sh [--dataset sst2|ag_news] [--graph-types all|constituency,syntactic,skipgrams,window]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
DATASETS="sst2,ag_news"
GRAPH_TYPES="constituency,syntactic,skipgrams,window"
OUTPUT_BASE="outputs/graphs"
DRY_RUN=false

show_help() {
    cat <<'USAGE'
Usage: 02_build_graphs.sh [options]

Build graph representations from text (Paper Section 3.1)

Graph Types (per paper):
  - constituency: Constituency parse trees
  - syntactic: Dependency parse trees  
  - skipgrams: Skip-gram co-occurrence graphs
  - window: Window-based proximity graphs

Options:
  --datasets LIST      Comma-separated: sst2,ag_news (default: both)
  --graph-types LIST   Comma-separated graph types (default: all four)
  --output-dir DIR     Output base directory (default: outputs/graphs)
  --dry-run            Print commands without executing
  --help               Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS="$2"; shift 2 ;;
        --graph-types) GRAPH_TYPES="$2"; shift 2 ;;
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
        sst2) HF_DATASET="stanfordnlp/sst2"; SPLITS="train,validation" ;;
        ag_news) HF_DATASET="SetFit/ag_news"; SPLITS="train,test" ;;
        *) HF_DATASET="$dataset"; SPLITS="train,validation" ;;
    esac

    for graph_type in "${GRAPH_ARR[@]}"; do
        echo "=========================================="
        echo "Building $graph_type graphs for $dataset"
        echo "=========================================="

        CMD="python -m src.graphtext.cli build-graphs \
            --graph_type $graph_type \
            --dataset $HF_DATASET \
            --output_dir $OUTPUT_BASE/$dataset/$graph_type"

        if [[ "$DRY_RUN" == true ]]; then
            echo "[dry-run] $CMD"
        else
            eval $CMD || echo "[warn] Failed to build $graph_type for $dataset"
        fi
    done
done

echo ""
echo "Graph building complete. Outputs in: $OUTPUT_BASE/"
