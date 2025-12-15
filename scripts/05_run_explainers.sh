#!/usr/bin/env bash
# ==============================================================================
# Step 5: Run Explainability (Section 3.4)
# Containers: subgraphx, graphsvx, tokenshap
# Usage: ./scripts/05_run_explainers.sh [--datasets sst2,ag_news] [--methods all]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_CMD="docker compose"
FAIR_FLAG="--fair"

# Defaults
DATASETS="sst2,ag_news"
METHODS="all"  # all, gnn, llm, subgraphx, graphsvx, tokenshap
DRY_RUN=false

show_help() {
    cat <<'USAGE'
Usage: 05_run_explainers.sh [options]

Run explainability modules in their respective containers (Paper Section 3.4)

Explainer Mapping:
  - SubgraphX → constituency, syntactic (tree graphs)
  - GraphSVX → skipgrams, window (non-tree graphs)
  - TokenSHAP → LLM baseline

Options:
  --datasets LIST      Comma-separated: sst2,ag_news (default: both)
  --methods LIST       Comma-separated: subgraphx,graphsvx,tokenshap (default: all)
  --dry-run            Print commands without executing
  --help               Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS="$2"; shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help) show_help ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

run_in_container() {
    local service=$1
    local cmd=$2
    echo "[$service] $cmd"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[dry-run] $COMPOSE_CMD run --rm $service bash -lc \"cd /app && $cmd\""
    else
        $COMPOSE_CMD run --rm --gpus all "$service" bash -lc "cd /app && $cmd"
    fi
}

run_subgraphx() {
    local dataset=$1
    local backbone=$2
    local graph_type=$3
    local split=$4
    run_in_container subgraphx \
        "python -m src.explain.gnn.subgraphx.main --dataset '$dataset' --graph-type '$graph_type' --backbone '$backbone' --split '$split' $FAIR_FLAG"
}

run_graphsvx() {
    local dataset=$1
    local backbone=$2
    local graph_type=$3
    local split=$4
    run_in_container graphsvx \
        "python -m src.explain.gnn.graphsvx.main --dataset '$dataset' --graph-type '$graph_type' --backbone '$backbone' --split '$split' $FAIR_FLAG"
}

run_tokenshap() {
    local profile=$1
    run_in_container tokenshap \
        "python -m src.explain.llm.main explain '$profile' $FAIR_FLAG"
}

should_run() {
    local method=$1
    [[ "$METHODS" == "all" ]] && return 0
    [[ "$METHODS" == *"$method"* ]] && return 0
    [[ "$METHODS" == "gnn" && ("$method" == "subgraphx" || "$method" == "graphsvx") ]] && return 0
    [[ "$METHODS" == "llm" && "$method" == "tokenshap" ]] && return 0
    return 1
}

IFS=',' read -ra DATASET_ARR <<< "$DATASETS"

echo "=========================================="
echo "Step 5: Running Explainability"
echo "  Datasets: $DATASETS"
echo "  Methods: $METHODS"
echo "=========================================="

for dataset in "${DATASET_ARR[@]}"; do
    case "$dataset" in
        sst2) 
            HF_DATASET="sst2"
            BACKBONE="stanfordnlp"
            SPLIT="validation"
            ;;
        ag_news)
            HF_DATASET="ag_news"
            BACKBONE="SetFit"
            SPLIT="test"
            ;;
        *) 
            HF_DATASET="$dataset"
            BACKBONE="stanfordnlp"
            SPLIT="validation"
            ;;
    esac

    echo ""
    echo "--- Dataset: $dataset ---"

    # SubgraphX for tree graphs
    if should_run "subgraphx"; then
        for graph_type in constituency syntactic; do
            run_subgraphx "$HF_DATASET" "$BACKBONE" "$graph_type" "$SPLIT"
        done
    fi

    # GraphSVX for non-tree graphs
    if should_run "graphsvx"; then
        for graph_type in skipgrams window; do
            run_graphsvx "$HF_DATASET" "$BACKBONE" "$graph_type" "$SPLIT"
        done
    fi

    # TokenSHAP for LLM baseline
    if should_run "tokenshap"; then
        run_tokenshap "${BACKBONE}/${HF_DATASET}"
    fi
done

echo ""
echo "Explainability complete."
