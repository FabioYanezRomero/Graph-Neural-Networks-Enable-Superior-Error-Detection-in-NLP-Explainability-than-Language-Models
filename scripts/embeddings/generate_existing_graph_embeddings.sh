#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_SCRIPT="$SCRIPT_DIR/generate_embeddings_pyg.sh"

BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs}
FORCE=false
DRY_RUN=false
DATASETS_OVERRIDE=""
GRAPH_TYPES_OVERRIDE=""

show_help() {
    cat <<'USAGE'
Usage: generate_existing_graph_embeddings.sh [options]

Scan outputs/graphs and generate embeddings for every dataset/graph type pair found.

Options:
  --batch_size N        Processing batch size (default: env BATCH_SIZE or 128)
  --device DEVICE       Device to use (default: env DEVICE or cuda:0)
  --output_base DIR     Base output directory (default: env OUT_BASE or outputs)
  --datasets LIST       Comma-separated dataset names to restrict processing
  --graph_types LIST    Comma-separated graph types to restrict (constituency, syntactic, window, ngrams, skipgrams, ...)
  --force               Overwrite existing embeddings
  --dry_run             Print the commands without executing
  --help                Show this message and exit
USAGE
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output_base)
            OUT_BASE="$2"
            shift 2
            ;;
        --datasets)
            DATASETS_OVERRIDE="$2"
            shift 2
            ;;
        --graph_types)
            GRAPH_TYPES_OVERRIDE="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

map_graph_dir_to_type() {
    local dir_name=$1
    case "$dir_name" in
        constituency)
            echo "constituency"
            ;;
        syntactic)
            echo "syntactic"
            ;;
        window.*)
            echo "window"
            ;;
        ngrams.*)
            echo "ngrams"
            ;;
        skipgrams.*)
            echo "skipgrams"
            ;;
        knowledge)
            echo "knowledge"
            ;;
        semantic*)
            echo "semantic"
            ;;
        fully*)
            echo "fully"
            ;;
        knn*)
            echo "knn"
            ;;
        *)
            echo ""
            ;;
    esac
}

IFS=',' read -r -a DATASET_FILTER <<< "$DATASETS_OVERRIDE"
declare -A DATASET_FILTER_MAP=()
for ds in "${DATASET_FILTER[@]}"; do
    [[ -n "$ds" ]] && DATASET_FILTER_MAP["$ds"]=1
done

IFS=',' read -r -a GRAPH_FILTER <<< "$GRAPH_TYPES_OVERRIDE"
declare -A GRAPH_FILTER_MAP=()
for gt in "${GRAPH_FILTER[@]}"; do
    [[ -n "$gt" ]] && GRAPH_FILTER_MAP["$gt"]=1
done

if [[ ! -d "$OUT_BASE/graphs" ]]; then
    echo "[error] No graphs directory found at $OUT_BASE/graphs" >&2
    exit 1
fi

declare -a DATASETS=()
if [[ -n "$DATASETS_OVERRIDE" ]]; then
    DATASETS=("${DATASET_FILTER[@]}")
else
    while IFS= read -r -d '' dataset_dir; do
        DATASETS+=("${dataset_dir##*/}")
    done < <(find "$OUT_BASE/graphs" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[dry-run] $*"
    else
        eval "$@"
    fi
}

for dataset in "${DATASETS[@]}"; do
    [[ -z "$dataset" ]] && continue
    if (( ${#DATASET_FILTER_MAP[@]} > 0 )) && [[ -z "${DATASET_FILTER_MAP[$dataset]:-}" ]]; then
        continue
    fi
    dataset_path="$OUT_BASE/graphs/$dataset"
    [[ ! -d "$dataset_path" ]] && continue

    declare -A GRAPH_TYPES_SEEN=()
    while IFS= read -r -d '' subset_dir; do
        subset_path="$subset_dir"
        while IFS= read -r -d '' graph_dir; do
            dir_name="${graph_dir##*/}"
            graph_type=$(map_graph_dir_to_type "$dir_name")
            [[ -z "$graph_type" ]] && continue
            if (( ${#GRAPH_FILTER_MAP[@]} > 0 )) && [[ -z "${GRAPH_FILTER_MAP[$graph_type]:-}" ]]; then
                continue
            fi
            GRAPH_TYPES_SEEN["$graph_type"]=1
        done < <(find "$subset_path" -mindepth 1 -maxdepth 1 -type d -print0)
    done < <(find "$dataset_path" -mindepth 1 -maxdepth 1 -type d -print0)

    if (( ${#GRAPH_TYPES_SEEN[@]} == 0 )); then
        echo "[warn] No graph types found for dataset $dataset in $dataset_path" >&2
    fi

    for graph_type in "${!GRAPH_TYPES_SEEN[@]}"; do
        cmd=("$BASE_SCRIPT" --dataset_name "$dataset" --graph_type "$graph_type" --batch_size "$BATCH_SIZE" --device "$DEVICE" --output_base "$OUT_BASE")
        if [[ "$FORCE" == true ]]; then
            cmd+=(--force)
        fi
        if [[ "$DRY_RUN" == true ]]; then
            echo "[dry-run] ${cmd[*]}"
            continue
        fi
        echo "[info] Running ${cmd[*]}"
        "${cmd[@]}"
    done
    unset GRAPH_TYPES_SEEN
done
