#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_MODULE="src.gnn_training.training"

PYG_BASE=${PYG_BASE:-"$REPO_ROOT/outputs/pyg_graphs"}
MODEL_BASE=${MODEL_BASE:-"$REPO_ROOT/outputs/gnn_models"}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PATIENCE=${PATIENCE:-5}
GRAD_ACC=${GRAD_ACC:-1}
DATASETS_FILTER=""
GRAPH_TYPES_FILTER=""
DRY_RUN=false
DEVICE_ENV=""

show_help() {
    cat <<'USAGE'
Usage: train_all_gnns.sh [options]

Iterate over all available PyG graph datasets and train a 2-layer GCN on each.
Requires embeddings generated under outputs/pyg_graphs/...

Options:
  --epochs N           Number of epochs (default: env EPOCHS or 20)
  --batch_size N       Training batch size (default: env BATCH_SIZE or 32)
  --learning_rate LR   Learning rate (default: env LEARNING_RATE or 0.001)
  --weight_decay WD    Weight decay (default: env WEIGHT_DECAY or 1e-4)
  --num_workers N      DataLoader workers (default: env NUM_WORKERS or 4)
  --patience N         Early stopping patience (default: env PATIENCE or 5)
  --grad_acc N         Gradient accumulation steps (default: env GRAD_ACC or 1)
  --datasets LIST      Comma-separated dataset names to include
  --graph_types LIST   Comma-separated graph types (constituency, syntactic, window, ngrams, skipgrams, knowledge, ...)
  --model_base DIR     Output directory for trained models (default: outputs/gnn_models)
  --pyg_base DIR       Base directory containing generated PyG graphs (default: outputs/pyg_graphs)
  --device DEV         Set CUDA_VISIBLE_DEVICES before training (e.g., 0)
  --dry_run            Print commands without executing
  --help               Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --grad_acc)
            GRAD_ACC="$2"
            shift 2
            ;;
        --datasets)
            DATASETS_FILTER="$2"
            shift 2
            ;;
        --graph_types)
            GRAPH_TYPES_FILTER="$2"
            shift 2
            ;;
        --model_base)
            MODEL_BASE="$2"
            shift 2
            ;;
        --pyg_base)
            PYG_BASE="$2"
            shift 2
            ;;
        --device)
            DEVICE_ENV="$2"
            shift 2
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

map_graph_type_to_subdir() {
    local graph_type=$1
    case "$graph_type" in
        constituency)
            echo "constituency"
            ;;
        syntactic)
            echo "syntactic"
            ;;
        semantic)
            echo "semantic"
            ;;
        fully)
            echo "fully"
            ;;
        knn)
            echo "knn"
            ;;
        knowledge)
            echo "knowledge"
            ;;
        *)
            echo "${graph_type}"
            ;;
    esac
}

dir_has_graphs() {
    local dir=$1
    [[ -d "$dir" ]] || return 1
    find "$dir" -maxdepth 1 \( -name '*.pt' -o -name '*.pkl' \) -print -quit | grep -q .
}

map_subdir_to_graph_type() {
    local dir_name=$1
    case "$dir_name" in
        constituency)
            echo "constituency"
            ;;
        syntactic)
            echo "syntactic"
            ;;
        window)
            echo "window"
            ;;
        window.*)
            echo "window"
            ;;
        ngrams)
            echo "ngrams"
            ;;
        ngrams.*)
            echo "ngrams"
            ;;
        skipgrams)
            echo "skipgrams"
            ;;
        skipgrams.*)
            echo "skipgrams"
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
        knowledge)
            echo "knowledge"
            ;;
        knowledge*)
            echo "knowledge"
            ;;
        *)
            echo ""
            ;;
    esac
}

IFS=',' read -r -a DATASET_FILTER_ARR <<< "$DATASETS_FILTER"
declare -A DATASET_FILTER_MAP=()
for ds in "${DATASET_FILTER_ARR[@]}"; do
    [[ -n "$ds" ]] && DATASET_FILTER_MAP["$ds"]=1
done

IFS=',' read -r -a GRAPH_FILTER_ARR <<< "$GRAPH_TYPES_FILTER"
declare -A GRAPH_FILTER_MAP=()
for gt in "${GRAPH_FILTER_ARR[@]}"; do
    [[ -n "$gt" ]] && GRAPH_FILTER_MAP["$gt"]=1
done

if [[ ! -d "$PYG_BASE" ]]; then
    echo "[error] PyG base directory not found: $PYG_BASE" >&2
    exit 1
fi

mapfile -t DATASETS < <(find "$PYG_BASE" -mindepth 2 -maxdepth 2 -type d -print | sort)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "[warn] No datasets found under $PYG_BASE" >&2
    exit 0
fi

echo "Training GNNs using PyG graphs from $PYG_BASE"
echo "Model outputs will be stored under $MODEL_BASE"

for dataset_path in "${DATASETS[@]}"; do
    dataset_name=${dataset_path#"$PYG_BASE/"}
    [[ -n "$DATASETS_FILTER" && -z "${DATASET_FILTER_MAP[$dataset_name]:-}" ]] && continue

    declare -A GRAPH_TYPES_PRESENT=()
    declare -A GRAPH_TYPE_DIR_MAP=()
    while IFS= read -r subset_dir; do
        while IFS= read -r graph_dir; do
            dir_name=$(basename "$graph_dir")
            graph_type=$(map_subdir_to_graph_type "$dir_name")
            [[ -z "$graph_type" ]] && continue
            if (( ${#GRAPH_FILTER_MAP[@]} > 0 )) && [[ -z "${GRAPH_FILTER_MAP[$graph_type]:-}" ]]; then
                continue
            fi
            if dir_has_graphs "$graph_dir"; then
                GRAPH_TYPES_PRESENT[$graph_type]=1
                GRAPH_TYPE_DIR_MAP[$graph_type]="$dir_name"
            fi
        done < <(find "$subset_dir" -mindepth 1 -maxdepth 1 -type d -print | sort)
    done < <(find "$dataset_path" -mindepth 1 -maxdepth 1 -type d -print | sort)

    if (( ${#GRAPH_TYPES_PRESENT[@]} == 0 )); then
        echo "[warn] No matching graph types for dataset $dataset_name"
        continue
    fi

    for graph_type in "${!GRAPH_TYPES_PRESENT[@]}"; do
        graph_subdir=${GRAPH_TYPE_DIR_MAP[$graph_type]:-$(map_graph_type_to_subdir "$graph_type")}
        train_dir="$dataset_path/train/$graph_subdir"
        if ! dir_has_graphs "$train_dir"; then
            echo "[warn] Skipping $dataset_name/$graph_type (no train graphs found)"
            continue
        fi

        val_dir=""
        for candidate in validation val dev; do
            cand_path="$dataset_path/$candidate/$graph_subdir"
            if dir_has_graphs "$cand_path"; then
                val_dir="$cand_path"
                break
            fi
        done

        test_dir=""
        if dir_has_graphs "$dataset_path/test/$graph_subdir"; then
            test_dir="$dataset_path/test/$graph_subdir"
        fi

        timestamp=$(date +'%Y%m%d_%H%M%S')
        run_dir="$MODEL_BASE/$dataset_name/$graph_type/$timestamp"
        mkdir -p "$run_dir"

        cmd=(python -m "$TRAIN_MODULE"
            --train_data_dir "$train_dir"
            --module GCNConv
            --num_layers 2
            --epochs "$EPOCHS"
            --batch_size "$BATCH_SIZE"
            --learning_rate "$LEARNING_RATE"
            --weight_decay "$WEIGHT_DECAY"
            --patience "$PATIENCE"
            --num_workers "$NUM_WORKERS"
            --gradient_accumulation_steps "$GRAD_ACC"
            --pooling mean
            --heads 1
            --dropout 0.5
            --output_dir "$run_dir")

        [[ -n "$val_dir" ]] && cmd+=(--val_data_dir "$val_dir")
        [[ -n "$test_dir" ]] && cmd+=(--test_data_dir "$test_dir")

        echo "[info] Training $dataset_name / $graph_type"
        if [[ "$DRY_RUN" == true ]]; then
            if [[ -n "$DEVICE_ENV" ]]; then
                printf '[dry-run] CUDA_VISIBLE_DEVICES=%s %s\n' "$DEVICE_ENV" "${cmd[*]}"
            else
                printf '[dry-run] %s\n' "${cmd[*]}"
            fi
        else
            if [[ -n "$DEVICE_ENV" ]]; then
                CUDA_VISIBLE_DEVICES="$DEVICE_ENV" "${cmd[@]}"
            else
                "${cmd[@]}"
            fi
        fi
    done
    unset GRAPH_TYPES_PRESENT GRAPH_TYPE_DIR_MAP
    declare -A GRAPH_TYPES_PRESENT
    declare -A GRAPH_TYPE_DIR_MAP
    echo

done
