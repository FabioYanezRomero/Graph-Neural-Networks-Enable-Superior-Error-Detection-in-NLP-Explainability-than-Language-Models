#!/usr/bin/env bash
set -euo pipefail

# Script to generate embeddings and PyTorch Geometric format for ALL datasets and graph types
# This script processes everything in batch mode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EMBEDDING_SCRIPT="$SCRIPT_DIR/generate_embeddings_pyg.sh"

# Configuration
DATASETS=("stanfordnlp/sst2" "SetFit/ag_news")
GRAPH_TYPES=("constituency" "syntactic" "window" "ngrams" "skipgrams" "knowledge")
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs}

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Generate embeddings and PyTorch Geometric format for ALL datasets and graph types."
    echo ""
    echo "Options:"
    echo "  --batch_size N           Processing batch size (default: 128)"
    echo "  --device DEVICE          Device to use (default: cuda:0)"
    echo "  --output_base DIR        Base output directory (default: outputs)"
    echo "  --datasets LIST          Comma-separated datasets (default: all)"
    echo "  --graph_types LIST       Comma-separated graph types (default: all)"
    echo "  --subsets LIST           Comma-separated subsets (default: auto-detect)"
    echo "  --force                  Overwrite existing embeddings"
    echo "  --dry_run                Show what would be done without executing"
    echo "  --parallel               Run in parallel (one process per dataset+graph_type)"
    echo "  --max_jobs N             Max parallel jobs (default: 4)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Process all datasets and graph types"
    echo "  $0 --datasets stanfordnlp/sst2       # Only SST-2 dataset"
    echo "  $0 --graph_types constituency,syntactic  # Only constituency and syntactic"
    echo "  $0 --parallel --max_jobs 2           # Run in parallel with max 2 jobs"
    echo ""
}

# Parse arguments
batch_size="$BATCH_SIZE"
device="$DEVICE"
output_base="$OUT_BASE"
datasets=""
graph_types=""
subsets=""
force=false
dry_run=false
parallel=false
max_jobs=4

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --datasets)
            datasets="$2"
            shift 2
            ;;
        --graph_types)
            graph_types="$2"
            shift 2
            ;;
        --subsets)
            subsets="$2"
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
        --parallel)
            parallel=true
            shift
            ;;
        --max_jobs)
            max_jobs="$2"
            shift 2
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

if [[ "$subsets" =~ ^[Aa][Ll][Ll]$ ]]; then
    subsets=""
fi

# Set datasets and graph types
if [[ -z "$datasets" ]]; then
    datasets_list=("${DATASETS[@]}")
else
    IFS=',' read -r -a datasets_list <<< "$datasets"
fi

if [[ -z "$graph_types" ]]; then
    graph_types_list=("${GRAPH_TYPES[@]}")
else
    IFS=',' read -r -a graph_types_list <<< "$graph_types"
fi

# Build common arguments
common_args="--batch_size $batch_size --device $device --output_base $output_base"
if [[ "$force" == true ]]; then
    common_args="$common_args --force"
fi
if [[ "$dry_run" == true ]]; then
    common_args="$common_args --dry_run"
fi
if [[ -n "$subsets" ]]; then
    common_args="$common_args --subsets $subsets"
fi

echo "Starting batch embedding generation..."
echo "Datasets: ${datasets_list[*]}"
echo "Graph types: ${graph_types_list[*]}"
echo "Common args: $common_args"
echo ""

# Function to check if graph type exists for dataset/subset
check_graph_exists() {
    local dataset=$1
    local graph_type=$2
    local subset=$3

    # Determine graph directory pattern
    local graph_dir_pattern
    case "$graph_type" in
        constituency)
            graph_dir_pattern="constituency"
            ;;
        syntactic)
            graph_dir_pattern="syntactic"
            ;;
        window)
            graph_dir_pattern="window.word.k1"
            ;;
        ngrams)
            graph_dir_pattern="ngrams.word.n2"
            ;;
        skipgrams)
            graph_dir_pattern="skipgrams.word.k1.n2"
            ;;
        knowledge)
            graph_dir_pattern="knowledge"
            ;;
        *)
            graph_dir_pattern="$graph_type"
            ;;
    esac

    local graph_dir="$output_base/graphs/$dataset/$subset/$graph_dir_pattern"
    [[ -d "$graph_dir" ]]
}

# Function to process a single combination
process_combination() {
    local dataset=$1
    local graph_type=$2

    echo "Processing $dataset + $graph_type..."

    # Build command
    local cmd="$EMBEDDING_SCRIPT --dataset_name $dataset --graph_type $graph_type $common_args"

    echo "[$(date)] Running: $cmd"
    if [[ "$dry_run" != true ]]; then
        if eval "$cmd"; then
            echo "[$(date)] SUCCESS: $dataset + $graph_type completed"
        else
            echo "[$(date)] ERROR: $dataset + $graph_type failed"
            return 1
        fi
    else
        echo "[$(date)] DRY RUN: Would run: $cmd"
    fi
}

# Function to get available subsets for a dataset
get_available_subsets() {
    local dataset=$1
    local graph_root="$output_base/graphs/$dataset"
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

# Collect all combinations to process
combinations=()
for dataset in "${datasets_list[@]}"; do
    echo "Checking available subsets for $dataset..."
    available_subsets=$(get_available_subsets "$dataset")
    echo "Available subsets: $available_subsets"

    for graph_type in "${graph_types_list[@]}"; do
        # Check if at least one subset has this graph type
        has_graph_type=false
        for subset in $available_subsets; do
            if check_graph_exists "$dataset" "$graph_type" "$subset"; then
                has_graph_type=true
                break
            fi
        done

        if [[ "$has_graph_type" == true ]]; then
            combinations+=("$dataset|$graph_type")
            echo "✓ Will process: $dataset + $graph_type"
        else
            echo "✗ Skipping: $dataset + $graph_type (no graphs found)"
        fi
    done
done

echo ""
echo "Found ${#combinations[@]} combinations to process:"
for combo in "${combinations[@]}"; do
    echo "  $combo"
done
echo ""

# Process combinations
if [[ "$parallel" == true ]]; then
    echo "Running in parallel mode with max $max_jobs jobs..."

    # Process in parallel with job control
    running_jobs=0
    job_pids=()

    for combo in "${combinations[@]}"; do
        IFS='|' read -r dataset graph_type <<< "$combo"

        # Wait for a job slot if needed
        while [[ $running_jobs -ge $max_jobs ]]; do
            # Check if any jobs have finished
            for i in "${!job_pids[@]}"; do
                if ! kill -0 "${job_pids[$i]}" 2>/dev/null; then
                    # Job finished
                    unset job_pids[$i]
                    ((running_jobs--))
                fi
            done
            # Clean up array
            job_pids=("${job_pids[@]}")
            # Wait a bit before checking again
            if [[ $running_jobs -ge $max_jobs ]]; then
                sleep 5
            fi
        done

        # Start new job
        process_combination "$dataset" "$graph_type" &
        job_pid=$!
        job_pids+=($job_pid)
        ((running_jobs++))

        echo "Started job (PID: $job_pid) for $dataset + $graph_type (running: $running_jobs/$max_jobs)"
    done

    # Wait for all remaining jobs to finish
    echo "Waiting for remaining jobs to finish..."
    for pid in "${job_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
        fi
    done

else
    # Sequential processing
    echo "Running in sequential mode..."

    for combo in "${combinations[@]}"; do
        IFS='|' read -r dataset graph_type <<< "$combo"

        if ! process_combination "$dataset" "$graph_type"; then
            echo "Failed to process $dataset + $graph_type, continuing with others..."
        fi
    done
fi

echo ""
echo "[$(date)] Batch processing completed!"
echo "Summary:"
echo "- Datasets processed: ${#datasets_list[@]}"
echo "- Graph types processed: ${#graph_types_list[@]}"
echo "- Combinations attempted: ${#combinations[@]}"
