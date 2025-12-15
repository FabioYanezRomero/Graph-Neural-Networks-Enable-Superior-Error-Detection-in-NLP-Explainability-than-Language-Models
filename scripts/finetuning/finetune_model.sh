#!/bin/bash

# Script to run the Model_Finetuning module
# This script should be placed in the Clean_Code directory

# Set the Python path to include the parent directory
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(realpath $0)))

# Parse command line arguments
dataset_name=""
model_name="google-bert/bert-base-uncased"
num_epochs=5
batch_size=16
learning_rate=1e-6
output_dir="outputs/llm"
fp16=true

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dataset_name NAME      Dataset name (e.g., snli) [required]"
    echo "  --model_name NAME        Model name or path (default: google-bert/bert-base-uncased)"
    echo "  --num_epochs N           Number of training epochs (default: 5)"
    echo "  --batch_size N           Training batch size (default: 16)"
    echo "  --learning_rate RATE     Learning rate (default: 1e-6)"
    echo "  --output_dir DIR         Output directory (default: outputs/llm)"
    echo "  --no_fp16                Disable mixed precision training"
    echo "  --help                   Show this help message"
    echo ""
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset_name)
            dataset_name="$2"
            shift 2
            ;;
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --learning_rate)
            learning_rate="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --no_fp16)
            fp16=false
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

# Check if required arguments are provided
if [ -z "$dataset_name" ]; then
    echo "Error: --dataset_name is required"
    show_help
fi

# Construct the command
cmd="python -m Clean_Code.Model_Finetuning.finetuner --dataset_name $dataset_name"

# Add optional arguments if provided
if [ ! -z "$model_name" ]; then
    cmd="$cmd --model_name $model_name"
fi

cmd="$cmd --num_epochs $num_epochs"
cmd="$cmd --batch_size $batch_size"
cmd="$cmd --learning_rate $learning_rate"
cmd="$cmd --output_dir $output_dir"

if [ "$fp16" = false ]; then
    cmd="$cmd --no_fp16"
fi

# Clear CUDA cache if using CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "Clearing CUDA cache..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "Failed to clear CUDA cache"
fi

# Run the command
echo "Running: $cmd"
eval $cmd

exit $?
