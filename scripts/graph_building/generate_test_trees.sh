#!/bin/bash
# Generate Small Test Constituency Trees
# This script generates constituency parse trees from a very small dataset for testing/demo purposes.

# Configuration Parameters
GRAPH_TYPE="constituency"  # Only constituency is supported
DATASET="SetFit/sst2"      # Use a small or demo dataset; change if needed
SUBSETS=("validation")           # Use the test subset (usually small)
BATCH_SIZE=2
DEVICE="cpu"               # Use CPU for portability
OUTPUT_DIR="outputs/test_text_trees"

SCRIPT_DIR="$(dirname "$0")"

# Make the script executable (if needed)
chmod +x "$SCRIPT_DIR/Tree_Generation/tree_generator.py"

# Set PYTHONPATH so Python can find Clean_Code as a module
export PYTHONPATH="$(realpath "$SCRIPT_DIR/../../")"

# Run the tree generator with the configuration parameters
python -m "Clean_Code.Tree_Generation.tree_generator" \
  --graph_type "$GRAPH_TYPE" \
  --dataset "$DATASET" \
  --subsets "${SUBSETS[@]}" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output_dir "$OUTPUT_DIR" \

echo "Test constituency tree generation complete!"
