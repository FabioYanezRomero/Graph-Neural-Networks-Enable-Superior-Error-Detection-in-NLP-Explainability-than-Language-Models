#!/usr/bin/env bash
set -euo pipefail

# Test script to validate the embedding generation pipeline
# Runs a quick test with a small subset of data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Testing embedding generation pipeline..."
echo "========================================"

# Test with a small dataset/graph combination
TEST_DATASET="stanfordnlp/sst2"
TEST_GRAPH_TYPE="constituency"
TEST_SUBSET="validation"
TEST_BATCH_SIZE=8

echo "Test parameters:"
echo "- Dataset: $TEST_DATASET"
echo "- Graph type: $TEST_GRAPH_TYPE"
echo "- Subset: $TEST_SUBSET"
echo "- Batch size: $TEST_BATCH_SIZE"
echo ""

# Create temporary test output directory
TEST_OUTPUT_BASE="/tmp/test_embeddings_$(date +%s)"
echo "Using test output directory: $TEST_OUTPUT_BASE"

# Clean up on exit
cleanup() {
    echo ""
    echo "Cleaning up test directory..."
    rm -rf "$TEST_OUTPUT_BASE"
}
trap cleanup EXIT

# Run the embedding generation script
echo "Step 1: Generating embeddings..."
"$SCRIPT_DIR/generate_embeddings_pyg.sh" \
    --dataset_name "$TEST_DATASET" \
    --graph_type "$TEST_GRAPH_TYPE" \
    --subsets "$TEST_SUBSET" \
    --batch_size "$TEST_BATCH_SIZE" \
    --output_base "$REPO_ROOT/outputs" \
    --force

echo ""
echo "Step 2: Validating generated embeddings..."
python3 "$SCRIPT_DIR/validate_embeddings_pyg.py" \
    --dataset_name "$TEST_DATASET" \
    --graph_type "$TEST_GRAPH_TYPE" \
    --subset "$TEST_SUBSET" \
    --output_base "/app/scripts/outputs"

echo ""
echo "Step 3: Checking output files..."
echo "Embedding files:"
find "/app/scripts/outputs/embeddings/$TEST_DATASET/$TEST_SUBSET/$TEST_GRAPH_TYPE" -name "*.pkl" | head -5

echo ""
echo "PyG files:"
find "/app/scripts/outputs/pyg_graphs/$TEST_DATASET/$TEST_SUBSET/$TEST_GRAPH_TYPE" -name "*.pt" | head -5

echo ""
echo "Step 4: Checking file sizes and basic structure..."
EMBEDDING_COUNT=$(find "/app/scripts/outputs/embeddings/$TEST_DATASET/$TEST_SUBSET/$TEST_GRAPH_TYPE" -name "*.pkl" | wc -l)
PYG_COUNT=$(find "/app/scripts/outputs/pyg_graphs/$TEST_DATASET/$TEST_SUBSET/$TEST_GRAPH_TYPE" -name "*.pt" | wc -l)

echo "Generated $EMBEDDING_COUNT embedding files"
echo "Generated $PYG_COUNT PyG files"

if [[ $EMBEDDING_COUNT -gt 0 && $PYG_COUNT -gt 0 ]]; then
    echo "✅ File generation successful!"

    # Quick check of first PyG file
    FIRST_PYG_FILE=$(find "/app/scripts/outputs/pyg_graphs/$TEST_DATASET/$TEST_SUBSET/$TEST_GRAPH_TYPE" -name "*.pt" | head -1)
    if [[ -n "$FIRST_PYG_FILE" ]]; then
        echo "Checking first PyG file structure..."
        python3 -c "
import torch
graphs = torch.load('$FIRST_PYG_FILE')
print(f'Loaded {len(graphs)} graphs')
if graphs:
    g = graphs[0]
    print(f'First graph: {g.x.size(0)} nodes, {g.edge_index.size(1)} edges, {g.x.size(1)}-dim embeddings')
    if hasattr(g, 'y') and g.y is not None:
        print(f'Labels available: {g.y}')
        print('✅ PyG structure with labels looks correct!')
    else:
        print('❌ Missing labels in PyG graphs')
"
    fi
else
    echo "❌ File generation failed!"
    exit 1
fi

echo ""
echo "🎉 Pipeline test completed successfully!"
echo ""
echo "To run the full pipeline on all data, use:"
echo "  $SCRIPT_DIR/generate_all_embeddings_pyg.sh --force"
echo ""
echo "To validate all generated embeddings:"
echo "  python3 $SCRIPT_DIR/validate_embeddings_pyg.py --all"
