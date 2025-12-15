#!/bin/bash
# Parallel TokenSHAP runner using sharding
# Usage: ./scripts/run_llm_sharded.sh <dataset> <num_shards> [additional args]

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset> <num_shards> [additional args]"
    echo "Example: $0 stanfordnlp/sst2 4"
    echo "Example: $0 setfit/ag_news 8 --max-samples 1000"
    exit 1
fi

DATASET=$1
NUM_SHARDS=$2
shift 2
EXTRA_ARGS="$@"

echo "=================================================="
echo "Running TokenSHAP with sharding"
echo "=================================================="
echo "Dataset: $DATASET"
echo "Number of shards: $NUM_SHARDS"
echo "Extra arguments: $EXTRA_ARGS"
echo "=================================================="
echo ""

# Array to store background process PIDs
PIDS=()

# Launch all shards in parallel
for ((i=0; i<NUM_SHARDS; i++)); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching shard $((i+1))/$NUM_SHARDS..."
    
    python3 -m src.explain.llm.main explain "$DATASET" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$i" \
        $EXTRA_ARGS &
    
    PID=$!
    PIDS+=($PID)
    echo "  └─ Shard $((i+1)) started (PID: $PID)"
    
    # Small delay to stagger starts
    sleep 2
done

echo ""
echo "All shards launched! Waiting for completion..."
echo ""

# Wait for all background processes and track failures
FAILED=0
for ((i=0; i<NUM_SHARDS; i++)); do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Shard $((i+1))/$NUM_SHARDS completed successfully (PID: $PID)"
    else
        EXIT_CODE=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Shard $((i+1))/$NUM_SHARDS failed with exit code $EXIT_CODE (PID: $PID)"
        FAILED=1
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=================================================="
    echo "✓ All shards completed successfully!"
    echo "=================================================="
    exit 0
else
    echo "=================================================="
    echo "✗ Some shards failed. Check logs above."
    echo "=================================================="
    exit 1
fi



