#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Smoke test runner for explainability pipelines.
# Processes a small sample (10 graphs/sentences per explainer) to validate the
# GPU execution path and sharded configuration without committing to the full
# dataset.
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the same defaults as the main runner unless explicitly overridden.
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export TOKENSHAP_PROCESSES="${TOKENSHAP_PROCESSES:-$NUM_PROCESSES}"
export GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
export MAX_GRAPHS_FLAG="--max-graphs 5"

echo ""
echo "===================================================================="
echo "Running explainability smoke test (10 samples per module)"
echo "GPU device: ${GPU_DEVICE}"
echo "Parallel shards per module: ${NUM_PROCESSES}"
echo "===================================================================="
echo ""

exec "${ROOT_DIR}/run_all_explainers.sh" "$@"
