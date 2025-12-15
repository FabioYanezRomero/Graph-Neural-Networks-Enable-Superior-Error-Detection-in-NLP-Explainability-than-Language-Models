#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Run all explainability methods with fair comparison mode
# ==============================================================================
# This script runs:
# - GraphSVX (GNN) on skipgrams and window graphs
# - SubgraphX (GNN) on constituency and syntactic graphs
# - TokenSHAP (LLM) on finetuned transformers
#
# All methods use --fair flag for 400 forward passes (default)
# Reduced from 2000 to make SubgraphX tractable while maintaining fairness
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_CMD="docker compose"
FAIR_FLAG="--fair"
FAIR_BUDGET=${FAIR_BUDGET:-2000}
NUM_PROCESSES=${NUM_PROCESSES:-3}
TOKENSHAP_PROCESSES=${TOKENSHAP_PROCESSES:-$NUM_PROCESSES}
GPU_DEVICE=${GPU_DEVICE:-cuda:0}

# Optional: Override to disable fair mode
# FAIR_FLAG=""

# Optional: Set custom target forward passes (e.g., higher budget)
# FAIR_FLAG="--fair --target-forward-passes 1000"

# Optional: Limit graphs for testing (e.g., first 100 samples)
# Export MAX_GRAPHS_FLAG="" to process all graphs.
if [[ -z "${MAX_GRAPHS_FLAG+x}" ]]; then
    MAX_GRAPHS_FLAG=""
fi

run_in_container() {
    local service=$1
    local use_gpu=$2
    local cmd=$3
    echo ""
    echo "===================================================================="
    echo "[${service}] ${cmd}"
    echo "===================================================================="
    local compose_parts=(${COMPOSE_CMD})
    local run_args=(run --rm -T)
    if [[ "${use_gpu}" == "true" ]]; then
        if [[ "${GPU_DEVICE}" =~ ^cuda:([0-9]+)$ ]]; then
            run_args+=(-e "CUDA_VISIBLE_DEVICES=${BASH_REMATCH[1]}")
        elif [[ "${GPU_DEVICE}" == "cuda" ]]; then
            run_args+=(-e "CUDA_VISIBLE_DEVICES=0")
        fi
    fi
    run_args+=(--entrypoint /bin/bash "${service}" -c "cd /app && ${cmd}")
    "${compose_parts[@]}" "${run_args[@]}"
}

run_sharded_module() {
    local service=$1
    local use_gpu=$2
    local base_cmd=$3
    local processes=${4:-$NUM_PROCESSES}

    if (( processes <= 1 )); then
        run_in_container "${service}" "${use_gpu}" "${base_cmd} --num-shards 1 --shard-index 0"
        return
    fi

    echo "Launching ${processes} parallel shards for ${service}..."
    local pids=()
    for (( shard = 0; shard < processes; ++shard )); do
        local shard_cmd="${base_cmd} --num-shards ${processes} --shard-index ${shard}"
        run_in_container "${service}" "${use_gpu}" "${shard_cmd}" &
        pids+=($!)
    done

    local status=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done

    if (( status != 0 )); then
        echo "Error: One or more shards failed for ${service}" >&2
        exit 1
    fi
}

run_graphsvx() {
    local dataset=$1
    local backbone=$2
    local graph_type=$3
    local split=$4
    local cmd="python -m src.explain.gnn.graphsvx.main --dataset '${dataset}' --graph-type '${graph_type}' --backbone '${backbone}' --split '${split}' --device '${GPU_DEVICE}' ${FAIR_FLAG} --target-forward-passes ${FAIR_BUDGET} ${MAX_GRAPHS_FLAG}"
    run_sharded_module graphsvx true "${cmd}"
}

run_subgraphx() {
    local dataset=$1
    local backbone=$2
    local graph_type=$3
    local split=$4
    local cmd="python -m src.explain.gnn.subgraphx.main --dataset '${dataset}' --graph-type '${graph_type}' --backbone '${backbone}' --split '${split}' --device '${GPU_DEVICE}' ${FAIR_FLAG} --target-forward-passes ${FAIR_BUDGET} ${MAX_GRAPHS_FLAG}"
    run_sharded_module subgraphx true "${cmd}"
}

run_tokenshap() {
    local profile=$1
    local max_samples_flag=${MAX_GRAPHS_FLAG/--max-graphs/--max-samples}
    local cmd="python -m src.explain.llm.main explain '${profile}' --device '${GPU_DEVICE}' ${FAIR_FLAG} --target-forward-passes ${FAIR_BUDGET} ${max_samples_flag}"
    local processes=${TOKENSHAP_PROCESSES}
    if (( processes > 1 )); then
        echo "Priming TokenSHAP cache for profile '${profile}'..."
        local warmup_cmd="python -m src.explain.llm.main explain '${profile}' --device '${GPU_DEVICE}' ${FAIR_FLAG} --target-forward-passes ${FAIR_BUDGET} --num-shards ${processes} --shard-index 0 --max-samples 1 --no-raw --no-progress --output-basename __warmup__"
        run_in_container tokenshap true "${warmup_cmd}"
    fi
    run_sharded_module tokenshap true "${cmd}" "${processes}"
}

main() {
    cd "${ROOT_DIR}" || exit 1

    echo ""
    echo "===================================================================="
    echo "Starting all explainability methods with fair comparison mode"
    echo "Target forward passes: 400 (default, optimized for SubgraphX)"
    echo "GPU device: ${GPU_DEVICE}"
    if (( TOKENSHAP_PROCESSES == NUM_PROCESSES )); then
        echo "Parallel shards per module: ${NUM_PROCESSES}"
    else
        echo "Parallel shards per module: ${NUM_PROCESSES} (TokenSHAP: ${TOKENSHAP_PROCESSES})"
    fi
    if [[ -n "${MAX_GRAPHS_FLAG}" ]]; then
        echo "Test mode: Processing first 100 samples only"
        echo "To process all samples, comment out MAX_GRAPHS_FLAG in script"
    fi
    echo "===================================================================="
    echo ""

    # GNN explainers - GraphSVX (4 runs)
    # echo ">>> Running GraphSVX on AG News and SST-2..."
    # run_graphsvx "ag_news" "SetFit" "skipgrams" "test"
    # run_graphsvx "ag_news" "SetFit" "window" "test"
    # run_graphsvx "sst2" "stanfordnlp" "skipgrams" "validation"
    # run_graphsvx "sst2" "stanfordnlp" "window" "validation"

    # GNN explainers - SubgraphX (4 runs)
    # echo ">>> Running SubgraphX on AG News and SST-2..."
    # run_subgraphx "ag_news" "SetFit" "constituency" "test"
    # run_subgraphx "ag_news" "SetFit" "syntactic" "test"
    # run_subgraphx "sst2" "stanfordnlp" "constituency" "validation"
    # run_subgraphx "sst2" "stanfordnlp" "syntactic" "validation"

    # LLM explainers - TokenSHAP (2 runs)
    echo ">>> Running TokenSHAP on AG News and SST-2..."
    # run_tokenshap "setfit/ag_news"
    run_tokenshap "stanfordnlp/sst2"

    echo ""
    echo "===================================================================="
    echo "✓ All explainability methods completed successfully!"
    echo "===================================================================="
    echo ""
    echo "Outputs generated:"
    echo "  - GNN insights: outputs/gnn_models/<backbone>/<dataset>/<graph_type>/<run_id>/explanations/"
    echo "  - LLM insights: outputs/insights/LLM/<backbone>/<dataset>/"
    echo ""
}

main "$@"
