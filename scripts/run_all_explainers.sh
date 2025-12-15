#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_CMD="docker compose"
FAIR_FLAG="--fair"

run_in_container() {
    local service=$1
    local cmd=$2
    local use_gpu=${3:-false}
    echo "[${service}] ${cmd}"
    if [[ "${use_gpu}" == true ]]; then
        ${COMPOSE_CMD} run --rm --gpus all "${service}" bash -lc "cd /app && ${cmd}"
    else
        ${COMPOSE_CMD} run --rm "${service}" bash -lc "cd /app && ${cmd}"
    fi
}

run_graphsvx() {
    local dataset=$1
    local backbone=$2
    local graph_type=$3
    local split=$4
    run_in_container graphsvx \
        "python -m src.explain.gnn.graphsvx.main --dataset '${dataset}' --graph-type '${graph_type}' --backbone '${backbone}' --split '${split}' ${FAIR_FLAG}" false
}

run_subgraphx() {
    local dataset=$1
    local backbone=$2
    local graph_type=$3
    local split=$4
    run_in_container subgraphx \
        "python -m src.explain.gnn.subgraphx.main --dataset '${dataset}' --graph-type '${graph_type}' --backbone '${backbone}' --split '${split}' ${FAIR_FLAG}" true
}

run_tokenshap() {
    local profile=$1
    run_in_container tokenshap \
        "python -m src.explain.llm.main explain '${profile}' ${FAIR_FLAG}" true
}

main() {
    cd "${ROOT_DIR}" || exit 1

    # GNN explainers
    # run_graphsvx "ag_news" "SetFit" "skipgrams" "test"
    # run_graphsvx "ag_news" "SetFit" "window" "test"
    # run_graphsvx "sst2" "stanfordnlp" "skipgrams" "validation"
    run_graphsvx "sst2" "stanfordnlp" "window" "validation"

    # run_subgraphx "ag_news" "SetFit" "constituency" "test"
    # run_subgraphx "ag_news" "SetFit" "syntactic" "test"
    # run_subgraphx "sst2" "stanfordnlp" "constituency" "validation"
    # run_subgraphx "sst2" "stanfordnlp" "syntactic" "validation"

    # LLM explainers
    # run_tokenshap "setfit/ag_news"
    # run_tokenshap "stanfordnlp/sst2"
}

main "$@"
