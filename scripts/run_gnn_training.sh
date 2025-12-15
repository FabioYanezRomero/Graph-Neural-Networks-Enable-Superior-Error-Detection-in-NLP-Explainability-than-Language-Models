#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_CMD="docker compose"

run_in_container() {
    local service=$1
    local cmd=$2
    local use_gpu=${3:-false}
    echo "[${service}] ${cmd}"
    if [[ "${use_gpu}" == true ]]; then
        ${COMPOSE_CMD} run --rm --gpus all "${service}" bash -lc "cd /app && ${cmd}"
    else:
        ${COMPOSE_CMD} run --rm "${service}" bash -lc "cd /app && ${cmd}"
    fi
}

run_legal_cases_gcn() {
    local train_dir="outputs/pyg_graphs/legal_cases/train"
    local val_dir="outputs/pyg_graphs/legal_cases/validation"
    local test_dir="outputs/pyg_graphs/legal_cases/test"
    local output_dir="outputs/legal_cases_gcn_run"

    local cmd="python src/gnn_training/training.py \
        --train_data_dir ${train_dir} \
        --val_data_dir ${val_dir} \
        --test_data_dir ${test_dir} \
        --module GCNConv \
        --num_layers 2 \
        --hidden_dim 256 \
        --dropout 0.2 \
        --batch_size 16 \
        --epochs 30 \
        --learning_rate 5e-4 \
        --weight_decay 1e-4 \
        --optimizer Adam \
        --scheduler ReduceLROnPlateau \
        --patience 8 \
        --num_workers 0 \
        --output_dir ${output_dir}"

    run_in_container app "${cmd}" true
}

main() {
    cd "${ROOT_DIR}" || exit 1
    run_legal_cases_gcn
}

main "$@"
