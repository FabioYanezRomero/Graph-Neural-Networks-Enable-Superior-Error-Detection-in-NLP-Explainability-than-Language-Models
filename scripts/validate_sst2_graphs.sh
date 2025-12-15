#!/usr/bin/env bash
set -euo pipefail

# Validate SST-2 graphs: syntactic, constituency, and common window sizes

SIZES=(${SIZES:-3 5})
GRAPH_TYPES=(syntactic constituency)
for k in "${SIZES[@]}"; do
  GRAPH_TYPES+=("window.word.k${k}")
done

bash scripts/validate_graphs.sh --dataset stanfordnlp/sst2 --graph_types "${GRAPH_TYPES[*]}"
