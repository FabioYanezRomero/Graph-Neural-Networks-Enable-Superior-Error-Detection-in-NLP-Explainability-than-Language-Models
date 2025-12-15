#!/usr/bin/env bash
set -euo pipefail

# Validate generated graphs for a dataset and a list of graph types.
# Usage:
#   scripts/validate_graphs.sh --dataset NAME [--graph_types "syntactic constituency window.word.k5"] [--out_base outputs/graphs]

DATASET=""
GRAPH_TYPES=(syntactic constituency window.word.k5)
OUT_BASE=${OUT_BASE:-outputs/graphs}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --graph_types)
      IFS=' ' read -r -a GRAPH_TYPES <<< "$2"; shift 2 ;;
    --out_base)
      OUT_BASE="$2"; shift 2 ;;
    --help)
      echo "Usage: $0 --dataset NAME [--graph_types \"syntactic constituency window.word.k5\"] [--out_base outputs/graphs]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Error: --dataset is required" >&2
  exit 1
fi

python3 scripts/validate_graph_outputs.py \
  --dataset "$DATASET" \
  --out_base "$OUT_BASE" \
  --graph_types "${GRAPH_TYPES[@]}"
