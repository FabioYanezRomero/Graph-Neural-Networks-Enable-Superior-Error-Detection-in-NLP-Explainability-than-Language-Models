#!/usr/bin/env bash
set -euo pipefail

# Generic graph-generation runner that wraps src.graph_builders.tree_generator
# Examples:
#   ./scripts/run_graph_generation.sh \
#       --graph_type syntactic \
#       --dataset stanfordnlp/sst2 \
#       --subsets "train validation" \
#       --batch_size 256 --device cuda:0
#
#   ./scripts/run_graph_generation.sh \
#       --graph_type window.word.k5 \
#       --dataset SetFit/ag_news

GRAPH_TYPE=""
DATASET=""
SUBSETS="train validation test"
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}

# Load central model config if present
if [ -f "scripts/models.env" ]; then
  # shellcheck disable=SC1091
  source scripts/models.env
fi

# Choose model name with fine-tuned preference
if [[ -n "${FINETUNED_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${FINETUNED_MODEL_NAME}"
elif [[ -n "${GRAPHTEXT_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${GRAPHTEXT_MODEL_NAME}"
else
  MODEL_NAME=${MODEL_NAME:-bert-base-uncased}
  echo "[warn] FINETUNED_MODEL_NAME/GRAPHTEXT_MODEL_NAME not set; using default '${MODEL_NAME}'." >&2
fi

# Auto-fallback to CPU if CUDA is not available to PyTorch
if [[ "$DEVICE" == cuda:* || "$DEVICE" == cuda ]]; then
  if ! python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
  then
    echo "[warn] torch.cuda.is_available()==False; falling back to CPU." >&2
    DEVICE=cpu
  fi
fi

usage() {
  cat <<EOF
Usage: $0 --graph_type TYPE --dataset NAME [--subsets "s1 s2 ..."] [--batch_size N] [--device DEV] [--out_base DIR] [--model_name HF_OR_PATH]

TYPE examples: syntactic, constituency, window.word.k3, window.token.k5
DATASET examples: stanfordnlp/sst2, SetFit/ag_news
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph_type)
      GRAPH_TYPE="$2"; shift 2;;
    --dataset)
      DATASET="$2"; shift 2;;
    --subsets)
      SUBSETS="$2"; shift 2;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --out_base|--output_dir)
      OUT_BASE="$2"; shift 2;;
    --model_name)
      MODEL_NAME="$2"; shift 2;;
    --help|-h)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$GRAPH_TYPE" || -z "$DATASET" ]]; then
  echo "Error: --graph_type and --dataset are required." >&2
  usage
  exit 1
fi

echo "Generating '$GRAPH_TYPE' graphs for dataset '$DATASET'"
echo "Subsets: $SUBSETS | Batch: $BATCH_SIZE | Device: $DEVICE"
echo "Output base: $OUT_BASE | Model: $MODEL_NAME"

# Convert subsets string to array for Python arg passing
read -r -a SUBSETS_ARR <<< "$SUBSETS"

python -m src.graph_builders.tree_generator \
  --graph_type "$GRAPH_TYPE" \
  --dataset "$DATASET" \
  --subsets "${SUBSETS_ARR[@]}" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output_dir "$OUT_BASE" \
  --model_name "$MODEL_NAME"

echo "Done. Graphs saved under ${OUT_BASE}/${DATASET}/<subset>/${GRAPH_TYPE}/"
