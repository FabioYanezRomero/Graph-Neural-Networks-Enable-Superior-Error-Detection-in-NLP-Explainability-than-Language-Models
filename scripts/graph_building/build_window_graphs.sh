#!/usr/bin/env bash
set -euo pipefail

# Build sliding window graphs where nodes are the original words (or tokenizer tokens)
# and edges connect words within configurable window sizes.

DATASET=""
WINDOW_SIZES_STR=${WINDOW_SIZES:-""}
GRAPH_UNITS_STR=${GRAPH_UNITS:-""}
SUBSETS_STR=${SUBSETS:-""}
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE_ENV_PRESET=0
if [[ -v DEVICE ]]; then
  DEVICE_ENV_PRESET=1
fi
DEVICE=${DEVICE:-cuda:0}
DEVICE_SET_EXPLICIT=$DEVICE_ENV_PRESET
OUT_BASE=${OUT_BASE:-outputs/graphs}
MODEL_OVERRIDE=""
MAX_BATCHES=${MAX_BATCHES:-""}
WEIGHTS_OVERRIDE=${WINDOW_WEIGHTS_PATH:-""}

usage() {
  cat <<'USAGE'
Usage: scripts/build_window_graphs.sh --dataset DATASET [options]

Required arguments:
  --dataset NAME            HuggingFace dataset identifier (e.g., SetFit/ag_news)

Optional arguments:
  --window_size K           Single window size (overrides --window_sizes)
  --window_sizes "K1 K2"     Space-separated list of window sizes (default: 1 2 3 5)
  --graph_unit UNIT         Shortcut for single unit ("word" or "token")
  --graph_units "U1 U2"     Space-separated list of units (default: word)
  --subsets "s1 s2"          Subsets to process (auto-detected if omitted)
  --batch_size N            Batch size for dataset loader (default: 256)
  --device DEV              Device for processing (default: cuda:0)
  --out_base DIR            Base output directory (default: outputs/graphs)
  --model_name NAME         Override model/tokenizer to reference (token graphs)
  --max_batches N           Limit batches per split (for smoke tests)
  --weights_path PATH       Override checkpoint/weights path to load (dataset-specific defaults otherwise)
  --help                    Show this message

Environment overrides:
  WINDOW_SIZES, GRAPH_UNIT, GRAPH_UNITS, SUBSETS, BATCH_SIZE, DEVICE, OUT_BASE,
  FINETUNED_MODEL_NAME, GRAPHTEXT_MODEL_NAME, MODEL_NAME, MAX_BATCHES,
  WINDOW_WEIGHTS_PATH.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --window_size)
      WINDOW_SIZES_STR="$2"; shift 2 ;;
    --window_sizes)
      WINDOW_SIZES_STR="$2"; shift 2 ;;
    --graph_unit)
      GRAPH_UNITS_STR="$2"; shift 2 ;;
    --graph_units)
      GRAPH_UNITS_STR="$2"; shift 2 ;;
    --subsets)
      SUBSETS_STR="$2"; shift 2 ;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; DEVICE_SET_EXPLICIT=1; shift 2 ;;
    --out_base|--output_dir)
      OUT_BASE="$2"; shift 2 ;;
    --model_name)
      MODEL_OVERRIDE="$2"; shift 2 ;;
    --max_batches)
      MAX_BATCHES="$2"; shift 2 ;;
    --weights_path)
      WEIGHTS_OVERRIDE="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Error: --dataset is required" >&2
  usage
  exit 1
fi

if [[ -z "$WINDOW_SIZES_STR" ]]; then
  WINDOW_SIZES_STR="1 2 3 5"
fi
IFS=' ' read -r -a WINDOW_SIZES <<< "$WINDOW_SIZES_STR"
if [[ ${#WINDOW_SIZES[@]} -eq 0 ]]; then
  echo "Error: at least one window size must be provided" >&2
  exit 1
fi

# Resolve graph units (default: word). Honor legacy GRAPH_UNIT env var as fallback.
if [[ -z "$GRAPH_UNITS_STR" && -n "${GRAPH_UNIT:-}" ]]; then
  GRAPH_UNITS_STR="$GRAPH_UNIT"
fi
if [[ -z "$GRAPH_UNITS_STR" ]]; then
  GRAPH_UNITS_STR="word"
fi
IFS=' ' read -r -a GRAPH_UNITS <<< "$GRAPH_UNITS_STR"
if [[ ${#GRAPH_UNITS[@]} -eq 0 ]]; then
  echo "Error: no graph units resolved" >&2
  exit 1
fi
for unit in "${GRAPH_UNITS[@]}"; do
  case "$unit" in
    word|token) ;;
    *)
      echo "Error: unsupported graph unit '$unit' (expected 'word' or 'token')" >&2
      exit 1 ;;
  esac
done

# Ensure every window size is an integer >=0
for k in "${WINDOW_SIZES[@]}"; do
  if ! [[ "$k" =~ ^[0-9]+$ ]]; then
    echo "Error: window size '$k' is not a non-negative integer" >&2
    exit 1
  fi
done

# Auto-detect subsets if not provided
if [[ -z "$SUBSETS_STR" ]]; then
  if ! SUBSETS_STR=$(python3 - <<PY
from datasets import get_dataset_split_names
try:
    splits = get_dataset_split_names("$DATASET")
    print(" ".join(splits))
except Exception:
    print("train validation test")
PY
  ); then
    echo "Error: failed to determine dataset splits" >&2
    exit 1
  fi
fi
IFS=' ' read -r -a SUBSETS <<< "$SUBSETS_STR"
if [[ ${#SUBSETS[@]} -eq 0 ]]; then
  echo "Error: no subsets resolved" >&2
  exit 1
fi

# Load optional model configuration (used for token graphs)
if [ -f "scripts/models.env" ]; then
  # shellcheck disable=SC1091
  source scripts/models.env
fi

# Resolve model name preference chain
if [[ -n "$MODEL_OVERRIDE" ]]; then
  MODEL_NAME="$MODEL_OVERRIDE"
elif [[ -n "${FINETUNED_MODEL_NAME:-}" ]]; then
  MODEL_NAME="$FINETUNED_MODEL_NAME"
elif [[ -n "${GRAPHTEXT_MODEL_NAME:-}" ]]; then
  MODEL_NAME="$GRAPHTEXT_MODEL_NAME"
else
  MODEL_NAME=${MODEL_NAME:-bert-base-uncased}
  echo "[warn] FINETUNED_MODEL_NAME/GRAPHTEXT_MODEL_NAME not set; using default '${MODEL_NAME}'." >&2
fi

# Resolve dataset-specific checkpoint/weights path
RESOLVED_WEIGHTS="$WEIGHTS_OVERRIDE"
if [[ -z "$RESOLVED_WEIGHTS" && -n "${GRAPHTEXT_WEIGHTS_PATH:-}" ]]; then
  RESOLVED_WEIGHTS="$GRAPHTEXT_WEIGHTS_PATH"
fi
case "$DATASET" in
  stanfordnlp/sst2|StanfordNLP/sst2|stanfordnlp/SST2|SST2|sst2)
    if [[ -n "${SST2_CHECKPOINT:-}" ]]; then
      RESOLVED_WEIGHTS="$SST2_CHECKPOINT"
    fi
    ;;
  SetFit/ag_news|setfit/ag_news|SetFit/AG_NEWS|AG_NEWS|ag_news|ag-news)
    if [[ -n "${AGNEWS_CHECKPOINT:-}" ]]; then
      RESOLVED_WEIGHTS="$AGNEWS_CHECKPOINT"
    fi
    ;;
esac

# If we have a checkpoint, try to infer a better model name from its config when no override provided.
if [[ -z "$MODEL_OVERRIDE" && -n "$RESOLVED_WEIGHTS" ]]; then
  if [[ -f "$RESOLVED_WEIGHTS" ]]; then
    CONFIG_MODEL=$(WINDOW_SCRIPT_WEIGHTS="$RESOLVED_WEIGHTS" python3 - <<'PY'
import json
import os
import sys
weights_path = os.environ.get('WINDOW_SCRIPT_WEIGHTS')
if not weights_path:
    sys.exit(0)
cfg_path = os.path.join(os.path.dirname(weights_path), 'config.json')
if not os.path.isfile(cfg_path):
    sys.exit(0)
try:
    with open(cfg_path) as f:
        data = json.load(f)
except Exception:
    sys.exit(0)
for key in ('model_name', 'model_name_or_path', 'pretrained_model_name'):
    val = data.get(key)
    if isinstance(val, str) and val.strip():
        print(val.strip())
        break
PY
    )
    if [[ -n "${CONFIG_MODEL:-}" ]]; then
      MODEL_NAME="$CONFIG_MODEL"
    fi
  fi
fi

if [[ -n "$RESOLVED_WEIGHTS" && ! -f "$RESOLVED_WEIGHTS" ]]; then
  echo "[warn] Expected checkpoint not found at $RESOLVED_WEIGHTS; proceeding with available model weights." >&2
fi

# Auto-fallback to CPU if CUDA is unavailable
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
    if [[ $DEVICE_SET_EXPLICIT -eq 0 ]]; then
      echo "[warn] torch.cuda.is_available()==False; falling back to CPU." >&2
      DEVICE=cpu
    else
      echo "[warn] torch.cuda.is_available()==False; continuing with requested device '$DEVICE'." >&2
    fi
  fi
fi

for unit in "${GRAPH_UNITS[@]}"; do
  for k in "${WINDOW_SIZES[@]}"; do
    graph_type="window.${unit}.k${k}"
    echo "Building ${graph_type} graphs for ${DATASET} [subsets: ${SUBSETS[*]}] -> ${OUT_BASE}"
    CMD=(python -m src.graph_builders.tree_generator
      --graph_type "$graph_type"
      --dataset "$DATASET"
      --subsets "${SUBSETS[@]}"
      --batch_size "$BATCH_SIZE"
      --device "$DEVICE"
      --output_dir "$OUT_BASE"
      --model_name "$MODEL_NAME")
    if [[ -n "$MAX_BATCHES" ]]; then
      CMD+=(--max_batches "$MAX_BATCHES")
    fi
    if [[ -n "$RESOLVED_WEIGHTS" ]]; then
      CMD+=(--weights_path "$RESOLVED_WEIGHTS")
    fi
    "${CMD[@]}"
  done
done

if [[ -n "$RESOLVED_WEIGHTS" ]]; then
  echo "Used checkpoint: $RESOLVED_WEIGHTS"
fi

echo "Done. Graphs saved under ${OUT_BASE}/${DATASET}/<subset>/window.<unit>.k<k>/ for each requested unit/size."
