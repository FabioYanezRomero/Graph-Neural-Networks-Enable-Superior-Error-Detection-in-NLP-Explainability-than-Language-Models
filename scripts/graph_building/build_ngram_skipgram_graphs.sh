#!/usr/bin/env bash
set -euo pipefail

# Build n-gram and skip-gram graphs (word- or token-based) for a HuggingFace dataset.

DATASET=""
FAMILIES_STR=${FAMILIES:-""}
NGRAM_SIZES_STR=${NGRAM_SIZES:-""}
SKIPGRAM_SKIPS_STR=${SKIPGRAM_SKIPS:-""}
SKIPGRAM_NS_STR=${SKIPGRAM_NS:-""}
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

usage() {
  cat <<'USAGE'
Usage: scripts/build_ngram_skipgram_graphs.sh --dataset DATASET [options]

Required arguments:
  --dataset NAME             HuggingFace dataset identifier (e.g., SetFit/ag_news)

Optional arguments:
  --families "F1 F2"          Graph families to build (default: ngrams skipgrams)
  --ngram_size N             Single n-gram length (overrides --ngram_sizes)
  --ngram_sizes "N1 N2"       Space-separated list of n-gram lengths (default: 2 3)
  --skipgram_skip K          Single maximum skip distance (overrides --skipgram_skips)
  --skipgram_skips "K1 K2"    Space-separated list of maximum skips for skip-grams (default: 0 1)
  --skipgram_n N             Single skip-gram n value (overrides --skipgram_ns)
  --skipgram_ns "N1 N2"       Space-separated list of skip-gram n values (default: 2)
  --graph_unit UNIT          Single unit (word or token)
  --graph_units "U1 U2"      Space-separated list of units (default: word)
  --subsets "s1 s2"           Subsets to process (auto-detected if omitted)
  --batch_size N             Batch size for dataset loader (default: 256)
  --device DEV               Device for processing (default: cuda:0)
  --out_base DIR             Base output directory (default: outputs/graphs)
  --model_name NAME          Override model/tokenizer to reference (token graphs)
  --max_batches N            Limit batches per split (for smoke tests)
  --help                     Show this message

Environment overrides:
  FAMILIES, NGRAM_SIZES, SKIPGRAM_SKIPS, SKIPGRAM_NS, GRAPH_UNITS, SUBSETS, BATCH_SIZE,
  DEVICE, OUT_BASE, MODEL_NAME, MAX_BATCHES.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --families)
      FAMILIES_STR="$2"; shift 2 ;;
    --ngram_size)
      NGRAM_SIZES_STR="$2"; shift 2 ;;
    --ngram_sizes)
      NGRAM_SIZES_STR="$2"; shift 2 ;;
    --skipgram_skip)
      SKIPGRAM_SKIPS_STR="$2"; shift 2 ;;
    --skipgram_skips)
      SKIPGRAM_SKIPS_STR="$2"; shift 2 ;;
    --skipgram_n)
      SKIPGRAM_NS_STR="$2"; shift 2 ;;
    --skipgram_ns)
      SKIPGRAM_NS_STR="$2"; shift 2 ;;
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

if [[ -z "$FAMILIES_STR" ]]; then
  FAMILIES_STR="ngrams skipgrams"
fi
IFS=' ' read -r -a FAMILIES <<< "$FAMILIES_STR"
if [[ ${#FAMILIES[@]} -eq 0 ]]; then
  echo "Error: at least one family must be provided" >&2
  exit 1
fi
for fam in "${FAMILIES[@]}"; do
  case "$fam" in
    ngrams|skipgrams) ;;
    *)
      echo "Error: unsupported family '$fam' (expected 'ngrams' or 'skipgrams')" >&2
      exit 1 ;;
  esac
done

if [[ -z "$NGRAM_SIZES_STR" ]]; then
  NGRAM_SIZES_STR="2 3"
fi
IFS=' ' read -r -a NGRAM_SIZES <<< "$NGRAM_SIZES_STR"
for n in "${NGRAM_SIZES[@]}"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]] || (( n < 2 )); then
    echo "Error: n-gram size '$n' must be an integer >= 2" >&2
    exit 1
  fi
done

if [[ -z "$SKIPGRAM_SKIPS_STR" ]]; then
  SKIPGRAM_SKIPS_STR="0 1"
fi
IFS=' ' read -r -a SKIPGRAM_SKIPS <<< "$SKIPGRAM_SKIPS_STR"
for k in "${SKIPGRAM_SKIPS[@]}"; do
  if ! [[ "$k" =~ ^[0-9]+$ ]]; then
    echo "Error: skip-gram skip '$k' must be a non-negative integer" >&2
    exit 1
  fi
done

if [[ -z "$SKIPGRAM_NS_STR" ]]; then
  SKIPGRAM_NS_STR="2"
fi
IFS=' ' read -r -a SKIPGRAM_NS <<< "$SKIPGRAM_NS_STR"
for n in "${SKIPGRAM_NS[@]}"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]] || (( n < 2 )); then
    echo "Error: skip-gram n '$n' must be an integer >= 2" >&2
    exit 1
  fi
done

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

if [ -f "scripts/models.env" ]; then
  # shellcheck disable=SC1091
  source scripts/models.env
fi

RESOLVED_WEIGHTS=""
case "$DATASET" in
  stanfordnlp/sst2|StanfordNLP/sst2|SST2|sst2)
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

RESOLVED_MODEL=""
if [[ -n "$MODEL_OVERRIDE" ]]; then
  MODEL_NAME="$MODEL_OVERRIDE"
elif [[ -n "${FINETUNED_MODEL_NAME:-}" ]]; then
  MODEL_NAME="$FINETUNED_MODEL_NAME"
elif [[ -n "${GRAPHTEXT_MODEL_NAME:-}" ]]; then
  MODEL_NAME="$GRAPHTEXT_MODEL_NAME"
else
  if [[ -z "$RESOLVED_MODEL" && -n "$RESOLVED_WEIGHTS" && -f "$RESOLVED_WEIGHTS" ]]; then
    CONFIG_MODEL=$(NGRAM_SKIPGRAM_WEIGHTS="$RESOLVED_WEIGHTS" python3 - <<'PY'
import json
import os
import sys
weights_path = os.environ.get('NGRAM_SKIPGRAM_WEIGHTS')
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
for key in ("model_name", "model_name_or_path", "pretrained_model_name"):
    val = data.get(key)
    if isinstance(val, str) and val.strip():
        print(val.strip())
        sys.exit(0)
PY
    )
    if [[ -n "${CONFIG_MODEL:-}" ]]; then
      RESOLVED_MODEL="$CONFIG_MODEL"
    fi
  fi
  if [[ -n "$RESOLVED_MODEL" ]]; then
    MODEL_NAME="$RESOLVED_MODEL"
  else
    MODEL_NAME=${MODEL_NAME:-bert-base-uncased}
    echo "[warn] FINETUNED_MODEL_NAME/GRAPHTEXT_MODEL_NAME not set; using default '${MODEL_NAME}'." >&2
  fi
fi

if [[ -n "$RESOLVED_WEIGHTS" && ! -f "$RESOLVED_WEIGHTS" ]]; then
  echo "[warn] Expected checkpoint not found at $RESOLVED_WEIGHTS; proceeding without it." >&2
  RESOLVED_WEIGHTS=""
elif [[ -n "$RESOLVED_WEIGHTS" ]]; then
  echo "[info] Using dataset-specific checkpoint: $RESOLVED_WEIGHTS" >&2
fi

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
  for fam in "${FAMILIES[@]}"; do
    case "$fam" in
      ngrams)
        for n in "${NGRAM_SIZES[@]}"; do
          graph_type="ngrams.${unit}.n${n}"
          echo "Building ${graph_type} graphs for ${DATASET} [subsets: ${SUBSETS[*]}] -> ${OUT_BASE}"
          CMD=(python -m src.graph_builders.tree_generator
            --graph_type "$graph_type"
            --dataset "$DATASET"
            --subsets "${SUBSETS[@]}"
            --batch_size "$BATCH_SIZE"
            --device "$DEVICE"
            --output_dir "$OUT_BASE"
            --model_name "$MODEL_NAME")
          if [[ -n "$RESOLVED_WEIGHTS" ]]; then
            CMD+=(--weights_path "$RESOLVED_WEIGHTS")
          fi
          if [[ -n "$MAX_BATCHES" ]]; then
            CMD+=(--max_batches "$MAX_BATCHES")
          fi
          "${CMD[@]}"
        done
        ;;
      skipgrams)
        for k in "${SKIPGRAM_SKIPS[@]}"; do
          for n in "${SKIPGRAM_NS[@]}"; do
            graph_type="skipgrams.${unit}.k${k}.n${n}"
            echo "Building ${graph_type} graphs for ${DATASET} [subsets: ${SUBSETS[*]}] -> ${OUT_BASE}"
            CMD=(python -m src.graph_builders.tree_generator
              --graph_type "$graph_type"
              --dataset "$DATASET"
              --subsets "${SUBSETS[@]}"
              --batch_size "$BATCH_SIZE"
              --device "$DEVICE"
              --output_dir "$OUT_BASE"
              --model_name "$MODEL_NAME")
            if [[ -n "$RESOLVED_WEIGHTS" ]]; then
              CMD+=(--weights_path "$RESOLVED_WEIGHTS")
            fi
            if [[ -n "$MAX_BATCHES" ]]; then
              CMD+=(--max_batches "$MAX_BATCHES")
            fi
            "${CMD[@]}"
          done
        done
        ;;
    esac
  done
done

echo "Done. Graphs saved under ${OUT_BASE}/${DATASET}/<subset>/<graph_type>/ for each requested configuration."
