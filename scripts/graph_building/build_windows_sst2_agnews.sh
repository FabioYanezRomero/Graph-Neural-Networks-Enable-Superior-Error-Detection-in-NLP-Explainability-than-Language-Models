#!/usr/bin/env bash
set -euo pipefail

# Build window graphs (word or token) for SST-2 and AG News across multiple window sizes.

SIZES=(${SIZES:-1 2 3 5 7})
DATASETS=("SetFit/ag_news" "stanfordnlp/sst2")
SUBSETS=(train validation test)
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}

# Prefer a fine-tuned model provided via env/route (used by token-based builders)
if [ -f "scripts/models.env" ]; then
  # shellcheck disable=SC1091
  source scripts/models.env
fi

GLOBAL_MODEL_PREF=""
if [[ -n "${FINETUNED_MODEL_NAME:-}" ]]; then
  GLOBAL_MODEL_PREF="${FINETUNED_MODEL_NAME}"
elif [[ -n "${GRAPHTEXT_MODEL_NAME:-}" ]]; then
  GLOBAL_MODEL_PREF="${GRAPHTEXT_MODEL_NAME}"
elif [[ -n "${MODEL_NAME:-}" ]]; then
  GLOBAL_MODEL_PREF="${MODEL_NAME}"
fi

DEFAULT_MODEL_NAME=${DEFAULT_MODEL_NAME:-bert-base-uncased}
if [[ -z "$GLOBAL_MODEL_PREF" ]]; then
  echo "[warn] FINETUNED_MODEL_NAME/GRAPHTEXT_MODEL_NAME not set; defaulting to '${DEFAULT_MODEL_NAME}'." >&2
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

for ds in "${DATASETS[@]}"; do
  RESOLVED_WEIGHTS=""
  case "$ds" in
    SetFit/ag_news)
      if [[ -n "${AGNEWS_CHECKPOINT:-}" ]]; then
        RESOLVED_WEIGHTS="$AGNEWS_CHECKPOINT"
      fi
      ;;
    stanfordnlp/sst2)
      if [[ -n "${SST2_CHECKPOINT:-}" ]]; then
        RESOLVED_WEIGHTS="$SST2_CHECKPOINT"
      fi
      ;;
  esac

  if [[ -n "$RESOLVED_WEIGHTS" && ! -f "$RESOLVED_WEIGHTS" ]]; then
    echo "[warn] Expected checkpoint not found at $RESOLVED_WEIGHTS for $ds; proceeding without it." >&2
    RESOLVED_WEIGHTS=""
  elif [[ -n "$RESOLVED_WEIGHTS" ]]; then
    echo "[info] Using dataset-specific checkpoint: $RESOLVED_WEIGHTS"
  else
    echo "[warn] No checkpoint configured for $ds; using base model weights." >&2
  fi

  MODEL_FOR_DATASET="$GLOBAL_MODEL_PREF"
  if [[ -z "$MODEL_FOR_DATASET" && -n "$RESOLVED_WEIGHTS" ]]; then
    CONFIG_MODEL=$(WINDOW_COMBO_WEIGHTS="$RESOLVED_WEIGHTS" python3 - <<'PY'
import json
import os
import sys
weights_path = os.environ.get('WINDOW_COMBO_WEIGHTS')
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
      MODEL_FOR_DATASET="$CONFIG_MODEL"
    fi
  fi
  if [[ -z "$MODEL_FOR_DATASET" ]]; then
    MODEL_FOR_DATASET="$DEFAULT_MODEL_NAME"
  fi

  for k in "${SIZES[@]}"; do
    echo "Building window.word.k${k} for ${ds} -> ${OUT_BASE}"
    CMD=(python -m src.graph_builders.tree_generator
      --graph_type "window.word.k${k}"
      --dataset "${ds}"
      --subsets "${SUBSETS[@]}"
      --batch_size "${BATCH_SIZE}"
      --device "${DEVICE}"
      --output_dir "${OUT_BASE}"
      --model_name "${MODEL_FOR_DATASET}")
    if [[ -n "$RESOLVED_WEIGHTS" ]]; then
      CMD+=(--weights_path "$RESOLVED_WEIGHTS")
    fi
    "${CMD[@]}"
  done
done

echo "Done. Graphs saved under ${OUT_BASE}/<dataset>/<subset>/window.word.k<k>/"
