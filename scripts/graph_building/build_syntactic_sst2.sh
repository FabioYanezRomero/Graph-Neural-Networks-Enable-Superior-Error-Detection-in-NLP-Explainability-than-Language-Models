#!/usr/bin/env bash
set -euo pipefail

# Build syntactic dependency trees for SST-2 using the registry-based generator.

DATASET="stanfordnlp/sst2"
# Auto-detect all splits available in the dataset (e.g., train validation test)
mapfile -t SUBSETS < <(python - <<'PY'
from datasets import get_dataset_split_names
import sys
name = "stanfordnlp/sst2"
try:
    names = get_dataset_split_names(name)
    print("\n".join(names))
except Exception as e:
    # Fallback to common splits if discovery fails
    print("train\nvalidation\ntest")
PY
)
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}

# Model selection is not needed for syntactic tree building (uses Stanza).
# Keep env sourcing for parity but avoid printing confusing BERT warnings.
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

CHECKPOINT_PATH=""
if [[ -n "${SST2_CHECKPOINT:-}" ]]; then
  CHECKPOINT_PATH="$SST2_CHECKPOINT"
fi

if [[ -n "$CHECKPOINT_PATH" && ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[warn] Expected checkpoint not found at $CHECKPOINT_PATH; proceeding without it." >&2
  CHECKPOINT_PATH=""
elif [[ -n "$CHECKPOINT_PATH" ]]; then
  echo "[info] Using dataset-specific checkpoint: $CHECKPOINT_PATH"
else
  echo "[warn] No SST-2 checkpoint configured; using base model weights." >&2
fi

MODEL_FOR_DATASET="$GLOBAL_MODEL_PREF"
if [[ -z "$MODEL_FOR_DATASET" ]]; then
  if [[ -n "$CHECKPOINT_PATH" ]]; then
    CONFIG_MODEL=$(SYN_SST_WEIGHTS="$CHECKPOINT_PATH" python3 - <<'PY'
import json
import os
import sys
weights_path = os.environ.get('SYN_SST_WEIGHTS')
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
    echo "[warn] FINETUNED_MODEL_NAME/GRAPHTEXT_MODEL_NAME not set; defaulting to '${MODEL_FOR_DATASET}'." >&2
  fi
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

echo "Building syntactic trees for ${DATASET} -> ${OUT_BASE}"
CMD=(python -m src.graph_builders.tree_generator
  --graph_type syntactic
  --dataset "${DATASET}"
  --subsets "${SUBSETS[@]}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --output_dir "${OUT_BASE}"
  --model_name "${MODEL_FOR_DATASET}")
if [[ -n "$CHECKPOINT_PATH" ]]; then
  CMD+=(--weights_path "$CHECKPOINT_PATH")
fi
"${CMD[@]}"

echo "Done. Trees saved under ${OUT_BASE}/${DATASET}/<subset>/syntactic/"
