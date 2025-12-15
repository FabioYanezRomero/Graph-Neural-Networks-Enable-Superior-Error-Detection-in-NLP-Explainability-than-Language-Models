#!/usr/bin/env bash
set -euo pipefail

echo "== nvidia-smi =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found in container PATH"
fi

echo
echo "== PyTorch CUDA report =="
python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))
PY

echo
echo "== Environment =="
env | grep -E 'CUDA|NVIDIA' | sort || true

