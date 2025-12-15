from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Dict

DEFAULT_BASE = os.environ.get("GRAPHTEXT_OUTPUT_DIR", "outputs")
METADATA_DIR = os.path.join(DEFAULT_BASE, "metadata")
METADATA_INDEX = os.path.join(METADATA_DIR, "index.json")


def _load_index() -> Dict[str, Any]:
    if not os.path.exists(METADATA_INDEX):
        return {"runs": []}
    try:
        with open(METADATA_INDEX, "r") as f:
            return json.load(f)
    except Exception:
        return {"runs": []}


def log_step(step: str, params: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    os.makedirs(METADATA_DIR, exist_ok=True)
    idx = _load_index()
    idx.setdefault("runs", [])

    def _serialise(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _serialise(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_serialise(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        if hasattr(value, "__fspath__"):
            try:
                return os.fspath(value)
            except Exception:
                pass
        return repr(value)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "params": _serialise(params),
        "outputs": _serialise(outputs),
    }
    idx["runs"].append(entry)
    with open(METADATA_INDEX, "w") as f:
        json.dump(idx, f, indent=2)
