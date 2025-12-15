#!/usr/bin/env python3

"""
Merge TokenSHAP summary pickles (which contain the new metrics such as insertion_auc)
with the raw per-record pickles from the previous run (which contain node-level data
like token_text and node_importance). The merged records are stored in the current
outputs tree as *_records_split_pickle directories so downstream analytics pick up
all samples without rerunning explainers.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple


MERGE_TARGETS: Iterable[Tuple[str, str, str]] = (
    ("SetFit/ag_news", "00", "03"),
    ("SetFit/ag_news", "02", "03"),
    ("stanfordnlp/sst2", "00", "03"),
    ("stanfordnlp/sst2", "01", "03"),
    ("stanfordnlp/sst2", "02", "03"),
)


def _dataset_path(dataset: str) -> Path:
    parts = dataset.split("/")
    if len(parts) == 2:
        return Path(parts[0]) / parts[1]
    if len(parts) > 2:
        return Path(parts[0]) / Path(*parts[1:])
    raise ValueError(f"Unexpected dataset identifier: {dataset!r}")


def _load_pickle(path: Path) -> Dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise TypeError(f"Unsupported pickle payload in {path}")


def _merge_records(new_record: Dict, old_record: Dict) -> Dict:
    merged = dict(old_record)  # start from raw record to preserve token/text fields
    for key, value in new_record.items():
        if value is None:
            continue
        if key == "extras":
            old_extras = merged.get("extras")
            if isinstance(old_extras, dict) and isinstance(value, dict):
                extras = dict(old_extras)
                extras.update(value)
                merged["extras"] = extras
            else:
                merged["extras"] = value
            continue
        merged[key] = value
    return merged


def merge_dataset(dataset: str, shard: str, total: str) -> None:
    dataset_path = _dataset_path(dataset)

    current_root = Path("outputs/insights/LLM") / dataset_path
    legacy_root = Path("outputs/insights/news/LLM") / dataset_path

    summary_dir = current_root / f"token_shap_shard{shard}of{total}_split_pickle"
    dest_dir = current_root / f"token_shap_shard{shard}of{total}_records_split_pickle"
    legacy_dir = legacy_root / f"token_shap_shard{shard}of{total}_records_split_pickle"

    if dest_dir.exists():
        print(f"[skip] {dest_dir} already exists")
        return

    if not summary_dir.exists():
        print(f"[skip] summary directory missing: {summary_dir}")
        return
    if not legacy_dir.exists():
        print(f"[skip] legacy records missing: {legacy_dir}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    for summary_path in sorted(summary_dir.glob("graph_*.pkl")):
        legacy_path = legacy_dir / summary_path.name
        if not legacy_path.exists():
            print(f"[warn] legacy record missing for {summary_path.name}, skipping")
            continue

        new_record = _load_pickle(summary_path)
        old_record = _load_pickle(legacy_path)
        merged = _merge_records(new_record, old_record)

        dest_path = dest_dir / summary_path.name
        with dest_path.open("wb") as handle:
            pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] merged records for {dataset} shard {shard}/{total} → {dest_dir}")


def main() -> None:
    for dataset, shard, total in MERGE_TARGETS:
        merge_dataset(dataset, shard, total)


if __name__ == "__main__":
    main()
