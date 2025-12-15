from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

from .metrics import minimal_sufficient_statistics, summarize_records
from .records import ExplanationRecord


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summaries_to_dataframe(summaries: List[dict]):
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required to build tabular summaries.")
    return pd.DataFrame(summaries)


def export_summaries_json(
    records: Iterable[ExplanationRecord],
    output_path: Path,
    *,
    sufficiency_threshold: float = 0.9,
    top_k: int = 10,
    summaries: Optional[List[dict]] = None,
) -> List[dict]:
    if summaries is None:
        summaries = summarize_records(
            records,
            sufficiency_threshold=sufficiency_threshold,
            top_k=top_k,
        )
    ensure_parent(output_path)
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def export_summaries_csv(
    summaries: List[dict],
    output_path: Path,
) -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required to export CSV summaries.")
    df = summaries_to_dataframe(summaries)
    ensure_parent(output_path)
    df.to_csv(output_path, index=False)


def export_minimal_size_histogram(
    records: Iterable[ExplanationRecord],
    output_path: Path,
    *,
    threshold: float = 0.9,
) -> None:
    histogram = minimal_sufficient_statistics(records, threshold=threshold)
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handler:
        handler.write("size,count\n")
        for size, count in histogram:
            handler.write(f"{size},{count}\n")


def export_agreement_json(
    entries: List[dict],
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    output_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def export_agreement_csv(
    entries: List[dict],
    output_path: Path,
) -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required to export CSV agreement summaries.")
    df = summaries_to_dataframe(entries)
    ensure_parent(output_path)
    df.to_csv(output_path, index=False)
