from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class MetricRecord:
    module: str
    method: str
    dataset: str
    graph: str
    metric: str
    group: str
    kl_correct_vs_incorrect: Optional[float]
    kl_incorrect_vs_correct: Optional[float]
    js_divergence: Optional[float]
    ks_statistic: Optional[float]
    source: Path

    def to_dict(self) -> Dict[str, object]:
        return {
            "module": self.module,
            "method": self.method,
            "dataset": self.dataset,
            "graph": self.graph,
            "metric": self.metric,
            "group": self.group,
            "kl_correct_vs_incorrect": self.kl_correct_vs_incorrect,
            "kl_incorrect_vs_correct": self.kl_incorrect_vs_correct,
            "js_divergence": self.js_divergence,
            "ks_statistic": self.ks_statistic,
            "source": str(self.source),
        }


def discover_metric_files(base_dir: Path) -> Iterable[Path]:
    return base_dir.rglob("metrics_kde_*.json")


def parse_metric_file(module: str, base_dir: Path, metrics_path: Path) -> MetricRecord:
    relative_parts = metrics_path.relative_to(base_dir).parts
    if len(relative_parts) < 3:
        raise ValueError(f"Unexpected path layout for {metrics_path}")

    method = relative_parts[0]
    dataset = relative_parts[1]

    graph = ""
    metric_name = ""
    if len(relative_parts) == 3:
        graph = ""
    elif len(relative_parts) == 4:
        graph = relative_parts[2]
    else:
        graph = relative_parts[2]
        metric_name = "/".join(relative_parts[3:-1])

    group_id = metrics_path.stem.replace("metrics_", "")

    with metrics_path.open("r", encoding="utf-8") as handler:
        payload = json.load(handler)

    return MetricRecord(
        module=module,
        method=method,
        dataset=dataset,
        graph=graph,
        metric=metric_name,
        group=group_id,
        kl_correct_vs_incorrect=payload.get("kl_correct_vs_incorrect"),
        kl_incorrect_vs_correct=payload.get("kl_incorrect_vs_correct"),
        js_divergence=payload.get("js_divergence"),
        ks_statistic=payload.get("ks_statistic"),
        source=metrics_path,
    )


def aggregate_module(module: str, base_dir: Path) -> pd.DataFrame:
    metric_files = list(discover_metric_files(base_dir))
    if not metric_files:
        raise FileNotFoundError(f"No divergence metrics (metrics_kde_*.json) found in {base_dir}")

    records: List[MetricRecord] = []
    for metrics_path in metric_files:
        record = parse_metric_file(module, base_dir, metrics_path)
        records.append(record)

    return pd.DataFrame([record.to_dict() for record in records])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate divergence metrics produced by analytics KDE generators."
    )
    parser.add_argument(
        "--module",
        required=True,
        help="Analytics module name (e.g., sequence, contrastivity, fidelity, auc).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override the base directory (defaults to outputs/analytics/<module>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (defaults to outputs/analytics/<module>/divergence_metrics.csv).",
    )

    args = parser.parse_args()

    module = args.module.strip().lower()
    base_dir = args.base_dir or Path("outputs/analytics") / module
    base_dir = base_dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    df = aggregate_module(module, base_dir)
    df.sort_values(
        ["method", "dataset", "graph", "metric", "group"],
        inplace=True,
        kind="mergesort",
    )

    output_path = args.output or base_dir / "divergence_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[aggregate] Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
