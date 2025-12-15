import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

STAT_COLUMNS = [
    "min_position_rank",
    "max_position_rank",
    "mean_position_rank",
    "median_position_rank",
    "std_position_rank",
    "position_spread",
]

DROP_COLUMNS = [
    "num_nodes",
    "num_edges",
    "total_nodes",
    "dataset_raw",
    "dataset_backbone",
    "run_id",
]


def parse_ranked_nodes(raw: object) -> List[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        candidates = list(raw)
    elif isinstance(raw, str):
        tokens = raw.strip().split()
        candidates = tokens
    else:
        return []

    nodes: List[int] = []
    for value in candidates:
        try:
            nodes.append(int(value))
        except Exception:
            continue
    return nodes


def infer_total_nodes(raw: object, nodes: Sequence[int]) -> Optional[int]:
    if raw is not None:
        try:
            if isinstance(raw, float) and np.isnan(raw):
                raise ValueError
            total = int(raw)
            if total > 0:
                return total
        except Exception:
            pass
    if nodes:
        surrogate = max(nodes) + 1
        if surrogate > 0:
            return surrogate
        return len(nodes)
    return None


def compute_position_ranks(nodes: Sequence[int], total_nodes: Optional[int]) -> List[float]:
    if not nodes or not total_nodes or total_nodes <= 0:
        return []
    return [node / float(total_nodes) for node in nodes]


def compute_stats(position_ranks: Sequence[float]) -> Mapping[str, Optional[float]]:
    if not position_ranks:
        return {key: np.nan for key in STAT_COLUMNS}
    ranks_array = np.asarray(position_ranks, dtype=float)
    min_rank = float(np.min(ranks_array))
    max_rank = float(np.max(ranks_array))
    spread = float(max_rank - min_rank)
    return {
        "min_position_rank": min_rank,
        "max_position_rank": max_rank,
        "mean_position_rank": float(np.mean(ranks_array)),
        "median_position_rank": float(np.median(ranks_array)),
        "std_position_rank": float(np.std(ranks_array)),
        "position_spread": spread,
    }


def format_nodes(nodes: Sequence[int]) -> str:
    if not nodes:
        return ""
    return " ".join(str(node) for node in nodes)


def format_ranks(ranks: Sequence[float]) -> str:
    if not ranks:
        return ""
    return " ".join(f"{rank:.6f}" for rank in ranks)


def process_csv(
    csv_path: Path,
    *,
    topks: Sequence[int],
    output_dir: Optional[Path],
    include_summary: bool,
) -> List[Path]:
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"– CSV produced no rows: {csv_path}")
        return []

    base_output_dir = output_dir or csv_path.parent
    base_output_dir.mkdir(parents=True, exist_ok=True)
    base_name = csv_path.stem

    result_df = df.copy()
    written: List[Path] = []

    for top_k in topks:
        if top_k <= 0:
            continue

        metrics: Dict[str, List[object]] = {
            f"top_nodes_top{top_k}": [],
            f"position_ranks_top{top_k}": [],
            f"top_nodes_count_top{top_k}": [],
        }
        for stat_name in STAT_COLUMNS:
            metrics[f"{stat_name}_top{top_k}"] = []

        iterator = tqdm(
            result_df.to_dict(orient="records"),
            desc=f"{base_name} | top-{top_k}",
            leave=False,
        )
        for row in iterator:
            ranked_nodes = parse_ranked_nodes(row.get("ranked_nodes"))
            if ranked_nodes:
                total_nodes = infer_total_nodes(row.get("num_nodes"), ranked_nodes)
                top_nodes = ranked_nodes[:top_k]
                position_ranks = compute_position_ranks(top_nodes, total_nodes)
            else:
                top_nodes = []
                position_ranks = []

            stats = compute_stats(position_ranks)
            metrics[f"top_nodes_top{top_k}"].append(format_nodes(top_nodes))
            metrics[f"position_ranks_top{top_k}"].append(format_ranks(position_ranks))
            metrics[f"top_nodes_count_top{top_k}"].append(len(top_nodes))
            for stat_name, value in stats.items():
                metrics[f"{stat_name}_top{top_k}"].append(value)

        for column_name, values in metrics.items():
            result_df[column_name] = values

        if include_summary:
            summary_source = result_df.drop(columns=DROP_COLUMNS, errors="ignore")
            stat_columns = [f"{name}_top{top_k}" for name in STAT_COLUMNS]
            available_stat_columns = [column for column in stat_columns if column in summary_source.columns]
            if not available_stat_columns:
                continue
            grouping_candidates = [
                "method",
                "dataset",
                "graph_type",
                "split",
                "is_correct",
                "label",
            ]
            group_columns = [
                column
                for column in grouping_candidates
                if column in summary_source.columns and summary_source[column].notna().any()
            ]
            if not group_columns:
                continue
            summary = (
                summary_source[group_columns + available_stat_columns]
                .groupby(group_columns)
                .agg(["mean", "median", "std"])
            )
            summary.columns = ["_".join(filter(None, column)).strip("_") for column in summary.columns]
            summary = summary.reset_index()
            summary_path = base_output_dir / f"{base_name}_top{top_k}_summary.csv"
            summary.to_csv(summary_path, index=False)
            written.append(summary_path)
            print(f"✓ Saved aggregated summary to {summary_path} ({len(summary)} rows)")

    result_df_output = result_df.drop(columns=DROP_COLUMNS, errors="ignore")
    enriched_path = base_output_dir / f"{base_name}_topk_stats.csv"
    result_df_output.to_csv(enriched_path, index=False)
    written.insert(0, enriched_path)
    print(f"✓ Saved enriched row-level stats to {enriched_path} ({len(result_df_output)} rows)")

    return written


BASE_OUTPUT_DIR = Path("outputs/analytics/sequence")


def _is_base_csv(path: Path) -> bool:
    name = path.name
    if name.endswith("_topk_stats.csv"):
        return False
    if name.endswith("_summary.csv"):
        return False
    if "_top" in name:
        return False
    return True


def discover_csv_files(inputs: Optional[Sequence[str]]) -> List[Path]:
    paths: List[Path] = []
    if not inputs:
        if not BASE_OUTPUT_DIR.exists():
            raise FileNotFoundError(
                "No inputs provided and default base directory outputs/analytics/sequence does not exist."
            )
        candidates = sorted(BASE_OUTPUT_DIR.rglob("*.csv"))
        paths = [path for path in candidates if _is_base_csv(path)]
        return paths

    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            if _is_base_csv(path):
                paths.append(path)
        elif path.is_dir():
            candidates = sorted(path.rglob("*.csv"))
            paths.extend(path for path in candidates if _is_base_csv(path))
        else:
            print(f"! Input path not found: {path}")
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute ranking statistics from GNN explanation CSVs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="List of CSV files or directories containing CSVs produced by extract_gnn.py. When omitted, defaults to all base CSVs beneath outputs/analytics/sequence.",
    )
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        required=True,
        help="One or more top-k thresholds to evaluate (e.g., --topk 5 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where statistics files will be written. Defaults to each CSV's directory.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also write aggregated summaries (mean, median, std) grouped by dataset/graph type.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    csv_paths = discover_csv_files(args.inputs)
    if not csv_paths:
        raise FileNotFoundError("No CSV files found for processing.")

    all_written: List[Path] = []
    for csv_path in tqdm(csv_paths, desc="Processing CSV files", leave=False):
        written = process_csv(
            csv_path,
            topks=args.topk,
            output_dir=args.output_dir,
            include_summary=args.summary,
        )
        all_written.extend(written)

    if all_written:
        print(f"\nCompleted statistics generation for {len(csv_paths)} CSV file(s).")
        for path in all_written:
            print(f"  - {path}")
    else:
        print("\nNo statistics files were generated.")


if __name__ == "__main__":
    main()
