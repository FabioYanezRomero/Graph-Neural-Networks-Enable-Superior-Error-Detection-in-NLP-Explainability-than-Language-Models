from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


STRUCTURAL_ROOT = Path("outputs/gnn_models")
OUTPUT_ROOT = Path("outputs/analytics/structural/substructures")

# Structural metrics to include in the correlation analysis (order preserved)
ALLOWED_FEATURES = [
    "avg_degree",
    "max_degree",
    "degree_variance",
    "degree_skewness",
    "n_components",
    "avg_betweenness",
    "max_betweenness",
    "min_betweenness",
    "avg_closeness",
    "max_closeness",
    "num_edges",
    "num_nodes",
]

# Columns that should never be considered for correlation computation
EXCLUDE_COLUMNS = {
    "dataset",
    "graph_type",
    "module",
    "shard",
    "pickle_file",
    "graph_index",
    "global_graph_index",
    "label",
    "prediction_class",
    "is_correct",
}


def slugify(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower().replace(" ", "_").replace("-", "_")
    allowed = [ch for ch in text if ch.isalnum() or ch in {"_", "."}]
    slug = "".join(allowed)
    return slug or "unknown"


def discover_structural_csvs(root: Path) -> List[Path]:
    return sorted(root.glob("**/*structural_properties.csv"))


def to_numeric_series(series: pd.Series) -> Optional[pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < 2:
        return None
    if numeric.nunique(dropna=True) < 2:
        return None
    return numeric


def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    columns: Dict[str, pd.Series] = {}
    for column in df.columns:
        if column in EXCLUDE_COLUMNS:
            continue
        if ALLOWED_FEATURES and column not in ALLOWED_FEATURES:
            continue
        numeric_series = to_numeric_series(df[column])
        if numeric_series is not None:
            columns[column] = numeric_series
    numeric_df = pd.DataFrame(columns, index=df.index)
    # Ensure the order matches ALLOWED_FEATURES
    ordered_cols = [col for col in ALLOWED_FEATURES if col in numeric_df.columns]
    numeric_df = numeric_df[ordered_cols]
    numeric_df = numeric_df.dropna(axis=1, how="all")
    return numeric_df


def parse_is_correct(series: pd.Series) -> pd.Series:
    def _convert(value: object) -> Optional[bool]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return None

    return series.map(_convert)


def infer_correctness(df: pd.DataFrame) -> pd.Series:
    if "label" not in df.columns or "prediction_class" not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    def _normalize(value: object) -> Optional[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip()
        if text == "":
            return None
        try:
            return str(int(float(text)))
        except Exception:
            return text

    labels = df["label"].map(_normalize)
    preds = df["prediction_class"].map(_normalize)
    correctness = labels == preds
    correctness[(labels.isna()) | (preds.isna())] = None
    return correctness


def compute_correlation(matrix: pd.DataFrame) -> Optional[pd.DataFrame]:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    corr = matrix.corr(method="pearson")
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if corr.shape[0] < 2 or corr.shape[1] < 2:
        return None
    return corr


def save_correlation(
    corr: pd.DataFrame,
    destination: Path,
    *,
    row_count: int,
    group_descriptor: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(destination)
    meta = {
        "rows": row_count,
        "features": corr.shape[0],
        "group": group_descriptor,
    }
    meta_path = destination.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[corr] {destination} (rows={row_count}, features={corr.shape[0]})")


def process_group(
    df: pd.DataFrame,
    numeric_matrix: pd.DataFrame,
    destination: Path,
    descriptor: str,
) -> None:
    if numeric_matrix.empty:
        return
    corr = compute_correlation(numeric_matrix)
    if corr is None:
        return
    save_correlation(corr, destination, row_count=numeric_matrix.shape[0], group_descriptor=descriptor)


def build_destinations(module: str, dataset: str, graph_type: str) -> Path:
    module_slug = slugify(module)
    dataset_slug = slugify(dataset)
    graph_slug = slugify(graph_type)
    return OUTPUT_ROOT / module_slug / dataset_slug / graph_slug


def process_file(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, sep=";")
    if df.empty:
        return

    module = df["module"].iloc[0]
    dataset = df["dataset"].iloc[0]
    graph_type = df["graph_type"].iloc[0]

    destination_root = build_destinations(module, dataset, graph_type)

    base_numeric = extract_numeric_features(df)
    process_group(
        df,
        base_numeric,
        destination_root / "correlation_overall.csv",
        descriptor="overall",
    )

    # Stratify by label (class)
    if "label" in df.columns:
        for label_value, subset in df.groupby("label"):
            matrix = base_numeric.loc[subset.index]
            if matrix.shape[0] < 2:
                continue
            label_slug = slugify(label_value)
            process_group(
                subset,
                matrix,
                destination_root / f"correlation_class_{label_slug}.csv",
                descriptor=f"class={label_value}",
            )

    # Stratify by correctness
    correctness_series: Optional[pd.Series] = None
    if "is_correct" in df.columns:
        parsed = parse_is_correct(df["is_correct"])
        if parsed.notna().sum() > 0:
            correctness_series = parsed

    if correctness_series is None:
        inferred = infer_correctness(df)
        if inferred.notna().sum() > 0:
            correctness_series = inferred

    if correctness_series is not None:
        df_correct = df.assign(_is_correct_bool=correctness_series)
        for value in correctness_series.dropna().unique():
            subset_index = df_correct.index[correctness_series == value]
            if len(subset_index) < 2:
                continue
            matrix = base_numeric.loc[subset_index]
            if matrix.shape[0] < 2:
                continue
            value_slug = "true" if value else "false"
            process_group(
                df_correct.loc[subset_index],
                matrix,
                destination_root / f"correlation_correct_{value_slug}.csv",
                descriptor=f"correct={value}",
            )

        if "label" in df.columns:
            valid_mask = correctness_series.notna()
            df_valid = df_correct.loc[valid_mask]
            for (label_value, correct_value), subset in df_valid.groupby(
                ["label", "_is_correct_bool"]
            ):
                if len(subset) < 2:
                    continue
                matrix = base_numeric.loc[subset.index]
                if matrix.shape[0] < 2:
                    continue
                label_slug = slugify(label_value)
                correct_slug = "true" if correct_value else "false"
                process_group(
                    subset,
                    matrix,
                    destination_root / f"correlation_class_{label_slug}_correct_{correct_slug}.csv",
                    descriptor=f"class={label_value}, correct={correct_value}",
                )


def main() -> None:
    csv_paths = discover_structural_csvs(STRUCTURAL_ROOT)
    if not csv_paths:
        print("No structural property CSV files found.")
        return

    for csv_path in csv_paths:
        try:
            process_file(csv_path)
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[warn] failed to process {csv_path}: {exc}")


if __name__ == "__main__":
    main()
