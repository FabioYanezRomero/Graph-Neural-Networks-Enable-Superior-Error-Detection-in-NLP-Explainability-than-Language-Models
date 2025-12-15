
"""
Comprehensive Explainability Metrics Analysis Script
Analyzes AUC and related metrics across multiple CSV files for GNN and LLM explainability modules
Generates interactive Plotly visualizations
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

DEFAULT_INPUT_ROOT = Path("outputs/analytics/auc")
DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/auc/plots/additional_metrics")

REQUIRED_COLUMNS: Set[str] = {
    "method",
    "dataset",
    "graph_type",
    "label",
    "prediction_confidence",
    "is_correct",
    "deletion_auc",
    "insertion_auc",
    "normalised_deletion_auc",
    "global_graph_index",
}

OPTIONAL_NUMERIC_COLUMNS: Sequence[str] = (
    "maskout_progression_len",
    "sufficiency_progression_len",
    "normalised_insertion_auc",
    "origin_confidence",
    "deletion_aac",
)

NUMERIC_COLUMNS: Sequence[str] = (
    "prediction_confidence",
    "origin_confidence",
    "deletion_auc",
    "insertion_auc",
    "normalised_deletion_auc",
    "normalised_insertion_auc",
    "maskout_progression_len",
    "sufficiency_progression_len",
    "global_graph_index",
    "deletion_aac",
)

BOOLEAN_TRUE_VALUES = {"true", "1", "yes", "y", "t"}
BOOLEAN_FALSE_VALUES = {"false", "0", "no", "n", "f"}

CORRECTNESS_COLOR_MAP = {
    "Correct": "#00CC96",
    "Incorrect": "#EF553B",
    "Unknown": "#636EFA",
}


def _slugify(value: Optional[str]) -> str:
    """Return a filesystem-safe identifier."""
    if not value:
        return "all"
    lowered = value.strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in lowered)
    cleaned = cleaned.strip("_")
    return cleaned or "all"


def _format_label(value: object) -> str:
    """Render labels in a dataset-agnostic, readable way."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    numeric_candidate = text.replace(".", "", 1)
    if numeric_candidate.isdigit():
        # Normalise trailing .0 in case of floats encoded as strings
        try:
            as_int = int(float(text))
            return f"Class {as_int}"
        except Exception:
            return f"Class {text}"
    return text


def _title_suffix(
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> str:
    """Build readable title suffixes that reflect the applied filters."""
    parts = [part for part in (method, dataset, graph_type) if part]
    return f" - {' / '.join(parts)}" if parts else ""


def _write_figure(fig: go.Figure, output_dir: Path, stem: str) -> Path:
    """Persist Plotly figures with consistent naming."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.html"
    fig.write_html(str(output_path))
    print(f"✓ Generated: {output_path.relative_to(output_dir)}")
    return output_path


def _filter_data(
    df: pd.DataFrame,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> pd.DataFrame:
    """Return a dataframe filtered by optional method, dataset, and graph type."""
    data = df.copy()
    if method and "method" in data.columns:
        data = data[data["method"].str.lower() == method.lower()]
    if dataset and "dataset" in data.columns:
        data = data[data["dataset"].str.lower() == dataset.lower()]
    if graph_type and "graph_type" in data.columns:
        data = data[data["graph_type"].str.lower() == graph_type.lower()]
    return data


def _context_label(
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> str:
    """Human-readable label for log messages."""
    parts = [part for part in (method, dataset, graph_type) if part]
    return " / ".join(parts) if parts else "all data"


def _method_dataset_graph_dir(base_dir: Path, method: str, dataset: str, graph: str) -> Path:
    """Return the output directory for a method/dataset/graph pairing."""
    if not method or not dataset or not graph:
        raise ValueError("Method, dataset, and graph type are required for output routing.")
    return base_dir / _slugify(method) / _slugify(dataset) / _slugify(graph)


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def _coerce_boolean(value: object):
    """Convert arbitrary truthy/falsy values into pandas-compatible booleans."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in BOOLEAN_TRUE_VALUES:
        return True
    if text in BOOLEAN_FALSE_VALUES:
        return False
    return pd.NA


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add optional numeric columns when they are missing from the CSV."""
    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and ensure required columns exist."""
    frame = _ensure_optional_columns(df.copy())

    for column in ("method", "dataset", "graph_type"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip()

    if "label" in frame.columns:
        frame["label"] = frame["label"].apply(lambda x: x if pd.isna(x) else str(x))

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "is_correct" in frame.columns:
        frame["is_correct"] = (
            frame["is_correct"].apply(_coerce_boolean).astype("boolean")
        )

    return frame


def _discover_per_graph_csvs(root: Path) -> List[Path]:
    """Locate per-instance explainability CSVs within the analytics directory."""
    csv_paths: List[Path] = []
    for csv_path in root.rglob("*.csv"):
        try:
            relative_parts = csv_path.relative_to(root).parts
        except ValueError:
            continue
        if len(relative_parts) == 3:
            csv_paths.append(csv_path)
    return sorted(csv_paths)


def _normalise_filter(values: Optional[Sequence[str]]) -> Optional[Set[str]]:
    """Prepare case-insensitive filters."""
    if values is None:
        return None
    filtered = {str(value).strip().lower() for value in values if str(value).strip()}
    return filtered or None


def load_all_csvs(
    input_root: Path = DEFAULT_INPUT_ROOT,
    *,
    methods: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    graph_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate per-instance explainability CSV files."""
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    candidates = _discover_per_graph_csvs(root)
    if not candidates:
        raise ValueError(f"No per-instance CSV files found under {root}")

    method_filter = _normalise_filter(methods)
    dataset_filter = _normalise_filter(datasets)
    graph_filter = _normalise_filter(graph_types)

    frames: List[pd.DataFrame] = []

    for csv_path in candidates:
        try:
            raw_df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Error loading {csv_path}: {exc}")
            continue

        missing = REQUIRED_COLUMNS - set(raw_df.columns)
        if missing:
            print(f"Skipping {csv_path}: missing columns {sorted(missing)}")
            continue

        frame = _sanitize_dataframe(raw_df)
        frame["source_csv"] = str(csv_path)

        if method_filter is not None and "method" in frame.columns:
            frame = frame[frame["method"].str.lower().isin(method_filter)]

        if dataset_filter is not None and "dataset" in frame.columns:
            frame = frame[frame["dataset"].str.lower().isin(dataset_filter)]

        if graph_filter is not None and "graph_type" in frame.columns:
            frame = frame[frame["graph_type"].str.lower().isin(graph_filter)]

        if frame.empty:
            continue

        frames.append(frame)
        print(f"Loaded: {csv_path} ({len(frame)} records)")

    if not frames:
        raise ValueError("No CSV files satisfied the requested filters.")

    combined_df = (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(
            by=["method", "dataset", "graph_type", "global_graph_index"],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    print(
        f"Total records loaded: {len(combined_df)} "
        f"from {len(frames)} CSV files across {combined_df['method'].nunique()} methods."
    )

    return combined_df


def parse_cli_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the analytics pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate advanced explainability analytics from per-instance AUC CSV files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root directory containing method/dataset graph CSVs (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination directory for generated HTML reports (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional list of method names to include (case-insensitive).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional list of dataset names to include (case-insensitive).",
    )
    parser.add_argument(
        "--graph-types",
        nargs="+",
        help="Optional list of graph types to include (case-insensitive).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Threshold for the high-confidence misprediction analysis (default: 0.9).",
    )
    parser.add_argument(
        "--progression-bins",
        type=int,
        default=10,
        help="Number of quantile bins for the instance complexity plot (default: 10).",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="Skip the aggregated plots across all methods.",
    )
    return parser.parse_args(args)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics for analysis"""
    # Deletion-Insertion divergence
    df['del_ins_spread'] = df['deletion_auc'] - df['insertion_auc']

    # Normalisation impact
    df['norm_diff'] = df['normalised_deletion_auc'] - df['deletion_auc']

    # AUC quality indicator (higher is better)
    df['auc_quality'] = df['deletion_auc'] * (1 - df['norm_diff'].abs())

    # Confidence-AUC alignment
    df['confidence_auc_product'] = df['prediction_confidence'] * df['deletion_auc']

    # Progression efficiency (lower progression_len = more efficient)
    df['progression_efficiency'] = 1.0 / (1.0 + df['maskout_progression_len'])

    return df


# ============================================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confidence_vs_auc(
    df: pd.DataFrame,
    output_dir: Path,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot 1: Prediction Confidence vs. Deletion AUC (stratified by correctness)
    """
    data = _filter_data(df, method=method, dataset=dataset, graph_type=graph_type)
    title_suffix = _title_suffix(method, dataset, graph_type)

    if data.empty:
        print(
            f"⚠️ Skipping confidence vs AUC plot for "
            f"{_context_label(method, dataset, graph_type)}: no data available."
        )
        return None

    data["correctness_name"] = (
        data["is_correct"]
        .map({True: "Correct", False: "Incorrect"})
        .fillna("Unknown")
    )

    fig = px.scatter(
        data,
        x="prediction_confidence",
        y="deletion_auc",
        color="correctness_name",
        facet_col="dataset",
        hover_data=[
            "label",
            "deletion_auc",
            "insertion_auc",
            "maskout_progression_len",
        ],
        labels={
            "prediction_confidence": "Prediction Confidence",
            "deletion_auc": "Deletion AUC",
            "correctness_name": "Prediction Outcome",
        },
        title=f"Confidence vs. AUC Reliability{title_suffix}",
        color_discrete_map=CORRECTNESS_COLOR_MAP,
    )

    fig.update_layout(
        height=600,
        hovermode="closest",
        template="plotly_white",
    )

    return _write_figure(fig, output_dir, "01_confidence_vs_auc")


def plot_deletion_insertion_divergence(
    df: pd.DataFrame,
    output_dir: Path,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot 2: Deletion-Insertion AUC Divergence by Label and Correctness
    """
    data = _filter_data(df, method=method, dataset=dataset, graph_type=graph_type)
    title_suffix = _title_suffix(method, dataset, graph_type)

    if data.empty:
        print(
            f"⚠️ Skipping deletion-insertion divergence plot for "
            f"{_context_label(method, dataset, graph_type)}: no data available."
        )
        return None

    data["label_name"] = data["label"].apply(_format_label)
    data["correctness_name"] = (
        data["is_correct"]
        .map({True: "Correct", False: "Incorrect"})
        .fillna("Unknown")
    )
    data["label_correctness"] = (
        data["label_name"] + " - " + data["correctness_name"]
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("By Label", "By Correctness", "By Label & Correctness"),
        specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}]],
    )

    # By label
    for label_name in sorted(data["label_name"].unique()):
        label_data = data[data["label_name"] == label_name]
        fig.add_trace(
            go.Box(
                y=label_data["del_ins_spread"],
                name=label_name,
                boxmean="sd",
            ),
            row=1,
            col=1,
        )

    # By correctness
    for correctness_name in ["Correct", "Incorrect", "Unknown"]:
        correctness_data = data[data["correctness_name"] == correctness_name]
        if correctness_data.empty:
            continue
        fig.add_trace(
            go.Box(
                y=correctness_data["del_ins_spread"],
                name=correctness_name,
                boxmean="sd",
            ),
            row=1,
            col=2,
        )

    # Combined view
    for combo in sorted(data["label_correctness"].unique()):
        combo_data = data[data["label_correctness"] == combo]
        fig.add_trace(
            go.Box(
                y=combo_data["del_ins_spread"],
                name=combo,
                boxmean="sd",
            ),
            row=1,
            col=3,
        )

    fig.update_yaxes(title_text="Deletion - Insertion AUC", row=1, col=1)
    fig.update_yaxes(title_text="Deletion - Insertion AUC", row=1, col=2)
    fig.update_yaxes(title_text="Deletion - Insertion AUC", row=1, col=3)
    fig.update_layout(
        title_text=f"Deletion-Insertion Divergence{title_suffix}",
        height=500,
        template="plotly_white",
    )

    return _write_figure(fig, output_dir, "02_deletion_insertion_divergence")


def plot_progression_length_impact(
    df: pd.DataFrame,
    output_dir: Path,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot 3: Graph Progression Length Impact on Fidelity
    """
    data = _filter_data(df, method=method, dataset=dataset, graph_type=graph_type)
    title_suffix = _title_suffix(method, dataset, graph_type)

    if data.empty:
        print(
            f"⚠️ Skipping progression length impact plot for "
            f"{_context_label(method, dataset, graph_type)}: no data available."
        )
        return None

    data["correctness_name"] = (
        data["is_correct"]
        .map({True: "Correct", False: "Incorrect"})
        .fillna("Unknown")
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Maskout Progression", "Sufficiency Progression"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    # Maskout progression vs deletion AUC
    fig.add_trace(
        go.Scatter(
            x=data["maskout_progression_len"],
            y=data["deletion_auc"],
            mode="markers",
            marker=dict(
                size=5,
                color=data["deletion_auc"],
                colorscale="Viridis",
                showscale=False,
            ),
            name="Deletion AUC",
            text=data["correctness_name"],
            hovertemplate="<b>Maskout:</b> %{x}<br><b>Deletion AUC:</b> %{y:.3f}"
            "<br><b>Outcome:</b> %{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Sufficiency progression vs deletion AUC
    fig.add_trace(
        go.Scatter(
            x=data["sufficiency_progression_len"],
            y=data["deletion_auc"],
            mode="markers",
            marker=dict(
                size=5,
                color=data["deletion_auc"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Deletion AUC"),
            ),
            name="Deletion AUC",
            text=data["correctness_name"],
            hovertemplate="<b>Sufficiency:</b> %{x}<br><b>Deletion AUC:</b> %{y:.3f}"
            "<br><b>Outcome:</b> %{text}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Maskout Progression Length", row=1, col=1)
    fig.update_xaxes(title_text="Sufficiency Progression Length", row=1, col=2)
    fig.update_yaxes(title_text="Deletion AUC", row=1, col=1)
    fig.update_yaxes(title_text="Deletion AUC", row=1, col=2)

    fig.update_layout(
        title_text=f"Progression Length Impact on Fidelity{title_suffix}",
        height=500,
        template="plotly_white",
    )

    return _write_figure(fig, output_dir, "03_progression_length_impact")


def plot_instance_complexity_sensitivity(
    df: pd.DataFrame,
    output_dir: Path,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    n_bins: int = 10,
) -> Optional[Path]:
    """
    Plot 4: Token-Position vs. AUC Sensitivity Profiles (binned by graph_index)
    """
    data = _filter_data(
        df, method=method, dataset=dataset, graph_type=graph_type
    )
    title_suffix = _title_suffix(method, dataset, graph_type)

    if data.empty:
        print(
            f"⚠️ Skipping instance complexity sensitivity plot for "
            f"{_context_label(method, dataset, graph_type)}: no data available."
        )
        return None

    # Create bins based on global_graph_index for complexity levels
    try:
        data["complexity_bin"] = pd.qcut(
            data["global_graph_index"], q=n_bins, duplicates="drop"
        )
    except ValueError:
        # Fallback to a single bin when qcut cannot create the requested quantiles
        data["complexity_bin"] = pd.cut(
            data["global_graph_index"], bins=min(len(data), 2), duplicates="drop"
        )

    bin_stats = (
        data.groupby("complexity_bin", observed=True)
        .agg({"deletion_auc": ["mean", "std", "count"], "insertion_auc": "mean"})
        .reset_index()
    )

    bin_stats.columns = [
        "complexity_bin",
        "del_auc_mean",
        "del_auc_std",
        "count",
        "ins_auc_mean",
    ]
    bin_stats["bin_center"] = bin_stats["complexity_bin"].apply(lambda x: x.mid)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bin_stats["bin_center"],
        y=bin_stats["del_auc_mean"],
        error_y=dict(type="data", array=bin_stats["del_auc_std"], visible=True),
        mode="lines+markers",
        name="Deletion AUC",
        marker=dict(size=8, color="#636EFA"),
        line=dict(width=2),
    ))

    fig.add_trace(go.Scatter(
        x=bin_stats["bin_center"],
        y=bin_stats["ins_auc_mean"],
        mode="lines+markers",
        name="Insertion AUC",
        marker=dict(size=8, color="#EF553B"),
        line=dict(width=2),
    ))

    fig.update_layout(
        title=f"AUC Sensitivity by Instance Complexity{title_suffix}",
        xaxis_title="Instance Complexity (Graph Index)",
        yaxis_title="AUC Score",
        template="plotly_white",
        height=600,
        hovermode="x unified",
    )

    return _write_figure(fig, output_dir, "04_instance_complexity_sensitivity")


def plot_normalised_auc_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot 5: Raw vs. Normalized AUC Distribution Analysis
    """
    data = _filter_data(df, method=method, dataset=dataset, graph_type=graph_type)
    title_suffix = _title_suffix(method, dataset, graph_type)

    if data.empty:
        print(
            f"⚠️ Skipping normalised AUC distribution plot for "
            f"{_context_label(method, dataset, graph_type)}: no data available."
        )
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Raw Deletion AUC Distribution',
            'Normalized Deletion AUC Distribution',
            'Raw vs. Normalized (Scatter)',
            'Normalization Impact Distribution'
        ),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'scatter'}, {'type': 'histogram'}]]
    )

    # Raw deletion AUC histogram
    fig.add_trace(
        go.Histogram(x=data['deletion_auc'], name='Raw', nbinsx=30, marker_color='#636EFA'),
        row=1, col=1
    )

    # Normalized deletion AUC histogram
    fig.add_trace(
        go.Histogram(x=data['normalised_deletion_auc'], name='Normalized', nbinsx=30, marker_color='#EF553B'),
        row=1, col=2
    )

    # Scatter: raw vs normalized
    fig.add_trace(
        go.Scatter(
            x=data['deletion_auc'],
            y=data['normalised_deletion_auc'],
            mode='markers',
            marker=dict(size=4, color='#00CC96', opacity=0.6),
            name='Raw vs Norm',
            hovertemplate='Raw: %{x:.3f}<br>Normalized: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Normalization impact
    fig.add_trace(
        go.Histogram(x=data['norm_diff'], name='Impact', nbinsx=30, marker_color='#AB63FA'),
        row=2, col=2
    )

    fig.update_xaxes(title_text='Deletion AUC', row=1, col=1)
    fig.update_xaxes(title_text='Normalized Deletion AUC', row=1, col=2)
    fig.update_xaxes(title_text='Raw AUC', row=2, col=1)
    fig.update_yaxes(title_text='Normalized AUC', row=2, col=1)
    fig.update_xaxes(title_text='Normalization Difference', row=2, col=2)

    fig.update_layout(
        title_text=f'Raw vs. Normalized AUC Distribution{title_suffix}',
        height=800,
        template='plotly_white',
        showlegend=True
    )

    return _write_figure(fig, output_dir, "05_normalised_auc_distribution")


def plot_label_stratified_fidelity(
    df: pd.DataFrame,
    output_dir: Path,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot 6: Label-Stratified Fidelity Boxplots (Deletion AUC by Label and Correctness)
    """
    data = _filter_data(df, method=method, dataset=dataset, graph_type=graph_type)
    title_suffix = _title_suffix(method, dataset, graph_type)

    if data.empty:
        print(
            f"⚠️ Skipping label-stratified fidelity plot for "
            f"{_context_label(method, dataset, graph_type)}: no data available."
        )
        return None

    data["label_name"] = data["label"].apply(_format_label)
    data["correctness_name"] = (
        data["is_correct"]
        .map({True: "Correct", False: "Incorrect"})
        .fillna("Unknown")
    )
    data["label_correctness"] = (
        data["label_name"] + " - " + data["correctness_name"]
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Deletion AUC by Label",
            "Deletion AUC by Correctness",
            "Deletion AUC by Label & Correctness",
        ),
        specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}]],
    )

    # By label
    for label in sorted(data["label_name"].unique()):
        label_data = data[data["label_name"] == label]
        if label_data.empty:
            continue
        fig.add_trace(
            go.Box(
                y=label_data["deletion_auc"],
                name=label,
                boxmean="sd",
            ),
            row=1,
            col=1,
        )

    # By correctness
    for correctness in ["Correct", "Incorrect", "Unknown"]:
        correctness_data = data[data["correctness_name"] == correctness]
        if correctness_data.empty:
            continue
        fig.add_trace(
            go.Box(
                y=correctness_data["deletion_auc"],
                name=correctness,
                boxmean="sd",
            ),
            row=1,
            col=2,
        )

    # Combined stratification
    for combo in sorted(data["label_correctness"].unique()):
        combo_data = data[data["label_correctness"] == combo]
        fig.add_trace(
            go.Box(
                y=combo_data["deletion_auc"],
                name=combo,
                boxmean="sd",
            ),
            row=1,
            col=3,
        )

    fig.update_yaxes(title_text="Deletion AUC", row=1, col=1)
    fig.update_yaxes(title_text="Deletion AUC", row=1, col=2)
    fig.update_yaxes(title_text="Deletion AUC", row=1, col=3)

    fig.update_layout(
        title_text=f"Label-Stratified Fidelity{title_suffix}",
        height=500,
        template="plotly_white",
    )

    return _write_figure(fig, output_dir, "06_label_stratified_fidelity")


def plot_confident_mispredictions_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    confidence_threshold: float = 0.9,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot 7: High-Confidence Misprediction Deep Dive
    """
    data = _filter_data(df, method=method, dataset=dataset, graph_type=graph_type)
    title_suffix = _title_suffix(method, dataset, graph_type)

    if data.empty:
        print(
            f"⚠️ Skipping confident misprediction analysis plot for "
            f"{_context_label(method, dataset, graph_type)}: no data available."
        )
        return None

    # Identify high-confidence mispredictions
    high_confidence = data["prediction_confidence"] >= confidence_threshold
    high_conf_errors = data[high_confidence & data["is_correct"].eq(False)]
    high_conf_correct = data[high_confidence & data["is_correct"].eq(True)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'AUC Distribution (High Confidence > {confidence_threshold})',
            'Deletion AUC Comparison',
            'Error Detection Rate',
            'Confidence vs AUC (High Confidence Errors)'
        ),
        specs=[[{'type': 'histogram'}, {'type': 'box'}],
               [{'type': 'indicator'}, {'type': 'scatter'}]]
    )

    # Histogram
    fig.add_trace(
        go.Histogram(x=high_conf_errors['deletion_auc'], name='Mispredictions', 
                    marker_color='#EF553B', nbinsx=20),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=high_conf_correct['deletion_auc'], name='Correct Predictions', 
                    marker_color='#00CC96', nbinsx=20),
        row=1, col=1
    )

    # Boxplot comparison
    fig.add_trace(
        go.Box(y=high_conf_errors['deletion_auc'], name='Errors', marker_color='#EF553B'),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=high_conf_correct['deletion_auc'], name='Correct', marker_color='#00CC96'),
        row=1, col=2
    )

    # Error detection rate (AUC < 0.6 for errors)
    if high_conf_errors.empty:
        error_detection_rate = 0.0
    else:
        error_detection_rate = (
            (high_conf_errors['deletion_auc'] < 0.6).sum() / len(high_conf_errors) * 100
        )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=error_detection_rate,
            title={'text': "Error Detection Rate (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightgreen"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}
        ),
        row=2, col=1
    )

    # Scatter: confidence vs AUC for high-confidence errors
    fig.add_trace(
        go.Scatter(
            x=high_conf_errors['prediction_confidence'],
            y=high_conf_errors['deletion_auc'],
            mode='markers',
            marker=dict(size=6, color=high_conf_errors['deletion_auc'], 
                       colorscale='RdYlGn', showscale=True),
            name='High Conf Errors',
            hovertemplate='Confidence: %{x:.3f}<br>Deletion AUC: %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )

    fig.update_xaxes(title_text='Deletion AUC', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_yaxes(title_text='Deletion AUC', row=1, col=2)
    fig.update_xaxes(title_text='Confidence', row=2, col=2)
    fig.update_yaxes(title_text='Deletion AUC', row=2, col=2)

    fig.update_layout(
        title_text=f'High-Confidence Misprediction Analysis{title_suffix}',
        height=900,
        template='plotly_white'
    )

    return _write_figure(fig, output_dir, "07_confident_mispredictions")


def compute_statistical_summary(
    df: pd.DataFrame,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute statistical metrics for the analysis
    """
    data = _filter_data(
        df, method=method, dataset=dataset, graph_type=graph_type
    )

    if data.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    summary = []

    # Confidence-AUC correlation
    try:
        corr_spearman, p_spearman = stats.spearmanr(
            data["prediction_confidence"], data["deletion_auc"], nan_policy="omit"
        )
    except Exception:
        corr_spearman, p_spearman = np.nan, np.nan

    try:
        corr_pearson, p_pearson = stats.pearsonr(
            data["prediction_confidence"].dropna(),
            data["deletion_auc"].dropna(),
        )
    except Exception:
        corr_pearson, p_pearson = np.nan, np.nan

    # Normalization impact
    try:
        norm_corr, norm_p = stats.pearsonr(
            data["deletion_auc"].dropna(),
            data["normalised_deletion_auc"].dropna(),
        )
    except Exception:
        norm_corr, norm_p = np.nan, np.nan

    # Error detection capability
    incorrect_mask = data["is_correct"].eq(False)
    incorrect_deletion = data.loc[incorrect_mask, "deletion_auc"]
    error_detection_auc_60 = (
        float((incorrect_deletion < 0.6).mean()) if not incorrect_deletion.empty else np.nan
    )
    error_detection_auc_70 = (
        float((incorrect_deletion < 0.7).mean()) if not incorrect_deletion.empty else np.nan
    )

    # Mean divergence
    mean_div_correct = data.loc[data["is_correct"].eq(True), "del_ins_spread"].mean()
    mean_div_incorrect = data.loc[data["is_correct"].eq(False), "del_ins_spread"].mean()

    summary_dict = {
        "Metric": [
            "Spearman Correlation (Conf-AUC)",
            "Pearson Correlation (Conf-AUC)",
            "Normalization Correlation",
            "Error Detection Rate (AUC<0.6)",
            "Error Detection Rate (AUC<0.7)",
            "Mean Divergence (Correct)",
            "Mean Divergence (Incorrect)",
            "Deletion AUC (Mean)",
            "Deletion AUC (Std)",
            "Total Records",
        ],
        "Value": [
            f"{corr_spearman:.4f} (p={p_spearman:.4e})"
            if not np.isnan(corr_spearman)
            else "N/A",
            f"{corr_pearson:.4f} (p={p_pearson:.4e})"
            if not np.isnan(corr_pearson)
            else "N/A",
            f"{norm_corr:.4f}" if not np.isnan(norm_corr) else "N/A",
            f"{error_detection_auc_60:.2%}"
            if not np.isnan(error_detection_auc_60)
            else "N/A",
            f"{error_detection_auc_70:.2%}"
            if not np.isnan(error_detection_auc_70)
            else "N/A",
            f"{mean_div_correct:.4f}"
            if not np.isnan(mean_div_correct)
            else "N/A",
            f"{mean_div_incorrect:.4f}"
            if not np.isnan(mean_div_incorrect)
            else "N/A",
            f"{data['deletion_auc'].mean():.4f}"
            if not data["deletion_auc"].dropna().empty
            else "N/A",
            f"{data['deletion_auc'].std():.4f}"
            if not data["deletion_auc"].dropna().empty
            else "N/A",
            f"{len(data)}",
        ],
    }

    return pd.DataFrame(summary_dict)


def plot_statistical_summary(
    df: pd.DataFrame,
    output_dir: Path,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[Path]]:
    """
    Plot 8: Statistical Summary Dashboard
    """
    summary_df = compute_statistical_summary(df, method, dataset, graph_type)

    if summary_df.empty:
        print(
            f"⚠️ Skipping statistical summary table for "
            f"{_context_label(method, dataset, graph_type)}: no data available."
        )
        return summary_df, None

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_df.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[summary_df['Metric'], summary_df['Value']],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])

    title_suffix = _title_suffix(method, dataset, graph_type)
    fig.update_layout(
        title_text=f'Statistical Summary{title_suffix}',
        height=500,
        template='plotly_white'
    )

    output_path = _write_figure(fig, output_dir, "08_statistical_summary")

    return summary_df, output_path


# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================

def main(args: Optional[Sequence[str]] = None):
    """Main execution function"""
    options = parse_cli_args(args)

    print("=" * 70)
    print("Explainability Metrics Analysis Pipeline")
    print("=" * 70)

    print("\n[1] Loading data from CSV files...")
    df = load_all_csvs(
        options.input_root,
        methods=options.methods,
        datasets=options.datasets,
        graph_types=options.graph_types,
    )
    print(f"Total records loaded: {len(df)}")
    print(f"Available datasets: {sorted(df['dataset'].unique())}")
    print(f"Available methods: {sorted(df['method'].unique())}")
    print(f"Available graph types: {sorted(df['graph_type'].unique())}")

    print("\n[2] Computing derived metrics...")
    df = compute_metrics(df)

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: List[Path] = []

    print("\n[3] Generating method/dataset/graph visualisations...")
    methods = sorted(df["method"].dropna().unique())

    for method in methods:
        method_df = df[df["method"].str.lower() == method.lower()]
        method_datasets = sorted(method_df["dataset"].dropna().unique())
        for dataset in method_datasets:
            dataset_df = method_df[method_df["dataset"].str.lower() == dataset.lower()]
            graph_types = sorted(dataset_df["graph_type"].dropna().unique())
            for graph_type in graph_types:
                print(f"\n  Processing: {method} / {dataset} / {graph_type}")
                target_dir = _method_dataset_graph_dir(output_dir, method, dataset, graph_type)
                outputs = [
                    plot_confidence_vs_auc(
                        df,
                        target_dir,
                        method=method,
                        dataset=dataset,
                        graph_type=graph_type,
                    ),
                    plot_deletion_insertion_divergence(
                        df,
                        target_dir,
                        method=method,
                        dataset=dataset,
                        graph_type=graph_type,
                    ),
                    plot_progression_length_impact(
                        df,
                        target_dir,
                        method=method,
                        dataset=dataset,
                        graph_type=graph_type,
                    ),
                    plot_instance_complexity_sensitivity(
                        df,
                        target_dir,
                        method=method,
                        dataset=dataset,
                        graph_type=graph_type,
                        n_bins=options.progression_bins,
                    ),
                    plot_normalised_auc_distribution(
                        df,
                        target_dir,
                        method=method,
                        dataset=dataset,
                        graph_type=graph_type,
                    ),
                    plot_label_stratified_fidelity(
                        df,
                        target_dir,
                        method=method,
                        dataset=dataset,
                        graph_type=graph_type,
                    ),
                    plot_confident_mispredictions_analysis(
                        df,
                        target_dir,
                        confidence_threshold=options.confidence_threshold,
                        method=method,
                        dataset=dataset,
                        graph_type=graph_type,
                    ),
                ]
                generated_files.extend(path for path in outputs if path is not None)

                summary_df, summary_path = plot_statistical_summary(
                    df,
                    target_dir,
                    method=method,
                    dataset=dataset,
                    graph_type=graph_type,
                )
                if summary_path:
                    generated_files.append(summary_path)
                if not summary_df.empty:
                    csv_path = target_dir / "08_statistical_summary.csv"
                    summary_df.to_csv(csv_path, index=False)
                    generated_files.append(csv_path)
                    print(
                        f"\n  Statistical Summary for {method} / {dataset} / {graph_type}:"
                    )
                    print(summary_df.to_string(index=False))

    summary_df_global = pd.DataFrame()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    if generated_files:
        unique_files: List[Path] = []
        seen: Set[Path] = set()
        for path in generated_files:
            if path not in seen:
                unique_files.append(path)
                seen.add(path)

        print("\nGenerated report files:")
        for idx, path in enumerate(unique_files, start=1):
            print(f"  {idx}. {path}")

    return df, summary_df_global


if __name__ == "__main__":
    df_final, summary_final = main()
