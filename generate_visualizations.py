#!/usr/bin/env python3
"""
Generate visualizations for GNN explanations.
"""

import sys
import csv
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visualisations import (
    generate_sparsity_visuals,
    generate_confidence_threshold_visuals,
    generate_token_score_densities,
    generate_token_score_differences,
    generate_token_score_ranking,
    generate_token_position_differences,
    generate_token_frequency_charts,
    generate_embedding_importance_correlation,
    generate_embedding_neighborhoods,
)

def _read_summary_metadata(summary_path: Path) -> tuple[str | None, str | None]:
    if not summary_path.exists():
        return None, None
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
            if row is None:
                return None, None
            dataset = row.get("dataset")
            graph_type = row.get("graph_type")
            return (dataset if dataset else None, graph_type if graph_type else None)
    except Exception:
        return None, None

def main():
    print("Generating visualizations for GNN analytics...")

    # Define the analytics roots
    analytics_root = Path("outputs/analytics/gnn")
    general_root = analytics_root / "general"
    pyg_root = Path("outputs/pyg_graphs")

    if not general_root.exists():
        print(f"No general analytics found under {general_root}.")
        return

    dataset_graph_dirs = sorted(path for path in general_root.iterdir() if path.is_dir())
    if not dataset_graph_dirs:
        print(f"No dataset graph directories found in {general_root}.")
        return

    # Generate visualizations for each dataset/graph type
    for folder in dataset_graph_dirs:
        summary_path = folder / "summary.csv"
        dataset_name, graph_type = _read_summary_metadata(summary_path)
        label = dataset_name if dataset_name else folder.name
        descriptor = f"{label}/{graph_type}" if graph_type else folder.name
        print(f"\nProcessing {descriptor}...")

        dataset_graph = folder.name

        # 1. Sparsity visualizations
        if summary_path.exists():
            print(f"  Generating sparsity visualizations...")
            generate_sparsity_visuals(
                summary_root=folder,
                output_root=analytics_root / "sparsity" / dataset_graph
            )

        # 2. Confidence visualizations
        if summary_path.exists():
            print(f"  Generating confidence visualizations...")
            generate_confidence_threshold_visuals(
                summary_root=folder,
                output_root=analytics_root / "confidence" / dataset_graph
            )

        # 3. Token score visualizations
        tokens_path = folder / "tokens.csv"
        token_exports_root = analytics_root / "token" / dataset_graph / "csv"
        if tokens_path.exists():
            print(f"  Generating token score visualizations...")
            generate_token_score_densities(
                tokens_root=folder,
                output_root=analytics_root / "score" / "density" / dataset_graph
            )
        if token_exports_root.exists():
            print(f"  Generating correct vs incorrect token comparisons...")
            generate_token_score_differences(
                tokens_root=token_exports_root,
                output_root=analytics_root / "score" / "difference" / dataset_graph
            )
            generate_token_position_differences(
                tokens_root=token_exports_root,
                output_root=analytics_root / "position" / "difference" / dataset_graph
            )

        if tokens_path.exists():
            generate_token_score_ranking(
                tokens_root=folder,
                output_root=analytics_root / "score" / "ranking" / dataset_graph
            )

        # 4. Token frequency charts
        if tokens_path.exists():
            print(f"  Generating token frequency charts...")
            generate_token_frequency_charts(
                tokens_root=folder,
                output_root=analytics_root / "token" / "frequency" / dataset_graph
            )

        # 5. Embedding visualizations (only for constituency and syntactic)
        if tokens_path.exists() and pyg_root.exists() and graph_type in {"constituency", "syntactic"}:
            print(f"  Generating embedding correlation visualizations...")
            generate_embedding_importance_correlation(
                tokens_root=folder,
                pyg_root=pyg_root,
                output_root=analytics_root / "embedding" / "correlation" / dataset_graph
            )

            print(f"  Generating embedding neighborhoods visualizations...")
            generate_embedding_neighborhoods(
                tokens_root=folder,
                pyg_root=pyg_root,
                output_root=analytics_root / "embedding" / "neighborhoods" / dataset_graph
            )

    print("\nAll visualizations generated!")

if __name__ == "__main__":
    main()




