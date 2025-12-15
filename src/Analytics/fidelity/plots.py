#!/usr/bin/env python3
"""
Correctness Discernability Analysis - Visualization Pipeline
=============================================================

This module generates comprehensive visualizations for the correctness discernability
analysis, demonstrating how GNN methods excel at error detection compared to TokenSHAP.

Key Metrics:
  - Fidelity+ Effect: Correctness discrimination via positive attribution
  - Fidelity- Effect: Correctness discrimination via deletion mechanism
  - Failure Rate Effect: Diagnostic anti-fidelity rate difference
  - Confidence Coupling: Mechanism enabling error detection

Author: Research Team
Date: 2025-11-02
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

DEFAULT_SUMMARY = Path("outputs/analytics/fidelity/fidelity_summary.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analytics/fidelity/plots/correctness_discernability")


class CorrectnessDiscernabilityVisualizations:
    """Generate visualizations for correctness discernability analysis."""

    def __init__(self, csv_path: Path, output_dir: Path) -> None:
        """
        Initialize visualization pipeline.

        Args:
            csv_path: Path to fidelity_summary.csv
            output_dir: Directory for output HTML files
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if self.df.empty:
            raise ValueError(f"No data found in summary CSV: {csv_path}")

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._parse_group_column()
        self._calculate_effects()

        self.method_order = ["graphsvx", "subgraphx", "token_shap_llm"]
        self.method_labels: Dict[str, str] = {
            "graphsvx": "GraphSVX",
            "subgraphx": "SubgraphX",
            "token_shap_llm": "TokenSHAP",
        }
        self.colors_correct = {"Correct": "#2ecc71", "Incorrect": "#e74c3c"}
        self.colors_method = {
            "graphsvx": "#3498db",
            "subgraphx": "#2ecc71",
            "token_shap_llm": "#e67e22",
        }

        print("✓ Initialized CorrectnessDiscernabilityVisualizations")
        print(
            f"  Rows: {len(self.df)} | Methods: {self.df['method'].nunique()} | "
            f"Datasets: {self.df['dataset'].nunique()}"
        )

    # ------------------------------------------------------------------ #
    # Data preparation helpers
    # ------------------------------------------------------------------ #

    def _parse_group_column(self) -> None:
        """Extract correctness label from the group column."""
        def parse_correctness(raw_group: str) -> bool | None:
            text = str(raw_group).lower()
            if "correct_true" in text:
                return True
            if "correct_false" in text:
                return False
            return None

        self.df["is_correct"] = self.df["group"].apply(parse_correctness)
        self.df["correctness_label"] = self.df["is_correct"].map(
            {True: "Correct", False: "Incorrect"}
        )

    def _calculate_effects(self) -> None:
        """Calculate correctness effects for each method-dataset combination."""
        records: list[dict[str, float | str]] = []

        grouped = self.df[self.df["is_correct"].notna()].groupby(["method", "dataset"])
        for (method, dataset), subset in grouped:
            correct_data = subset[subset["is_correct"] == True]  # noqa: E712
            incorrect_data = subset[subset["is_correct"] == False]  # noqa: E712

            if correct_data.empty or incorrect_data.empty:
                continue

            # Fidelity metrics
            fid_plus_correct = correct_data["fidelity_plus_mean"].mean()
            fid_plus_incorrect = incorrect_data["fidelity_plus_mean"].mean()
            fid_minus_correct = correct_data["fidelity_minus_mean"].mean()
            fid_minus_incorrect = incorrect_data["fidelity_minus_mean"].mean()

            # Effects
            fid_plus_effect = abs(fid_plus_correct - fid_plus_incorrect)
            fid_minus_effect = abs(fid_minus_correct - fid_minus_incorrect)

            # Failure rates
            fail_correct = correct_data["fidelity_plus_negative_pct"].mean()
            fail_incorrect = incorrect_data["fidelity_plus_negative_pct"].mean()
            fail_effect = fail_incorrect - fail_correct

            # Confidence gap
            conf_correct = correct_data["mean_prediction_confidence"].mean()
            conf_incorrect = incorrect_data["mean_prediction_confidence"].mean()
            conf_gap = conf_correct - conf_incorrect

            # Coupling
            corr_conf_plus = correct_data["corr_confidence_fidelity_plus"].mean()

            records.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "fid_plus_correct": fid_plus_correct,
                    "fid_plus_incorrect": fid_plus_incorrect,
                    "fid_plus_effect": fid_plus_effect,
                    "fid_minus_correct": fid_minus_correct,
                    "fid_minus_incorrect": fid_minus_incorrect,
                    "fid_minus_effect": fid_minus_effect,
                    "fail_correct": fail_correct,
                    "fail_incorrect": fail_incorrect,
                    "fail_effect": fail_effect,
                    "conf_gap": conf_gap,
                    "corr_conf_plus": corr_conf_plus,
                }
            )

        if not records:
            raise ValueError("Unable to compute effects: missing correctness stratification.")

        self.effects_df = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    def _save_figure(self, fig: go.Figure, filename: str) -> None:
        """Save Plotly figure to HTML."""
        path = self.output_dir / filename
        fig.write_html(str(path))
        print(f"  ✓ Saved: {path.relative_to(self.output_dir)}")

    def _method_category_orders(self) -> Dict[str, Iterable[str]]:
        """Provide consistent ordering for categorical axes."""
        return {
            "method": self.method_order,
            "Method": [self.method_labels[m] for m in self.method_order],
            "Signal": ["Fidelity+ Effect", "Fidelity- Effect", "Failure Rate Effect"],
        }

    # ------------------------------------------------------------------ #
    # Tier-1 visualizations
    # ------------------------------------------------------------------ #

    def plot_d1_fidelity_plus_correctness_discrimination(self) -> None:
        """Grouped bar chart: Fidelity+ discrimination by correctness."""
        print("\n[D.1] Generating Fidelity+ Correctness Discrimination...")

        data = self.df[self.df["is_correct"].notna()].copy()
        if data.empty:
            print("  ! Skipping D.1: no correctness-labelled data found.")
            return

        fig = px.bar(
            data,
            x="method",
            y="fidelity_plus_mean",
            color="correctness_label",
            facet_col="dataset",
            barmode="group",
            title=(
                "<b>D.1: Fidelity+ Correctness Discrimination</b><br>"
                "<sub>GNNs show larger fidelity gaps between correct and incorrect predictions</sub>"
            ),
            labels={
                "fidelity_plus_mean": "Fidelity+ (mean)",
                "method": "Method",
                "correctness_label": "Prediction",
            },
            color_discrete_map=self.colors_correct,
            template="plotly_white",
            category_orders=self._method_category_orders(),
        )
        fig.update_traces(marker=dict(line=dict(width=0)))

        #fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            height=600,
            width=1400,
            font=dict(size=11),
            hovermode="closest",
            showlegend=True,
            legend=dict(title="Prediction Correctness", x=1.02, y=1),
        )
        fig.update_xaxes(tickangle=0)

        self._save_figure(fig, "D1_fidelity_plus_correctness_discrimination.html")

    def plot_d2_error_detection_signal_strength(self) -> None:
        """Comparison of error detection signal strengths across methods."""
        print("[D.2] Generating Error Detection Signal Strength...")

        records: list[dict[str, float | str]] = []
        for _, row in self.effects_df.iterrows():
            method = row["method"]
            dataset = row["dataset"]
            records.extend(
                [
                    {
                        "Method": self.method_labels[method],
                        "Signal": "Fidelity+ Effect",
                        "Value": row["fid_plus_effect"],
                        "Dataset": dataset,
                    },
                    {
                        "Method": self.method_labels[method],
                        "Signal": "Fidelity- Effect",
                        "Value": row["fid_minus_effect"],
                        "Dataset": dataset,
                    },
                    {
                        "Method": self.method_labels[method],
                        "Signal": "Failure Rate Effect",
                        "Value": row["fail_effect"],
                        "Dataset": dataset,
                    },
                ]
            )

        signal_df = pd.DataFrame(records)
        fig = px.bar(
            signal_df,
            x="Signal",
            y="Value",
            color="Method",
            facet_col="Dataset",
            barmode="group",
            title=(
                "<b>D.2: Error Detection Signal Strength Comparison</b><br>"
                "<sub>GNNs exhibit larger effects across all error diagnostics</sub>"
            ),
            labels={"Value": "Effect Magnitude"},
            color_discrete_map={
                self.method_labels[k]: v for k, v in self.colors_method.items()
            },
            template="plotly_white",
            category_orders=self._method_category_orders(),
        )

        fig.update_layout(
            height=600,
            width=1400,
            font=dict(size=10),
            hovermode="x unified",
            showlegend=True,
        )
        fig.update_xaxes(tickangle=45)

        self._save_figure(fig, "D2_error_detection_signal_strength.html")

    def plot_d3_confidence_coupling_as_feature(self) -> None:
        """Scatter plot: confidence coupling vs. correctness effect."""
        print("[D.3] Generating Confidence Coupling as Feature...")

        method_effects = (
            self.effects_df.groupby("method")
            .agg({"fid_plus_effect": "mean", "corr_conf_plus": "mean"})
            .reset_index()
        )
        method_effects["Method_Label"] = method_effects["method"].map(self.method_labels)

        fig = px.scatter(
            method_effects,
            x="corr_conf_plus",
            y="fid_plus_effect",
            color="method",
            size="fid_plus_effect",
            hover_name="Method_Label",
            hover_data={
                "corr_conf_plus": ":.4f",
                "fid_plus_effect": ":.4f",
                "method": False,
            },
            title=(
                "<b>D.3: Confidence Coupling Enables Error Detection</b><br>"
                "<sub>Higher coupling strength aligns with stronger correctness discrimination</sub>"
            ),
            labels={
                "corr_conf_plus": "ρ(Confidence, Fidelity+) · Coupling Strength",
                "fid_plus_effect": "Correctness Effect (Error Detection Power)",
            },
            color_discrete_map=self.colors_method,
            template="plotly_white",
            category_orders=self._method_category_orders(),
        )

        fig.add_hline(y=0.08, line_dash="dash", line_color="gray", opacity=0.3)
        fig.add_vline(x=0.1, line_dash="dash", line_color="gray", opacity=0.3)

        fig.add_annotation(
            x=0.7,
            y=0.19,
            text="<b>GNNs</b><br>High Coupling<br>Strong Error Detection",
            showarrow=False,
            bgcolor="rgba(52, 152, 219, 0.1)",
            bordercolor="#3498db",
            borderwidth=2,
        )
        fig.add_annotation(
            x=0.01,
            y=0.075,
            text="<b>TokenSHAP</b><br>No Coupling<br>Weak Error Detection",
            showarrow=False,
            bgcolor="rgba(230, 126, 34, 0.1)",
            bordercolor="#e67e22",
            borderwidth=2,
        )

        fig.update_layout(
            height=700,
            width=1000,
            font=dict(size=11),
            hovermode="closest",
            showlegend=False,
        )
        fig.update_xaxes(range=[-0.05, 0.85])
        fig.update_yaxes(range=[0.05, 0.22])

        self._save_figure(fig, "D3_confidence_coupling_as_feature.html")

    # ------------------------------------------------------------------ #
    # Tier-2 visualizations
    # ------------------------------------------------------------------ #

    def plot_d4_fidelity_minus_deletion_discrimination(self) -> None:
        """Deletion-based fidelity discrimination."""
        print("[D.4] Generating Fidelity- Deletion-Based Discrimination...")

        data = self.df[self.df["is_correct"].notna()].copy()
        if data.empty:
            print("  ! Skipping D.4: no correctness-labelled data found.")
            return

        fig = px.bar(
            data,
            x="method",
            y="fidelity_minus_mean",
            color="correctness_label",
            facet_col="dataset",
            barmode="group",
            title=(
                "<b>D.4: Deletion-Based Fidelity Discrimination (Fidelity-)</b><br>"
                "<sub>GNNs produce larger deletion-based effects for incorrect predictions</sub>"
            ),
            labels={
                "fidelity_minus_mean": "Fidelity- (mean)",
                "method": "Method",
                "correctness_label": "Prediction",
            },
            color_discrete_map=self.colors_correct,
            template="plotly_white",
            category_orders=self._method_category_orders(),
        )

        #fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            height=600,
            width=1400,
            font=dict(size=11),
            hovermode="closest",
        )

        self._save_figure(fig, "D4_fidelity_minus_deletion_discrimination.html")

    def plot_d5_failure_rate_quality_distribution(self) -> None:
        """Stacked bar chart of explanation quality tiers by method and correctness."""
        print("[D.5] Generating Failure Rate Quality Distribution...")

        def categorize_fidelity(fid: float) -> str:
            if fid > 0.1:
                return "Good (Fid+ > 0.1)"
            if fid > 0:
                return "Fair (0 < Fid+ ≤ 0.1)"
            if fid > -0.1:
                return "Poor (-0.1 ≤ Fid+ ≤ 0)"
            return "Anti-faithful (Fid+ < -0.1)"

        data = self.df[self.df["is_correct"].notna()].copy()
        if data.empty:
            print("  ! Skipping D.5: no correctness-labelled data found.")
            return

        data["fidelity_quality"] = data["fidelity_plus_mean"].apply(categorize_fidelity)
        data["Method_Correctness"] = data["method"].map(self.method_labels) + " · " + data[
            "correctness_label"
        ]

        quality_counts = (
            data.groupby(["Method_Correctness", "fidelity_quality"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            quality_counts.groupby("Method_Correctness")["count"]
            .sum()
            .reset_index(name="total")
        )
        quality_counts = quality_counts.merge(totals, on="Method_Correctness")
        quality_counts["percentage"] = (quality_counts["count"] / quality_counts["total"]) * 100

        color_map = {
            "Good (Fid+ > 0.1)": "#2ecc71",
            "Fair (0 < Fid+ ≤ 0.1)": "#f1c40f",
            "Poor (-0.1 ≤ Fid+ ≤ 0)": "#e67e22",
            "Anti-faithful (Fid+ < -0.1)": "#e74c3c",
        }

        fig = px.bar(
            quality_counts,
            x="Method_Correctness",
            y="percentage",
            color="fidelity_quality",
            title=(
                "<b>D.5: Explanation Quality Distribution by Prediction Correctness</b><br>"
                "<sub>GNN failure modes act as diagnostic error signals</sub>"
            ),
            labels={"percentage": "Percentage of Explanations (%)"},
            color_discrete_map=color_map,
            template="plotly_white",
            category_orders={
                "fidelity_quality": [
                    "Good (Fid+ > 0.1)",
                    "Fair (0 < Fid+ ≤ 0.1)",
                    "Poor (-0.1 ≤ Fid+ ≤ 0)",
                    "Anti-faithful (Fid+ < -0.1)",
                ]
            },
        )

        fig.update_layout(
            height=600,
            width=1400,
            font=dict(size=10),
            hovermode="x",
            xaxis=dict(tickangle=45),
            barmode="stack",
        )

        self._save_figure(fig, "D5_failure_rate_quality_distribution.html")

    def plot_d6_two_explanation_paradigms(self) -> None:
        """Quadrant analysis contrasting explanation paradigms."""
        print("[D.6] Generating Two Explanation Paradigms Quadrant...")

        method_stats = (
            self.effects_df.groupby("method")
            .agg({"fid_plus_correct": "mean", "fid_plus_effect": "mean"})
            .reset_index()
        )
        method_stats["Method_Label"] = method_stats["method"].map(self.method_labels)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=method_stats["fid_plus_correct"],
                y=method_stats["fid_plus_effect"],
                mode="markers+text",
                marker=dict(
                    size=20,
                    color=[self.colors_method[m] for m in method_stats["method"]],
                    line=dict(width=2, color="white"),
                ),
                text=method_stats["Method_Label"],
                textposition="top center",
                textfont=dict(size=12, color="black", family="Arial Black"),
                hovertemplate="<b>%{text}</b><br>Fidelity+: %{x:.4f}<br>Effect: %{y:.4f}<extra></extra>",
                showlegend=False,
            )
        )

        fig.add_hline(y=0.08, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0.1, line_dash="dash", line_color="gray", opacity=0.5)

        fig.add_annotation(
            x=0.025,
            y=0.18,
            text="<b>Error-Sensitive<br>Low-Quality</b>",
            showarrow=False,
            bgcolor="rgba(52, 152, 219, 0.1)",
            bordercolor="#3498db",
            borderwidth=2,
            font=dict(size=11, color="#3498db"),
        )
        fig.add_annotation(
            x=0.22,
            y=0.05,
            text="<b>High-Quality<br>Not Diagnostic</b>",
            showarrow=False,
            bgcolor="rgba(230, 126, 34, 0.1)",
            bordercolor="#e67e22",
            borderwidth=2,
            font=dict(size=11, color="#e67e22"),
        )

        fig.update_layout(
            title=(
                "<b>D.6: Two Explanation Paradigms - Trade-off Analysis</b><br>"
                "<sub>Methods optimize for complementary objectives</sub>"
            ),
            xaxis_title="Absolute Fidelity+ (Faithfulness Quality)",
            yaxis_title="Correctness Effect (Error Detection Power)",
            height=700,
            width=1000,
            template="plotly_white",
            font=dict(size=11),
            hovermode="closest",
        )

        self._save_figure(fig, "D6_two_explanation_paradigms.html")

    # ------------------------------------------------------------------ #
    # Pipeline orchestration
    # ------------------------------------------------------------------ #

    def generate_all_tier1_visualizations(self) -> None:
        """Generate all Tier-1 (critical) visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING TIER-1 CRITICAL VISUALIZATIONS")
        print("=" * 80)

        self.plot_d1_fidelity_plus_correctness_discrimination()
        self.plot_d2_error_detection_signal_strength()
        self.plot_d3_confidence_coupling_as_feature()

    def generate_all_tier2_visualizations(self) -> None:
        """Generate all Tier-2 (complementary) visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING TIER-2 COMPLEMENTARY VISUALIZATIONS")
        print("=" * 80)

        self.plot_d4_fidelity_minus_deletion_discrimination()
        self.plot_d5_failure_rate_quality_distribution()
        self.plot_d6_two_explanation_paradigms()

    def generate_all_visualizations(self) -> None:
        """Generate all visualizations."""
        self.generate_all_tier1_visualizations()
        self.generate_all_tier2_visualizations()

        print("\n" + "=" * 80)
        print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")

        for rel_path in sorted(p.name for p in self.output_dir.glob("*.html")):
            print(f"  - {rel_path}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate correctness discernability visualizations from fidelity summary."
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY),
        help="Path to fidelity_summary.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store generated HTML plots (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CORRECTNESS DISCERNABILITY ANALYSIS - VISUALIZATION PIPELINE")
    print("=" * 80)

    viz_pipeline = CorrectnessDiscernabilityVisualizations(
        csv_path=Path(args.summary),
        output_dir=Path(args.output),
    )
    viz_pipeline.generate_all_visualizations()


if __name__ == "__main__":
    main()
