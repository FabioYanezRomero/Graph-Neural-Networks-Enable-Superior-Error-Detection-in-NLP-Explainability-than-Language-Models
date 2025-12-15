#!/usr/bin/env python3
"""
Render a combined dashboard per dataset that stacks the trade-off scatter
plots (top) and the quadrant distribution bars (bottom) in a single HTML
document.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

import plotly.io as pio

from Analytics.consistency.plots import (
    DEFAULT_ROOT,
    PLOTS_ROOT,
    build_tradeoff_scatter_figure,
    dataset_label,
    load_instance_records,
)
from Analytics.consistency.final_plot import quadrant_distribution_figure

COMBINED_FILENAME = "vizB_scatter_quadrant_combined_{dataset}.html"


def build_combined(dataset: str, root: Path) -> Path | None:
    dataset_dir = (root / PLOTS_ROOT / dataset).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instance_records(root, dataset)
    scatter_fig = build_tradeoff_scatter_figure(instances, dataset)
    if scatter_fig is None:
        print(f"! {dataset}: skipping combined plot (no scatter data)")
        return None

    try:
        quadrant_fig = quadrant_distribution_figure(dataset)
    except ValueError as exc:
        print(f"! {dataset}: skipping combined plot ({exc})")
        return None

    scatter_html = pio.to_html(
        scatter_fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id=f"scatter-{dataset}",
        config=dict(
            displaylogo=False,
            modeBarButtonsToRemove=["toImage"],
        ),
    )
    quadrant_html = pio.to_html(
        quadrant_fig,
        include_plotlyjs=False,
        full_html=False,
        div_id=f"quadrant-{dataset}",
        config=dict(
            displaylogo=False,
            modeBarButtonsToRemove=["toImage"],
        ),
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{dataset_label(dataset)} · Trade-off & Quadrant Dashboard</title>
  <style>
    body {{
      font-family: "Inter", "Segoe UI", sans-serif;
      margin: 24px;
      background: #fafafa;
      color: #1f2933;
    }}
    .section {{
      margin-bottom: 48px;
      padding: 24px;
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    .section h2 {{
      margin-top: 0;
      margin-bottom: 16px;
      font-size: 1.35rem;
      font-weight: 600;
    }}
  </style>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      const button = document.getElementById("download-combined-{dataset}");
      async function downloadCombined() {{
        const figureIds = ["scatter-{dataset}", "quadrant-{dataset}"];
        const images = await Promise.all(
          figureIds.map(id => Plotly.toImage(document.getElementById(id), {{format: "png", scale: 2}}))
        );
        const loaded = await Promise.all(images.map(src => new Promise(resolve => {{
          const img = new Image();
          img.onload = () => resolve(img);
          img.src = src;
        }})));
        const gap = 40;
        const targetWidth = Math.max(...loaded.map(img => img.width));
        const scaledHeights = loaded.map(img => (img.height * targetWidth) / img.width);
        const totalHeight = scaledHeights.reduce((acc, h) => acc + h, 0) + (loaded.length - 1) * gap;
        const canvas = document.createElement("canvas");
        canvas.width = targetWidth;
        canvas.height = totalHeight;
        const ctx = canvas.getContext("2d");
        let offsetY = 0;
        loaded.forEach((img, index) => {{
          const drawHeight = (img.height * targetWidth) / img.width;
          ctx.drawImage(img, 0, offsetY, targetWidth, drawHeight);
          offsetY += drawHeight + gap;
        }});
        const link = document.createElement("a");
        link.href = canvas.toDataURL("image/png");
        link.download = "{dataset}_tradeoff_quadrant.png";
        link.click();
      }}
      button.addEventListener("click", downloadCombined);
    }});
  </script>
</head>
<body>
  <div class="section">
    <h2>{dataset_label(dataset)} · Sufficiency–Necessity Trade-off Scatter</h2>
    {scatter_html}
  </div>
  <div class="section" style="padding-top:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <h2 style="margin:0;">{dataset_label(dataset)} · Quadrant Distribution</h2>
      <button id="download-combined-{dataset}" style="padding:10px 16px;border:none;border-radius:6px;background:#2563eb;color:#fff;font-size:0.95rem;cursor:pointer;">
        Download Combined PNG
      </button>
    </div>
    {quadrant_html}
  </div>
</body>
</html>
"""

    output_path = dataset_dir / COMBINED_FILENAME.format(dataset=dataset)
    output_path.write_text(html_doc, encoding="utf-8")
    print(f"✓ {dataset}: combined plot -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined scatter + quadrant dashboards.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing consistency outputs (default: outputs/analytics/consistency).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset slug(s) to render (default: all available in root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.dataset:
        datasets = args.dataset
    else:
        datasets = sorted(
            p.name
            for p in (root / PLOTS_ROOT).glob("*")
            if p.is_dir()
        )
    if not datasets:
        raise SystemExit("No datasets found to render.")

    for dataset in datasets:
        build_combined(dataset, root)


if __name__ == "__main__":  # pragma: no cover
    main()
