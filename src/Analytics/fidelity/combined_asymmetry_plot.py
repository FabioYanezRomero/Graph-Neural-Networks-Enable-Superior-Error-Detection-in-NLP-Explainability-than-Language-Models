#!/usr/bin/env python3
"""
Render a two-panel fidelity asymmetry dashboard (AG News vs. SST-2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.io as pio

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from Analytics.fidelity.final_plot import (
    DATASET_LABELS,
    INSTANCE_ROOT,
    OUTPUT_ROOT,
    asymmetry_figure,
)

DEFAULT_LEFT = "setfit_ag_news"
DEFAULT_RIGHT = "stanfordnlp_sst2"
OUTPUT_NAME = "fidelity_asymmetry_combined.html"


def build_combined(left: str, right: str, root: Path, output_dir: Path) -> Path:
    fig_left = asymmetry_figure(left, root, show_legend=False)
    fig_right = asymmetry_figure(right, root, show_legend=False)
    fig_left.update_layout(title=dict(text=""))
    fig_right.update_layout(title=dict(text=""))

    left_html = pio.to_html(
        fig_left,
        include_plotlyjs="cdn",
        full_html=False,
        div_id=f"left-{left}",
        config=dict(displaylogo=False, modeBarButtonsToRemove=["toImage"]),
    )
    right_html = pio.to_html(
        fig_right,
        include_plotlyjs=False,
        full_html=False,
        div_id=f"right-{right}",
        config=dict(displaylogo=False, modeBarButtonsToRemove=["toImage"]),
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Fidelity Asymmetry · {DATASET_LABELS.get(left, left)} vs {DATASET_LABELS.get(right, right)}</title>
  <style>
    body {{
      margin: 24px;
      background: #f8fafc;
      font-family: "Inter", "Segoe UI", sans-serif;
      color: #0f172a;
    }}
    .grid {{
      display: flex;
      gap: 24px;
      align-items: flex-start;
    }}
    .panel {{
      flex: 1;
      min-width: 0;
      background: #ffffff;
      padding: 16px 16px 32px;
      border-radius: 14px;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.12);
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 1.25rem;
    }}
    .plotly-graph-div {{
      width: 100% !important;
      height: 620px !important;
    }}
    .toolbar {{
      display: flex;
      justify-content: flex-end;
      margin-bottom: 18px;
    }}
    .toolbar button {{
      padding: 10px 16px;
      border: none;
      border-radius: 8px;
      background: #2563eb;
      color: #fff;
      font-size: 0.95rem;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }}
  </style>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      const button = document.getElementById("download-fidelity-combined");
      async function downloadCombined() {{
        const ids = ["left-{left}", "right-{right}"];
        const images = await Promise.all(
          ids.map(id =>
            Plotly.toImage(document.getElementById(id), {{ format: "png", scale: 2 }})
          )
        );
        const loaded = await Promise.all(
          images.map(
            src =>
              new Promise(resolve => {{
                const img = new Image();
                img.onload = () => resolve(img);
                img.src = src;
              }})
          )
        );
        const gap = 60;
        const totalWidth = loaded.reduce((acc, img) => acc + img.width, 0) + gap;
        const height = Math.max(...loaded.map(img => img.height));
        const canvas = document.createElement("canvas");
        canvas.width = totalWidth;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        let offset = 0;
        loaded.forEach(img => {{
          ctx.drawImage(img, offset, 0);
          offset += img.width + gap;
        }});
        const link = document.createElement("a");
        link.href = canvas.toDataURL("image/png");
        link.download = "fidelity_asymmetry_combined.png";
        link.click();
      }}
      button.addEventListener("click", downloadCombined);
    }});
  </script>
</head>
<body>
  <div class="toolbar">
    <button id="download-fidelity-combined">Download Combined PNG</button>
  </div>
  <div class="grid">
    <div class="panel">
      <h2>{DATASET_LABELS.get(left, left.replace('_', ' ').title())}</h2>
      {left_html}
    </div>
    <div class="panel">
      <h2>{DATASET_LABELS.get(right, right.replace('_', ' ').title())}</h2>
      {right_html}
    </div>
  </div>
</body>
</html>
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    output_path.write_text(html_doc, encoding="utf-8")
    print(f"✓ Combined asymmetry plot -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render combined fidelity asymmetry dashboard.")
    parser.add_argument("--root", type=Path, default=INSTANCE_ROOT, help="Instance root (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT, help="Output directory (default: %(default)s)")
    parser.add_argument("--left", type=str, default=DEFAULT_LEFT, help="Dataset slug for the left panel.")
    parser.add_argument("--right", type=str, default=DEFAULT_RIGHT, help="Dataset slug for the right panel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_combined(args.left, args.right, args.root, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
