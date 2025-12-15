from __future__ import annotations
from dataclasses import dataclass
import sys
import subprocess


@dataclass
class BatchConvertConfig:
    label_source: str = "llm"  # or "original"
    use_pred: bool = True
    hf_dataset_name: str = "stanfordnlp/sst2"
    graph_type: str = "syntactic"
    in_base: str | None = None
    out_base: str | None = None
    llm_dir: str | None = None


class GraphBatchConverter:
    """Wrapper for batch conversion of NetworkX to PyG with labels."""

    def run(self, cfg: BatchConvertConfig) -> None:
        args = [
            sys.executable,
            "-m",
            "src.convert.batch_convert",
            "--label-source", cfg.label_source,
            "--hf-dataset-name", cfg.hf_dataset_name,
            "--graph-type", cfg.graph_type,
        ]
        if cfg.use_pred:
            args.append("--use-pred")
        if cfg.in_base:
            args.extend(["--in-base", cfg.in_base])
        if cfg.out_base:
            args.extend(["--out-base", cfg.out_base])
        if cfg.llm_dir:
            args.extend(["--llm-dir", cfg.llm_dir])

        # Note: legacy script accepts only dataset + global graph_type; it reads its own dirs.
        # If needed, adjust environment variables or directory structure upstream.
        subprocess.run(args, check=True)
