"""
Simple programmatic pipeline runner for GraphText.

Usage:
    python -m src.graphtext.cli run --config configs/example_pipeline.json
"""
from __future__ import annotations
import json
from pathlib import Path
from .cli import cmd_finetune, cmd_build, cmd_embed, cmd_convert, cmd_train, cmd_explain
import argparse


def run_config(path: str):
    cfg_path = Path(path)
    with cfg_path.open("r") as f:
        cfg = json.load(f)
    if "finetune" in cfg:
        cmd_finetune(argparse.Namespace(**cfg["finetune"]))
    if "build" in cfg:
        cmd_build(argparse.Namespace(**cfg["build"]))
    if "embed" in cfg:
        cmd_embed(argparse.Namespace(**cfg["embed"]))
    if "convert" in cfg:
        cmd_convert(argparse.Namespace(**cfg["convert"]))
    if "train" in cfg:
        cmd_train(argparse.Namespace(**cfg["train"]))
    if "explain" in cfg:
        cmd_explain(argparse.Namespace(**cfg["explain"]))

