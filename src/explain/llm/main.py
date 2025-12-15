"""CLI entry point for LLM explainability using TokenSHAP."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import (
    DEFAULT_FINETUNED_ROOT,
    DEFAULT_INSIGHTS_ROOT,
    LLMExplainerRequest,
    build_default_profiles,
)
from .model_loader import load_dataset_split, load_finetuned_model
from .token_shap_runner import collect_token_shap_hyperparams, token_shap_explain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _list_datasets(args: argparse.Namespace) -> int:
    """List available dataset profiles."""
    profiles = build_default_profiles(finetuned_root=Path(args.finetuned_root))
    
    print("Available LLM dataset profiles:")
    print("-" * 80)
    for key, profile in profiles.items():
        print(f"\n  Key: {key}")
        print(f"  Dataset: {profile.dataset_name}")
        print(f"  Split: {profile.split}")
        print(f"  Num Labels: {profile.num_labels}")
        print(f"  Checkpoint: {profile.checkpoint_dir}")
        print(f"  Backbone: {profile.derive_backbone()}")
        print(f"  Graph Type: {profile.graph_type}")
        print(f"  Max Tokens: {profile.max_tokens}")
        print(f"  Max Length: {profile.max_length}")
    print()
    return 0


def _collect_hyperparams(args: argparse.Namespace) -> int:
    """Collect suggested hyperparameters for each sample."""
    profiles = build_default_profiles(finetuned_root=Path(args.finetuned_root))
    
    profile_key = args.dataset
    if profile_key not in profiles:
        LOGGER.error("Unknown dataset profile '%s'. Use --list-datasets to see available profiles.", profile_key)
        return 1
    
    profile = profiles[profile_key]

    if args.num_shards < 1:
        LOGGER.error("num-shards must be at least 1 (received %d).", args.num_shards)
        return 1
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        LOGGER.error(
            "shard-index must be in the range [0, %d) (received %d).",
            args.num_shards,
            args.shard_index,
        )
        return 1
    
    request = LLMExplainerRequest(
        profile=profile,
        device=args.device,
        sampling_override=args.sampling_override,
        max_samples=args.max_samples,
        sufficiency_threshold=args.sufficiency_threshold,
        top_k_nodes=args.top_k,
        insights_root=Path(args.output_dir) if args.output_dir else DEFAULT_INSIGHTS_ROOT,
        output_basename=args.output_basename,
        store_raw=not args.no_raw,
        fair_comparison=getattr(args, "fair", False),
        target_forward_passes=getattr(args, "target_forward_passes", 2000),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    
    LOGGER.info("Loading model and dataset for profile '%s'...", profile_key)
    model_bundle = load_finetuned_model(profile, device=request.resolve_device())
    dataset = load_dataset_split(profile)
    
    output_path = Path(args.hyperparams_output) if args.hyperparams_output else None
    
    LOGGER.info("Collecting hyperparameters...")
    hparam_path, per_sample = collect_token_shap_hyperparams(
        request,
        model_bundle,
        dataset,
        output_path=output_path,
        max_samples=args.max_samples,
        progress=not args.no_progress,
    )
    
    print(f"\nCollected hyperparameters for {len(per_sample)} samples.")
    print(f"Saved to: {hparam_path}")
    return 0


def _explain(args: argparse.Namespace) -> int:
    """Run TokenSHAP explainability on the specified dataset."""
    profiles = build_default_profiles(finetuned_root=Path(args.finetuned_root))
    
    profile_key = args.dataset
    if profile_key not in profiles:
        LOGGER.error("Unknown dataset profile '%s'. Use --list-datasets to see available profiles.", profile_key)
        return 1
    
    profile = profiles[profile_key]

    if args.num_shards < 1:
        LOGGER.error("num-shards must be at least 1 (received %d).", args.num_shards)
        return 1
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        LOGGER.error(
            "shard-index must be in the range [0, %d) (received %d).",
            args.num_shards,
            args.shard_index,
        )
        return 1
    
    request = LLMExplainerRequest(
        profile=profile,
        device=args.device,
        sampling_override=args.sampling_override,
        max_samples=args.max_samples,
        sufficiency_threshold=args.sufficiency_threshold,
        top_k_nodes=args.top_k,
        insights_root=Path(args.output_dir) if args.output_dir else DEFAULT_INSIGHTS_ROOT,
        output_basename=args.output_basename,
        store_raw=not args.no_raw,
        fair_comparison=getattr(args, "fair", False),
        target_forward_passes=getattr(args, "target_forward_passes", 2000),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    
    LOGGER.info("Loading model and dataset for profile '%s'...", profile_key)
    model_bundle = load_finetuned_model(profile, device=request.resolve_device())
    dataset = load_dataset_split(profile)
    
    if request.fair_comparison:
        LOGGER.info("Fair comparison mode enabled with target_forward_passes=%d", request.target_forward_passes)
    
    LOGGER.info("Running TokenSHAP explainability...")
    (
        records,
        summaries,
        summary_json,
        summary_csv,
        raw_json,
        raw_pickle,
    ) = token_shap_explain(
        request,
        model_bundle,
        dataset,
        progress=not args.no_progress,
        use_advisor=not args.no_advisor,
    )
    
    print(f"\nExplainability complete!")
    print(f"  Processed {len(records)} samples")
    print(f"  Summary JSON: {summary_json}")
    print(f"  Summary CSV: {summary_csv}")
    if raw_json:
        print(f"  Raw JSON: {raw_json}")
    if raw_pickle:
        print(f"  Raw pickle: {raw_pickle}")
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM explainability using TokenSHAP on finetuned transformers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List datasets command
    list_parser = subparsers.add_parser(
        "list-datasets",
        help="List available dataset profiles",
    )
    list_parser.add_argument(
        "--finetuned-root",
        default=str(DEFAULT_FINETUNED_ROOT),
        help="Root directory for finetuned LLM checkpoints",
    )
    
    # Collect hyperparams command
    collect_parser = subparsers.add_parser(
        "collect-hyperparams",
        help="Collect suggested hyperparameters for each sample",
    )
    collect_parser.add_argument(
        "dataset",
        help="Dataset profile key (e.g., 'setfit/ag_news', 'stanfordnlp/sst2')",
    )
    collect_parser.add_argument(
        "--finetuned-root",
        default=str(DEFAULT_FINETUNED_ROOT),
        help="Root directory for finetuned LLM checkpoints",
    )
    collect_parser.add_argument(
        "--hyperparams-output",
        help="Path to save hyperparameters JSON",
    )
    collect_parser.add_argument(
        "--device",
        help="Device to use (cuda/cpu)",
    )
    collect_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process",
    )
    collect_parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shards to split the dataset",
    )
    collect_parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard (0-based)",
    )
    collect_parser.add_argument(
        "--sampling-override",
        type=float,
        help="Override sampling ratio for all samples",
    )
    collect_parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=0.9,
        help="Sufficiency threshold for minimal coalition",
    )
    collect_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top important tokens to extract",
    )
    collect_parser.add_argument(
        "--output-dir",
        help="Override output directory",
    )
    collect_parser.add_argument(
        "--output-basename",
        help="Override output file basename",
    )
    collect_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    collect_parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not store raw pickle files",
    )
    collect_parser.add_argument(
        "--fair",
        action="store_true",
        help="Use fair comparison mode with formula-based sampling ratios (default: 400 forward passes)",
    )
    collect_parser.add_argument(
        "--target-forward-passes",
        type=int,
        default=400,
        help="Target number of forward passes for fair comparison mode (default: 400)",
    )
    
    # Explain command
    explain_parser = subparsers.add_parser(
        "explain",
        help="Run TokenSHAP explainability on a dataset",
    )
    explain_parser.add_argument(
        "dataset",
        help="Dataset profile key (e.g., 'setfit/ag_news', 'stanfordnlp/sst2')",
    )
    explain_parser.add_argument(
        "--finetuned-root",
        default=str(DEFAULT_FINETUNED_ROOT),
        help="Root directory for finetuned LLM checkpoints",
    )
    explain_parser.add_argument(
        "--device",
        help="Device to use (cuda/cpu)",
    )
    explain_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process",
    )
    explain_parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shards to split the dataset",
    )
    explain_parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard (0-based)",
    )
    explain_parser.add_argument(
        "--sampling-override",
        type=float,
        help="Override sampling ratio for all samples",
    )
    explain_parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=0.9,
        help="Sufficiency threshold for minimal coalition",
    )
    explain_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top important tokens to extract",
    )
    explain_parser.add_argument(
        "--output-dir",
        help="Override output directory",
    )
    explain_parser.add_argument(
        "--output-basename",
        help="Override output file basename",
    )
    explain_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    explain_parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not store raw pickle files",
    )
    explain_parser.add_argument(
        "--no-advisor",
        action="store_true",
        help="Disable hyperparameter advisor (use fixed sampling ratios)",
    )
    explain_parser.add_argument(
        "--fair",
        action="store_true",
        help="Use fair comparison mode with formula-based sampling ratios (default: 400 forward passes)",
    )
    explain_parser.add_argument(
        "--target-forward-passes",
        type=int,
        default=400,
        help="Target number of forward passes for fair comparison mode (default: 400)",
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "list-datasets":
        return _list_datasets(args)
    elif args.command == "collect-hyperparams":
        return _collect_hyperparams(args)
    elif args.command == "explain":
        return _explain(args)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
