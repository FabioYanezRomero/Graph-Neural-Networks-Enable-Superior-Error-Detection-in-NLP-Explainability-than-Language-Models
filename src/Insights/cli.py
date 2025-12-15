from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import networkx as nx

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

from .metrics import (
    feature_overlap_ratio,
    kendall_rank_correlation,
    pairwise_agreement,
    rank_biased_overlap,
    spearman_rank_correlation,
    summarize_record,
    summarize_records,
    top_nodes,
)
from .providers import GraphArtifactProvider
from .llm_providers import LLMExplanationProvider
from .readers import (
    discover_llm_records,
    discover_graphsvx_runs,
    discover_subgraphx_runs,
    iter_graphsvx_run_records,
    load_llm_records,
    load_graphsvx_records,
    load_subgraphx_records,
    load_subgraphx_run_records,
)
from .reporting import (
    ensure_parent,
    export_agreement_csv,
    export_agreement_json,
    export_minimal_size_histogram,
    export_summaries_csv,
    export_summaries_json,
)
from .records import ExplanationRecord

DEFAULT_GRAPH_TYPES = ("skipgrams", "window", "constituency")


class SummaryShardWriter:
    def __init__(self, base_path: Path, shard_size: int):
        if shard_size <= 0:
            raise ValueError("Shard size must be a positive integer.")
        self.base_path = base_path
        self.shard_size = shard_size
        self._batch: List[dict] = []
        self._shard_index = 0
        self._total = 0
        self._shard_paths: List[str] = []
        ensure_parent(self.base_path)

    def add(self, summary: dict) -> None:
        self._batch.append(summary)
        if len(self._batch) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._batch:
            return
        self._shard_index += 1
        shard_name = f"{self.base_path.stem}.part{self._shard_index:04d}{self.base_path.suffix}"
        shard_path = self.base_path.with_name(shard_name)
        shard_path.write_text(json.dumps(self._batch, indent=2), encoding="utf-8")
        self._total += len(self._batch)
        self._shard_paths.append(shard_path.name)
        self._batch = []

    def finalize(self) -> dict:
        self.flush()
        manifest = {
            "total_records": self._total,
            "shard_size": self.shard_size,
            "shards": self._shard_paths,
        }
        self.base_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def _infer_graph_type(path: Path) -> str | None:
    for part in path.parts:
        lower = part.lower()
        for candidate in DEFAULT_GRAPH_TYPES:
            if candidate in lower:
                return candidate
    return None


def _infer_run_id(path: Path) -> str | None:
    for parent in path.parents:
        if parent.name and parent.name[0].isdigit():
            return parent.name
    return None


def iter_records(args: argparse.Namespace) -> Iterator[ExplanationRecord]:
    coalition_base = Path(args.coalition_base) if args.coalition_base else None

    enable_subgraphx_predictions = not getattr(args, "disable_subgraphx_predictions", False)

    for path_str in args.graphsvx_json:
        path = Path(path_str)
        inferred_graph_type = _infer_graph_type(path)
        inferred_run = _infer_run_id(path)
        dataset = args.graphsvx_dataset
        graph_type = args.graphsvx_graph_type or inferred_graph_type
        run_id = args.graphsvx_run_id or inferred_run

        loaded = load_graphsvx_records(
            path,
            dataset=dataset,
            graph_type=graph_type,
            run_id=run_id,
            coalition_base=coalition_base,
        )
        for record in loaded:
            yield record

    run_dirs: List[Path] = [Path(path_str) for path_str in getattr(args, "graphsvx_run_dirs", [])]
    if getattr(args, "graphsvx_root", None):
        run_dirs.extend(discover_graphsvx_runs(Path(args.graphsvx_root)))

    if run_dirs:
        grouped: Dict[Tuple[Path, str], List[Path]] = defaultdict(list)
        for candidate in run_dirs:
            name = candidate.name
            if "_shard" in name:
                base = name.split("_shard", 1)[0]
            else:
                base = name
            grouped[(candidate.parent, base)].append(candidate)
        normalized_run_dirs: List[Path] = []
        for paths in grouped.values():
            shard_paths = [p for p in paths if "_shard" in p.name]
            if shard_paths:
                normalized_run_dirs.extend(sorted(shard_paths))
            else:
                normalized_run_dirs.extend(sorted(paths))
        run_dirs = normalized_run_dirs

    seen: set[Path] = set()
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        for record in iter_graphsvx_run_records(
            resolved,
            dataset=args.graphsvx_dataset,
            graph_type=args.graphsvx_graph_type,
            run_id=args.graphsvx_run_id,
        ):
            yield record

    for path_str in args.subgraphx_json:
        path = Path(path_str)
        inferred_graph_type = _infer_graph_type(path)
        inferred_run = _infer_run_id(path)
        dataset = args.subgraphx_dataset
        graph_type = args.subgraphx_graph_type or inferred_graph_type
        run_id = args.subgraphx_run_id or inferred_run

        loaded = load_subgraphx_records(
            path,
            dataset=dataset,
            graph_type=graph_type,
            run_id=run_id,
        )
        for record in loaded:
            yield record

    subgraphx_run_dirs: List[Path] = [Path(p) for p in getattr(args, "subgraphx_run_dirs", [])]
    if getattr(args, "subgraphx_root", None):
        subgraphx_run_dirs.extend(discover_subgraphx_runs(Path(args.subgraphx_root)))

    seen_sx: set[Path] = set()
    for run_dir in subgraphx_run_dirs:
        resolved = run_dir.resolve()
        if not resolved.exists():
            continue

        candidate_dirs: List[Path]
        results_path = resolved / "results.pkl"
        if results_path.exists():
            candidate_dirs = [resolved]
        else:
            candidate_dirs = [
                candidate.resolve()
                for candidate in discover_subgraphx_runs(resolved)
            ]

        for candidate in candidate_dirs:
            candidate_path = candidate.resolve()
            if candidate_path in seen_sx or not candidate_path.exists():
                continue
            seen_sx.add(candidate_path)
            loaded = load_subgraphx_run_records(
                candidate_path,
                dataset=args.subgraphx_dataset,
                graph_type=args.subgraphx_graph_type,
                run_id=args.subgraphx_run_id,
                enable_predictions=enable_subgraphx_predictions,
            )
            for record in loaded:
                yield record

    llm_paths: List[Path] = [Path(p) for p in getattr(args, "llm_records", [])]
    if getattr(args, "llm_root", None):
        llm_paths.extend(discover_llm_records(Path(args.llm_root)))

    seen_llm: set[Path] = set()
    for raw_path in llm_paths:
        resolved = raw_path.resolve()
        if resolved in seen_llm or not resolved.exists():
            continue
        seen_llm.add(resolved)
        loaded = load_llm_records(
            resolved,
            dataset=args.llm_dataset,
            graph_type=args.llm_graph_type,
            run_id=args.llm_run_id,
        )
        for record in loaded:
            yield record


def collect_records(args: argparse.Namespace) -> List[ExplanationRecord]:
    return list(iter_records(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise explainability artefacts from GraphSVX/SubgraphX/TokenSHAP.",
    )
    parser.add_argument(
        "--graphsvx-json",
        nargs="*",
        default=[],
        help="Paths to GraphSVX result JSON files.",
    )
    parser.add_argument(
        "--graphsvx-run-dirs",
        nargs="*",
        default=[],
        help="GraphSVX run directories under outputs/gnn_models (each containing results.pkl).",
    )
    parser.add_argument(
        "--graphsvx-root",
        help="Root directory to discover GraphSVX runs automatically (e.g., outputs/gnn_models).",
    )
    parser.add_argument(
        "--graph-root",
        default="outputs/graphs",
        help="Root directory containing NetworkX graph artefacts.",
    )
    parser.add_argument(
        "--pyg-root",
        default="outputs/pyg_graphs",
        help="Root directory containing PyG graph tensors.",
    )
    parser.add_argument(
        "--subgraphx-json",
        nargs="*",
        default=[],
        help="Paths to SubgraphX result JSON files.",
    )
    parser.add_argument(
        "--subgraphx-run-dirs",
        nargs="*",
        default=[],
        help="SubgraphX run directories under outputs/gnn_models (each containing results.pkl).",
    )
    parser.add_argument(
        "--subgraphx-root",
        help="Root directory to discover SubgraphX runs automatically.",
    )
    parser.add_argument("--graphsvx-dataset", help="Override dataset name for GraphSVX records.")
    parser.add_argument("--graphsvx-graph-type", help="Override graph type for GraphSVX records.")
    parser.add_argument("--graphsvx-run-id", help="Override run id for GraphSVX records.")
    parser.add_argument("--subgraphx-dataset", help="Override dataset name for SubgraphX records.")
    parser.add_argument("--subgraphx-graph-type", help="Override graph type for SubgraphX records.")
    parser.add_argument("--subgraphx-run-id", help="Override run id for SubgraphX records.")
    parser.add_argument(
        "--llm-records",
        nargs="*",
        default=[],
        help="Paths to TokenSHAP LLM record files (JSON or pickle).",
    )
    parser.add_argument(
        "--llm-root",
        help="Root directory to discover TokenSHAP record files automatically.",
    )
    parser.add_argument("--llm-dataset", help="Override dataset name for LLM records.")
    parser.add_argument("--llm-graph-type", help="Override graph type for LLM records.")
    parser.add_argument("--llm-run-id", help="Override run id for LLM records.")
    parser.add_argument(
        "--coalition-base",
        help="Base directory to resolve GraphSVX coalition CSV paths against.",
    )
    parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=0.9,
        help="Threshold (fraction of origin confidence) for minimal sufficiency.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top nodes to store per record.",
    )
    parser.add_argument("--output-json", help="Path to write detailed record summaries (JSON).")
    parser.add_argument("--output-csv", help="Path to write summaries as CSV (requires pandas).")
    parser.add_argument("--minimal-csv", help="Path to write minimal coalition size histogram CSV.")
    parser.add_argument(
        "--summary-shard-size",
        type=int,
        help="When set, write summary JSON in shards of this size and store a manifest at --output-json.",
    )
    parser.add_argument(
        "--agreement-json",
        help="Path to write pairwise agreement metrics across explainers (JSON).",
    )
    parser.add_argument(
        "--agreement-csv",
        help="Path to write pairwise agreement metrics across explainers (CSV, requires pandas).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary statistics without writing files.",
    )
    parser.add_argument(
        "--disable-structure",
        action="store_true",
        help="Skip loading graph artefacts (structural metrics and token mapping remain null).",
    )
    parser.add_argument(
        "--disable-centrality",
        action="store_true",
        help="Disable centrality alignment even when structural graphs are available.",
    )
    parser.add_argument(
        "--disable-subgraphx-predictions",
        action="store_true",
        help="Skip recomputing predictions for SubgraphX shards (avoids large tensor loads).",
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=0.9,
        help="Rank-biased overlap persistence parameter (0 < p < 1).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators during processing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    shard_size = getattr(args, "summary_shard_size", None)
    use_streaming = shard_size is not None and shard_size > 0
    if shard_size is not None and shard_size <= 0:
        parser.error("--summary-shard-size must be a positive integer.")
    if use_streaming and not args.output_json:
        parser.error("--summary-shard-size requires --output-json.")
    if use_streaming and (args.output_csv or args.minimal_csv or args.agreement_csv):
        parser.error("--summary-shard-size is incompatible with CSV outputs; remove the shard option or CSV flags.")

    record_iterator: Iterable[ExplanationRecord]
    if use_streaming:
        iterator = iter(iter_records(args))
        try:
            first_record = next(iterator)
        except StopIteration:
            parser.error("No explanations found. Provide at least one JSON path.")
        record_iterator = itertools.chain([first_record], iterator)
    else:
        records = collect_records(args)
        if not records:
            parser.error("No explanations found. Provide at least one JSON path.")
        record_iterator = records

    if use_streaming:
        sample_count = 0
    else:
        sample_count = len(records)  # type: ignore[arg-type]

    if not use_streaming and sample_count == 0:
        parser.error("No explanations found. Provide at least one JSON path.")

    graph_provider: GraphArtifactProvider | None = None
    centrality_funcs: Dict[str, Callable[[nx.Graph], Dict[int, float]]] | None = None
    if not args.disable_structure:
        try:
            graph_provider = GraphArtifactProvider(args.graph_root, args.pyg_root, strict=False)
            if not args.disable_centrality:
                centrality_funcs = {
                    "degree": lambda g: nx.degree_centrality(g.to_undirected()),
                    "betweenness": lambda g: nx.betweenness_centrality(g.to_undirected(), normalized=True),
                    "closeness": lambda g: nx.closeness_centrality(g.to_undirected()),
                }
        except Exception:  # pragma: no cover - graceful fallback
            graph_provider = None
            centrality_funcs = None

    llm_provider = LLMExplanationProvider()

    def _graph_lookup(record: ExplanationRecord):
        if graph_provider is None:
            return None
        try:
            return graph_provider(record)
        except Exception:
            return None

    def combined_provider(record: ExplanationRecord):
        graph_info = _graph_lookup(record)
        if graph_info is not None:
            return graph_info
        return llm_provider(record)

    provider_callable = combined_provider if (graph_provider or llm_provider) else (lambda _: None)
    centrality_callable = centrality_funcs if graph_provider and centrality_funcs else None
    minimal_counter: Counter[int] = Counter()
    minimal_sum = 0.0
    minimal_count = 0
    summaries: List[dict] | None = None
    agreement_entries: List[dict] | None = None

    if not use_streaming:
        summaries = summarize_records(
            record_iterator,
            sufficiency_threshold=args.sufficiency_threshold,
            top_k=args.top_k,
            graph_provider=provider_callable,
            centrality_funcs=centrality_callable,
        )
        sample_count = len(summaries)
        minimal_sizes = [
            summary["minimal_coalition_size"]
            for summary in summaries
            if summary["minimal_coalition_size"] is not None
        ]
        if minimal_sizes:
            minimal_sum = float(sum(minimal_sizes))
            minimal_count = len(minimal_sizes)
            minimal_counter.update(minimal_sizes)
        agreement_entries = pairwise_agreement(
            record_iterator,  # type: ignore[arg-type]
            top_k=args.top_k,
            rbo_p=args.rbo_p,
        )
    else:
        assert use_streaming and shard_size
        summary_writer = SummaryShardWriter(Path(args.output_json), shard_size)
        agreement_groups: Dict[Tuple[object, object, object, object], List[Dict[str, object]]] = defaultdict(list)
        sample_count = 0
        iterator_for_loop: Iterable[ExplanationRecord] = record_iterator
        progress_bar = None
        if not args.no_progress and tqdm is not None:
            progress_bar = tqdm(record_iterator, desc="Summarising", unit="record")
            iterator_for_loop = progress_bar
        for record in iterator_for_loop:
            summary = summarize_record(
                record,
                sufficiency_threshold=args.sufficiency_threshold,
                top_k=args.top_k,
                graph_provider=provider_callable,
                centrality_funcs=centrality_callable,
            )
            summary_writer.add(summary)
            sample_count += 1
            minimal_size = summary.get("minimal_coalition_size")
            if minimal_size is not None:
                minimal_counter[minimal_size] += 1
                minimal_sum += float(minimal_size)
                minimal_count += 1
            if args.agreement_json:
                key = (record.dataset, record.graph_index, record.label, record.is_correct)
                agreement_groups[key].append(
                    {
                        "dataset": record.dataset,
                        "graph_index": record.graph_index,
                        "label": record.label,
                        "is_correct": record.is_correct,
                        "method": record.method,
                        "graph_type": record.graph_type,
                        "run_id": record.run_id,
                        "top_nodes": list(top_nodes(record, k=args.top_k)),
                    }
                )
        summary_writer.finalize()
        if args.agreement_json:
            agreement_entries = []
            for recs in agreement_groups.values():
                if len(recs) < 2:
                    continue
                for left, right in itertools.combinations(recs, 2):
                    top_left = left["top_nodes"]
                    top_right = right["top_nodes"]
                    if not isinstance(top_left, Sequence) or not isinstance(top_right, Sequence):
                        continue
                    entry = {
                        "dataset": left["dataset"],
                        "graph_index": left["graph_index"],
                        "label": left["label"],
                        "is_correct": left["is_correct"],
                        "method_a": left["method"],
                        "method_b": right["method"],
                        "graph_type_a": left["graph_type"],
                        "graph_type_b": right["graph_type"],
                        "run_id_a": left["run_id"],
                        "run_id_b": right["run_id"],
                        "top_k": args.top_k,
                        "overlap_count": len(set(top_left) & set(top_right)),
                        "rbo": rank_biased_overlap(top_left, top_right, p=args.rbo_p, k=args.top_k),
                        "spearman": spearman_rank_correlation(top_left, top_right, k=args.top_k),
                        "kendall": kendall_rank_correlation(top_left, top_right, k=args.top_k),
                        "feature_overlap_ratio": feature_overlap_ratio(top_left, top_right, k=args.top_k),
                        "stability_jaccard": stability_jaccard_simple(top_left, top_right, args.top_k),
                    }
                    agreement_entries.append(entry)
        else:
            agreement_entries = []
        if progress_bar is not None:
            progress_bar.close()

    if args.dry_run or (not args.output_json and not args.output_csv and not use_streaming):
        print(f"Loaded {sample_count} explanations.", file=sys.stdout)
        if minimal_count:
            avg_size = minimal_sum / minimal_count
            print(
                f"Average minimal coalition size (threshold={args.sufficiency_threshold}): {avg_size:.2f}",
                file=sys.stdout,
            )
        else:
            print("No minimal coalitions satisfied the threshold.", file=sys.stdout)

    if not use_streaming and args.output_json:
        assert summaries is not None
        export_summaries_json(
            record_iterator,  # type: ignore[arg-type]
            Path(args.output_json),
            sufficiency_threshold=args.sufficiency_threshold,
            top_k=args.top_k,
            summaries=summaries,
        )

    if not use_streaming and args.output_csv:
        assert summaries is not None
        try:
            export_summaries_csv(summaries, Path(args.output_csv))
        except RuntimeError as exc:
            parser.error(str(exc))

    if args.minimal_csv:
        export_minimal_size_histogram(
            record_iterator,  # type: ignore[arg-type]
            Path(args.minimal_csv),
            threshold=args.sufficiency_threshold,
        )

    if args.agreement_json and agreement_entries is not None:
        export_agreement_json(agreement_entries, Path(args.agreement_json))

    if args.agreement_csv and not use_streaming:
        assert agreement_entries is not None
        try:
            export_agreement_csv(agreement_entries, Path(args.agreement_csv))
        except RuntimeError as exc:
            parser.error(str(exc))

    if args.dry_run and agreement_entries:
        print(f"Computed {len(agreement_entries)} pairwise agreement entries.", file=sys.stdout)

    return 0


def stability_jaccard_simple(top_left: Sequence[int], top_right: Sequence[int], k: int) -> Optional[float]:
    limited_left = list(top_left)[:k]
    limited_right = list(top_right)[:k]
    if not limited_left or not limited_right:
        return None
    intersection = len(set(limited_left) & set(limited_right))
    union = len(set(limited_left) | set(limited_right))
    return intersection / union if union else None


if __name__ == "__main__":
    raise SystemExit(main())
