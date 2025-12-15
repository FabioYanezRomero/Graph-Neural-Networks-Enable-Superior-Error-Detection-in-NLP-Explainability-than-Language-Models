from __future__ import annotations

import ast
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:  # tqdm is optional; silently disable progress if unavailable.
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

from src.explain.gnn.config import ExplainerRequest
from src.explain.gnn.model_loader import load_graph_split, load_gnn_model

from .records import Coalition, ExplanationRecord, RelatedPrediction


def _compute_is_correct(label: Optional[object], predicted: Optional[object]) -> Optional[bool]:
    if label is None or predicted is None:
        return None
    try:
        return int(label) == int(predicted)
    except (TypeError, ValueError):
        return label == predicted


def _resolve_path(raw: str, base_dir: Optional[Path]) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return (base_dir / path).resolve()


def _read_coalitions(csv_path: Path) -> List[Coalition]:
    coalitions: List[Coalition] = []
    if not csv_path.exists():
        return coalitions

    with csv_path.open("r", encoding="utf-8") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            combination_raw = row.get("combination_id")
            size_raw = row.get("coalition_size")

            try:
                nodes = ast.literal_eval(row.get("coalition_nodes", "[]"))
            except (SyntaxError, ValueError):
                nodes = []
            try:
                mask = ast.literal_eval(row.get("coalition_binary", "[]"))
            except (SyntaxError, ValueError):
                mask = None

            coalition = Coalition.from_iterable(
                nodes=nodes,
                confidence=float(row.get("confidence", 0.0)),
                combination_id=int(combination_raw) if combination_raw not in (None, "", "None") else None,
                binary_mask=mask,
                size=int(size_raw) if size_raw not in (None, "", "None") else None,
            )
            coalitions.append(coalition)
    coalitions.sort(key=lambda c: (c.size, -c.confidence, c.combination_id or 0))
    return coalitions


def _graphsvx_item_to_record(
    item: Mapping[str, Any],
    *,
    dataset: str,
    graph_type: Optional[str],
    run_id: Optional[str],
    coalition_base: Optional[Path],
    method: str = "graphsvx",
    extra_extras: Optional[Mapping[str, Any]] = None,
) -> ExplanationRecord:
    coalitions_path = item.get("coalitions_path")
    coalitions: List[Coalition] = []
    if coalitions_path:
        resolved = _resolve_path(coalitions_path, coalition_base)
        coalitions = _read_coalitions(resolved)

    prediction = item.get("prediction") or {}
    label = item.get("label")
    predicted_class = prediction.get("class")
    is_correct = item.get("is_correct")
    if is_correct is None:
        is_correct = _compute_is_correct(label, predicted_class)

    extras = {
        key: value
        for key, value in item.items()
        if key
        not in {
            "graph_index",
            "label",
            "num_nodes",
            "num_edges",
            "prediction",
            "node_importance",
            "top_nodes",
            "related_prediction",
            "hyperparams",
            "coalitions_path",
        }
    }
    if extra_extras:
        extras.update(extra_extras)

    return ExplanationRecord(
        dataset=dataset,
        graph_type=graph_type,
        method=method,
        run_id=run_id,
        graph_index=item.get("graph_index"),
        label=label,
        prediction_class=predicted_class,
        prediction_confidence=prediction.get("confidence"),
        is_correct=is_correct,
        num_nodes=item.get("num_nodes"),
        num_edges=item.get("num_edges"),
        node_importance=item.get("node_importance"),
        top_nodes=tuple(item.get("top_nodes", [])),
        related_prediction=RelatedPrediction.from_mapping(item.get("related_prediction")),
        hyperparams=item.get("hyperparams") or {},
        coalitions=coalitions,
        extras=extras,
    )


def load_graphsvx_records(
    json_path: Path,
    *,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    run_id: Optional[str] = None,
    coalition_base: Optional[Path] = None,
) -> List[ExplanationRecord]:
    """
    Load a collection of ExplanationRecord instances from a GraphSVX JSON dump.
    """
    json_path = Path(json_path)
    dataset = dataset or json_path.stem.replace("graphsvx_results_", "")

    with json_path.open("r", encoding="utf-8") as handler:
        raw_items = json.load(handler)

    records: List[ExplanationRecord] = []
    for item in raw_items:
        record = _graphsvx_item_to_record(
            item,
            dataset=dataset,
            graph_type=graph_type,
            run_id=run_id,
            coalition_base=coalition_base or json_path.parent,
        )
        records.append(record)
    return records


def _infer_context_from_graphsvx_run(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to infer (dataset, graph_type, run_id) from a standard outputs/gnn_models layout.
    """
    run_dir = run_dir.resolve()
    parts = run_dir.parts
    dataset: Optional[str] = None
    graph_type: Optional[str] = None
    run_id: Optional[str] = None
    try:
        idx = parts.index("gnn_models")
    except ValueError:
        return dataset, graph_type, run_id

    if len(parts) > idx + 4:
        backbone = parts[idx + 1]
        dataset_part = parts[idx + 2]
        graph_type = parts[idx + 3]
        run_id = parts[idx + 4]
        dataset = f"{backbone}/{dataset_part}"
    return dataset, graph_type, run_id


def iter_graphsvx_run_records(
    run_dir: Path,
    *,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    run_id: Optional[str] = None,
    prefer_summary: bool = True,
) -> Iterator[ExplanationRecord]:
    """
    Yield ExplanationRecord instances from a GraphSVX run directory produced under outputs/gnn_models.

    The directory is expected to contain ``summary.json`` and optionally ``results.pkl`` for legacy support.
    """
    run_dir = Path(run_dir)
    results_path = run_dir / "results.pkl"
    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handler:
            summary = json.load(handler)
        dataset = dataset or summary.get("dataset")
        graph_type = graph_type or summary.get("graph_type")
        run_id = run_id or summary.get("run_id")

    inferred_dataset, inferred_graph_type, inferred_run_id = _infer_context_from_graphsvx_run(run_dir)
    dataset = dataset or inferred_dataset or "unknown"
    graph_type = graph_type or inferred_graph_type
    run_id = run_id or inferred_run_id

    method = summary.get("method", "graphsvx")
    split = summary.get("split")
    shard_index = summary.get("shard_index")
    num_shards = summary.get("num_shards")
    backbone = summary.get("backbone")

    graphs = summary.get("graphs") if summary_path.exists() else None
    if prefer_summary and isinstance(graphs, list):
        for item in graphs:
            extra_extras = {
                "split": split,
                "backbone": backbone,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "hyperparam_source": item.get("hyperparam_source"),
            }
            record = _graphsvx_item_to_record(
                item,
                dataset=dataset,
                graph_type=graph_type,
                run_id=run_id,
                coalition_base=run_dir,
                method=method,
                extra_extras=extra_extras,
            )
            yield record
        return

    if not results_path.exists():
        raise FileNotFoundError(f"GraphSVX run directory missing results.pkl: {run_dir}")

    # Import locally to avoid heavy module import unless needed.
    from src.explain.gnn.graphsvx.main import GraphSVXResult  # type: ignore
    import sys
    import types

    # Historical pickles may reference GraphSVXResult under __main__.
    main_module = sys.modules.get("__main__")
    if main_module is None:
        main_module = types.ModuleType("__main__")
        sys.modules["__main__"] = main_module
    setattr(main_module, "GraphSVXResult", GraphSVXResult)

    with results_path.open("rb") as handler:
        raw_results: List[GraphSVXResult] = pickle.load(handler)

    iterator = raw_results
    if tqdm is not None:
        iterator = tqdm(
            raw_results,
            desc=f"GraphSVX[{run_dir.name}]",
            unit="record",
            leave=False,
        )

    for item in iterator:
        explanation: Dict[str, Any] = item.explanation or {}
        prediction = explanation.get("original_prediction") or {}
        coalition_path_raw = explanation.get("combinations_path") or explanation.get("coalitions_path")
        coalitions: List[Coalition] = []
        if coalition_path_raw:
            coalitions = _read_coalitions(Path(coalition_path_raw))

        related = item.related_prediction or explanation.get("related_prediction")

        extras: Dict[str, Any] = {
            "split": split,
            "backbone": summary.get("backbone"),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "max_combinations": explanation.get("max_combinations"),
            "sampled": explanation.get("sampled"),
            "hyperparam_source": item.source,
        }

        importance = explanation.get("node_importance")
        if torch is not None and isinstance(importance, torch.Tensor):
            importance = importance.tolist()

        label = item.label
        predicted_class = prediction.get("class")
        is_correct = getattr(item, "is_correct", None)
        if is_correct is None:
            is_correct = _compute_is_correct(label, predicted_class)
        record = ExplanationRecord(
            dataset=dataset,
            graph_type=graph_type,
            method=method,
            run_id=run_id,
            graph_index=item.graph_index,
            label=label,
            prediction_class=predicted_class,
            prediction_confidence=prediction.get("confidence"),
            is_correct=is_correct,
            num_nodes=explanation.get("num_nodes"),
            num_edges=explanation.get("num_edges"),
            node_importance=importance,
            top_nodes=tuple(explanation.get("top_nodes", ())),
            related_prediction=RelatedPrediction.from_mapping(related),
            hyperparams=item.hyperparams or {},
            coalitions=coalitions,
            extras=extras,
        )
        yield record


def load_graphsvx_run_records(
    run_dir: Path,
    *,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    run_id: Optional[str] = None,
    prefer_summary: bool = True,
) -> List[ExplanationRecord]:
    """
    Load ExplanationRecord instances from a GraphSVX run directory produced under outputs/gnn_models.
    """
    return list(
        iter_graphsvx_run_records(
            run_dir,
            dataset=dataset,
            graph_type=graph_type,
            run_id=run_id,
            prefer_summary=prefer_summary,
        )
    )


def discover_graphsvx_runs(root: Path) -> List[Path]:
    """
    Discover GraphSVX run directories beneath a root (typically outputs/gnn_models).
    """
    root = Path(root)
    candidates: Set[Path] = set()
    for results_path in root.rglob("results.pkl"):
        if results_path.parent.parent.name == "graphsvx":
            candidates.add(results_path.parent)
    return sorted(candidates)


def load_subgraphx_records(
    json_path: Path,
    *,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    run_id: Optional[str] = None,
) -> List[ExplanationRecord]:
    """
    Load ExplanationRecord instances from a SubgraphX summary JSON.
    """
    json_path = Path(json_path)
    dataset = dataset or json_path.stem.replace("subgraphx_results_", "")

    with json_path.open("r", encoding="utf-8") as handler:
        raw_items = json.load(handler)

    records: List[ExplanationRecord] = []
    for item in raw_items:
        label = item.get("label")
        prediction_class = item.get("prediction_class")
        is_correct = item.get("is_correct")
        if is_correct is None:
            is_correct = _compute_is_correct(label, prediction_class)
        record = ExplanationRecord(
            dataset=dataset,
            graph_type=graph_type,
            method="subgraphx",
            run_id=run_id,
            graph_index=item.get("graph_index"),
            label=label,
            prediction_class=prediction_class,
            prediction_confidence=item.get("prediction_confidence"),
            is_correct=is_correct,
            num_nodes=item.get("num_nodes"),
            num_edges=item.get("num_edges"),
            related_prediction=RelatedPrediction.from_mapping(item.get("related_prediction")),
            hyperparams=item.get("hyperparams") or {},
            coalitions=[],
            extras={
                key: value
                for key, value in item.items()
                if key
                not in {
                    "graph_index",
                    "label",
                    "num_nodes",
                    "num_edges",
                    "related_prediction",
                    "hyperparams",
                }
            },
        )
        records.append(record)
    return records


def _ensure_subgraphx_stub() -> None:
    import sys
    import types

    module = sys.modules.get("src.explain.gnn.subgraphx.main")
    if module is None:
        module = types.ModuleType("src.explain.gnn.subgraphx.main")
        sys.modules["src.explain.gnn.subgraphx.main"] = module

    if not hasattr(module, "SubgraphXResult"):
        class SubgraphXResult:  # type: ignore[too-many-instance-attributes]
            def __init__(
                self,
                graph_index: int,
                label: Optional[int],
                explanation,
                related_prediction: Dict[str, float],
                num_nodes: int,
                num_edges: int,
                hyperparams: Dict[str, float],
            ) -> None:
                self.graph_index = graph_index
                self.label = label
                self.explanation = self._sanitize_explanation(explanation)
                self.related_prediction = related_prediction
                self.num_nodes = num_nodes
                self.num_edges = num_edges
                self.hyperparams = hyperparams

            @staticmethod
            def _sanitize_explanation(raw) -> List[Dict[str, object]]:
                cleaned: List[Dict[str, object]] = []
                if not isinstance(raw, list):
                    return cleaned
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    coalition = entry.get("coalition") or entry.get("node_idx") or []
                    try:
                        coalition = [int(v) for v in coalition]
                    except Exception:
                        coalition = []
                    info: Dict[str, object] = {
                        "coalition": coalition,
                        "P": entry.get("P"),
                        "W": entry.get("W"),
                        "N": entry.get("N"),
                    }
                    cleaned.append(info)
                return cleaned

        module.SubgraphXResult = SubgraphXResult  # type: ignore[attr-defined]
        main_module = sys.modules.get("__main__")
        if main_module is None:
            main_module = types.ModuleType("__main__")
            sys.modules["__main__"] = main_module
        setattr(main_module, "SubgraphXResult", SubgraphXResult)


def _infer_context_from_subgraphx_run(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    run_dir = run_dir.resolve()
    parts = run_dir.parts
    dataset: Optional[str] = None
    graph_type: Optional[str] = None
    run_id: Optional[str] = None
    try:
        idx = parts.index("gnn_models")
    except ValueError:
        return dataset, graph_type, run_id

    if len(parts) > idx + 4:
        backbone = parts[idx + 1]
        dataset_part = parts[idx + 2]
        graph_type = parts[idx + 3]
        run_id = parts[idx + 4]
        dataset = f"{backbone}/{dataset_part}"
    return dataset, graph_type, run_id


def load_subgraphx_run_records(
    run_dir: Path,
    *,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    run_id: Optional[str] = None,
    enable_predictions: bool = True,
) -> List[ExplanationRecord]:
    run_dir = Path(run_dir)
    results_path = run_dir / "results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"SubgraphX run directory missing results.pkl: {run_dir}")

    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handler:
            summary = json.load(handler)
        dataset = dataset or summary.get("dataset")
        graph_type = graph_type or summary.get("graph_type")
        run_id = run_id or summary.get("run_id")

    inferred_dataset, inferred_graph_type, inferred_run_id = _infer_context_from_subgraphx_run(run_dir)
    dataset = dataset or inferred_dataset or "unknown"
    graph_type = graph_type or inferred_graph_type
    run_id = run_id or inferred_run_id

    _ensure_subgraphx_stub()

    split = summary.get("split")
    shard_index_raw = summary.get("shard_index")
    num_shards_raw = summary.get("num_shards")
    backbone = summary.get("backbone")

    def _coerce_int(value: object, default: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    num_shards_safe = _coerce_int(num_shards_raw, 1)
    if num_shards_safe < 1:
        num_shards_safe = 1
    shard_index_safe = _coerce_int(shard_index_raw, 0)
    if shard_index_safe < 0:
        shard_index_safe = 0
    if shard_index_safe >= num_shards_safe:
        shard_index_safe = num_shards_safe - 1

    with results_path.open("rb") as handler:
        raw_results = pickle.load(handler)

    predictions: Dict[int, Tuple[int, float]] = {}
    if (
        enable_predictions
        and torch is not None
        and backbone
        and dataset
        and graph_type
        and split
    ):
        dataset_obj = None
        loader = None
        model = None
        try:
            request = ExplainerRequest(
                dataset=dataset,
                graph_type=graph_type,
                backbone=backbone,
                split=split,
                method="subgraphx",
                num_shards=num_shards_safe,
                shard_index=shard_index_safe,
            )
            dataset_obj, loader = load_graph_split(request, batch_size=1, shuffle=False)
            model, _, _ = load_gnn_model(request, dataset=dataset_obj)
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(loader):
                    data = batch.to(device)
                    logits = model(data=data)
                    probs = torch.softmax(logits, dim=1)
                    pred_cls = int(torch.argmax(probs, dim=1)[0].item())
                    confidence = float(probs[0, pred_cls].item())
                    predictions[idx] = (pred_cls, confidence)
        except Exception:
            predictions = {}
        finally:
            if loader is not None:
                loader = None
            if dataset_obj is not None:
                dataset_obj = None
            if model is not None:
                model = None

    records: List[ExplanationRecord] = []
    for item in raw_results:
        explanation_entries = item.explanation or []
        coalitions: List[Coalition] = []
        for entry in explanation_entries:
            nodes = entry.get("coalition") or entry.get("node_idx") or []
            nodes = [int(n) for n in nodes]
            confidence = float(entry.get("P") or entry.get("W") or 0.0)
            size_override = entry.get("N")
            metadata = {
                key: entry.get(key)
                for key in ("W", "N", "P")
                if key in entry
            }
            coalitions.append(
                Coalition.from_iterable(
                    nodes=nodes,
                    confidence=confidence,
                    size=int(size_override) if size_override is not None else None,
                    **metadata,
                )
            )
            # Release heavy tensors/graphs promptly
            entry.pop("data", None)
            entry.pop("ori_graph", None)

        top_nodes: Tuple[int, ...] = tuple(coalitions[0].nodes) if coalitions else tuple()

        extras: Dict[str, Any] = {
            "split": split,
            "backbone": backbone,
            "shard_index": shard_index_safe if shard_index_raw is not None else None,
            "num_shards": num_shards_safe if num_shards_raw is not None else None,
        }

        pred_entry = predictions.get(item.graph_index)
        prediction_class = pred_entry[0] if pred_entry else None
        prediction_confidence = pred_entry[1] if pred_entry else None
        is_correct = getattr(item, "is_correct", None)
        if is_correct is None:
            is_correct = _compute_is_correct(item.label, prediction_class)

        related = RelatedPrediction.from_mapping(item.related_prediction)

        node_importance: Optional[List[float]] = None
        if related.maskout_progression_drop and top_nodes:
            drops = [float(value) for value in related.maskout_progression_drop]
        elif related.sufficiency_progression_drop and top_nodes:
            drops = [float(value) for value in related.sufficiency_progression_drop]
        else:
            drops = None

        if drops is not None and top_nodes:
            incremental: List[float] = []
            previous = 0.0
            for value in drops:
                incremental.append(value - previous)
                previous = value
            length = item.num_nodes or (max(top_nodes) + 1 if top_nodes else len(incremental))
            node_importance = [0.0] * int(length)
            for node_idx, score in zip(top_nodes, incremental):
                if isinstance(node_idx, int) and 0 <= node_idx < len(node_importance):
                    node_importance[int(node_idx)] = float(score)

        record = ExplanationRecord(
            dataset=dataset,
            graph_type=graph_type,
            method="subgraphx",
            run_id=run_id,
            graph_index=item.graph_index,
            label=item.label,
            prediction_class=prediction_class,
            prediction_confidence=prediction_confidence,
            is_correct=is_correct,
            num_nodes=item.num_nodes,
            num_edges=item.num_edges,
            node_importance=node_importance,
            top_nodes=top_nodes,
            related_prediction=related,
            hyperparams=item.hyperparams or {},
            coalitions=coalitions,
            extras=extras,
        )
        records.append(record)
        item.explanation = None  # type: ignore[attr-defined]
        if hasattr(item, "related_prediction"):
            item.related_prediction = None  # type: ignore[attr-defined]
    return records


def discover_subgraphx_runs(root: Path) -> List[Path]:
    root = Path(root)
    candidates: Set[Path] = set()
    for results_path in root.rglob("results.pkl"):
        if results_path.parent.parent.name == "subgraphx":
            candidates.add(results_path.parent)
    return sorted(candidates)


def _load_llm_json(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handler:
        payload = json.load(handler)
    if not isinstance(payload, list):
        raise ValueError(f"TokenSHAP records JSON must contain a list of records: {path}")
    return payload


def _load_llm_pickle(path: Path) -> List[Mapping[str, Any]]:
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, list):
        if payload and isinstance(payload[0], ExplanationRecord):
            return [record.to_dict() for record in payload]  # type: ignore[attr-defined]
        if payload and isinstance(payload[0], Mapping):
            return list(payload)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported TokenSHAP pickle payload: {path}")


def load_llm_records(
    path: Path,
    *,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    run_id: Optional[str] = None,
) -> List[ExplanationRecord]:
    """
    Load ExplanationRecords produced by TokenSHAP LLM explainers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LLM records file not found: {path}")

    if path.suffix.lower() == ".json":
        raw_items = _load_llm_json(path)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        raw_items = _load_llm_pickle(path)
    else:
        raise ValueError(f"Unsupported TokenSHAP record format: {path.suffix}")

    records: List[ExplanationRecord] = []
    for item in raw_items:
        if isinstance(item, ExplanationRecord):
            record = item
        else:
            record = ExplanationRecord.from_mapping(item)
        if dataset:
            record.dataset = dataset
        if graph_type:
            record.graph_type = graph_type
        if run_id:
            record.run_id = run_id
        if record.graph_type is None:
            record.graph_type = "tokens"
        if record.is_correct is None:
            record.is_correct = _compute_is_correct(record.label, record.prediction_class)
        records.append(record)
    return records


def discover_llm_records(root: Path) -> List[Path]:
    """
    Discover TokenSHAP record files (JSON or pickle) beneath a root directory.
    """
    root = Path(root)
    candidates: Set[Path] = set()
    for pattern in ("*_records.json", "*_records.pkl", "*_records.pickle"):
        for candidate in root.rglob(pattern):
            candidates.add(candidate)
    return sorted(candidates)
