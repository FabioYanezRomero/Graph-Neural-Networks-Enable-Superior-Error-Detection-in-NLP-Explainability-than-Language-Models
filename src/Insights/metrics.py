from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import itertools
import math

import networkx as nx

from .records import Coalition, ExplanationRecord

try:
    from .providers import GraphInfo  # type: ignore
    from .llm_providers import TokenInfo  # type: ignore
except ImportError:  # pragma: no cover
    GraphInfo = None  # type: ignore
    TokenInfo = None  # type: ignore


def _compute_auc(values: Mapping[int, float], *, max_size: Optional[int]) -> Optional[float]:
    if not values or max_size in (None, 0):
        return None
    sorted_items = sorted(values.items())
    total = 0.0
    last_size = 0
    for size, value in sorted_items:
        width = size - last_size
        total += width * value
        last_size = size
    return total / max_size if max_size else None


@dataclass(frozen=True)
class CurveResult:
    """Stores size-confidence curves (e.g., insertion/deletion)."""

    values: Dict[int, float]
    origin: Optional[float]
    normalized: bool
    auc: Optional[float]

    def as_series(self) -> List[Tuple[int, float]]:
        """Return sorted (size, value) pairs."""
        return sorted(self.values.items(), key=lambda item: item[0])


def minimal_sufficient_size(record: ExplanationRecord, *, threshold: float = 0.9) -> Optional[int]:
    coalition = record.minimal_coalition(threshold)
    return coalition.size if coalition else None


def minimal_sufficient_statistics(
    records: Iterable[ExplanationRecord],
    *,
    threshold: float = 0.9,
) -> List[Tuple[int, int]]:
    stats: Dict[int, int] = {}
    for record in records:
        size = minimal_sufficient_size(record, threshold=threshold)
        if size is None:
            continue
        stats[size] = stats.get(size, 0) + 1
    return sorted(stats.items())


def insertion_curve(record: ExplanationRecord, *, normalize: bool = True) -> CurveResult:
    if not record.coalitions:
        return CurveResult(values={}, origin=None, normalized=normalize, auc=None)

    origin = record.related_prediction.origin
    curve: Dict[int, float] = {}
    for coalition in record.coalitions:
        best = curve.get(coalition.size)
        value = coalition.confidence
        if normalize and origin:
            value = value / origin
        if best is None or value > best:
            curve[coalition.size] = value

    max_size = record.num_nodes or (max(curve) if curve else None)
    auc = _compute_auc(curve, max_size=max_size)
    return CurveResult(values=curve, origin=origin, normalized=normalize, auc=auc)


def deletion_curve(record: ExplanationRecord, *, normalize: bool = True) -> CurveResult:
    """
    Uses mask-out confidence (if available) as a proxy deletion baseline.
    """
    origin = record.related_prediction.origin
    maskout = record.related_prediction.maskout
    progression = record.related_prediction.maskout_progression_confidence
    curve: Dict[int, float] = {}

    def _normalise(value: float) -> float:
        if not normalize:
            return value
        if origin in (None, 0):
            return value
        return value / origin

    if normalize and origin is not None:
        curve[0] = 1.0
    else:
        curve[0] = origin if origin is not None else (maskout if maskout is not None else 0.0)

    max_size: Optional[int] = None

    if progression:
        for idx, conf in enumerate(progression, start=1):
            if conf is None:
                continue
            curve[idx] = _normalise(float(conf))
        if maskout is not None:
            final_idx = len(progression)
            curve[final_idx] = _normalise(float(maskout))
        max_size = max(curve) if len(curve) > 1 else len(progression)
    elif maskout is not None:
        removal_size = 0
        if record.top_nodes:
            removal_size = len(record.top_nodes)
        elif record.related_prediction.sparsity is not None and record.num_nodes:
            removal_size = max(
                1,
                int(round(record.related_prediction.sparsity * record.num_nodes)),
            )
        elif record.num_nodes:
            removal_size = record.num_nodes
        else:
            removal_size = 1

        curve[removal_size] = _normalise(float(maskout))
        max_size = max(removal_size, 1)
    else:
        return CurveResult(values={}, origin=origin, normalized=normalize, auc=None)

    if max_size in (None, 0):
        max_size = record.num_nodes or 1

    auc = _compute_auc(curve, max_size=max_size)
    return CurveResult(values=curve, origin=origin, normalized=normalize, auc=auc)


def _coerce_sequence(values: Optional[Sequence[float]]) -> List[float]:
    cleaned: List[float] = []
    if values is None:
        return cleaned
    for value in values:
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    return cleaned


def _monotonicity_score(values: Sequence[float], *, direction: str) -> Optional[float]:
    if not values or len(values) < 2:
        return None
    comparisons = 0
    satisfied = 0
    tolerance = 1e-9
    for previous, current in zip(values, values[1:]):
        comparisons += 1
        if direction == "non_decreasing":
            if current + tolerance >= previous:
                satisfied += 1
        elif direction == "non_increasing":
            if current - tolerance <= previous:
                satisfied += 1
        else:
            raise ValueError(f"Unsupported monotonicity direction: {direction}")
    if comparisons == 0:
        return None
    return satisfied / comparisons


def _monotonicity_metrics(record: ExplanationRecord) -> Dict[str, Optional[float]]:
    maskout_drop = _coerce_sequence(record.related_prediction.maskout_progression_drop)
    maskout_conf = _coerce_sequence(record.related_prediction.maskout_progression_confidence)
    suff_conf = _coerce_sequence(record.related_prediction.sufficiency_progression_confidence)
    suff_drop = _coerce_sequence(record.related_prediction.sufficiency_progression_drop)

    metrics = {
        "maskout_drop_monotonicity": _monotonicity_score(maskout_drop, direction="non_decreasing") if maskout_drop else None,
        "maskout_conf_monotonicity": _monotonicity_score(maskout_conf, direction="non_increasing") if maskout_conf else None,
        "sufficiency_conf_monotonicity": _monotonicity_score(suff_conf, direction="non_decreasing") if suff_conf else None,
        "sufficiency_drop_monotonicity": _monotonicity_score(suff_drop, direction="non_increasing") if suff_drop else None,
    }

    available = [value for value in metrics.values() if value is not None]
    metrics["faithfulness_monotonicity"] = (sum(available) / len(available)) if available else None
    return metrics


def robustness_score(record: ExplanationRecord) -> Optional[float]:
    origin = record.related_prediction.origin
    maskout = record.related_prediction.maskout
    if origin in (None, 0) or maskout is None:
        return None
    return 1.0 - (maskout / origin)


def _normalised_drop(baseline: Optional[float], value: Optional[float], *, normalise: bool = True) -> Optional[float]:
    if baseline is None or value is None:
        return None
    drop = baseline - value
    if not normalise or baseline == 0:
        return drop
    return drop / abs(baseline)


def fidelity_plus(record: ExplanationRecord, *, normalise: bool = True) -> Optional[float]:
    """Normalised drop when retaining only the important elements (sufficiency)."""

    return _normalised_drop(record.related_prediction.origin, record.related_prediction.masked, normalise=normalise)


def fidelity_minus(record: ExplanationRecord, *, normalise: bool = True) -> Optional[float]:
    """Normalised drop when masking out the important elements (necessity)."""

    return _normalised_drop(record.related_prediction.origin, record.related_prediction.maskout, normalise=normalise)


def faithfulness(record: ExplanationRecord, *, normalise: bool = True) -> Optional[float]:
    """Contrast between keeping and dropping important elements."""

    masked = record.related_prediction.masked
    maskout = record.related_prediction.maskout
    origin = record.related_prediction.origin
    if masked is None or maskout is None:
        return None
    score = masked - maskout
    if not normalise or origin in (None, 0):
        return score
    return score / abs(origin)


def deletion_auc(record: ExplanationRecord, *, normalise: bool = True) -> Optional[float]:
    """Convenience wrapper that returns the deletion AUC."""

    return deletion_curve(record, normalize=normalise).auc


def insertion_auc(record: ExplanationRecord, *, normalize: bool = True) -> Optional[float]:
    return insertion_curve(record, normalize=normalize).auc


def jaccard_overlap(nodes_a: Sequence[int], nodes_b: Sequence[int]) -> Optional[float]:
    if not nodes_a or not nodes_b:
        return None
    set_a = set(nodes_a)
    set_b = set(nodes_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else None


def top_nodes(record: ExplanationRecord, *, k: int) -> Sequence[int]:
    if record.top_nodes:
        return record.top_nodes[:k]
    if not record.node_importance:
        return ()
    ranked = sorted(
        enumerate(record.node_importance),
        key=lambda item: item[1],
        reverse=True,
    )
    return tuple(idx for idx, _ in ranked[:k])


def stability_jaccard(
    record_a: ExplanationRecord,
    record_b: ExplanationRecord,
    *,
    k: int = 10,
) -> Optional[float]:
    top_a = top_nodes(record_a, k=k)
    top_b = top_nodes(record_b, k=k)
    return jaccard_overlap(top_a, top_b)


def induced_subgraph_metrics(graph: nx.Graph, nodes: Sequence[int]) -> Dict[str, Optional[float]]:
    if not nodes:
        return {
            "induced_num_nodes": 0,
            "induced_num_edges": 0,
            "components": 0,
            "density": None,
            "boundary_edges": 0,
            "cut_ratio": None,
            "avg_shortest_path": None,
        }

    node_set = set(nodes)
    subgraph = graph.subgraph(node_set).copy()
    undirected = subgraph.to_undirected()
    full_nodes = graph.number_of_nodes()

    boundary_edges = sum(
        1
        for u, v in graph.edges()
        if (u in node_set) ^ (v in node_set)
    )

    denominator = len(node_set) * (full_nodes - len(node_set))

    if subgraph.is_directed():
        component_count = nx.number_weakly_connected_components(subgraph)
    else:
        component_count = nx.number_connected_components(subgraph)

    metrics = {
        "induced_num_nodes": subgraph.number_of_nodes(),
        "induced_num_edges": subgraph.number_of_edges(),
        "components": component_count,
        "density": nx.density(undirected),
        "boundary_edges": boundary_edges,
        "cut_ratio": boundary_edges / denominator if denominator else None,
        "avg_shortest_path": None,
    }

    try:
        if component_count == 1 and undirected.number_of_nodes() > 1:
            metrics["avg_shortest_path"] = nx.average_shortest_path_length(undirected)
    except (nx.NetworkXError, ZeroDivisionError):
        metrics["avg_shortest_path"] = None

    return metrics


def stability_average(records: Sequence[ExplanationRecord], *, k: int = 10) -> Optional[float]:
    """Average pairwise stability (Jaccard of top-k nodes) across records."""

    pairs = list(itertools.combinations(records, 2))
    if not pairs:
        return None
    scores = [stability_jaccard(a, b, k=k) for a, b in pairs]
    scores = [score for score in scores if score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def centrality_alignment(
    node_importance: Mapping[int, float],
    centrality_scores: Mapping[int, float],
) -> Optional[float]:
    import math

    common_nodes = set(node_importance) & set(centrality_scores)
    if len(common_nodes) < 2:
        return None

    importance_values = [node_importance[node] for node in common_nodes]
    centrality_values = [centrality_scores[node] for node in common_nodes]

    ranks_importance = _rank_values(importance_values)
    ranks_centrality = _rank_values(centrality_values)

    mean_rank_importance = sum(ranks_importance) / len(ranks_importance)
    mean_rank_centrality = sum(ranks_centrality) / len(ranks_centrality)

    numerator = sum(
        (r_i - mean_rank_importance) * (r_j - mean_rank_centrality)
        for r_i, r_j in zip(ranks_importance, ranks_centrality)
    )
    denominator = math.sqrt(
        sum((r - mean_rank_importance) ** 2 for r in ranks_importance)
        * sum((r - mean_rank_centrality) ** 2 for r in ranks_centrality)
    )
    return numerator / denominator if denominator else None


def _rank_values(values: Sequence[float]) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1])

    ranks = [0.0] * len(values)
    current_rank = 1
    while current_rank <= len(values):
        start = current_rank - 1
        end = start
        while end + 1 < len(values) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        avg_rank = (current_rank + end + 1) / 2
        for idx in range(start, end + 1):
            ranks[indexed[idx][0]] = avg_rank
        current_rank = end + 2
    return ranks


def _default_graph_provider(_: ExplanationRecord) -> Optional[nx.Graph]:
    return None


def summarize_record(
    record: ExplanationRecord,
    *,
    sufficiency_threshold: float = 0.9,
    top_k: int = 10,
    graph: Optional[nx.Graph] = None,
    graph_provider: Callable[[ExplanationRecord], Optional[nx.Graph]] = _default_graph_provider,
    centrality_funcs: Optional[Mapping[str, Callable[[nx.Graph], Mapping[int, float]]]] = None,
) -> Dict[str, Any]:
    """
    Produce a dictionary summarising key metrics for an explanation record.
    """

    node_text: Optional[Sequence[str]] = None
    graph_payload = graph or (graph_provider(record) if graph_provider else None)

    # Handle both GraphInfo (for GNN explanations) and TokenInfo (for LLM explanations)
    if GraphInfo is not None and isinstance(graph_payload, GraphInfo):
        graph_obj = graph_payload.graph
        node_text = tuple(graph_payload.node_text)
    elif TokenInfo is not None and isinstance(graph_payload, TokenInfo):
        # For LLM explanations, we have tokens but no graph structure
        graph_obj = None
        node_text = tuple(graph_payload.tokens)
    else:
        graph_obj = graph_payload  # type: ignore[assignment]
    minimal = record.minimal_coalition(sufficiency_threshold)
    insertion = insertion_curve(record)
    deletion = deletion_curve(record)

    origin_conf = record.related_prediction.origin
    masked_conf = record.related_prediction.masked
    maskout_conf = record.related_prediction.maskout
    masked_delta = origin_conf - masked_conf if origin_conf is not None and masked_conf is not None else None
    maskout_delta = origin_conf - maskout_conf if origin_conf is not None and maskout_conf is not None else None
    robustness = robustness_score(record)
    monotonicity = _monotonicity_metrics(record)

    summary: Dict[str, Any] = {
        "dataset": record.dataset,
        "graph_type": record.graph_type,
        "method": record.method,
        "run_id": record.run_id,
        "graph_index": record.graph_index,
        "label": record.label,
        "prediction_class": record.prediction_class,
        "prediction_confidence": record.prediction_confidence,
        "is_correct": record.is_correct,
        "origin_confidence": record.related_prediction.origin,
        "masked_confidence": record.related_prediction.masked,
        "maskout_confidence": record.related_prediction.maskout,
        "origin_distribution": list(record.related_prediction.origin_distribution)
        if record.related_prediction.origin_distribution is not None
        else None,
        "masked_distribution": list(record.related_prediction.masked_distribution)
        if record.related_prediction.masked_distribution is not None
        else None,
        "maskout_distribution": list(record.related_prediction.maskout_distribution)
        if record.related_prediction.maskout_distribution is not None
        else None,
        "sparsity": record.related_prediction.sparsity,
        "minimal_coalition_size": minimal.size if minimal else None,
        "minimal_coalition_confidence": minimal.confidence if minimal else None,
        "insertion_auc": insertion.auc,
        "deletion_auc": deletion.auc,
        "insertion_curve": insertion.as_series(),
        "deletion_curve": deletion.as_series(),
        "top_nodes": list(top_nodes(record, k=top_k)),
        "num_nodes": record.num_nodes,
        "num_edges": record.num_edges,
        "origin_contrastivity": record.related_prediction.origin_contrastivity,
        "masked_contrastivity": record.related_prediction.masked_contrastivity,
        "maskout_contrastivity": record.related_prediction.maskout_contrastivity,
        "origin_second_class": record.related_prediction.origin_second_class,
        "origin_second_confidence": record.related_prediction.origin_second_confidence,
        "maskout_progression_confidence": list(record.related_prediction.maskout_progression_confidence)
        if record.related_prediction.maskout_progression_confidence is not None
        else None,
        "maskout_progression_drop": list(record.related_prediction.maskout_progression_drop)
        if record.related_prediction.maskout_progression_drop is not None
        else None,
        "sufficiency_progression_confidence": list(record.related_prediction.sufficiency_progression_confidence)
        if record.related_prediction.sufficiency_progression_confidence is not None
        else None,
        "sufficiency_progression_drop": list(record.related_prediction.sufficiency_progression_drop)
        if record.related_prediction.sufficiency_progression_drop is not None
        else None,
        "masked_delta": masked_delta,
        "maskout_delta": maskout_delta,
        "robustness_score": robustness,
        "maskout_drop_monotonicity": monotonicity["maskout_drop_monotonicity"],
        "maskout_conf_monotonicity": monotonicity["maskout_conf_monotonicity"],
        "sufficiency_conf_monotonicity": monotonicity["sufficiency_conf_monotonicity"],
        "sufficiency_drop_monotonicity": monotonicity["sufficiency_drop_monotonicity"],
    }

    if node_text:
        summary["top_tokens"] = [node_text[idx] for idx in summary["top_nodes"] if 0 <= idx < len(node_text)]
    else:
        summary["top_tokens"] = None

    # Handle minimal coalition tokens
    if minimal and node_text:
        summary["minimal_coalition_tokens"] = [node_text[idx] for idx in minimal.nodes if 0 <= idx < len(node_text)]
    else:
        summary["minimal_coalition_tokens"] = None

    # Structural metrics only available for graph-based explanations
    if minimal and graph_obj is not None:
        summary["structural_metrics"] = induced_subgraph_metrics(graph_obj, minimal.nodes)
    else:
        summary["structural_metrics"] = None

    if graph_obj is not None and centrality_funcs and record.node_importance:
        importance_map = {idx: float(score) for idx, score in enumerate(record.node_importance)}
        centrality_results: Dict[str, Optional[float]] = {}
        for name, func in centrality_funcs.items():
            try:
                scores = func(graph_obj)
                centrality_results[name] = centrality_alignment(importance_map, scores)
            except Exception:
                centrality_results[name] = None
        summary["centrality_alignment"] = centrality_results
    else:
        summary["centrality_alignment"] = None

    summary["fidelity_plus"] = fidelity_plus(record)
    summary["fidelity_minus"] = fidelity_minus(record)
    summary["faithfulness"] = faithfulness(record)
    summary["faithfulness_monotonicity"] = monotonicity["faithfulness_monotonicity"]

    return summary


def summarize_records(
    records: Iterable[ExplanationRecord],
    *,
    sufficiency_threshold: float = 0.9,
    top_k: int = 10,
    graph_provider: Callable[[ExplanationRecord], Optional[nx.Graph]] = _default_graph_provider,
    centrality_funcs: Optional[Mapping[str, Callable[[nx.Graph], Mapping[int, float]]]] = None,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for record in records:
        summary = summarize_record(
            record,
            sufficiency_threshold=sufficiency_threshold,
            top_k=top_k,
            graph_provider=graph_provider,
            centrality_funcs=centrality_funcs,
        )
        summaries.append(summary)
    return summaries


def _limit_ranking(values: Sequence[int], k: Optional[int]) -> List[int]:
    if k is not None:
        return list(values[:k])
    return list(values)


def _ranking_positions(values: Sequence[int]) -> Dict[int, int]:
    return {value: index + 1 for index, value in enumerate(values)}


def rank_biased_overlap(
    ranking_a: Sequence[int],
    ranking_b: Sequence[int],
    *,
    p: float = 0.9,
    k: Optional[int] = None,
) -> Optional[float]:
    if not ranking_a or not ranking_b:
        return None
    if not (0 < p < 1):
        return None

    limited_a = _limit_ranking(ranking_a, k)
    limited_b = _limit_ranking(ranking_b, k)
    depth = max(len(limited_a), len(limited_b))
    if depth == 0:
        return None

    seen_a: set[int] = set()
    seen_b: set[int] = set()
    cumulative = 0.0

    for depth_index in range(1, depth + 1):
        if depth_index <= len(limited_a):
            seen_a.add(limited_a[depth_index - 1])
        if depth_index <= len(limited_b):
            seen_b.add(limited_b[depth_index - 1])
        overlap = len(seen_a & seen_b) / depth_index
        cumulative += overlap * (p ** (depth_index - 1))

    return (1 - p) * cumulative


def spearman_rank_correlation(
    ranking_a: Sequence[int],
    ranking_b: Sequence[int],
    *,
    k: Optional[int] = None,
) -> Optional[float]:
    limited_a = _limit_ranking(ranking_a, k)
    limited_b = _limit_ranking(ranking_b, k)
    if not limited_a or not limited_b:
        return None

    union = set(limited_a) | set(limited_b)
    if len(union) < 2:
        return None

    pos_a = _ranking_positions(limited_a)
    pos_b = _ranking_positions(limited_b)
    default_rank = max(len(limited_a), len(limited_b)) + 1

    ranks_a = [pos_a.get(node, default_rank) for node in union]
    ranks_b = [pos_b.get(node, default_rank) for node in union]

    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)

    numerator = sum((ra - mean_a) * (rb - mean_b) for ra, rb in zip(ranks_a, ranks_b))
    denom_a = math.sqrt(sum((ra - mean_a) ** 2 for ra in ranks_a))
    denom_b = math.sqrt(sum((rb - mean_b) ** 2 for rb in ranks_b))
    denominator = denom_a * denom_b
    if denominator == 0:
        return None
    return numerator / denominator


def kendall_rank_correlation(
    ranking_a: Sequence[int],
    ranking_b: Sequence[int],
    *,
    k: Optional[int] = None,
) -> Optional[float]:
    limited_a = _limit_ranking(ranking_a, k)
    limited_b = _limit_ranking(ranking_b, k)
    if not limited_a or not limited_b:
        return None

    union = list(set(limited_a) | set(limited_b))
    if len(union) < 2:
        return None

    pos_a = _ranking_positions(limited_a)
    pos_b = _ranking_positions(limited_b)
    default_rank = max(len(limited_a), len(limited_b)) + 1

    concordant = 0
    discordant = 0

    for left, right in itertools.combinations(union, 2):
        diff_a = pos_a.get(left, default_rank) - pos_a.get(right, default_rank)
        diff_b = pos_b.get(left, default_rank) - pos_b.get(right, default_rank)
        if diff_a == 0 or diff_b == 0:
            continue
        if diff_a > 0 and diff_b > 0:
            concordant += 1
        elif diff_a < 0 and diff_b < 0:
            concordant += 1
        else:
            discordant += 1

    total = concordant + discordant
    if total == 0:
        return None
    return (concordant - discordant) / total


def feature_overlap_ratio(
    ranking_a: Sequence[int],
    ranking_b: Sequence[int],
    *,
    k: Optional[int] = None,
) -> Optional[float]:
    limited_a = _limit_ranking(ranking_a, k)
    limited_b = _limit_ranking(ranking_b, k)
    if not limited_a or not limited_b:
        return None
    intersection = len(set(limited_a) & set(limited_b))
    baseline = min(len(limited_a), len(limited_b))
    if baseline == 0:
        return None
    return intersection / baseline


def pairwise_agreement(
    records: Iterable[ExplanationRecord],
    *,
    top_k: int = 10,
    rbo_p: float = 0.9,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any, Any, Any], List[ExplanationRecord]] = {}
    for record in records:
        key = (record.dataset, record.graph_index, record.label, record.is_correct)
        grouped.setdefault(key, []).append(record)

    entries: List[Dict[str, Any]] = []
    for key, recs in grouped.items():
        if len(recs) < 2:
            continue
        for left, right in itertools.combinations(recs, 2):
            top_left = top_nodes(left, k=top_k)
            top_right = top_nodes(right, k=top_k)
            entry = {
                "dataset": left.dataset,
                "graph_index": left.graph_index,
                "label": left.label,
                "is_correct": left.is_correct,
                "method_a": left.method,
                "method_b": right.method,
                "graph_type_a": left.graph_type,
                "graph_type_b": right.graph_type,
                "run_id_a": left.run_id,
                "run_id_b": right.run_id,
                "top_k": top_k,
                "overlap_count": len(set(top_left) & set(top_right)),
                "rbo": rank_biased_overlap(top_left, top_right, p=rbo_p, k=top_k),
                "spearman": spearman_rank_correlation(top_left, top_right, k=top_k),
                "kendall": kendall_rank_correlation(top_left, top_right, k=top_k),
                "feature_overlap_ratio": feature_overlap_ratio(top_left, top_right, k=top_k),
                "stability_jaccard": stability_jaccard(left, right, k=top_k),
            }
            entries.append(entry)
    return entries
