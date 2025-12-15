from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RelatedPrediction:
    """Stores faithfulness-related prediction scores."""

    origin: Optional[float] = None
    masked: Optional[float] = None
    maskout: Optional[float] = None
    sparsity: Optional[float] = None
    origin_distribution: Optional[Tuple[float, ...]] = None
    masked_distribution: Optional[Tuple[float, ...]] = None
    maskout_distribution: Optional[Tuple[float, ...]] = None
    origin_second_class: Optional[int] = None
    origin_second_confidence: Optional[float] = None
    origin_contrastivity: Optional[float] = None
    masked_second_confidence: Optional[float] = None
    masked_contrastivity: Optional[float] = None
    maskout_second_confidence: Optional[float] = None
    maskout_contrastivity: Optional[float] = None
    maskout_progression_confidence: Optional[Tuple[float, ...]] = None
    maskout_progression_drop: Optional[Tuple[float, ...]] = None
    sufficiency_progression_confidence: Optional[Tuple[float, ...]] = None
    sufficiency_progression_drop: Optional[Tuple[float, ...]] = None

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "RelatedPrediction":
        if not data:
            return cls()

        def _coerce_sequence(key: str) -> Optional[Tuple[float, ...]]:
            values = data.get(key)
            if values is None:
                return None
            if isinstance(values, (list, tuple)):
                try:
                    return tuple(float(v) for v in values)
                except (TypeError, ValueError):
                    return None
            return None
        return cls(
            origin=data.get("origin"),
            masked=data.get("masked"),
            maskout=data.get("maskout"),
            sparsity=data.get("sparsity"),
            origin_distribution=_coerce_sequence("origin_distribution"),
            masked_distribution=_coerce_sequence("masked_distribution"),
            maskout_distribution=_coerce_sequence("maskout_distribution"),
            origin_second_class=data.get("origin_second_class"),
            origin_second_confidence=data.get("origin_second_confidence"),
            origin_contrastivity=data.get("origin_contrastivity"),
            masked_second_confidence=data.get("masked_second_confidence"),
            masked_contrastivity=data.get("masked_contrastivity"),
            maskout_second_confidence=data.get("maskout_second_confidence"),
            maskout_contrastivity=data.get("maskout_contrastivity"),
            maskout_progression_confidence=_coerce_sequence("maskout_progression_confidence"),
            maskout_progression_drop=_coerce_sequence("maskout_progression_drop"),
            sufficiency_progression_confidence=_coerce_sequence("sufficiency_progression_confidence"),
            sufficiency_progression_drop=_coerce_sequence("sufficiency_progression_drop"),
        )


@dataclass(frozen=True)
class Coalition:
    """Represents a single coalition (subset of nodes) produced by an explainer."""

    nodes: Tuple[int, ...]
    confidence: float
    size: int
    combination_id: Optional[int] = None
    binary_mask: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_iterable(
        cls,
        nodes: Iterable[int],
        confidence: float,
        combination_id: Optional[int] = None,
        binary_mask: Optional[Sequence[int]] = None,
        size: Optional[int] = None,
        **metadata: Any,
    ) -> "Coalition":
        node_tuple = tuple(int(n) for n in nodes)
        coalition_size = size if size is not None else len(node_tuple)
        return cls(
            nodes=node_tuple,
            confidence=float(confidence),
            size=coalition_size,
            combination_id=combination_id,
            binary_mask=tuple(int(v) for v in binary_mask) if binary_mask is not None else None,
            metadata=metadata or {},
        )


@dataclass
class ExplanationRecord:
    """
    Canonical container for a single explanation instance.

    Attributes:
        dataset: Name of the dataset (e.g., 'ag-news').
        graph_type: Graph construction variant (e.g., 'skipgrams', 'window').
        method: Explainer name (e.g., 'graphsvx').
        run_id: Identifier for the experiment run that produced the explanation.
        graph_index: Index of the original graph/sample in the dataset.
        label: Ground-truth label (if available).
        prediction_class: Predicted label.
        prediction_confidence: Confidence assigned to the predicted label.
        num_nodes / num_edges: Graph sizes captured by the explainer artefact.
        node_importance: Continuous importance scores per node (GraphSVX-style).
        top_nodes: Ranked list of high-importance nodes.
        related_prediction: Faithfulness metrics (origin/masked/maskout/sparsity/distributions).
        hyperparams: Hyper-parameters used by the explainer.
        coalitions: List of coalitions evaluated by the explainer.
        extras: Additional raw fields not explicitly modelled above.
    """

    dataset: str
    graph_type: Optional[str]
    method: str
    run_id: Optional[str]
    graph_index: int
    label: Optional[int]
    prediction_class: Optional[int]
    prediction_confidence: Optional[float]
    is_correct: Optional[bool] = None
    num_nodes: Optional[int] = None
    num_edges: Optional[int] = None
    node_importance: Optional[Sequence[float]] = None
    top_nodes: Tuple[int, ...] = field(default_factory=tuple)
    related_prediction: RelatedPrediction = field(default_factory=RelatedPrediction)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    coalitions: List[Coalition] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExplanationRecord":
        """
        Reconstruct an ExplanationRecord from a JSON-friendly mapping.
        """
        if payload is None:
            raise ValueError("Cannot build ExplanationRecord from None payload.")

        dataset = payload.get("dataset")
        graph_type = payload.get("graph_type")
        method = payload.get("method")
        run_id = payload.get("run_id")
        graph_index = payload.get("graph_index")
        label = payload.get("label")
        prediction_class = payload.get("prediction_class")
        prediction_confidence = payload.get("prediction_confidence")
        is_correct = payload.get("is_correct")
        num_nodes = payload.get("num_nodes")
        num_edges = payload.get("num_edges")
        node_importance = payload.get("node_importance")
        top_nodes = payload.get("top_nodes") or ()
        hyperparams = payload.get("hyperparams") or {}
        extras = payload.get("extras") or {}

        related_prediction = RelatedPrediction.from_mapping(payload.get("related_prediction"))

        coalitions_raw = payload.get("coalitions") or []
        coalitions: List[Coalition] = []
        for entry in coalitions_raw:
            if not isinstance(entry, Mapping):
                continue
            nodes_raw = entry.get("nodes") or ()
            binary_mask_raw = entry.get("binary_mask")
            coalition = Coalition(
                nodes=tuple(int(n) for n in nodes_raw),
                confidence=float(entry.get("confidence", 0.0)),
                size=int(entry.get("size")) if entry.get("size") is not None else len(nodes_raw),
                combination_id=entry.get("combination_id"),
                binary_mask=tuple(int(v) for v in binary_mask_raw) if isinstance(binary_mask_raw, (list, tuple)) else None,
                metadata=dict(entry.get("metadata") or {}),
            )
            coalitions.append(coalition)

        return cls(
            dataset=dataset,
            graph_type=graph_type,
            method=method,
            run_id=run_id,
            graph_index=int(graph_index) if graph_index is not None else 0,
            label=label,
            prediction_class=prediction_class,
            prediction_confidence=prediction_confidence,
            is_correct=is_correct,
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_importance=tuple(node_importance) if isinstance(node_importance, (list, tuple)) else node_importance,
            top_nodes=tuple(top_nodes),
            related_prediction=related_prediction,
            hyperparams=dict(hyperparams),
            coalitions=coalitions,
            extras=dict(extras),
        )

    def minimal_coalition(
        self,
        threshold: float,
        *,
        origin_confidence: Optional[float] = None,
    ) -> Optional[Coalition]:
        """
        Returns the smallest coalition whose confidence reaches the specified
        fraction of the origin confidence. If origin confidence is not provided,
        the value stored in related_prediction.origin is used.
        """
        if not self.coalitions:
            return None
        baseline = (
            origin_confidence
            if origin_confidence is not None
            else self.related_prediction.origin
        )
        if baseline is None:
            return None

        required = baseline * threshold
        eligible = [c for c in self.coalitions if c.confidence >= required]
        if not eligible:
            # Fall back to the highest-confidence coalition even if the threshold is unmet.
            best = max(self.coalitions, key=lambda c: (c.confidence, -c.size))
            return best

        eligible.sort(key=lambda c: (c.size, -c.confidence))
        return eligible[0]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record into a JSON-friendly dictionary."""
        return {
            "dataset": self.dataset,
            "graph_type": self.graph_type,
            "method": self.method,
            "run_id": self.run_id,
            "graph_index": self.graph_index,
            "label": self.label,
            "prediction_class": self.prediction_class,
            "prediction_confidence": self.prediction_confidence,
            "is_correct": self.is_correct,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_importance": list(self.node_importance) if self.node_importance is not None else None,
            "top_nodes": list(self.top_nodes),
        "related_prediction": {
            "origin": self.related_prediction.origin,
            "masked": self.related_prediction.masked,
            "maskout": self.related_prediction.maskout,
            "sparsity": self.related_prediction.sparsity,
            "origin_distribution": list(self.related_prediction.origin_distribution)
            if self.related_prediction.origin_distribution is not None
            else None,
            "masked_distribution": list(self.related_prediction.masked_distribution)
            if self.related_prediction.masked_distribution is not None
            else None,
            "maskout_distribution": list(self.related_prediction.maskout_distribution)
            if self.related_prediction.maskout_distribution is not None
            else None,
            "origin_second_class": self.related_prediction.origin_second_class,
            "origin_second_confidence": self.related_prediction.origin_second_confidence,
            "origin_contrastivity": self.related_prediction.origin_contrastivity,
            "masked_second_confidence": self.related_prediction.masked_second_confidence,
            "masked_contrastivity": self.related_prediction.masked_contrastivity,
            "maskout_second_confidence": self.related_prediction.maskout_second_confidence,
            "maskout_contrastivity": self.related_prediction.maskout_contrastivity,
            "maskout_progression_confidence": list(self.related_prediction.maskout_progression_confidence)
            if self.related_prediction.maskout_progression_confidence is not None
            else None,
            "maskout_progression_drop": list(self.related_prediction.maskout_progression_drop)
            if self.related_prediction.maskout_progression_drop is not None
            else None,
            "sufficiency_progression_confidence": list(self.related_prediction.sufficiency_progression_confidence)
            if self.related_prediction.sufficiency_progression_confidence is not None
            else None,
            "sufficiency_progression_drop": list(self.related_prediction.sufficiency_progression_drop)
            if self.related_prediction.sufficiency_progression_drop is not None
            else None,
        },
            "hyperparams": self.hyperparams,
            "coalitions": [
                {
                    "combination_id": c.combination_id,
                    "nodes": list(c.nodes),
                    "confidence": c.confidence,
                    "size": c.size,
                    "binary_mask": list(c.binary_mask) if c.binary_mask is not None else None,
                    "metadata": c.metadata,
                }
                for c in self.coalitions
            ],
            "extras": self.extras,
        }
