import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch, Data

from dig.xgraph.method import SubgraphX


def _contrastive_stats(
    distribution: Optional[Sequence[float]],
    target_class: Optional[int],
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if distribution is None:
        return None, None, None
    try:
        values = [float(v) for v in distribution]
    except (TypeError, ValueError):
        return None, None, None
    if not values:
        return None, None, None

    if target_class is not None and 0 <= int(target_class) < len(values):
        target_idx = int(target_class)
        target_conf = values[target_idx]
        others = [(idx, val) for idx, val in enumerate(values) if idx != target_idx]
        if not others:
            return None, None, None
        second_idx, second_val = max(others, key=lambda item: item[1])
        contrast = target_conf - second_val if target_conf is not None else None
        return second_idx, second_val, contrast

    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    if len(ordered) < 2:
        return None, None, None
    _, best_val = ordered[0]
    second_idx, second_val = ordered[1]
    contrast = best_val - second_val if best_val is not None else None
    return second_idx, second_val, contrast


def _extract_node_tokens(data: Data) -> Optional[List[str]]:
    token_keys = ("node_tokens", "tokens", "token_text", "words", "token_strings")
    for key in token_keys:
        value = getattr(data, key, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if torch.is_tensor(value):
            flat = value.detach().cpu().tolist()
            if isinstance(flat, list):
                return [str(item) for item in flat]
    mapping = getattr(data, "token_map", None)
    if isinstance(mapping, dict):
        num_nodes = int(getattr(data, "num_nodes", len(mapping)))
        return [str(mapping.get(idx)) for idx in range(num_nodes)]
    return None


class CustomSubgraphX(SubgraphX):
    """Augmented SubgraphX explainer that captures probability distributions."""

    def __init__(self, *args, value_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_value_func = value_func
        if value_func is not None:
            warnings.warn(
                "Custom value_func injected into SubgraphX. This will override internal model calls."
            )

    def explain(  # type: ignore[override]
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label: Optional[int],
        max_nodes: int = 5,
        node_idx: Optional[int] = None,
        saved_MCTSInfo_list: Optional[List[List]] = None,
        **kwargs,
    ):
        call_kwargs: Dict[str, object] = {
            "x": x,
            "edge_index": edge_index,
            "label": label,
            "max_nodes": max_nodes,
            "node_idx": node_idx,
            "saved_MCTSInfo_list": saved_MCTSInfo_list,
        }
        call_kwargs.update(kwargs)
        if self._custom_value_func is not None:
            call_kwargs["value_func"] = self._custom_value_func

        def _filtered(include_value_func: bool) -> Dict[str, object]:
            filtered: Dict[str, object] = {}
            for key, value in call_kwargs.items():
                if key == "value_func" and not include_value_func:
                    continue
                if key in {"x", "edge_index", "label", "value_func"} or value is not None:
                    filtered[key] = value
            return filtered

        try:
            results, related_pred = super().explain(**_filtered(include_value_func=True))
        except TypeError as exc:
            if "value_func" in call_kwargs and "unexpected keyword argument 'value_func'" in str(exc):
                results, related_pred = super().explain(**_filtered(include_value_func=False))
            else:
                raise

        try:
            self._augment_related_prediction(results, related_pred, x, edge_index, label)
        except Exception as exc:  # pragma: no cover - defensive guard
            warnings.warn(f"Failed to augment SubgraphX probabilities: {exc}")
        return results, related_pred

    def _augment_related_prediction(
        self,
        results,
        related_pred: Dict[str, object],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label: Optional[int],
    ) -> None:
        if not isinstance(related_pred, dict):
            return

        base_data, coalition = self._extract_primary_data(results)
        if isinstance(coalition, list):
            related_pred.setdefault("top_nodes", [int(idx) for idx in coalition])

        origin_probs = self._predict_probs_from_inputs(x, edge_index)
        origin_distribution = origin_probs.detach().cpu().tolist()
        if origin_distribution:
            related_pred["origin_distribution"] = [float(v) for v in origin_distribution]

        target_index = self._resolve_target_index(label, origin_probs)
        if target_index is not None and 0 <= target_index < len(origin_distribution):
            related_pred["origin"] = float(origin_distribution[target_index])

        masked_distribution, maskout_distribution = self._compute_masked_distributions(results)

        if masked_distribution is not None:
            related_pred["masked_distribution"] = [float(v) for v in masked_distribution]
            if target_index is not None and 0 <= target_index < len(masked_distribution):
                related_pred["masked"] = float(masked_distribution[target_index])
        else:
            related_pred.setdefault("masked_distribution", None)

        if maskout_distribution is not None:
            related_pred["maskout_distribution"] = [float(v) for v in maskout_distribution]
            if target_index is not None and 0 <= target_index < len(maskout_distribution):
                related_pred["maskout"] = float(maskout_distribution[target_index])
        else:
            related_pred.setdefault("maskout_distribution", None)

        second_idx, second_conf, contrast = _contrastive_stats(origin_distribution, target_index)
        related_pred["origin_second_class"] = second_idx
        related_pred["origin_second_confidence"] = second_conf
        related_pred["origin_contrastivity"] = contrast

        if masked_distribution is not None:
            _, masked_second_conf, masked_contrast = _contrastive_stats(masked_distribution, target_index)
            related_pred["masked_second_confidence"] = masked_second_conf
            related_pred["masked_contrastivity"] = masked_contrast
        else:
            related_pred.setdefault("masked_second_confidence", None)
            related_pred.setdefault("masked_contrastivity", None)

        if maskout_distribution is not None:
            _, maskout_second_conf, maskout_contrast = _contrastive_stats(maskout_distribution, target_index)
            related_pred["maskout_second_confidence"] = maskout_second_conf
            related_pred["maskout_contrastivity"] = maskout_contrast
        else:
            related_pred.setdefault("maskout_second_confidence", None)
            related_pred.setdefault("maskout_contrastivity", None)

        node_tokens = None
        if base_data is not None:
            node_tokens = _extract_node_tokens(base_data)
        if node_tokens:
            related_pred["node_tokens"] = node_tokens
            ranked_nodes = related_pred.get("top_nodes", [])
            related_pred["ranked_tokens"] = [
                node_tokens[idx] for idx in ranked_nodes if isinstance(idx, int) and 0 <= idx < len(node_tokens)
            ]
            related_pred["top_token_text"] = [
                node_tokens[idx] for idx in related_pred.get("top_nodes", [])
                if isinstance(idx, int) and 0 <= idx < len(node_tokens)
            ]

        related_pred.setdefault("ranked_nodes", related_pred.get("top_nodes", []))

    def _resolve_target_index(
        self, label: Optional[int], origin_probs: torch.Tensor
    ) -> Optional[int]:
        if label is not None:
            try:
                idx = int(label)
                if 0 <= idx < origin_probs.numel():
                    return idx
            except (TypeError, ValueError):
                pass
        if origin_probs.numel() == 0:
            return None
        return int(torch.argmax(origin_probs).item())

    def _compute_masked_distributions(
        self, results
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        base_data, coalition = self._extract_primary_data(results)
        if base_data is None or not coalition:
            return None, None

        num_nodes = base_data.num_nodes or base_data.x.size(0)
        if num_nodes <= 0:
            return None, None

        valid_nodes = sorted({idx for idx in coalition if 0 <= idx < num_nodes})
        if not valid_nodes:
            return None, None

        mask_keep = torch.zeros(num_nodes, dtype=torch.float32)
        mask_keep[valid_nodes] = 1.0
        mask_drop = torch.ones(num_nodes, dtype=torch.float32)
        mask_drop[valid_nodes] = 0.0

        masked_batch = self._build_batch(base_data, mask_keep)
        maskout_batch = self._build_batch(base_data, mask_drop)

        masked_probs = self._predict_probs_from_inputs(masked_batch)
        maskout_probs = self._predict_probs_from_inputs(maskout_batch)

        return (
            masked_probs.detach().cpu().tolist(),
            maskout_probs.detach().cpu().tolist(),
        )

    def _extract_primary_data(
        self, results
    ) -> Tuple[Optional[Data], Optional[Sequence[int]]]:
        if not results:
            return None, None
        entry = results[0]
        if isinstance(entry, list) and entry:
            entry = entry[0]
        if not isinstance(entry, dict):
            return None, None

        data_obj = entry.get("data")
        coalition = entry.get("coalition") or []
        if data_obj is None:
            return None, None

        if isinstance(data_obj, Batch):
            data_list = data_obj.to_data_list()
            if not data_list:
                return None, None
            base_data = data_list[0]
        elif isinstance(data_obj, Data):
            base_data = data_obj
        else:
            return None, None

        indices: List[int] = []
        for value in coalition:
            try:
                indices.append(int(value))
            except (TypeError, ValueError):
                continue
        return base_data, indices

    def _build_batch(self, data: Data, mask: torch.Tensor) -> Batch:
        masked = data.clone().cpu()
        mask = mask.to(masked.x.device, dtype=masked.x.dtype)
        masked.x = masked.x * mask.unsqueeze(1)

        if self.subgraph_building_method == "split":
            node_mask = mask > 0.5
            row, col = masked.edge_index
            edge_mask = node_mask[row] & node_mask[col]
            masked.edge_index = masked.edge_index[:, edge_mask]
            if getattr(masked, "edge_attr", None) is not None:
                masked.edge_attr = masked.edge_attr[edge_mask]

        batch = Batch.from_data_list([masked])
        return batch.to(self.device)

    def cumulative_maskout_confidence(
        self,
        data: Data,
        ordered_nodes: Sequence[int],
        target_index: Optional[int],
    ) -> List[float]:
        if target_index is None:
            return []
        num_nodes = getattr(data, "num_nodes", None) or data.x.size(0)
        if num_nodes <= 0:
            return []
        drop_nodes: List[int] = []
        confidences: List[float] = []
        for node_idx in ordered_nodes:
            if node_idx is None:
                continue
            try:
                node_int = int(node_idx)
            except (TypeError, ValueError):
                continue
            if node_int < 0 or node_int >= num_nodes:
                continue
            drop_nodes.append(node_int)
            mask = torch.ones(num_nodes, dtype=torch.float32, device=self.device)
            mask[drop_nodes] = 0.0
            batch = self._build_batch(data, mask)
            probs = self._predict_probs_from_inputs(batch)
            if probs.numel() <= target_index:
                continue
            confidences.append(float(probs[target_index]))
        return confidences

    def cumulative_sufficiency_confidence(
        self,
        data: Data,
        ordered_nodes: Sequence[int],
        target_index: Optional[int],
    ) -> List[float]:
        if target_index is None:
            return []
        num_nodes = getattr(data, "num_nodes", None) or data.x.size(0)
        if num_nodes <= 0:
            return []
        kept_nodes: List[int] = []
        confidences: List[float] = []
        for node_idx in ordered_nodes:
            if node_idx is None:
                continue
            try:
                node_int = int(node_idx)
            except (TypeError, ValueError):
                continue
            if node_int < 0 or node_int >= num_nodes:
                continue
            kept_nodes.append(node_int)
            mask = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
            mask[kept_nodes] = 1.0
            batch = self._build_batch(data, mask)
            probs = self._predict_probs_from_inputs(batch)
            if probs.numel() <= target_index:
                continue
            confidences.append(float(probs[target_index]))
        return confidences

    def _predict_probs_from_inputs(self, *model_args, **model_kwargs) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(*model_args, **model_kwargs)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.squeeze()
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if probs.dim() > 1:
            probs = probs[0]
        return probs
