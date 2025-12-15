from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
import warnings

from .config import ExplainerRequest, clone_request_with_method
from .graphsvx.main import explain_request as run_graphsvx


@dataclass
class ExplainerOutput:
    method: str
    results: List[Any]
    artifact_dir: Path
    summary_path: Path
    raw_path: Optional[Path]


def run(request: ExplainerRequest, *, progress: bool = True) -> ExplainerOutput:
    method = request.resolved_method()
    chosen_method = method
    if method == "subgraphx":
        try:
            from .subgraphx.main import explain_request as run_subgraphx  # type: ignore

            results, artifact_dir, summary_path, raw_path = run_subgraphx(request, progress=progress)
        except Exception as exc:
            if request.method:  # user explicitly asked for SubgraphX
                raise
            warnings.warn(
                "SubgraphX dependencies unavailable, falling back to GraphSVX. "
                f"Original error: {exc}"
            )
            fallback_request = clone_request_with_method(request, "graphsvx")
            results, artifact_dir, summary_path, raw_path = run_graphsvx(
                fallback_request, progress=progress
            )
            chosen_method = "graphsvx"
    elif method == "graphsvx":
        results, artifact_dir, summary_path, raw_path = run_graphsvx(request, progress=progress)
    else:
        raise ValueError(f"Unsupported explainer method: {method}")
    return ExplainerOutput(
        method=chosen_method,
        results=results,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
        raw_path=raw_path,
    )
