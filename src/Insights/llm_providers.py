"""Providers for LLM-based explanations that don't rely on graph structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .records import ExplanationRecord


@dataclass(frozen=True)
class TokenInfo:
    """Container for token-level information from LLM explanations."""

    tokens: Sequence[str]
    num_tokens: int

    def text_for_indices(self, indices: Sequence[int]) -> List[str]:
        """Return token text for the given indices."""
        return [self.tokens[i] for i in indices if 0 <= i < len(self.tokens)]


class LLMExplanationProvider:
    """
    Provider for LLM explainability that extracts token text from record extras.

    This is analogous to GraphArtifactProvider but for LLM explanations where
    tokens are stored directly in the ExplanationRecord.extras dictionary.
    """

    def __call__(self, record: ExplanationRecord) -> Optional[TokenInfo]:
        """
        Extract token information from an LLM explanation record.

        Args:
            record: Explanation record with token_text in extras

        Returns:
            TokenInfo if available, None otherwise
        """
        try:
            return self._load_token_info(record)
        except Exception:
            return None

    def _load_token_info(self, record: ExplanationRecord) -> Optional[TokenInfo]:
        """Load token information from record extras."""
        if not record.extras or not isinstance(record.extras, dict):
            return None

        token_text = record.extras.get("token_text")
        if not token_text or not isinstance(token_text, (list, tuple)):
            return None

        return TokenInfo(
            tokens=tuple(token_text),
            num_tokens=len(token_text),
        )
