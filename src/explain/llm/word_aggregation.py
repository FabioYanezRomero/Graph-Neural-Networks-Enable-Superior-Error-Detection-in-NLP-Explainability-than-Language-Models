"""Word-level aggregation of subword token importance scores."""

from __future__ import annotations

from typing import Dict, List, Sequence

from src.Insights.records import ExplanationRecord


def _aggregate_subwords_to_words(
    tokens: Sequence[str],
    importance: Sequence[float],
    *,
    aggregation: str = "mean",
) -> tuple[List[str], List[float]]:
    """
    Aggregate subword tokens (e.g., WordPiece) into word-level importance.

    Args:
        tokens: List of subword tokens
        importance: Importance scores for each token
        aggregation: How to aggregate ('mean', 'max', 'sum')

    Returns:
        Tuple of (word_list, word_importance)
    """
    if not tokens or not importance:
        return [], []

    words: List[str] = []
    word_importance: List[float] = []
    current_word_tokens: List[str] = []
    current_word_scores: List[float] = []

    for token, score in zip(tokens, importance):
        # Check if token is a continuation (starts with ##)
        if token.startswith("##"):
            # Continuation of previous word
            current_word_tokens.append(token[2:])  # Remove ##
            current_word_scores.append(score)
        else:
            # Start of new word - finalize previous word if exists
            if current_word_tokens:
                word = "".join(current_word_tokens)
                if aggregation == "mean":
                    agg_score = sum(current_word_scores) / len(current_word_scores)
                elif aggregation == "max":
                    agg_score = max(current_word_scores)
                elif aggregation == "sum":
                    agg_score = sum(current_word_scores)
                else:
                    agg_score = sum(current_word_scores) / len(current_word_scores)

                words.append(word)
                word_importance.append(agg_score)

            # Start new word
            current_word_tokens = [token]
            current_word_scores = [score]

    # Finalize last word
    if current_word_tokens:
        word = "".join(current_word_tokens)
        if aggregation == "mean":
            agg_score = sum(current_word_scores) / len(current_word_scores)
        elif aggregation == "max":
            agg_score = max(current_word_scores)
        elif aggregation == "sum":
            agg_score = sum(current_word_scores)
        else:
            agg_score = sum(current_word_scores) / len(current_word_scores)

        words.append(word)
        word_importance.append(agg_score)

    return words, word_importance


def create_word_level_summary(
    record: ExplanationRecord,
    *,
    aggregation: str = "mean",
    top_k: int = 5,
) -> Dict[str, object]:
    """
    Create a word-level summary from token-level explanation.

    Args:
        record: Explanation record with token-level data
        aggregation: Aggregation method ('mean', 'max', 'sum')
        top_k: Number of top words to extract

    Returns:
        Dictionary with word-level metrics
    """
    if not record.extras or not isinstance(record.extras, dict):
        return {}

    tokens = record.extras.get("token_text", [])
    importance = record.node_importance

    if not tokens or not importance or len(tokens) != len(importance):
        return {}

    words, word_scores = _aggregate_subwords_to_words(
        tokens,
        importance,
        aggregation=aggregation,
    )

    if not words:
        return {}

    # Sort by importance and get top k
    word_score_pairs = sorted(
        zip(words, word_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    top_words = [word for word, _ in word_score_pairs[:top_k]]
    top_word_scores = [score for _, score in word_score_pairs[:top_k]]

    return {
        "num_words": len(words),
        "top_words": top_words,
        "top_word_scores": top_word_scores,
        "word_aggregation_method": aggregation,
    }
