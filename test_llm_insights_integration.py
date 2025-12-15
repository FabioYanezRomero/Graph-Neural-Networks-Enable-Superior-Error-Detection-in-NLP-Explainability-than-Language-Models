#!/usr/bin/env python3
"""Test LLM Insights integration - verify token text is properly saved and extracted."""

from src.Insights.llm_providers import LLMExplanationProvider, TokenInfo
from src.Insights.metrics import summarize_record
from src.Insights.records import Coalition, ExplanationRecord, RelatedPrediction


def test_token_info():
    """Test TokenInfo creation and methods."""
    print("Testing TokenInfo...")
    
    token_text = ["this", "is", "a", "test", "sentence"]
    prompt = "this is a test sentence"
    
    token_info = TokenInfo(
        token_text=tuple(token_text),
        prompt=prompt,
        num_tokens=5,
    )
    
    assert token_info.num_tokens == 5
    assert len(token_info.token_text) == 5
    assert token_info.prompt == prompt
    
    # Test text_for_indices
    top_indices = [0, 2, 4]
    top_tokens = token_info.text_for_indices(top_indices)
    assert top_tokens == ["this", "a", "sentence"]
    
    print("✅ TokenInfo works correctly")


def test_llm_provider():
    """Test LLMExplanationProvider extraction."""
    print("\nTesting LLMExplanationProvider...")
    
    # Create a mock explanation record
    record = ExplanationRecord(
        dataset="stanfordnlp/sst2",
        graph_type="tokens",
        method="token_shap_llm",
        run_id="test_run",
        graph_index=0,
        label=1,
        prediction_class=1,
        prediction_confidence=0.95,
        num_nodes=5,
        num_edges=0,
        node_importance=[0.1, 0.3, 0.9, 0.5, 0.2],
        top_nodes=(2, 3, 1, 4, 0),
        extras={
            "prompt": "this is a test sentence",
            "token_text": ["this", "is", "a", "test", "sentence"],
            "elapsed_time": 1.5,
        }
    )
    
    # Use provider
    provider = LLMExplanationProvider()
    token_info = provider(record)
    
    assert token_info is not None
    assert isinstance(token_info, TokenInfo)
    assert token_info.num_tokens == 5
    assert len(token_info.token_text) == 5
    assert token_info.prompt == "this is a test sentence"
    
    print("✅ LLMExplanationProvider extracts token info correctly")


def test_summarize_with_tokens():
    """Test that summarize_record populates top_tokens automatically."""
    print("\nTesting summarize_record with LLM provider...")
    
    # Create coalitions
    coalitions = [
        Coalition.from_iterable(
            nodes=[2, 3],
            confidence=0.92,
            combination_id=0,
        ),
        Coalition.from_iterable(
            nodes=[2],
            confidence=0.85,
            combination_id=1,
        ),
    ]
    
    # Create record
    record = ExplanationRecord(
        dataset="stanfordnlp/sst2",
        graph_type="tokens",
        method="token_shap_llm",
        run_id="test_run",
        graph_index=0,
        label=1,
        prediction_class=1,
        prediction_confidence=0.95,
        num_nodes=5,
        num_edges=0,
        node_importance=[0.1, 0.3, 0.9, 0.5, 0.2],
        top_nodes=(2, 3, 1, 4, 0),
        related_prediction=RelatedPrediction(origin=0.95),
        coalitions=coalitions,
        extras={
            "prompt": "this is a great test",
            "token_text": ["this", "is", "a", "great", "test"],
        }
    )
    
    # Summarize with LLM provider
    provider = LLMExplanationProvider()
    summary = summarize_record(
        record,
        graph_provider=provider,
        sufficiency_threshold=0.9,
        top_k=3,
    )
    
    # Check that top_tokens is populated
    assert "top_tokens" in summary
    assert summary["top_tokens"] is not None
    # Top nodes are [2, 3, 1], which map to ["a", "great", "is"]
    expected_top_tokens = ["a", "great", "is"]
    assert summary["top_tokens"] == expected_top_tokens
    
    # Check minimal coalition tokens
    assert "minimal_coalition_tokens" in summary
    # Minimal coalition should be [2, 3] -> ["a", "great"]
    expected_minimal = ["a", "great"]
    print(f"   DEBUG: minimal_coalition_tokens = {summary['minimal_coalition_tokens']}")
    print(f"   DEBUG: expected = {expected_minimal}")
    # The order might differ depending on how minimal_coalition is selected
    assert set(summary["minimal_coalition_tokens"]) == set(expected_minimal)
    
    print("✅ summarize_record correctly populates top_tokens and minimal_coalition_tokens")
    print(f"   Top tokens: {summary['top_tokens']}")
    print(f"   Minimal coalition tokens: {summary['minimal_coalition_tokens']}")


def test_without_provider():
    """Test that records without token_text don't crash."""
    print("\nTesting summarize_record without provider...")
    
    record = ExplanationRecord(
        dataset="stanfordnlp/sst2",
        graph_type="tokens",
        method="token_shap_llm",
        run_id="test_run",
        graph_index=0,
        label=1,
        prediction_class=1,
        prediction_confidence=0.95,
        num_nodes=5,
        num_edges=0,
        top_nodes=(2, 3, 1),
        extras={}  # No token_text
    )
    
    # Should work without crashing, but top_tokens will be None
    provider = LLMExplanationProvider(strict=False)
    summary = summarize_record(
        record,
        graph_provider=provider,
        top_k=3,
    )
    
    assert "top_tokens" in summary
    assert summary["top_tokens"] is None  # No token text available
    
    print("✅ Gracefully handles records without token_text")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing LLM Insights Integration")
    print("=" * 80)
    
    test_token_info()
    test_llm_provider()
    test_summarize_with_tokens()
    test_without_provider()
    
    print("\n" + "=" * 80)
    print("✅ All LLM Insights integration tests passed!")
    print("=" * 80)
    print("\nThe LLM explainability module is now fully integrated with Insights:")
    print("- Token text is automatically extracted from ExplanationRecords")
    print("- top_tokens and minimal_coalition_tokens are populated in summaries")
    print("- Compatible with existing analytics infrastructure")
    print()


if __name__ == "__main__":
    main()

