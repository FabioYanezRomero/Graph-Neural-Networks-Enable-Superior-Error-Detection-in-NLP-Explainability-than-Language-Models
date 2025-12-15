#!/usr/bin/env python3
"""Test and demonstrate the TokenSHAP hyperparameter advisor."""

from __future__ import annotations

import json
from typing import List

from .hyperparam_advisor import (
    DatasetContext,
    ModelSpec,
    SentenceStats,
    TokenSHAPHyperparameterAdvisor,
)


def test_sentence_stats():
    """Test SentenceStats computation."""
    print("=" * 80)
    print("Testing SentenceStats")
    print("=" * 80)
    
    # Test with various token lists
    test_cases = [
        (["this", "is", "short"], "Very short sentence"),
        (["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"], "Medium sentence"),
        (["a"] * 20, "Long repetitive sentence"),
        (["##word", "##piece", "##token", "##ization"], "Subword tokens"),
    ]
    
    for tokens, description in test_cases:
        stats = SentenceStats.from_tokens(tokens)
        print(f"\n{description}:")
        print(f"  Tokens: {tokens}")
        print(f"  Num tokens: {stats.num_tokens}")
        print(f"  Num chars: {stats.num_chars}")
        print(f"  Avg token length: {stats.avg_token_length:.2f}")
        print(f"  Max token length: {stats.max_token_length}")
        print(f"  Is very short: {stats.is_very_short}")
        print(f"  Is short: {stats.is_short}")
        print(f"  Is medium: {stats.is_medium}")
        print(f"  Is long: {stats.is_long}")
        print(f"  Has subword tokens: {stats.has_subword_tokens}")


def test_advisor_suggestions():
    """Test hyperparameter suggestions for different sentence types."""
    print("\n" + "=" * 80)
    print("Testing TokenSHAPHyperparameterAdvisor")
    print("=" * 80)
    
    # Create model spec and context
    model_spec = ModelSpec(
        base_model_name="google-bert/bert-base-uncased",
        num_labels=2,
        max_length=128,
    )
    
    context = DatasetContext(
        dataset="stanfordnlp/sst2",
        task_type="classification",
        backbone="stanfordnlp",
    )
    
    # Create advisor
    advisor = TokenSHAPHyperparameterAdvisor(
        model_spec=model_spec,
        context=context,
    )
    
    # Test with various sentence lengths
    test_sentences = [
        ["good", "movie"],  # 2 tokens
        ["this", "is", "great", "!"],  # 4 tokens
        ["the", "acting", "was", "superb", "and", "engaging"],  # 6 tokens
        ["i", "really", "enjoyed", "watching", "this", "film", "last", "night"],  # 8 tokens
        ["the"] * 12,  # 12 tokens
        ["token"] * 16,  # 16 tokens
        ["word"] * 20,  # 20 tokens
    ]
    
    results = []
    for tokens in test_sentences:
        params = advisor.suggest(tokens)
        stats = SentenceStats.from_tokens(tokens)
        
        print(f"\nTokens: {len(tokens)}")
        print(f"  Sentence: {' '.join(tokens[:5])}{'...' if len(tokens) > 5 else ''}")
        print(f"  Sampling ratio: {params['sampling_ratio']:.4f}")
        print(f"  Num samples override: {params['num_samples_override']}")
        print(f"  Top K nodes: {params['top_k_nodes']}")
        print(f"  Max combinations: 2^{len(tokens)} = {2**min(len(tokens), 20):,}")
        if params['num_samples_override'] is not None:
            coverage = (params['num_samples_override'] / 2**min(len(tokens), 20)) * 100
            print(f"  Coverage: {coverage:.2f}%")
        
        results.append({
            "num_tokens": len(tokens),
            "hyperparams": params,
            "max_combinations": 2**min(len(tokens), 20),
        })
    
    return results


def test_multi_class():
    """Test hyperparameter suggestions for multi-class problems."""
    print("\n" + "=" * 80)
    print("Testing Multi-Class Settings (AG News - 4 classes)")
    print("=" * 80)
    
    # Create model spec for multi-class
    model_spec = ModelSpec(
        base_model_name="google-bert/bert-base-uncased",
        num_labels=4,  # AG News has 4 classes
        max_length=128,
    )
    
    context = DatasetContext(
        dataset="ag_news",
        task_type="classification",
        backbone="SetFit",
    )
    
    advisor = TokenSHAPHyperparameterAdvisor(
        model_spec=model_spec,
        context=context,
    )
    
    # Compare with binary classification
    model_spec_binary = ModelSpec(
        base_model_name="google-bert/bert-base-uncased",
        num_labels=2,
        max_length=128,
    )
    
    advisor_binary = TokenSHAPHyperparameterAdvisor(
        model_spec=model_spec_binary,
        context=context,
    )
    
    test_tokens = ["this", "is", "a", "news", "article", "about", "sports"]
    
    params_multi = advisor.suggest(test_tokens)
    params_binary = advisor_binary.suggest(test_tokens)
    
    print(f"\nSentence: {' '.join(test_tokens)}")
    print(f"Num tokens: {len(test_tokens)}")
    print("\nMulti-class (4 classes):")
    print(f"  Sampling ratio: {params_multi['sampling_ratio']:.4f}")
    print(f"  Num samples: {params_multi['num_samples_override']}")
    print("\nBinary (2 classes):")
    print(f"  Sampling ratio: {params_binary['sampling_ratio']:.4f}")
    print(f"  Num samples: {params_binary['num_samples_override']}")
    
    if params_multi['num_samples_override'] and params_binary['num_samples_override']:
        ratio = params_multi['num_samples_override'] / params_binary['num_samples_override']
        print(f"\nMulti-class uses {ratio:.2f}x more samples than binary")


def test_locked_params():
    """Test locked parameters functionality."""
    print("\n" + "=" * 80)
    print("Testing Locked Parameters")
    print("=" * 80)
    
    model_spec = ModelSpec(
        base_model_name="google-bert/bert-base-uncased",
        num_labels=2,
        max_length=128,
    )
    
    context = DatasetContext(
        dataset="stanfordnlp/sst2",
        task_type="classification",
        backbone="stanfordnlp",
    )
    
    # Create advisor with locked sampling ratio
    locked_params = {"sampling_ratio": 0.5}
    advisor = TokenSHAPHyperparameterAdvisor(
        model_spec=model_spec,
        context=context,
        locked_params=locked_params,
    )
    
    test_tokens = ["this"] * 10  # 10 tokens
    params = advisor.suggest(test_tokens)
    
    print(f"\nWith locked sampling_ratio=0.5:")
    print(f"  Tokens: {len(test_tokens)}")
    print(f"  Sampling ratio: {params['sampling_ratio']:.4f} (locked)")
    print(f"  Num samples: {params['num_samples_override']} (adaptive)")
    print(f"  Top K: {params['top_k_nodes']} (adaptive)")


def save_results(results: List[dict], output_path: str = "test_hyperparam_results.json"):
    """Save test results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"Results saved to {output_path}")
    print("=" * 80)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TokenSHAP Hyperparameter Advisor Test Suite")
    print("=" * 80)
    
    test_sentence_stats()
    results = test_advisor_suggestions()
    test_multi_class()
    test_locked_params()
    
    # Save results
    save_results(results)
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()





