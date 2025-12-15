#!/usr/bin/env python3
"""Quick demonstration of the TokenSHAP hyperparameter advisor."""

from src.explain.llm.hyperparam_advisor import (
    DatasetContext,
    ModelSpec,
    SentenceStats,
    TokenSHAPHyperparameterAdvisor,
)


def main():
    print("\n" + "=" * 80)
    print("TokenSHAP Hyperparameter Advisor - Quick Demo")
    print("=" * 80)
    
    # Example sentences from SST-2
    example_sentences = [
        "good",
        "great movie",
        "this film is excellent",
        "the acting was superb and really engaging",
        "i thoroughly enjoyed watching this movie last night with my friends",
    ]
    
    # Setup for SST-2 (binary sentiment classification)
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
    
    print("\nDataset: SST-2 (Binary Sentiment Classification)")
    print("Model: BERT-base-uncased")
    print("\nSuggested hyperparameters for example sentences:\n")
    
    # Analyze each sentence
    for i, sentence in enumerate(example_sentences, 1):
        # Simple tokenization (space-split for demo)
        tokens = sentence.split()
        
        # Get statistics
        stats = SentenceStats.from_tokens(tokens)
        
        # Get suggested hyperparameters
        params = advisor.suggest(tokens)
        
        # Calculate combinatorial space
        max_combinations = 2 ** len(tokens)
        
        print(f"{i}. Sentence: \"{sentence}\"")
        print(f"   Tokens: {len(tokens)}")
        print(f"   Max combinations: {max_combinations:,}")
        print(f"   Suggested sampling ratio: {params['sampling_ratio']:.4f}")
        print(f"   Suggested num samples: {params['num_samples_override']}")
        print(f"   Suggested top K: {params['top_k_nodes']}")
        
        if params['num_samples_override']:
            actual_samples = min(params['num_samples_override'], max_combinations)
            coverage_pct = (actual_samples / max_combinations) * 100
            print(f"   Coverage: {coverage_pct:.2f}%")
        print()
    
    # Compare with AG News (multi-class)
    print("\n" + "-" * 80)
    print("Comparison: Multi-class (AG News - 4 classes) vs Binary (SST-2)")
    print("-" * 80 + "\n")
    
    # AG News setup
    context_agnews = DatasetContext(
        dataset="ag_news",
        task_type="classification",
        backbone="SetFit",
    )
    
    model_spec_agnews = ModelSpec(
        base_model_name="google-bert/bert-base-uncased",
        num_labels=4,  # 4 classes
        max_length=128,
    )
    
    advisor_agnews = TokenSHAPHyperparameterAdvisor(
        model_spec=model_spec_agnews,
        context=context_agnews,
    )
    
    test_sentence = "this is a news article about sports"
    tokens = test_sentence.split()
    
    params_binary = advisor.suggest(tokens)
    params_multi = advisor_agnews.suggest(tokens)
    
    print(f"Sentence: \"{test_sentence}\"")
    print(f"Tokens: {len(tokens)}\n")
    
    print("Binary classification (SST-2):")
    print(f"  Sampling ratio: {params_binary['sampling_ratio']:.4f}")
    print(f"  Num samples: {params_binary['num_samples_override']}")
    print(f"  Top K: {params_binary['top_k_nodes']}\n")
    
    print("Multi-class classification (AG News):")
    print(f"  Sampling ratio: {params_multi['sampling_ratio']:.4f}")
    print(f"  Num samples: {params_multi['num_samples_override']}")
    print(f"  Top K: {params_multi['top_k_nodes']}\n")
    
    if params_binary['num_samples_override'] and params_multi['num_samples_override']:
        ratio = params_multi['num_samples_override'] / params_binary['num_samples_override']
        print(f"Multi-class uses {ratio:.2f}x samples (more classes = more variation)")
    
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run full test suite: python -m src.explain.llm.test_hyperparam_advisor")
    print("2. Collect hyperparams: python -m src.explain.llm.main hyperparams stanfordnlp/sst2")
    print("3. Run explanations: python -m src.explain.llm.main explain stanfordnlp/sst2")
    print()


if __name__ == "__main__":
    main()





