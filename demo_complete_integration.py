#!/usr/bin/env python3
"""Demo: Complete integration of hyperparameter advisor + Insights."""

from src.Insights.llm_providers import LLMExplanationProvider
from src.Insights.metrics import summarize_record
from src.Insights.records import Coalition, ExplanationRecord, RelatedPrediction
from src.explain.llm.hyperparam_advisor import (
    DatasetContext,
    ModelSpec,
    TokenSHAPHyperparameterAdvisor,
)


def main():
    print("=" * 80)
    print("Complete Integration Demo: Hyperparameter Advisor + Insights")
    print("=" * 80)
    
    # Example sentences
    sentences = [
        "good",
        "this movie is great",
        "i really enjoyed watching this excellent film last night",
    ]
    
    # Setup hyperparameter advisor
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
    
    advisor = TokenSHAPHyperparameterAdvisor(
        model_spec=model_spec,
        context=context,
    )
    
    # Setup Insights provider
    llm_provider = LLMExplanationProvider()
    
    print("\nProcessing example sentences...\n")
    
    for i, sentence in enumerate(sentences, 1):
        # Tokenize (simple space split for demo)
        tokens = sentence.split()
        
        # Step 1: Get suggested hyperparameters from advisor
        suggested_params = advisor.suggest(tokens)
        
        print(f"Example {i}: \"{sentence}\"")
        print(f"  Tokens: {tokens}")
        print(f"  Num tokens: {len(tokens)}")
        print(f"  Suggested hyperparameters:")
        print(f"    - sampling_ratio: {suggested_params['sampling_ratio']:.4f}")
        print(f"    - num_samples: {suggested_params['num_samples_override']}")
        print(f"    - top_k: {suggested_params['top_k_nodes']}")
        
        # Step 2: Create mock ExplanationRecord (simulating TokenSHAP output)
        # In real usage, this would be created by token_shap_runner
        coalitions = [
            Coalition.from_iterable(
                nodes=[idx for idx in range(min(3, len(tokens)))],
                confidence=0.92,
                combination_id=0,
            ),
        ]
        
        # Mock importance scores (in real usage, from TokenSHAP)
        importance = [0.1] * len(tokens)
        if len(tokens) > 0:
            importance[0] = 0.9  # First token has high importance
        if len(tokens) > 2:
            importance[2] = 0.8  # Third token has high importance
        
        record = ExplanationRecord(
            dataset="stanfordnlp/sst2",
            graph_type="tokens",
            method="token_shap_llm",
            run_id="demo",
            graph_index=i-1,
            label=1,
            prediction_class=1,
            prediction_confidence=0.95,
            num_nodes=len(tokens),
            num_edges=0,
            node_importance=importance,
            top_nodes=tuple(sorted(range(len(tokens)), key=lambda idx: importance[idx], reverse=True)[:min(3, len(tokens))]),
            related_prediction=RelatedPrediction(origin=0.95),
            hyperparams=suggested_params,  # Store suggested hyperparams
            coalitions=coalitions,
            extras={
                "prompt": sentence,
                "token_text": tokens,
                "elapsed_time": 1.5,
            }
        )
        
        # Step 3: Summarize using Insights module
        summary = summarize_record(
            record,
            graph_provider=llm_provider,  # Automatically extracts token text
            sufficiency_threshold=0.9,
            top_k=3,
        )
        
        # Step 4: Display Insights output
        print(f"  Insights summary:")
        print(f"    - Top token indices: {summary['top_nodes']}")
        print(f"    - Top tokens: {summary['top_tokens']}")  # ← Human-readable!
        print(f"    - Minimal coalition size: {summary['minimal_coalition_size']}")
        print(f"    - Minimal coalition tokens: {summary['minimal_coalition_tokens']}")  # ← Human-readable!
        print()
    
    print("=" * 80)
    print("Demo Summary")
    print("=" * 80)
    print()
    print("✅ Hyperparameter Advisor:")
    print("   - Analyzed each sentence")
    print("   - Suggested adaptive hyperparameters based on length")
    print("   - Shorter sentences get higher sampling ratios")
    print()
    print("✅ Insights Integration:")
    print("   - LLMExplanationProvider extracted token text from records")
    print("   - top_tokens automatically populated (human-readable)")
    print("   - minimal_coalition_tokens automatically populated")
    print("   - Same infrastructure as GNN explanations")
    print()
    print("Complete integration enables:")
    print("• Per-sample hyperparameter optimization")
    print("• Human-readable token text in all outputs")
    print("• Unified analytics for GNN and LLM explanations")
    print("• Transparent and explainable results")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()





