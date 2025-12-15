#!/usr/bin/env python3
"""Demo: Word-level aggregation for TokenSHAP explanations."""

from src.explain.llm.word_aggregation import (
    detect_word_boundaries,
    aggregate_token_scores_to_words,
    get_top_words,
)


def main():
    print("=" * 80)
    print("Word-Level Aggregation Demo")
    print("=" * 80)
    print()
    print("TokenSHAP works at the TOKEN level (including subword pieces).")
    print("But for human interpretation, WORD-level scores are more natural!")
    print()
    
    # Example 1: BERT WordPiece tokenization
    print("-" * 80)
    print("Example 1: BERT WordPiece Tokens")
    print("-" * 80)
    
    # Simulating BERT tokenization: "The movie was incredibly amazing"
    # → ["the", "movie", "was", "in", "##credibly", "amazing"]
    tokens = ["the", "movie", "was", "in", "##credibly", "amazing"]
    token_scores = [0.1, 0.3, 0.05, 0.6, 0.5, 0.9]
    
    print(f"\nTokens: {tokens}")
    print(f"Token scores: {token_scores}")
    
    # Detect word boundaries
    words = detect_word_boundaries(tokens)
    print(f"\nDetected {len(words)} words:")
    for i, word in enumerate(words):
        token_indices = word.token_indices
        word_tokens = [tokens[idx] for idx in token_indices]
        print(f"  Word {i+1}: '{word.word_text}'")
        print(f"    Tokens: {word_tokens} (indices: {token_indices})")
    
    # Aggregate scores
    word_spans, word_scores = aggregate_token_scores_to_words(
        token_scores, tokens, aggregation="mean"
    )
    
    print(f"\nWord-level scores (mean aggregation):")
    for word, score in zip(word_spans, word_scores):
        print(f"  '{word.word_text}': {score:.3f}")
    
    # Get top words
    top_words, top_scores, _ = get_top_words(
        token_scores, tokens, k=3, aggregation="mean"
    )
    
    print(f"\nTop 3 most important words:")
    for i, (word, score) in enumerate(zip(top_words, top_scores), 1):
        print(f"  {i}. '{word}': {score:.3f}")
    
    # Example 2: Different aggregation methods
    print()
    print("-" * 80)
    print("Example 2: Different Aggregation Methods")
    print("-" * 80)
    
    # Word "incredibly" is split into ["in", "##credibly"] with scores [0.6, 0.5]
    print(f"\nWord 'incredibly' = tokens ['in', '##credibly']")
    print(f"Token scores: [0.6, 0.5]")
    print()
    
    for agg_method in ["mean", "sum", "max", "first"]:
        word_spans, word_scores = aggregate_token_scores_to_words(
            token_scores, tokens, aggregation=agg_method
        )
        # Find "incredibly" in the results
        for word, score in zip(word_spans, word_scores):
            if word.word_text == "incredibly":
                print(f"  {agg_method:8s} → {score:.3f}")
                break
    
    print()
    print("Which aggregation to use?")
    print("  • mean:  Good default - averages across subwords")
    print("  • sum:   Favors longer words (more tokens)")
    print("  • max:   Takes highest-scoring subword")
    print("  • first: Uses only the first subword token")
    
    # Example 3: Practical use case
    print()
    print("-" * 80)
    print("Example 3: Practical Sentiment Analysis Example")
    print("-" * 80)
    
    # Sentence: "This is an absolutely fantastic and incredible film"
    # Tokens might be: ["this", "is", "an", "absolute", "##ly", "fantastic", "and", "in", "##credible", "film"]
    tokens_real = ["this", "is", "an", "absolute", "##ly", "fantastic", "and", "in", "##credible", "film"]
    # High scores for positive words
    scores_real = [0.05, 0.02, 0.01, 0.7, 0.75, 0.95, 0.03, 0.8, 0.85, 0.4]
    
    print(f"\nSentence: 'This is an absolutely fantastic and incredible film'")
    print(f"Tokens: {tokens_real}")
    print(f"Scores: {[f'{s:.2f}' for s in scores_real]}")
    
    # Word-level aggregation
    word_spans, word_scores = aggregate_token_scores_to_words(
        scores_real, tokens_real, aggregation="mean"
    )
    
    print(f"\nWord-level results:")
    for word, score in zip(word_spans, word_scores):
        print(f"  '{word.word_text:12s}': {score:.3f}")
    
    # Top words
    top_words, top_scores, _ = get_top_words(
        scores_real, tokens_real, k=5, aggregation="mean"
    )
    
    print(f"\nTop 5 most important words (for sentiment):")
    for i, (word, score) in enumerate(zip(top_words, top_scores), 1):
        print(f"  {i}. '{word}': {score:.3f}")
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("✅ Word-level aggregation converts subword token scores to word scores")
    print("✅ Handles BERT WordPiece (##), GPT (Ġ), and SentencePiece (▁) tokens")
    print("✅ Supports multiple aggregation methods (mean, sum, max, first)")
    print("✅ Provides human-readable explanations")
    print()
    print("Token-level output:")
    print("  ['in', '##credible'] → scores: [0.8, 0.85]")
    print()
    print("Word-level output:")
    print("  'incredible' → score: 0.825 (mean)")
    print()
    print("This makes explanations much more interpretable! 🎉")
    print()


if __name__ == "__main__":
    main()





