# TokenSHAP Hyperparameter Advisor

This document describes the hyperparameter advisor for TokenSHAP, which automatically suggests optimal hyperparameters for each individual sentence based on its characteristics.

## Overview

Similar to the hyperparameter advisors implemented for SubgraphX and GraphSVX in the graph explainability modules, the TokenSHAP hyperparameter advisor analyzes each sentence and suggests appropriate hyperparameters to balance computational cost with explanation quality.

## Architecture

The implementation follows the same pattern as the graph explainer advisors:

```
src/explain/llm/
├── hyperparam_advisor.py      # Core advisor implementation
├── token_shap_runner.py        # Integration with TokenSHAP
├── main.py                     # CLI entry points
├── test_hyperparam_advisor.py  # Test suite
└── README_HYPERPARAM_ADVISOR.md
```

## Key Components

### 1. SentenceStats

Analyzes sentence characteristics similar to `GraphStats`:

```python
@dataclass(frozen=True)
class SentenceStats:
    num_tokens: int
    num_chars: int
    avg_token_length: float
    max_token_length: int
```

**Properties:**
- `is_very_short`: ≤4 tokens
- `is_short`: 5-8 tokens
- `is_medium`: 9-16 tokens
- `is_long`: >16 tokens
- `has_subword_tokens`: Heuristic for subword tokenization

### 2. ModelSpec

Minimal model description for tuning:

```python
@dataclass(frozen=True)
class ModelSpec:
    base_model_name: str
    num_labels: int
    max_length: int
```

### 3. DatasetContext

Dataset-level metadata:

```python
@dataclass(frozen=True)
class DatasetContext:
    dataset: str
    task_type: str
    backbone: str
```

### 4. TokenSHAPHyperparameterAdvisor

Main advisor class that suggests hyperparameters based on sentence characteristics.

## Hyperparameters

### sampling_ratio (float)

Controls what fraction of all possible token combinations to sample.

**Strategy:**
- Very short sentences (≤4 tokens): 0.8 (80% coverage)
- Short sentences (5-8 tokens): 0.5-0.6
- Medium sentences (9-12 tokens): 0.1-0.2
- Long sentences (13-16 tokens): 0.01-0.05
- Very long sentences (>16 tokens): 0.001-0.005

**Adjustments:**
- Subword tokens: Reduce by 10% (more redundancy)
- Multi-class (>2 classes): Increase by 10% (more variation needed)

**Rationale:** The combinatorial space is 2^N where N is the number of tokens. For N=10, this is 1,024 combinations; for N=20, it's over 1 million. Aggressive reduction is necessary for longer sentences.

### num_samples_override (int)

Explicit number of samples to use, overriding the sampling ratio.

**Strategy:**
- ≤6 tokens: 128 samples
- 7-8 tokens: 256 samples
- 9-10 tokens: 512 samples
- 11-12 tokens: 768 samples
- 13-14 tokens: 1,024 samples
- 15-16 tokens: 1,280 samples
- >16 tokens: 1,536 samples

**Caps:**
- Maximum: 2,048 samples
- Minimum: 32 samples
- Never exceeds 2^num_tokens

**Adjustments:**
- Multi-class: Increase by 10% per additional class
- Subword tokens: Reduce by 10%

### top_k_nodes (int)

Number of top important tokens to highlight.

**Strategy:**
- Very short (≤4 tokens): ~75% of tokens
- Short (5-8 tokens): ~60% of tokens
- Medium (9-12 tokens): ~50% of tokens
- Long (13-16 tokens): ~40% of tokens (max 8)
- Very long (17-20 tokens): ~30% of tokens (max 10)
- Extremely long (>20 tokens): ~25% of tokens (max 12)

## Usage

### Basic Usage (Integrated)

The advisor is automatically used when running TokenSHAP explanations:

```python
from src.explain.llm.config import LLMExplainerRequest, build_default_profiles
from src.explain.llm.model_loader import load_finetuned_model, load_dataset_split
from src.explain.llm.token_shap_runner import token_shap_explain

# Load configuration
profiles = build_default_profiles()
profile = profiles["stanfordnlp/sst2"]

# Load model and data
model_bundle = load_finetuned_model(profile)
dataset = load_dataset_split(profile)

# Create request
request = LLMExplainerRequest(profile=profile, max_samples=10)

# Run with advisor (default)
records, summaries, *paths = token_shap_explain(
    request=request,
    model_bundle=model_bundle,
    dataset=dataset,
    use_advisor=True,  # This is the default
)
```

### Collect Hyperparameters Only

To analyze what hyperparameters would be suggested without running full explanations:

```python
from src.explain.llm.token_shap_runner import collect_token_shap_hyperparams

# Collect hyperparameters for all samples
output_path, per_sample = collect_token_shap_hyperparams(
    request=request,
    model_bundle=model_bundle,
    dataset=dataset,
    max_samples=100,
)

# Output saved to: outputs/insights/LLM/{backbone}/{dataset}/hyperparams/token_shap_hyperparams.json
```

### Using Locked Parameters

Override specific hyperparameters while keeping others adaptive:

```python
from src.explain.llm.hyperparam_advisor import TokenSHAPHyperparameterAdvisor

advisor = TokenSHAPHyperparameterAdvisor(
    model_spec=model_spec,
    context=context,
    locked_params={
        "sampling_ratio": 0.2,  # Fixed at 20%
        # top_k_nodes and num_samples_override remain adaptive
    },
)
```

### CLI Usage

```bash
# Run explanations with advisor
python -m src.explain.llm.main explain stanfordnlp/sst2 --max-samples 50

# Disable advisor (use fixed sampling profile)
python -m src.explain.llm.main explain stanfordnlp/sst2 --no-advisor --sampling-ratio 0.1

# Collect hyperparameters
python -m src.explain.llm.main hyperparams stanfordnlp/sst2 --max-samples 100

# For AG News (4 classes)
python -m src.explain.llm.main explain setfit/ag_news --max-samples 50
```

## Output Format

### Hyperparameters JSON

The `collect_token_shap_hyperparams` function produces a JSON file with this structure:

```json
{
  "method": "token_shap",
  "dataset": "stanfordnlp/sst2",
  "backbone": "stanfordnlp",
  "split": "validation",
  "max_tokens": 21,
  "max_length": 128,
  "num_labels": 2,
  "base_model_name": "google-bert/bert-base-uncased",
  "total_samples": 100,
  "per_sample": [
    {
      "sample_index": 0,
      "num_tokens": 8,
      "num_chars": 42,
      "avg_token_length": 5.25,
      "max_token_length": 9,
      "hyperparams": {
        "sampling_ratio": 0.5,
        "num_samples_override": 256,
        "top_k_nodes": 5
      },
      "prompt": "it's a charming and often affecting journey..."
    }
  ]
}
```

## Comparison with Graph Advisors

| Aspect | SubgraphX/GraphSVX | TokenSHAP |
|--------|-------------------|-----------|
| Input | Graph (nodes, edges) | Sentence (tokens) |
| Stats | num_nodes, avg_degree, density | num_tokens, avg_token_length |
| Main challenge | Graph structure complexity | Combinatorial explosion (2^N) |
| Key parameter | max_nodes, num_hops | sampling_ratio |
| Tuning basis | Graph topology | Sentence length |
| Per-sample | Yes | Yes |

## Design Rationale

### Why Per-Sample Hyperparameters?

Just as graphs vary greatly in size and structure, sentences vary in:
- **Length**: 3 tokens vs. 20 tokens
- **Complexity**: Simple vs. compound sentences
- **Tokenization**: Full words vs. subword pieces

A fixed sampling ratio is inefficient:
- Too high for long sentences → Memory explosion
- Too low for short sentences → Poor coverage

### Why Similar to Graph Advisors?

The pattern is proven effective:
1. **Modularity**: Advisor is separate from main explainer logic
2. **Flexibility**: Supports locked parameters for experiments
3. **Transparency**: Outputs suggested hyperparameters for analysis
4. **Consistency**: Same API pattern across explanation methods

## Testing

Run the test suite:

```bash
python -m src.explain.llm.test_hyperparam_advisor
```

This will:
1. Test SentenceStats computation
2. Test hyperparameter suggestions for various sentence lengths
3. Compare binary vs. multi-class settings
4. Test locked parameter functionality
5. Save results to `test_hyperparam_results.json`

## Integration Points

The advisor integrates at these points in the codebase:

1. **token_shap_runner.py**: 
   - `token_shap_explain()`: Uses advisor for per-sample hyperparameters
   - `collect_token_shap_hyperparams()`: Collects hyperparameters without explaining

2. **main.py**: 
   - CLI commands for running explanations and collecting hyperparameters

3. **config.py**: 
   - `TOKEN_SHAP_DEFAULTS`: Default hyperparameter values

## Future Enhancements

Potential improvements:
1. **Learning-based tuning**: Use historical data to refine suggestions
2. **Task-specific profiles**: Different defaults for sentiment vs. topic classification
3. **Budget constraints**: Optimize for fixed time/memory budget
4. **Adaptive sampling**: Adjust sampling during explanation based on variance

## References

- SubgraphX hyperparameter advisor: `src/explain/gnn/subgraphx/hyperparam_advisor.py`
- GraphSVX hyperparameter advisor: `src/explain/gnn/graphsvx/hyperparam_advisor.py`
- TokenSHAP paper: [Original TokenSHAP implementation](https://github.com/peterbhase/Token-SHAP)





