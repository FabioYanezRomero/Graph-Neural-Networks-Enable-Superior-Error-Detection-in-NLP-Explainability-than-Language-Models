<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# make sure the planning is across the 4 dimensions considered: AUC, fidelity, consistency and progression

Based on your paper's comprehensive evaluation framework, here's an experimental plan for error detection across the **four evaluation dimensions** you established: **AUC**, **Fidelity**, **Consistency**, and **Progression**.[^1]

## Experimental Plan: Multi-Dimensional Error Detection System

### **Overview**

Design a binary classifier that distinguishes correct from incorrect predictions using explainability-derived features from all four evaluation dimensions, demonstrating practical applicability of your theoretical findings.[^1]

***

## **Phase 1: Feature Extraction Across Four Dimensions**

For each prediction in AG News test set and SST-2 validation set, extract features organized by dimension:

### **Dimension 1: AUC-Based Features (Error Detection Capacity)**

**Deletion \& Insertion Metrics**:[^1]

- **Deletion AUC**: $\text{AUC}_{\text{del}} = \int_0^1 \text{conf}(x \setminus \text{features}_{1:k}) \, dk$
- **Insertion AUC**: $\text{AUC}_{\text{ins}} = \int_0^1 \text{conf}(x_{\text{neutral}} \cup \text{features}_{1:k}) \, dk$
- **AUC Separation**: $\Delta\text{AUC} = \text{AUC}^{\text{incorrect}}_{\text{del}} - \text{AUC}^{\text{correct}}_{\text{del}}$

**Key Discriminative Power**: Your results show GraphSVX achieves 60-70 percentage point Deletion AUC separation between correct/incorrect predictions, making this your strongest single feature.[^1]

### **Dimension 2: Fidelity-Based Features (Behavioral Faithfulness)**

**Necessity \& Sufficiency Metrics**:[^1]

- **Deletion Fidelity**: $M^- = \text{conf}(x) - \text{conf}(x \setminus \text{features}_{1:K})$
- **Insertion Fidelity**: $M^+ = \text{conf}(x_{\text{neutral}} \cup \text{features}_{1:K}) - \text{conf}(x_{\text{neutral}})$
- **Asymmetry Index**: $A = \frac{M^- - M^+}{M^- + M^+}$, range $[-1, 1]$

**Architectural Signature**: GNNs show pronounced deletion > insertion asymmetry ($A \gg 0$) for correct predictions, while incorrect predictions show balanced profiles.[^1]

**Quadrant Assignment**:

- Sufficient-Necessary: High $M^+$ AND high $M^-$ → **85-92% correct predictions cluster here**[^1]
- Sufficient-Redundant: High $M^+$, low $M^-$
- Insufficient-Necessary: Low $M^+$, high $M^-$
- Insufficient-Redundant: Low $M^+$ AND low $M^-$


### **Dimension 3: Consistency Features (Prediction Outcome Stability)**

**Margin Analysis Across Perturbations**:[^1]

- **Origin Margin**: $M_{\text{origin}} = P(y|x) - P(y_2|x)$
- **Masked Margin**: $M_{\text{masked}} = P(y|x_{\text{masked}}) - P(y_2|x_{\text{masked}})$
    - Only important features retained
- **Maskout Margin**: $M_{\text{maskout}} = P(y|x_{\text{maskout}}) - P(y_2|x_{\text{maskout}})$
    - Important features removed

**Derived Features**:

- **Margin Collapse**: $M_{\text{origin}} - M_{\text{maskout}}$ (sharp collapse → correct prediction)
- **Margin Preservation**: $M_{\text{masked}} / M_{\text{origin}}$ (high ratio → features sufficient)
- **Consistency Ratio**: $\frac{M_{\text{maskout}}}{M_{\text{masked}}}$ (approaching 0 → strong necessity signal)

**Expected Patterns**: Correct predictions show "catastrophic margin collapse" when necessary features removed; incorrect predictions maintain higher maskout margins.[^1]

### **Dimension 4: Progression Features (Feature Ranking Stability)**

**Importance Concentration Metrics**:[^1]

**A. Maskout Drop Progression** at thresholds $k \in \{1, 3, 5, 10\}$:
$M^-(k) = P(y|x) - P(y|x \setminus \text{features}_{1:k})$

**B. Sufficiency Drop Progression** at same thresholds:
$M^+(k) = P(y|x_{\text{neutral}} \cup \text{features}_{1:k}) - P(y|x_{\text{neutral}})$

**C. Cumulative Importance Concentration**:
$C(N) = \frac{\sum_{i=1}^N \text{importance}_i}{\sum_{i=1}^K \text{importance}_i}$

- Compute at $N \in \{1, 3, 5, 10\}$

**Derived Metrics**:

- **Early Concentration Rate**: $C(1)$ (GraphSVX: 50-55% vs TokenSHAP: 38% )[^1]
- **Progression Steepness**: $\frac{C(3) - C(1)}{2}$ (measures feature concentration gradient)
- **Monotonicity Score**: Correlation coefficient between rank position and drop magnitude
- **Divergence Score**: $|M^-(k) - M^+(k)|$ averaged across thresholds (discrete architectures should show $\approx 0$ divergence [^1])

**Key Finding**: GraphSVX achieves **2.4-2.5× faster feature concentration** than TokenSHAP, with maskout and sufficiency progressions remaining virtually identical (divergence < 2%).[^1]

***

## **Phase 2: Feature Engineering \& Transformation**

### **2.1 Dimension-Specific Feature Sets**

Create modular feature groups:

**AUC Feature Set** (3 features):

```python
auc_features = [
    'deletion_auc',
    'insertion_auc', 
    'auc_separation'
]
```

**Fidelity Feature Set** (4 features):

```python
fidelity_features = [
    'deletion_fidelity',      # M^-
    'insertion_fidelity',     # M^+
    'asymmetry_index',        # A
    'quadrant_label'          # Categorical: 4 quadrants
]
```

**Consistency Feature Set** (6 features):

```python
consistency_features = [
    'origin_margin',
    'masked_margin',
    'maskout_margin',
    'margin_collapse',        # origin - maskout
    'margin_preservation',    # masked / origin
    'consistency_ratio'       # maskout / masked
]
```

**Progression Feature Set** (11 features):

```python
progression_features = [
    'maskout_drop_k1', 'maskout_drop_k3', 'maskout_drop_k5', 'maskout_drop_k10',
    'sufficiency_drop_k1', 'sufficiency_drop_k3', 'sufficiency_drop_k5', 'sufficiency_drop_k10',
    'concentration_top1', 'concentration_top3', 'concentration_top5', 'concentration_top10',
    'progression_steepness',
    'monotonicity_score',
    'divergence_score'
]
```


### **2.2 Interaction Features**

Capture cross-dimensional relationships:

**AUC × Fidelity**:

- `deletion_auc × deletion_fidelity`
- `asymmetry_index × auc_separation`

**Consistency × Progression**:

- `margin_collapse × concentration_top1`
- `maskout_margin × divergence_score`

**Fidelity × Consistency**:

- `asymmetry_index × consistency_ratio`
- `quadrant × margin_collapse` (categorical interaction)

***

## **Phase 3: Multi-Level Classifier Architecture**

### **3.1 Architecture Options**

**Option A: Ensemble of Dimension-Specific Classifiers**

Train 4 separate classifiers, one per dimension:

```python
clf_auc = LogisticRegression()        # On AUC features only
clf_fidelity = RandomForest()         # On Fidelity features only  
clf_consistency = XGBoost()           # On Consistency features only
clf_progression = GradientBoosting()  # On Progression features only

# Final prediction via weighted voting
final_pred = weighted_vote([clf_auc, clf_fidelity, clf_consistency, clf_progression])
```

**Weights determined by**:

- Per-dimension AUROC on validation set
- Your empirical findings (e.g., AUC dimension shows strongest separation )[^1]

**Option B: Hierarchical Feature Fusion**

```python
# Stage 1: Dimension-level embeddings
embed_auc = MLP(auc_features) → 8 dims
embed_fidelity = MLP(fidelity_features) → 8 dims
embed_consistency = MLP(consistency_features) → 8 dims
embed_progression = MLP(progression_features) → 8 dims

# Stage 2: Cross-dimensional attention
attention_weights = SoftAttention([embed_auc, embed_fidelity, embed_consistency, embed_progression])
fused_repr = Σ(attention_weights * embeds)

# Stage 3: Final classification
error_prob = Sigmoid(Linear(fused_repr))
```

**Option C: Unified Gradient Boosting**

Single XGBoost/LightGBM model on all 24+ features:

```python
all_features = auc_features + fidelity_features + consistency_features + progression_features
clf = XGBoost(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=200,
    objective='binary:logistic'
)
clf.fit(all_features, is_error_label)
```

Advantages: Built-in feature importance ranking, handles interactions automatically.

### **3.2 Training Protocol**

**Data Splitting**:

- **70% meta-training**: Train error detector
- **15% meta-validation**: Tune hyperparameters, select model
- **15% meta-test**: Final evaluation

**Class Balancing**:

- Your GNN models achieve 93.8-96.5% accuracy on AG News, 91.7-94.3% on SST-2[^1]
- This creates severe class imbalance (errors are minority class)
- Apply **SMOTE** or **class weighting** to balance training

**Cross-Validation**:

- Stratified 5-fold CV on meta-training set
- Stratify by: dataset, class label, correctness
- Ensures each fold maintains representative distributions

***

## **Phase 4: Evaluation Protocol Across Dimensions**

### **4.1 Overall Performance Metrics**

- **AUROC**: Threshold-independent discrimination
- **AUPRC**: Critical given class imbalance
- **Precision@k**: For top-k flagged errors (k=50, 100, 200)
- **Recall@90% Precision**: How many errors caught at 90% precision threshold?


### **4.2 Dimension-Specific Ablation**

Measure contribution of each dimension:


| **Features Used** | **AUROC** | **AUPRC** | **F1** |
| :-- | :-- | :-- | :-- |
| AUC only | ? | ? | ? |
| Fidelity only | ? | ? | ? |
| Consistency only | ? | ? | ? |
| Progression only | ? | ? | ? |
| AUC + Fidelity | ? | ? | ? |
| AUC + Consistency | ? | ? | ? |
| AUC + Progression | ? | ? | ? |
| All 4 dimensions | ? | ? | ? |

**Expected Hierarchy** based on your findings:[^1]

1. **AUC dimension**: Strongest individual predictor (60-70 pt separation)
2. **Fidelity dimension**: Strong quadrant clustering (85-92% correct in Sufficient-Necessary)
3. **Progression dimension**: 2.4-2.5× concentration advantage
4. **Consistency dimension**: Sharp margin collapse for correct predictions

### **4.3 Stratified Analysis**

Following your paper's methodology, stratify error detection performance by:[^1]

**By Graph Type** (for GNN):

- Constituency (SubgraphX)
- Dependency/Syntactic (SubgraphX)
- Window (GraphSVX)
- Skip-gram (GraphSVX)

**By Architecture**:

- GNN + GraphSVX/SubgraphX vs LLM + TokenSHAP
- Expected: GNN shows **48% stronger error detection signal**[^1]

**By Dataset**:

- AG News (4-class, longer text)
- SST-2 (2-class, shorter text)

**By Class Label**:

- Per-class error detection AUROC
- Check for class-dependent bias patterns

**By Confidence Buckets**:

- High confidence (>0.9): Can we detect over-confident errors?
- Medium confidence (0.7-0.9)
- Low confidence (<0.7)

***

## **Phase 5: Dimension-Specific Insights**

### **5.1 AUC Dimension Analysis**

**Research Questions**:

- Does Deletion AUC alone achieve competitive error detection?
- What's the optimal threshold for Deletion AUC to flag errors?
- How does AUC separation vary by graph type?

**Visualization**:

- Deletion AUC distributions: correct vs incorrect[^1]
- ROC curves per graph type
- Threshold sweep analysis


### **5.2 Fidelity Dimension Analysis**

**Research Questions**:

- Does quadrant membership predict errors effectively?
- What % of errors fall outside Sufficient-Necessary quadrant?
- Does asymmetry index $A$ differ significantly for correct/incorrect?

**Visualization**:

- Scatter: $M^+$ vs $M^-$, colored by correctness[^1]
- Quadrant population heatmaps
- Asymmetry index distributions


### **5.3 Consistency Dimension Analysis**

**Research Questions**:

- Does margin collapse magnitude predict correctness?
- Can we detect errors via consistency ratio alone?
- How does margin preservation differ across architectures?

**Visualization**:

- Margin trajectory plots: origin → masked → maskout
- Collapse magnitude histograms by correctness
- Consistency ratio vs error probability


### **5.4 Progression Dimension Analysis**

**Research Questions**:

- Does early concentration ($C(1)$) reliably predict correctness?
- At what threshold $k$ does progression best discriminate errors?
- Does monotonicity correlate with prediction confidence?

**Visualization**:

- Concentration curves: $C(k)$ for $k=1..10$
- Steepness distributions by correctness
- Divergence score vs AUROC correlation

***

## **Phase 6: Cross-Dimensional Interaction Analysis**

### **6.1 Dimension Correlation Study**

Compute pairwise correlations between dimension-level scores:

```python
dimension_scores = {
    'auc_score': deletion_auc,
    'fidelity_score': asymmetry_index,
    'consistency_score': margin_collapse,
    'progression_score': concentration_top1
}

corr_matrix = np.corrcoef(list(dimension_scores.values()))
```

**Expected**: Low correlation → dimensions capture orthogonal information.[^1]

### **6.2 Dimension Agreement Analysis**

For each prediction, classify error likelihood by each dimension independently:


| **Prediction** | **AUC Says** | **Fidelity Says** | **Consistency Says** | **Progression Says** | **Ground Truth** |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Sample 1 | Error | Correct | Error | Error | Error (3/4 agree) |
| Sample 2 | Correct | Correct | Correct | Correct | Correct (unanimous) |
| Sample 3 | Error | Error | Correct | Error | Correct (majority wrong) |

**Analysis Questions**:

- What % of errors have unanimous agreement across all 4 dimensions?
- Which dimension is most reliable when others disagree?
- Do certain error types confuse specific dimensions?

***

## **Phase 7: Practical Application Scenarios**

### **7.1 Confidence Calibration**

Use error detection score to adjust model confidence:

```python
calibrated_conf = original_conf * (1 - error_probability)
```

**Evaluation**: Compare calibration metrics (ECE, Brier score) before/after.

### **7.2 Active Learning Pipeline**

**Selection Strategies**:

1. **High Error Probability**: $P(\text{error}) > 0.7$
2. **Unanimous Dimension Agreement**: All 4 dimensions flag as error
3. **High Confidence + High Error Score**: Over-confident mistakes
4. **Dimension Disagreement**: 2 dimensions say error, 2 say correct → boundary cases

**Evaluation**: Measure label efficiency (accuracy improvement per labeled sample).

### **7.3 Model Debugging Dashboard**

Create interpretable error profiles:

```
Prediction ID: 12345
True Label: Sports | Predicted: Business | Confidence: 0.87

Error Detection Score: 0.92 (HIGH RISK)

Dimension Breakdown:
├─ AUC: Deletion AUC = 0.23 (vs. 0.78 avg for correct) → STRONG ERROR SIGNAL
├─ Fidelity: Quadrant = Sufficient-Redundant (85% errors here) → ERROR LIKELY
├─ Consistency: Margin collapse = 0.12 (vs. 0.89 avg) → WEAK ERROR SIGNAL  
└─ Progression: Concentration@top1 = 0.31 (vs. 0.54 avg) → MODERATE ERROR SIGNAL

Verdict: 3/4 dimensions indicate error. Recommend human review.
```


***

## **Phase 8: Expected Outcomes \& Validation**

### **8.1 Quantitative Expectations**

Based on your findings:[^1]

**GNN-based Error Detection**:

- Overall AUROC: **0.75-0.85** (AG News), **0.70-0.80** (SST-2)
- AUC dimension alone: **0.70-0.78** AUROC
- All 4 dimensions: **+8-12% improvement** over AUC alone
- Precision@100: **≥ 80%** (i.e., 80+ flagged predictions are actual errors)

**LLM Baseline Performance**:

- Overall AUROC: **0.55-0.65** (48% weaker than GNN )[^1]
- Minimal improvement from multi-dimensional features (distributed importance obscures signals)

**Dimension Contributions**:

- AUC: **40-45%** of total predictive power
- Fidelity: **25-30%**
- Consistency: **15-20%**
- Progression: **10-15%**


### **8.2 Qualitative Validation**

**Case Studies**:

- Analyze 50 high-error-score predictions manually
- Confirm architectural patterns match theoretical predictions[^1]
- Document failure modes (which errors escape all 4 dimensions?)

**Error Taxonomy**:

- **Type I**: All 4 dimensions correctly identify (easiest)
- **Type II**: 3/4 dimensions identify (moderate)
- **Type III**: 2/4 dimensions identify (challenging)
- **Type IV**: ≤1 dimension identifies (hardest, potential model limitations)

***

## **Implementation Pseudocode**

```python
# Phase 1: Feature Extraction
for sample in dataset:
    # Dimension 1: AUC
    features['deletion_auc'] = compute_deletion_auc(sample)
    features['insertion_auc'] = compute_insertion_auc(sample)
    
    # Dimension 2: Fidelity
    features['M_minus'] = compute_deletion_fidelity(sample)
    features['M_plus'] = compute_insertion_fidelity(sample)
    features['asymmetry'] = (M_minus - M_plus) / (M_minus + M_plus)
    features['quadrant'] = assign_quadrant(M_plus, M_minus)
    
    # Dimension 3: Consistency
    features['origin_margin'] = P(y|x) - P(y2|x)
    features['masked_margin'] = P(y|x_masked) - P(y2|x_masked)
    features['maskout_margin'] = P(y|x_maskout) - P(y2|x_maskout)
    features['margin_collapse'] = origin_margin - maskout_margin
    
    # Dimension 4: Progression
    for k in [1, 3, 5, 10]:
        features[f'maskout_k{k}'] = compute_maskout_drop(sample, k)
        features[f'sufficiency_k{k}'] = compute_sufficiency_drop(sample, k)
        features[f'concentration_top{k}'] = compute_concentration(sample, k)

# Phase 2: Train Error Detector
X_train, y_train = prepare_features(train_samples)  # y_train: 1=error, 0=correct
error_detector = XGBoost(...)
error_detector.fit(X_train, y_train)

# Phase 3: Evaluate Across Dimensions
for dimension in ['auc', 'fidelity', 'consistency', 'progression']:
    features_subset = select_dimension_features(dimension)
    auroc = evaluate_classifier(features_subset, test_samples)
    print(f"{dimension} AUROC: {auroc}")

# Phase 4: Stratified Analysis
for graph_type in ['constituency', 'dependency', 'window', 'skipgram']:
    for dataset in ['ag_news', 'sst2']:
        subset = filter_samples(graph_type, dataset)
        performance = evaluate_error_detection(subset)
        report_results(graph_type, dataset, performance)
```


***

## **Summary: Four-Dimensional Error Detection Framework**

This experimental plan operationalizes your theoretical findings  into a practical error detection system by:[^1]

1. **AUC Dimension**: Leverages 60-70 pt Deletion AUC separation as primary signal
2. **Fidelity Dimension**: Exploits 85-92% clustering in Sufficient-Necessary quadrant
3. **Consistency Dimension**: Uses catastrophic margin collapse for correct predictions
4. **Progression Dimension**: Harnesses 2.4-2.5× faster feature concentration in GNNs

The multi-dimensional approach ensures robust, interpretable error detection that reflects fundamental architectural differences between discrete graph structures and continuous token distributions, validated across datasets, graph types, and class labels.

<div align="center">⁂</div>

[^1]: UPDATED__From_text_to_Graph-11.pdf

