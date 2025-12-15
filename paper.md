# Graph Neural Networks Enable Superior Error

# Detection in NLP Explainability than Language Models

```
Fabio Yáñez-Romeroa,1,∗, Andrés Montoyob,1, Armando Suárezb,1, Yoan
Gutiérrezb,1, Ruslan Mitkovc,
aUniversity Institute for Computer Research, University of Alicante, Carretera San Vicente del
Raspeig s/n, 03690 San Vicente del Raspeig, Alicante, Spain
bDepartment of Computing Information Systems, University of Alicante, Carretera San Vicente
del Raspeig s/n, 03690 San Vicente del Raspeig, Alicante, Spain
cLancaster University, Bailrigg, Lancaster, LA1 4YW, United Kingdom
```
∗Corresponding author: Fabio Yáñez-Romero.
Email addresses: fabio.yanez@ua.es (Fabio Yáñez-Romero),
montoyo@dlsi.ua.es (Andrés Montoyo), armando@dlsi.ua.es (Armando Suárez),
ygutierrez@dlsi.ua.es (Yoan Gutiérrez), r.mitkov@lancaster.ac.uk (Ruslan
Mitkov)


Abstract

Large language models have become the dominant paradigm in Natural Language
Processing. Yet, their explainability remains fundamentally limited by the token-
level interactions enabled by the attention mechanism, making it difficult to detect
errors automatically in the model answers. This study conducts a systematic com-
parison of explainability signatures between discrete graph-based and continuous
token-based architectures for error detection in text classification. Using meth-
ods such as text-to-graph conversion, LLM-as-teacher distillation, and unified
Shapley-based attribution, we demonstrate that GNN-based explainers achieve
substantially superior error-detection performance (99.7–100.0% accuracy) com-
pared to LLM-based explainers (88.1–89.6% accuracy), under a fair comparison.
This performance gap persists across datasets, graph topologies, and prediction
correctness conditions.
These findings highlight core architectural differences: discrete graph struc-
tures encode necessity through node participation and prioritise non-redundant
features. In contrast, continuous token embeddings encode sufficiency, main-
taining predictions via redundant pathways. For high-stakes applications where
transparent error detection is critical, graph-based explainability offers clear ad-
vantages over LLM-based approaches, even when LLMs yield high attribution
confidence.

Keywords: Graph Neural Networks, Explainability, Error Detection, Natural
Language Processing

1. Introduction

Large Language Models (LLMs) have become the cornerstone of contempo-
rary natural language processing, enabling state-of-the-art performance across a
wide range of text understanding and classification tasks. This progress has mo-
tivated the design and broad adoption of explainability modules around LLMs,
which quantify the importance of individual tokens or words in model predictions
and offer rapid, localised interpretability.
Despite these strengths, the interpretability of LLM decisions remains funda-
mentally constrained by transformer architecture, including tokenisation fragmen-
tation, distributed representations, and local attribution scope, as further discussed
in Section 2.1. These limitations motivate exploring discrete graph-based archi-


tectures as alternatives to continuous token representations for transparent error
detection.
We evaluate on AG-News and SST-2 as foundational testbeds, establishing
graph-based explainability capabilities on general-domain benchmarks before fu-
ture adoption in specialised high-stakes applications.

1.1. Fundamental Hypothesis and Research Questions

We hypothesise that discrete graph-based architectures and continuous
token-based architectures produce fundamentally different explanation sig-
natures. This core hypothesis generates four specific, testable predictions:

1. Error Detection Capacity—GNNs show strong Deletion and Insertion Area
    Under the Curve (AUC) separation between correct and incorrect predic-
    tions. In contrast, LLMs show weaker separation due to multi-pathway
    contributions.
2. Feature Ranking Stability—GNNs exhibit steep, monotonic confidence
    degradation reflecting discrete message passing, while LLMs show gradual,
    distributed degradation.
3. Consistency across Prediction Outcomes—GNNs demonstrate sharp mar-
    gin collapse when necessary features are removed, while LLMs show softer
    degradation
4. Behavioural Faithfulness—GNNs exhibit pronounced deletion-insertion
    asymmetry (M− ≫ M+) due to binary feature participation in discrete
    structures.

By stratifying analysis across graph types, datasets, class labels, and prediction
correctness, we isolate whether observed differences reflect genuine architectural
properties or experimental confounds. If all four dimensions show consistent pat-
terns across these stratifications, we have strong evidence that architecture—not
task, dataset, or evaluation methodology—drives explanation behaviour.

1.2. Proposed Solution and Contributions

This study introduces a modular, architecture-neutral methodology for NLP
that achieves explainability and model traceability by transforming input text into
structured graph representations (constituency, dependency, window, skip-gram)


and systematically comparing their explanation signatures with those of LLM-
based alternatives through unified Shapley-based attribution.
Key contributions include:

- LLM-as-teacher distillation framework enabling GNN surrogates to repli-
    cate LLM predictions while offering topology-specific interpretability un-
    available to sequence models, paired with standardised cross-architecture
    evaluation via unified Shapley methods (SubgraphX, GraphSVX, Token-
    SHAP), ensuring methodological parity.
- Four-dimensional evaluation protocol isolating architectural effects from
    experimental confounds through stratification by graph type, dataset, class
    label, and prediction correctness—transforming vague architectural intu-
    itions into falsifiable predictions with complementary metrics (Deletion/Insertion
    AUC, progression patterns, margin collapse, fidelity asymmetry).
- Empirical demonstration that discrete graph structures achieve 99.7–100%
    error detection accuracy compared to 88.1–89.6% for continuous token dis-
    tributions, with this gap persisting across datasets, graph topologies, and
    prediction correctness conditions—revealing fundamental architectural dif-
    ferences in how necessity versus sufficiency are encoded.
2. Related Work

To contextualise the proposed modular pipeline, we review (I) recent advances
and limitations in language model explainability, highlighting evaluation chal-
lenges and the need for structured alternatives, (II) the state of explainability
methods for graph neural networks, with emphasis on recent benchmarking frame-
works that demonstrate advantages for structured inputs in both general-domain
and specialised NLP tasks, and (III) modern approaches for automatic, high-
fidelity structured parsing and text-to-graph translation.

2.1. Explainability in Language Models

The rise of large language models (LLMs) has brought urgency for trans-
parency, bias mitigation, and responsible AI deployment across domains Luo and
Specia (2024); Zhao et al. (2023); Palikhe et al. (2025). Despite their state-of-the-
art performance in generation and classification tasks Touvron et al. (2023), three
core architectural limitations fundamentally constrain transformer-based explain-
ability: (1) tokenisation fragmentation—subword tokenisation distributes word


meaning across multiple tokens, obscuring compositional semantic understand-
ing, particularly in morphologically complex or underrepresented languages Luo
and Specia (2024); Zhao et al. (2023); (2) distributed relational representa-
tion—transformers encode relational patterns through attention mechanisms dis-
tributed across layers and heads, requiring aggregation methods rather than direct
inspection to trace semantic or linguistic relations Kobayashi et al. (2020); and (3)
post-hoc locality—token-level importance attribution (via perturbation Modar-
ressi et al. (2023); Hooker et al. (2019), gradient-based methods Kindermans et al.
(2017), or activation analysis Geva et al. (2022)) effectively captures local influ-
ences but obscures higher-level structural reasoning, multi-hop relational infer-
ence, and compositional auditability Cambria et al. (2024); Dhaini et al. (2024);
Zhao et al. (2023).
Recent benchmarking efforts have systematically demonstrated these chal-
lenges. ALMANACS Mills et al. (2025) reveals that current LLM explanation
methods fail simulatability tests, where humans cannot reliably predict model
behaviour from provided explanations. Similarly, PRobELM Yuan et al. (2025)
shows that plausibility rankings for LLM explanations frequently diverge from
human intuitions, particularly for complex reasoning tasks. These evaluation
frameworks underscore the pressing need for more interpretable architectures and
structured explanation paradigms that bridge the gap between model internals and
human comprehension.
Structured graph-based representations have re-emerged as essential founda-
tions for improved explainability, offering node- and relation-level granularity that
enables explicit, traceable pathways between input elements and predicted out-
puts Palikhe et al. (2025); Luo and Specia (2024).

2.2. Explainability for Graph Neural Networks

Graph neural networks (GNNs) offer unique advantages for explainability due
to their inherently structured, node and edge-based architecture, where each node
and relationship corresponds to a human-understandable entity or feature Kakkad
et al. (2023); Agarwal et al. (2023). Within the factual family of GNN explainabil-
ity, perturbation-based methods have emerged as the dominant post-hoc paradigm Kosan
et al. (2024). These approaches infer the importance of graph components by sys-
tematically perturbing the input—adding or removing nodes, edges, or features—
and observing the effect on the model’s prediction Yuan et al. (2021); Duval and
Malliaros (2021); Kosan et al. (2024). Perturbation-based methods have emerged
as model-agnostic explanations grounded in empirical sensitivity analysis.


Among perturbation-based explainers, Shapley-value methods provide attri-
bution through cooperative game theory Yuan et al. (2021); Duval and Malliaros
(2021). In this sense, SubgraphX employs Monte Carlo Tree Search guided by
Shapley values to identify contiguous explanatory subgraphs in tree-structured
graphs Yuan et al. (2021), while GraphSVX generalises Shapley reasoning to dis-
tributed node coalitions suitable for non-hierarchical graph topologies Duval and
Malliaros (2021). This methodological diversity motivates modular approaches
that select explainers based on graph topology and task requirements Agarwal
et al. (2023); Bonabi Mobaraki and Khan (2023).
Recent systematic benchmarks—including GraphFramEx Amara et al. (2024)
and GNNX-BENCH Kosan et al. (2024)—demonstrate that, for structured inputs,
post-hoc GNN explainers provide explanations more aligned with explicit seman-
tic features and exhibiting greater stability across domain variations than token-
based neural explainers. However, explanation effectiveness depends critically
on the quality and semantic richness of the underlying graph representation Xue
et al. (2025); Cedro and Martens (2025), which motivates a systematic evaluation
across diverse text-to-graph conversion strategies.

2.3. Structured Graph Representations in NLP

Systematic text-to-graph conversion enables structured neural reasoning by
preserving linguistic relationships through explicit node-edge topologies. Hi-
erarchical representations—constituency and dependency trees—capture syntac-
tic structure through phrase boundaries and head-dependent relations, respec-
tively de Marneffe et al. (2021); Yang et al. (2020). Non-hierarchical representations—window-
based and skip-gram graphs—encode distributional semantics through co-occurrence
and positional proximity Brannon et al. (2024); Nonkes et al. (2024). Advances in
neural parsing Bauer and Manning (2025) and GNN-integrated workflows Yang
and Deng (2020) now enable reliable, large-scale graph extraction across lan-
guages and domains.
The architectural distinction between hierarchical and non-hierarchical graphs
directly influences explainability behaviour: tree structures impose discrete message-
passing paths where node removal severs information flow, while proximity graphs
distribute semantic relations across overlapping neighbourhoods Brannon et al.
(2024); Yang and Deng (2020). This topological diversity motivates modular ex-
plainability frameworks that adapt explanation methods to graph structure—SubgraphX
for contiguous subgraphs in trees and GraphSVX for distributed node coalitions
in non-hierarchical graphs—enabling the systematic evaluation of whether archi-


tectural differences in discrete versus continuous representations affect error de-
tection capacity across structural paradigms.
While prior work has independently demonstrated (1) limitations of token-
based explainability in transformers, (2) advantages of GNN explainers for struc-
tured inputs, and (3) diverse text-to-graph conversion methods, no systematic
comparison has evaluated if discrete graph architectures and continuous token
architectures produce different error detection signatures that can be used in
advantage. Existing benchmarks evaluate explainability methods within archi-
tectures but lack cross-architectural protocols that isolate whether observed dif-
ferences reflect genuine architectural properties—necessity encoding in discrete
structures versus sufficiency encoding in continuous representations—or merely
dataset-specific or method-specific artifacts. This study addresses this gap through
a unified Shapley-based evaluation framework with stratified analysis across graph
types, datasets, and prediction correctness, showing a larger margin in error de-
tection rate on GNNs compared to Language Models.

3. Methodology

Our methodology re-frames text classification as a graph-based modelling
and explainability challenge. Rather than relying exclusively on token-level or
transformer-based representations, we construct a range of graph structures de-
rived from text—constituency trees, syntactic trees, window graphs, and skip-
gram graphs—to preserve both compositional and contextual information at vary-
ing levels of granularity.
This approach enables categorisation of sentences as a graph classification
problem, leveraging advances in graph neural networks (GNNs) and the wealth of
resources from graph-based learning. We adopt an LLM-as-a-teacher framework
for GNNs, where a fine-tuned LLM provides node-level labels and embeddings.
A GNN surrogate is then trained to approximate the teacher’s outputs, offering a
more human-interpretable model whose internal reasoning can be more easily ex-
amined. This setup facilitates direct comparison between the LLM and the GNN,
as both are subjected to Shapley-based post-hoc analysis for attributing feature
importance. Building on recent advances in GNN distillation Mavromatis et al.
(2023); Li et al. (2024); Hu et al. (2024), our approach extends this paradigm to
multiple graph types while emphasising transparency and diagnostic interpretabil-
ity.
Figure 1 illustrates the overall methodology and internal workflow of our
modular framework, designed for flexible adaptation across diverse graph for-


mulations, GNN architectures, and post-hoc explainability modules. Each stage
contributes to building a transparent pipeline linking linguistic structure, model
behaviour, and interpretability outcomes.
The following subsections describe each component in detail:

1. Text-to-Graph Conversion: Systematic transformation of each dataset in-
    stance into multiple graph representations—constituency, dependency, window-
    based, and skip-gram graphs (3.1).
2. Language Model Fine-tuning and Embedding Generation: The LLM is
    first fine-tuned on the target classification task to align its representations
    with domain-specific semantics. Subsequently, we extract node-level em-
    beddings and the model’s predictions, which serve as supervisory signals
    for the GNN under the LLM-as-a-teacher paradigm (3.2).
3. Graph Neural Network Training: Specialise GNN architectures to ap-
    proximate the LLM’s outputs. LLM is used as a teacher for the obtained
    graphs, both using its specific embeddings and labels in the process (3.3).
4. Post-Hoc Explainability: Applying perturbation-based explainability tech-
    niques to both the GNN and the LLM, enabling a systematic comparison
    between their attribution behaviours under a unified mathematical principle
    (Shapley values). The resulting contributions are analysed at the node/word
    level, as each graph node corresponds to a token in the original text (3.4).
5. Explainability Dimensions Evaluation: Performing comparative distribu-
    tion analyses of attribution outputs from each model, integrating Shapley-
    based importance measures across all the different dimensions considered
    (3.5).

To validate the importance of the different dimensions considered in our evalu-
ation of the explanations, we trained one logistic regression model per architecture
and dataset, evaluating the performance and the relative weight of each dimension
in the process (see 3.6). A general overview of the entire process is shown in
Figure 2, indicating the interaction between the explainability modules and the
logistic regression for the given predictions.
The code for reproducing all experiments is available via GitHub^1.

(^1) Repository for Experiment Reproducibility


Figure 1: End-to-end pipeline: text instances are converted to multiple graph types, embedded
via fine-tuned LLM, trained as GNN surrogates via Graph Convolutional layers, and evaluated
through Shapley-based explainability under unified fairness constraints.

Figure 2: Each trained model is wrapped with their respective explainability module. From these
modules, we obtain agnostic metrics that can be compared across modules, those metrics are used
for train each logistic regression model for error detection based on our base model prediction.


3.1. Text-to-Graph Conversion

To enable text classification with graph neural networks, each sentence is sys-
tematically transformed into multiple graph representations capturing both hierar-
chical and non-hierarchical linguistic structures.
Hierarchical graphs (constituency and dependency trees) are extracted via
the Stanza library Qi et al. (2020). Dependency parsing extracts head-dependent
relations with morphological annotations (POS tags, lemmas, morphological fea-
tures), while constituency parsing produces Penn Treebank-style phrase struc-
tures. Both yield tree structures where nodes represent linguistically-enriched
words and edges encode hierarchical relationships.
Non-hierarchical graphs (window-based and skip-gram) capture distribu-
tional semantics through co-occurrence patterns. Window graphs connect tokens
within a specified context window; skip-gram graphs connect tokens separated by
fixed intervals. Nodes carry POS annotations, and edges represent proximity or
co-occurrence relations.
All graphs are configured with unique node identifiers and topology-appropriate
directionality, preserving explicit linguistic hierarchies and contextual patterns for
downstream explainability analysis (see Figure 1, Section 1).

3.2. Language Model Fine-tuning and Embedding Generation

Task-adapted embeddings are extracted from Transformer models fine-tuned
on classification targets. We use models with [CLS] tokens to capture global
sequence semantics, which also serve as embeddings for special tokens in con-
stituency graphs, ensuring consistent handling of structural information. Such
models as BERT Devlin et al. (2019) encode both sentence-level and word-level
features effectively (see Figure 1, Section 2).

3.3. Graph Neural Network Training

Graphs are converted to GNN-compatible tensor format with node embed-
dings extracted from the fine-tuned language model. Word nodes use averaged
subtoken hidden states, preserving semantic and positional information. Phrase
nodes in tree-based graphs use [CLS] token embeddings, capturing global se-
quence meaning. This unified pipeline ensures consistent node features across
all graph types.
Training Under LLM-as-Teacher Paradigm: The GNN is trained to repli-
cate the fine-tuned LLM’s predictions using Graph Convolutional Network (GCN)
layers Kipf and Welling (2017) for message passing, where each layer aggregates


neighbouring node representations via normalised graph convolution. LLM clas-
sifications serve as supervisory labels, transferring knowledge to each GNN. Iden-
tical training and evaluation metrics enable direct performance comparison. This
setup facilitates analysis of how linguistic structure and graph topology influence
explainability results (see Figure 1, Section 3).

3.4. Post-Hoc Explainability

Post-hoc explainability techniques identify which input elements have the most
significant influence on a model already trained. The selected explainers—SubgraphX
Yuan et al. (2021), GraphSVX Duval and Malliaros (2021), and TokenSHAP
Goldshmidt and Horovicz (2024)—cover complementary structural configurations
across graph and language representations, ensuring conceptual consistency and
faithfulness across modalities (see Figure 1, Section 4).

3.4.1. Explainability Modules
For tree-structured graphs (constituency and dependency), we adopt SubgraphX,
which uses Monte Carlo Tree Search to identify contiguous explanatory sub-
graphs that preserve syntactic coherence. For non-tree graphs (window-based and
skip-gram), we employ GraphSVX, which generalises Shapley-value reasoning
to distributed node coalitions suitable for semantic co-occurrence patterns. For
language models, we use TokenSHAP , aggregating token-level Shapley attribu-
tions at the word level for structural alignment with graph explainers. This uni-
fied Shapley-based foundation enables fair cross-architecture comparison (Section
3.4.2).

3.4.2. Cross-Model Fair Explainability

Our experimental design adapts GraphFramEx Amara et al. (2024) fairness
principles to multimodal settings following the M4 Benchmark Li et al. (2023).
The framework operates along five harmonised dimensions, ensuring evaluation
parity across LLMs and GNNs:

- Budget: 2000 forward passes per instance
- Sampling: Normalised to combinatorial domain (exponential for TokenSHAP,
    linear for GraphSVX, polynomial for SubgraphX)
- Sparsity: Fixed at 20% of input elements (tokens/nodes)
- Context: Aligned dependency fields (full self-attention for LLMs ⇔ two-hop
    receptive fields for GNNs)
- Granularity: Atomic perturbations or single-node expansions per step


Appendix B details dimension specifications, isolating intrinsic explainer be-
haviour from computational asymmetries as well as the general nomenclature
across models and architectures for the explainability modules.

3.5. Multidimensional Evaluation Framework

This framework evaluates whether explanations enable reliable error detection
through inherent structural properties across four complementary dimensions (see
Figure 1, Section 5).

3.5.1. Stratified Evaluation Protocol
We employ stratified evaluation across five complementary dimensions to iso-
late explainability patterns from dataset and model-specific biases: (1) By Ex-
plainability Module (SubgraphX for tree graphs, GraphSVX for non-tree graphs,
TokenSHAP for LLMs); (2) By Input Construction Method (syntactic, con-
stituency, window-based, skipgram); (3) By Dataset (AG-News, SST-2); (4) By
Class Label (per-class stratification); (5) By Correctness (correct vs. incorrect
predictions). Core terminology is standardised in Table A.7, enabling direct met-
ric comparison between heterogeneous architectures.

3.5.2. Evaluated Dimensions
Table 1 summarises the four evaluation dimensions. Mathematical formula-
tions and detailed metrics are provided in Appendix B.


Table 1: Four-dimensional evaluation framework for cross-architecture explainability comparison

```
Dimension Objective Methodology Interpretation
AUC Discriminative
Capacity
```
```
Measure whether aggregate
AUC values—computed by
progressively adding (inser-
tion) or removing (deletion)
features—can discriminate be-
tween correct and incorrect
predictions
```
```
Compute per-instance Deletion
AUC (M−) and Insertion AUC
(M+) as holistic metrics inte-
grating confidence changes. Ap-
ply thresholds [0.1, ..., 1.0] to
AUC distributions and measure
error/correctness detection rates
```
```
AUC distributions that dif-
fer significantly between
correct and incorrect
predictions enable reliable
threshold-based error
detection.
```
```
Feature Ranking
Stability
```
```
Verify consistency in impor-
tance concentration according
to the top-k features
```
```
Measure sufficiency drop pro-
gression (Mk−) and maskout
drop progression (Mk+) for k ∈
{ 1 , 3 , 5 , 10 }
```
```
Progression patterns re-
veal importance distribu-
tion: steep drops indi-
cate concentrated features
(discrete structures), grad-
ual decay indicates dis-
tributed importance (con-
tinuous representations)
Consistency across
Outcomes
```
```
Investigate whether explana-
tions preserve decision confi-
dence via margin between top
class and second candidate
```
```
Compute origin, masked, and
maskout margins using confi-
dence difference between top
class and second-highest class
```
```
If features are necessary,
maskout margin collapses
sharply while masked
margin remains elevated.
Correct predictions show
robust margins; incorrect
predictions show brittle
margins
Behavioural Faith-
fulness
```
```
Quantify whether features are
causally necessary and suffi-
cient through complementary
operations
```
```
Measure deletion fidelity (M−),
insertion fidelity (M+), and
asymmetry index: A = (M−−
M+)/(|M−| +|M+|)
```
```
High M−indicates neces-
sary features; high M+in-
dicates sufficient features.
GNNs show high asymme-
try (A > 0 ); LLMs show
balanced fidelity (A≈ 0 )
```
Separability Metric for Error Detection. For Dimensions 3 and 4, which evaluate
quadrant-based classification patterns (sufficiency-necessity and faithfulness cat-
egories), we introduce a Separability metric to quantify error detection capacity.
This metric combines the weighted standard deviations of correct and incorrect
prediction distributions across categorical boundaries:

```
Separability =
```
```
q
SD^2 correct,weighted+ SD^2 incorrect,weighted
where:
```
- SDcorrect,weightedis the weighted standard deviation of correct prediction per-
    centages across categories
- SDincorrect,weightedis the weighted standard deviation of incorrect prediction
    percentages across categories


- Weights are determined by each category’s prevalence (% of Total instances)

Higher Separability values indicate sharper categorical boundaries that reli-
ably distinguish correct from incorrect predictions. This metric captures whether
explainability-derived categories encode diagnostic signals: if correct predictions
concentrate in theoretically "good" categories (e.g., Sufficient-Necessary, Faith-
ful) while incorrect predictions concentrate in "bad" categories (e.g., Insufficient-
Redundant, Unfaithful), Separability will be high. Conversely, uniform correct-
ness distribution across all categories yields low Separability, indicating weak er-
ror detection capacity. This metric is architecture-agnostic and enables quantita-
tive comparison of how discrete graph structures versus continuous token repre-
sentations create categorical distinctions for automated quality control.
Each dimension considered evaluates distinct properties. These are intention-
ally orthogonal—success in one does not guarantee success in others. Consistency
across all four dimensions, multiple graph types, both datasets, and stratified by
class and correctness constitutes strong evidence that observed differences reflect
fundamental architectural properties.

3.6. Error Signal Analysis

To assess whether the four evaluation dimensions collectively encode suffi-
cient information to discriminate correct from incorrect predictions, we apply
stratified logistic regression that consolidates dimension-specific metrics into a
unified error detection framework. This approach quantifies the relative contribu-
tion of each dimension to error detection capacity and validates whether observed
patterns reflect genuine architectural signatures.
For each test instance i, we construct a standardised feature vector aggregating
the metrics defined in Appendix Table B.8 from all four evaluation dimensions.
We model the probability of correct classification via binary logistic regression:

```
log
```
## 

```
P(correct = 1|xi)
1 − P(correct = 1|xi)
```
## 

```
= β 0 +
```
## X

```
d∈D
```
```
βd· fd(xi) (1)
```
whereD = {AUC, Fidelity, Consistency, Progression} and fd(xi) aggregates
features from dimension d.
To isolate architectural effects from confounds, we train separate models for
each combination of dataset (AG-News, SST-2), explainability module (SubgraphX,
GraphSVX, TokenSHAP), graph type, and class label. Statistical validation em-
ploys 200 bootstrap resamples to compute 95% confidence intervals for coeffi-
cients {βd}, with significance assessed via t-statistics under H 0 : βd= 0, com-


plemented by 10-fold stratified cross-validation to evaluate generalisation capac-
ity. The dimension-level coefficients quantify which architectural properties con-
tribute most decisively to error detection, with systematic differences between
GNN-based and LLM-based explainers revealing whether discrete graph struc-
tures and continuous token representations produce fundamentally distinct error
detection signatures.

4. Experiments

The modular methodology described above allows us to swap different com-
ponents—language models, text-to-graph conversions, GNN architectures, and
explainability modules—adapting our study to a range of underlying graph struc-
tures and tasks. Below, we review our experimental setup, data, and results.

4.1. Configuration

For all experiments, we begin by fine-tuning a language model for each NLP
task; the best model is used to generate embeddings for the GNN input graphs.
BERT serves as the backbone of the LM. GNN training is performed using a
Graph Convolutional Network (GCN) with a multilayer perceptron (MLP) classi-
fier head, with GNN targets derived from the LM outputs.
To ensure high-quality model interpretability, we select the explainability
module based on the graph structure: SubgraphX is used for tree-structured
graphs (constituency/syntactic), while GraphSVX is used for window and skip-
gram graphs. This ensures that, regardless of topology, the most relevant set of
nodes/subgraphs—either connected or distributed—is identified and scored for
each prediction.
For each dataset and explainability module, hyperparameters were determined
empirically through targeted pilot experiments. Specifically, we used a valida-
tion subset from each dataset to systematically explore a small range of plausible
parameter settings for each module (e.g., subgraph size and MCTS rollouts for
SubgraphX; node/neighbourhood selection criteria for GraphSVX). The chosen
settings were balanced to achieve fidelity, sparsity, and stability, as specified in
the module documentation and recent literature. They were fixed for the full eval-
uation set to ensure a fair and reproducible comparison across tasks and graph
types. This manual search yielded robust, interpretable explanations without re-
sorting to automated hyperparameter optimisation frameworks.


4.2. Datasets

We perform experiments on the AG News Zhang et al. (2015) and SST-2 Socher
et al. (2013) datasets, which cover topic classification and sentiment analysis, re-
spectively. Both datasets are well-studied, enabling results to be compared to past
work—while providing a rich testbed to validate interpretability in both tree-based
and proximity-based graph structures.

5. Results and Discussion

5.1. LLM as a Teacher

Table 2 reports the precision, recall, and F1 scores for both the fine-tuned
language model and the corresponding GNN for each task. In this case, the per-
formance against the ground truth for each dataset is reported in the corresponding
column. The capability of each GNN in replicating the LLM responses is indi-
cated in the “Teacher” column. In general, the replication of the LLM responses
by each GNN is easier to achieve for the Ag-news dataset; nevertheless, the agree-
ment between both models is always over 90%. The worst performant GNN in
this sense uses the constituency graphs, which add noise due to the special nodes
considered.

Table 2: Classification performance for BERT teacher and GCN models with different graph types.
Teacher: replicating BERT predictions; GT: ground truth labels.

AG News
Teacher Ground Truth
Model Type R P F 1 R P F 1
BERT Tokens – – – .938 .938.

GCN

```
Syntactic .958 .958 .958 .921 .920.
Constituency .938 .938 .938 .905 .904.
Skip-grams .965 .966 .965 .926 .926.
Window .965 .965 .965 .926 .926.
```
```
SST-
Teacher Ground Truth
Model Type R P F 1 R P F 1
BERT Tokens – – – .917 .917.
GCN
```
```
Syntactic .919 .919 .919 .875 .875.
Constituency .929 .930 .929 .890 .890.
Skip-grams .917 .917 .917 .878 .878.
Window .933 .934 .933 .888 .888.
```
5.2. Dimension 1: AUC Discriminative Capacity

We evaluate discriminative capacity through Area Under the Curve (AUC)
as a holistic metric quantifying total confidence change during feature addition
(insertion) or removal (deletion):

- Insertion AUC: Features ranked by importance are progressively added to
    a completely masked baseline (all features initially removed), measuring
    confidence recovery as each feature is reintroduced.


- Deletion AUC: Features ranked by importance are progressively removed
    from the complete original input (all features initially present), measuring
    confidence degradation as each feature is deleted.
Detection Rate Calculation. Across 10 evenly spaced AUC thresholds [0. 1 , 0. 2 ,..., 1 .0],
we compute correctness and error detection rates by comparing AUC distributions
with each instance’s correctness, based on the ground truth and the given predic-
tion. The detection rate is defined as the proportion of instances that fall below
the AUC threshold.
- Error Detection Rate: The proportion of incorrect predictions successfully
identified at a given threshold = (flagged incorrect predictions) / (total in-
correct predictions).
- Correctness Detection Rate: The proportion of correct predictions prop-
erly retained at a given threshold = (unflagged correct predictions) / (total
correct predictions).
Example. Consider a test set of 100 instances with 20 incorrect predictions.
At an insertion AUC threshold of 0.7, suppose 15 incorrect predictions have AUC
< 0. 7 while 5 have AUC ≥ 0. 7. The Error Detection Rate at this threshold is
15 /20 = 75%.
These detection rates are plotted across all thresholds (Figures 3 and 4), yield-
ing Equal Error Rate (EER) analysis. The intersection point identifies the optimal
threshold that maximises simultaneous detection of both correct and erroneous
classifications. Higher y-axis values indicate the stronger discriminative capacity
between correct and incorrect predictions.

Figure 3: Error and correctness detection rates plotted across insertion AUC thresholds, indicating
the mean per dataset.


Figure 4: Error and correctness detection rates plotted across deletion AUC thresholds, indicating
the mean per dataset.

Key Finding. Graph-based explainers produce AUC distributions with signifi-
cantly stronger separation between correct and incorrect predictions than token-
based explainers, though the discriminative capacity varies by metric type.
Deletion AUC favours GraphSVX (proximity-based graphs), where correct
predictions consistently yield higher values than incorrect ones, enabling reli-
able error detection. TokenSHAP exhibits moderate discrimination, whereas Sub-
graphX shows limited deletion-based separation.
Insertion AUC favours SubgraphX (hierarchical trees), which produces clearly
distinct distributions for correct versus incorrect predictions. TokenSHAP main-
tains moderate discrimination, while GraphSVX shows minimal insertion-based
separation.
These complementary patterns across AG News and SST-2 confirm a funda-
mental architectural difference: discrete graph structures enable stronger AUC-
based discrimination than continuous token representations.

5.3. Dimension 2: Feature Ranking Stability and Importance Concentration

This dimension examines how importance concentrates within the top-k fea-
tures for k ∈{ 1 , 3 , 5 , 10 }, providing a granular view of deletion and insertion op-
erations. We analyse maskout progression—cumulative confidence drop as top-
k features are removed—and sufficiency progression—cumulative confidence
gain as top-k features are added. The key diagnostic signal is the separation be-
tween correct and incorrect predictions.


Concentration Patterns Across Architectures and Datasets. Figure 5 reveals concentration-
based separation that reinforces the architectural differences observed in Dimen-
sion 1.
For maskout progression (Figure 5a), GNN-based methods demonstrate clear
separation between correct and incorrect predictions across top-k values, partic-
ularly pronounced in AG News. This separation indicates that correct and incor-
rect predictions exhibit distinct concentration profiles when features are removed.
SST-2 shows similar but less pronounced patterns. In contrast, TokenSHAP ex-
hibits weaker separation between correct and incorrect predictions, indicating that
LLM-based explainers produce less distinguishable concentration patterns for er-
ror detection.
For sufficiency progression (Figure 5b), GNN-based methods maintain some
separation between correct and incorrect predictions, though less pronounced than
in maskout progression. TokenSHAP shows minimal separation, with correct
and incorrect predictions following similar concentration trajectories during re-
construction.

Key Finding. Granular top-k concentration analysis validates Dimension 1 find-
ings through feature-level separation patterns. GNN-based explainers consistently
produce distinct concentration profiles for correct versus incorrect predictions in
both maskout and sufficiency progression, with maskout showing the strongest
discriminative signals. This granular separation explains the superior AUC-based
discrimination observed in Dimension 1—graph structures enable architecturally
distinct feature importance patterns that manifest both holistically (Dimension 1’s
AUC) and granularly (Dimension 2’s top-k concentration). LLM-based explainers
show weaker concentration-based separation, consistent with their more limited
AUC discrimination. These complementary views converge on the same archi-
tectural insight: discrete graph structures enable clearer differentiation between
correct and incorrect predictions through concentrated feature dependencies.


```
(a) Maskout Progression drop concentration.
```
```
(b) Sufficiency Progression drop concentration.
```
Figure 5: Progression drop concentration for the top k features across the evaluated datasets and
explainability modules.


```
Table 3: Sufficiency-necessity classification patterns with separability metrics across datasets
(a) AG News
Method Category %Correct Incorrect |∆|
SubgraphX(Constituency)
```
```
Insufficient-RedundantInsufficient-Necessary 2.050.59 60.000.00 100.0040.00 100.0020.00
Sufficient-RedundantSufficient-Necessary 24.8772.47 92.1297.31 7.882.69 84.2394.63
```
Mean Sep. (Weighted)Separability ± SD –– 14.04– 14.04– 91. 71 ± (^77) 19.85. 47
SubgraphX(Syntactic)
Insufficient-RedundantInsufficient-Necessary 1.990.38 37.0975.86 62.9124.14 25.8351.72
Sufficient-RedundantSufficient-Necessary 38.5459.09 97.7896.66 2.223.34 95.5693.32
Mean Sep. (Weighted)Separability ± SD –– 8.48– 8.48– 92. 68 ± (^49) 12.00. 12
GraphSVX(Window)
Insufficient-RedundantInsufficient-Necessary 0.000.00 0.000.00 0.000.00 0.000.00
Sufficient-RedundantSufficient-Necessary 88.7611.24 99.1076.00 24.000.90 98.1951.99
Mean Sep. (Weighted)Separability ± SD –– 7.30– 7.30– 93. 00 ± (^23) 10.32. 10
GraphSVX(Skipgrams)
Insufficient-RedundantInsufficient-Necessary 0.080.18 100.0050.00 50.000.00 100.000.00
Sufficient-RedundantSufficient-Necessary 88.5311.21 98.7779.69 20.311.23 97.5359.39
Mean Sep. (Weighted)Separability ± SD –– 6.34– 6.34– 93. 08 ± (^40) 8.97. 43
TokenSHAP
Insufficient-RedundantInsufficient-Necessary 22.591.64 90.7459.20 40.809.26 81.4818.40
Sufficient-RedundantSufficient-Necessary 73.632.13 96.0774.69 25.313.93 92.1449.38
Mean Sep. (Weighted)Separability ± SD –– 5.78– 5.78– 87. 61 ± (^28) 8.17. 88
(b) SST-2
Method Category % CorrectIncorrect |∆|
SubgraphX(Constituency)
Insufficient-RedundantInsufficient-Necessary 4.010.23 100.000.00 100.000.00 100.00100.00
Sufficient-RedundantSufficient-Necessary 44.9550.80 87.7697.07 12.242.93 75.5194.13
Mean Sep. (Weighted)Separability ± SD –– 18.76– 18.76– 86. 01 ± (^82) 26.53. 72
SubgraphX(Syntactic)
Insufficient-RedundantInsufficient-Necessary 3.100.00 0.000.00 100.000.00 100.000.00
Sufficient-RedundantSufficient-Necessary 43.2353.67 94.4391.86 5.578.14 88.8683.72
Mean Sep. (Weighted)Separability ± SD –– 16.42– 16.42– 89. 91 ± (^89) 23.23. 34
GraphSVX(Window)
Insufficient-RedundantInsufficient-Necessary 0.110.00 100.000.00 0.000.00 100.000.00
Sufficient-RedundantSufficient-Necessary 83.4916.40 98.2168.53 31.471.79 96.4337.06
Mean Sep. (Weighted)Separability ± SD –– 10.99– 10.99– 86. 70 ± (^28) 15.54. 86
GraphSVX(Skipgrams)
Insufficient-RedundantInsufficient-Necessary 0.110.00 100.000.00 0.000.00 100.000.00
Sufficient-RedundantSufficient-Necessary 84.5215.37 97.1561.94 38.062.85 94.3023.88
Mean Sep. (Weighted)Separability ± SD –– 12.70– 12.70– 83. 49 ± (^34) 17.96. 62
TokenSHAP
Insufficient-RedundantInsufficient-Necessary 31.653.44 91.6776.67 23.338.33 83.3353.33
Sufficient-RedundantSufficient-Necessary 55.739.17 94.4480.00 20.005.56 88.8960.00
Mean Sep. (Weighted)Separability ± SD –– 4.95– 4.95– 83. 26 ± (^15) 7.00. 04
5.4. Dimension 3: Consistency Across Prediction Outcomes
We evaluate explainability consistency by measuring how explanation-identified
features affect the decision boundary between the predicted class and the second-
highest class probability. For each instance, we compute sufficiency preservation
(masked margin / original margin) and necessity preservation (maskout margin /
original margin). High sufficiency preservation indicates that the highlighted fea-
tures alone maintain the decision boundary; low necessity preservation indicates
margin collapse when features are removed, providing evidence of true necessity.
According to these definitions, we can normalise and classify explanations
into four quadrants: Sufficient-Necessary (high sufficiency, low necessity—ideal
precise identification), Sufficient-Redundant (both high—alternative pathways ex-
ist), Insufficient-Necessary (low sufficiency, high necessity—features required but
insufficient alone), and Insufficient-Redundant (both low—features neither neces-
sary nor sufficient alone).
Quadrant Distribution and Separation Capacity. Table 3 reveals distinct margin
preservation patterns across architectures. Hierarchical GNN explainers (Sub-
graphX) predominantly concentrate instances in the Sufficient-Necessary quad-
rant, indicating precise feature identification where highlighted subgraphs both


maintain margins when isolated and collapse margins when removed. Non-hierarchical
GNN explainers (GraphSVX) exhibit intense concentration in the Sufficient-Redundant
regime, where distributed node coalitions in proximity-based graphs preserve mar-
gins through overlapping receptive fields. TokenSHAP exhibits a broader distri-
bution across quadrants, reflecting continuous token representations that preserve
predictions via distributed semantic pathways.
The Separability metric confirms GNN superiority across both datasets. All
GNN-based explainers—both hierarchical and non-hierarchical—substantially out-
perform TokenSHAP, with the performance gap widening on SST-2 compared to
AG News. This demonstrates that discrete graph structures enable sharper cate-
gorical boundaries for distinguishing correct from incorrect predictions than con-
tinuous token embeddings.

Key Finding. Margin consistency evaluation confirms that discrete graph struc-
tures create superior error-detection signals compared to continuous token repre-
sentations. Hierarchical structures achieve the highest separability through binary
feature participation, where node removal severs syntactic pathways, leading to
clear margin collapses. Non-hierarchical proximity graphs maintain strong sepa-
rability despite preserving margins through distributed coalitions, as their graph-
structured representations still encode more distinctive error patterns than token-
based alternatives. These findings align with Dimensions 1 and 2, confirming
that graph-based explainability architectures consistently outperform LLM-based
approaches across complementary error-detection mechanisms.

5.5. Dimension 4: Behavioural Faithfulness and Fidelity Asymmetry

We assess faithfulness through per-instance fidelity metrics that quantify necessity-
sufficiency structure. Fidelity-plus (F+) measures confidence gain when inserting
highlighted features into a neutral baseline (feature sufficiency); Fidelity-minus
(F-) measures confidence drop when removing features from the original input
(feature necessity).
Instances are classified into four quadrants: Faithful (high F+, high F-—features
both necessary and sufficient), Redundant (high F+, low F-—sufficient but not
necessary), Incomplete (low F+, high F-—necessary but insufficient alone), and
Unfaithful (low F+, low F-—neither necessary nor sufficient).

Quadrant Distribution and Error Signal Strength. Table 4 demonstrates that all
GNN-based explainers achieve substantially higher Separability than TokenSHAP
across both datasets. Hierarchical explainers (SubgraphX) show selective Faith-
ful categorisation with strong predictive power—instances classified as Faithful


```
Table 4: Faithfulness classification patterns with separability metrics across datasets
(a) AG News
Method Category %Correct Incorrect |∆|
SubgraphX(Constituency)
```
```
UnfaithfulIncomplete 41.200.38 13.7991.09 86.218.91 72.4182.18
RedundantFaithful 57.560.86 15.3897.44 84.622.56 69.2394.88
```
Mean Sep. (Weighted)Separability ± SD –– 9.36– 9.36– 89. 34 ± (^79) 13.24. 81
SubgraphX(Syntactic)
UnfaithfulIncomplete 21.341.05 48.7589.83 51.2510.17 79.652.50
RedundantFaithful 75.841.76 53.7399.15 46.270.85 98.307.46
Mean Sep. (Weighted)Separability ± SD –– 8.40– 8.40– 91. 71 ± (^43) 11.89. 89
GraphSVX(Window)
UnfaithfulIncomplete 17.180.00 82.010.00 17.990.00 64.010.00
RedundantFaithful 82.490.33 100.0099.51 0.000.49 100.0099.01
Mean Sep. (Weighted)Separability ± SD –– 6.60– 6.60– 93. 00 ± (^40) 9.34. 64
GraphSVX(Skipgrams)
UnfaithfulIncomplete 16.260.09 100.0082.52 17.480.00 100.0065.05
RedundantFaithful 83.040.61 93.4899.30 6.520.70 86.9698.61
Mean Sep. (Weighted)Separability ± SD –– 6.19– 6.19– 93. 08 ± (^14) 8.76. 00
TokenSHAP
UnfaithfulIncomplete 10.208.87 89.6189.03 10.3910.97 79.2378.06
RedundantFaithful 35.7145.22 94.6695.02 5.344.98 89.3190.05
Mean Sep. (Weighted)Separability ± SD –– 2.19– 2.19– 87. 61 ± (^5) 3.10. 54
(b) SST-2
Method Category % CorrectIncorrect |∆|
SubgraphX(Constituency)
UnfaithfulIncomplete 55.051.61 83.757.14 92.8616.25 85.7167.50
RedundantFaithful 42.780.57 80.0098.93 20.001.07 60.0097.86
Mean Sep. (Weighted)Separability ± SD –– 12.86– 12.86– 80. 73 ± (^71) 18.19. 07
SubgraphX(Syntactic)
UnfaithfulIncomplete 33.941.26 89.860.00 100.0010.14 100.0079.73
RedundantFaithful 61.353.44 10.0099.44 90.000.56 80.0098.88
Mean Sep. (Weighted)Separability ± SD –– 19.33– 19.33– 91. 74 ± (^90) 27.33. 18
GraphSVX(Window)
UnfaithfulIncomplete 22.130.11 100.0074.09 25.910.00 100.0048.19
RedundantFaithful 77.520.23 100.0098.82 0.001.18 100.0097.63
Mean Sep. (Weighted)Separability ± SD –– 10.27– 10.27– 86. 70 ± (^22) 14.52. 12
GraphSVX(Skipgrams)
UnfaithfulIncomplete 19.150.11 67.070.00 100.0032.93 100.0034.13
RedundantFaithful 79.930.80 85.7197.85 14.292.15 71.4395.70
Mean Sep. (Weighted)Separability ± SD –– 12.50– 12.50– 83. 72 ± (^75) 17.68. 60
TokenSHAP
UnfaithfulIncomplete 11.012.64 69.5782.29 30.4317.71 39.1364.58
RedundantFaithful 24.0862.27 90.4894.66 9.525.34 80.9589.32
Mean Sep. (Weighted)Separability ± SD –– 5.34– 5.34– 83. 26 ± (^19) 7.55. 15
exhibit near-perfect correctness rates while Unfaithful categories concentrate er-
rors. Non-hierarchical explainers (GraphSVX) demonstrate broader Faithful cate-
gorisation while maintaining superior separability compared to TokenSHAP. The
high Faithful concentration in GraphSVX, coupled with moderate separability,
indicates that proximity-based graphs identify joint necessity-sufficiency for pre-
dicted class confidence in most instances, though with less extreme differentiation
in correctness than tree-structured approaches.
TokenSHAP exhibits substantially weaker categorical boundaries. While achiev-
ing comparable Faithful concentration on SST-2, its correctness rates remain rel-
atively uniform across all faithfulness categories—the Faithful quadrant provides
only marginal improvement over Unfaithful classifications. This uniform distri-
bution indicates that continuous token representations maintain predicted class
confidence through compensatory pathways that obscure the causal necessity-
sufficiency structure captured by discrete graph architectures.
Key Finding. Faithfulness evaluation via predicted class-confidence dynamics
yields complementary insights to margin-based consistency (Dimension 3). Graph-
based explainers draw sharp distinctions between faithful and unfaithful explana-
tions, reliably separating correct from incorrect predictions. In contrast, token-


based explainers exhibit weak coupling between faithfulness and correctness through
distributed compensatory representations. These findings align with Dimensions
1-3, confirming that graph architectures consistently encode stronger error-detection
signals across aggregate confidence trajectories, decision boundary stability, and
causal feature identification.

5.6. Integrated Error Signal Analysis via Logistic Regression
Table 5 presents classification accuracy for error detection via stratified logis-
tic regression across all explainability configurations. SubgraphX on constituency
and syntactic graphs achieves near-perfect error detection (99.7–100.0%), while
GraphSVX on skipgram and window graphs attains intermediate performance
(86.5–93.6%), and TokenSHAP exhibits substantially lower accuracy (88.1–89.6%).
This 10–13 percentage point gap between discrete graph-based and continuous
token-based explainability confirms that structured representations expose model
decision boundaries with greater transparency. Detailed coefficient patterns and
dimension-specific significance analysis are provided in Appendix B.

Table 5: Classification accuracy for error detection via stratified logistic regression across explain-
ability modules and graph representations. Best results are in bold, second-best are underlined.
Values represent mean ± standard deviation across 10-fold cross-validation and 200 bootstrap re-
samples.

```
Dataset Graph Type CV Accuracy CV Std Bootstrap Accuracy Bootstrap Std
Graph Neural Network Methods
AG News Constituency 0.999 0.002 1.000 0.000
AG News Syntactic 0.997 0.004 0.998 0.002
AG News Skipgrams 0.936 0.042 0.942 0.037
AG News Window 0.925 0.049 0.931 0.042
SST-2 Constituency 1.000 0.000 1.000 0.000
SST-2 Syntactic 0.999 0.005 0.999 0.001
SST-2 Skipgrams 0.886 0.065 0.900 0.017
SST-2 Window 0.865 0.057 0.882 0.032
Large Language Model Method
AG News Tokens 0.881 0.053 0.887 0.045
SST-2 Tokens 0.896 0.043 0.914 0.021
```
6. Conclussion

The widespread adoption of large language models (LLMs) has made model
explainability essential for ensuring trustworthy and accountable AI systems. This


study investigates whether discrete graph-based architectures and continuous token-
based architectures exhibit fundamentally different explainability signatures that
enable robust error detection. Our multidimensional evaluation framework reveals
a clear distinction between the two approaches, underscoring crucial ramifications
for explainability in high-risk application domains.
The central finding challenges conventional assumptions about explanation
quality: GNN-based explainers achieve near-perfect error detection (99.7–100.0%
classification accuracy for SubgraphX on tree structures, 86.5–93.6% for GraphSVX
on non-tree graphs) while LLM-based explainers show substantially weaker per-
formance (88.1–89.6% for TokenSHAP), despite LLMs often maintaining higher
confidence in their importance attributions and producing more distributed, seem-
ingly robust explanations. This 10–13 percentage point gap demonstrates that
explanation confidence does not translate to diagnostic utility—a decisive insight
for practitioners weighing architectural trade-offs.
This architectural divergence manifests consistently across all four evalua-
tion dimensions. In error detection capacity (Dimension 1), GNN explanations
produce strong Deletion AUC separation between correct and incorrect predic-
tions because discrete message-passing paths sever information flow when criti-
cal nodes are removed, creating clear necessity signals. LLM explanations show
weaker separation because continuous token embeddings maintain predictions via
overlapping representational pathways, attenuating diagnostic signals even as at-
tribution confidence remains high. In feature ranking stability (Dimension 2),
GNNs exhibit steep, concentrated importance degradation, where removing top-
ranked features causes a sharp collapse in confidence. In contrast, LLMs display
gradual, distributed degradation, reflecting multi-pathway contributions. Consis-
tency analysis (Dimension 3) reveals that GNN explanations preserve decision
margins with higher standard deviation separation between correct and incorrect
predictions (SubgraphX-Constituency: 44.72% on AG-News, 49.69% on SST-
2) compared to LLMs (TokenSHAP: 16.68% and 8.21% respectively), demon-
strating that discrete structures impose sharper constraints on feature participa-
tion. Behavioural faithfulness (Dimension 4) establishes explanation fidelity it-
self as an error detection signal—GNN methods achieve standard deviations of
45.44% (SubgraphX-Constituency, AG-News) versus 3.20% (TokenSHAP), indi-
cating that discrete graph structures create distinctions between faithful and un-
faithful explanations that reliably discriminate prediction correctness.
The integration of these four dimensions through stratified logistic regression
(3.6) validates that these patterns aggregate into robust architectural signatures.
Near-perfect classification accuracy combined with dimension-specific coefficient


significance (Appendix C) confirms that each evaluation dimension contributes in-
dependent evidence to error detection capacity. High positive coefficients across
AUC, Fidelity, Consistency, and Progression dimensions demonstrate that the
four-dimensional framework successfully captures fundamental architectural dif-
ferences in how necessity versus sufficiency are encoded. The low bootstrap vari-
ance (0.65% mean divergence) and high cross-validation stability establish that
these error detection signals generalise reliably across experimental conditions,
confirming that observed differences reflect genuine architectural properties rather
than spurious correlations or task-specific artefacts.
Notably, these patterns remained consistent across both datasets (AG-News
and SST-2), all four graph construction methods (constituency, dependency, win-
dow, and skip-gram), across class labels, and across prediction correctness strata.
This cross-dataset, cross-topology consistency validates that observed differences
reflect genuine architectural properties—GNNs encode necessity through binary
node participation, LLMs encode sufficiency through distributed continuous rep-
resentations—rather than experimental artefacts or task-specific confounds.
For practical deployment, this necessity-sufficiency dichotomy suggests that
explainability research should distinguish between diagnostic architectures and
explanatory architectures. Graph-based approaches provide necessity-focused ex-
planations with superior error detection capabilities, making them particularly
suitable for legal, medical, and financial domains where transparent quality con-
trol and auditable traceability are paramount. Token-based approaches capture
distributed sufficiency relationships that reflect robust reasoning but may provide
false assurance through high-confidence explanations that fail to identify system-
atic errors.

7. Future Work

While this study establishes that discrete graph-based architectures provide su-
perior error detection compared to continuous token-based architectures on general-
domain benchmarks, several extensions are necessary to validate the generaliz-
ability and robustness of these findings.
First, evaluating the framework across domain-specific datasets—including
legal document analysis, clinical notes, financial regulatory texts, and scientific
literature—would test whether the necessity-sufficiency dichotomy persists when
specialised semantic relationships and domain terminologies are prevalent. Sec-
ond, extending the evaluation to diverse NLP tasks that preserve the classifica-
tion paradigm—such as question answering with answer selection, named entity


recognition, relation extraction, and textual entailment—would establish whether
architectural advantages for error detection generalise across different reason-
ing patterns. For generation tasks (e.g., summarisation, machine translation),
graph-based explainability could serve as a post-hoc quality judge by extracting
graphs from LLM-generated outputs and performing classification-based error de-
tection on structural or semantic properties, enabling explainable quality assess-
ment without requiring graph-native generation architectures.
Third, and critically, investigating language models with alternative atten-
tion mechanisms—particularly window-based attention (e.g., Longformer, Big-
Bird) and other local attention variants—would test whether restricting contextual
scope dilutes the observed error detection gap. Full self-attention allows LLMs to
maintain distributed sufficiency through global token interactions; local attention
mechanisms may reduce these multi-pathway contributions, potentially strength-
ening necessity signals and narrowing the performance gap with GNN explain-
ers. This investigation would isolate whether the architectural divergence stems
fundamentally from discrete-versus-continuous representations or can be modu-
lated through attention design choices, providing actionable insights for develop-
ing explainability-optimised language model architectures.

Acknowledgments

This work was supported by the University of Alicante, the Spanish Ministry
of Science and Innovation, the Generalitat Valenciana, and the European Regional
Development Fund (ERDF) through the following funding: COOLANG: Techno-
logical Resources for Intelligent Viral Analysis, with grant reference PID2021-
122263OB-C22. VIVES: ‘Pla de Tecnologies de la Llengua per al valencià’
project (2022/TL22/00215334), funded by MCIN/AEI/10.13039/501100011033
and “ERDF A way of making Europe”. NL4DISMIS: Natural Language Tech-
nologies for dealing with dis- and misinformation with grant reference CIPROM/2021/021.

References

Agarwal, C., Queen, O., Lakkaraju, H., Zitnik, M., 2023. Evaluating explainabil-
ity for graph neural networks. URL: https://arxiv.org/abs/2208.
09339 , arXiv:2208.09339.

Amara, K., Ying, R., Zhang, Z., Han, Z., Shan, Y., Brandes, U., Schemm, S.,
Zhang, C., 2024. Graphframex: Towards systematic evaluation of explainabil-


```
ity methods for graph neural networks. URL: https://arxiv.org/abs/
2206.09677, arXiv:2206.09677.
```
Bauer, J., Manning, C.D., 2025. High-accuracy transition-based constituency
parsing, in: Sagae, K., Oepen, S. (Eds.), Proceedings of the 18th Interna-
tional Conference on Parsing Technologies (IWPT, SyntaxFest 2025), Asso-
ciation for Computational Linguistics, Ljubljana, Slovenia. pp. 26–39. URL:
https://aclanthology.org/2025.iwpt-1.4/.

Bonabi Mobaraki, E., Khan, A., 2023. A demonstration of interpretability meth-
ods for graph neural networks, in: Proceedings of the 6th Joint Workshop on
Graph Data Management Experiences & Systems (GRADES) and Network
Data Analytics (NDA), Association for Computing Machinery, New York,
NY, USA. URL: https://doi.org/10.1145/3594778.3594880,
doi:10.1145/3594778.3594880.

Brannon, W., Kang, W., Fulay, S., Jiang, H., Roy, B., Roy, D., Kabbara, J., 2024.
ConGraT: Self-supervised contrastive pretraining for joint graph and text em-
beddings, in: Ustalov, D., Gao, Y., Panchenko, A., Tutubalina, E., Nikishina,
I., Ramesh, A., Sakhovskiy, A., Usbeck, R., Penn, G., Valentino, M. (Eds.),
Proceedings of TextGraphs-17: Graph-based Methods for Natural Language
Processing, Association for Computational Linguistics, Bangkok, Thailand. pp.
19–39. URL: https://aclanthology.org/2024.textgraphs-1.
2/.

Cambria, E., Malandri, L., Mercorio, F., Nobani, N., Seveso, A., 2024.
Xai meets llms: A survey of the relation between explainable ai and
large language models. URL: https://arxiv.org/abs/2407.15248,
arXiv:2407.15248.

Cedro, M., Martens, D., 2025. Graphxain: Narratives to explain graph
neural networks. URL: https://arxiv.org/abs/2411.02540,
arXiv:2411.02540.

Devlin, J., Chang, M.W., Lee, K., Toutanova, K., 2019. Bert: Pre-
training of deep bidirectional transformers for language understanding.
arXiv:1810.04805.

Dhaini, M., Erdogan, E., Bakshi, S., Kasneci, G., 2024. Explainability meets
text summarization: A survey, in: Mahamood, S., Minh, N.L., Ippolito, D.


```
(Eds.), Proceedings of the 17th International Natural Language Generation
Conference, Association for Computational Linguistics, Tokyo, Japan. pp. 631–
```
645. URL: https://aclanthology.org/2024.inlg-main.49/,
doi:10.18653/v1/2024.inlg-main.49.

Duval, A., Malliaros, F.D., 2021. Graphsvx: Shapley value explanations for
graph neural networks. URL: https://arxiv.org/abs/2104.10482,
arXiv:2104.10482.

Geva, M., Caciularu, A., Wang, K., Goldberg, Y., 2022. Transformer feed-
forward layers build predictions by promoting concepts in the vocabulary space,
in: Goldberg, Y., Kozareva, Z., Zhang, Y. (Eds.), Proceedings of the 2022
Conference on Empirical Methods in Natural Language Processing, Associ-
ation for Computational Linguistics, Abu Dhabi, United Arab Emirates. pp.
30–45. URL: https://aclanthology.org/2022.emnlp-main.3,
doi:10.18653/v1/2022.emnlp-main.3.

Goldshmidt, R., Horovicz, M., 2024. Tokenshap: Interpreting large language
models with monte carlo shapley value estimation. URL: https://arxiv.
org/abs/2407.10114, arXiv:2407.10114.

Hooker, S., Erhan, D., Kindermans, P.J., Kim, B., 2019. A benchmark for inter-
pretability methods in deep neural networks. URL: https://arxiv.org/
abs/1806.10758, arXiv:1806.10758.

Hu, S., Zou, G., Yang, S., Gan, Y., Zhang, B., Chen, Y., 2024. Large language
model meets graph neural network in knowledge distillation. URL: https:
//arxiv.org/abs/2402.05894, arXiv:2402.05894.

Kakkad, J., Jannu, J., Sharma, K., Aggarwal, C., Medya, S., 2023. A survey
on explainability of graph neural networks. URL: https://arxiv.org/
abs/2306.01958, arXiv:2306.01958.

Kindermans, P.J., Hooker, S., Adebayo, J., Alber, M., Schütt, K.T., Dähne, S.,
Erhan, D., Kim, B., 2017. The (un)reliability of saliency methods. URL:
https://arxiv.org/abs/1711.00867, arXiv:1711.00867.

Kipf, T.N., Welling, M., 2017. Semi-supervised classification with graph con-
volutional networks. URL: https://arxiv.org/abs/1609.02907,
arXiv:1609.02907.


Kobayashi, G., Kuribayashi, T., Yokoi, S., Inui, K., 2020. Attention is not
only a weight: Analyzing transformers with vector norms, in: Webber, B.,
Cohn, T., He, Y., Liu, Y. (Eds.), Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP), Association
for Computational Linguistics, Online. pp. 7057–7075. URL: https://
aclanthology.org/2020.emnlp-main.574, doi:10.18653/v1/
2020.emnlp-main.574.

Kosan, M., Verma, S., Armgaan, B., Pahwa, K., Singh, A., Medya, S., Ranu, S.,

2024. Gnnx-bench: Unravelling the utility of perturbation-based gnn explain-
ers through in-depth benchmarking. URL: https://arxiv.org/abs/
2310.01794, arXiv:2310.01794.

Li, Q., Zhao, T., Chen, L., Xu, J., Wang, S., 2024. Enhancing graph neu-
ral networks with limited labeled data by actively distilling knowledge from
large language models. URL: https://arxiv.org/abs/2407.13989,
arXiv:2407.13989.

Li, X., Du, M., Chen, J., Chai, Y., Lakkaraju, H., Xiong, H., 2023. M4: a uni-
fied xai benchmark for faithfulness evaluation of feature attribution methods
across metrics, modalities and models, in: Proceedings of the 37th International
Conference on Neural Information Processing Systems, Curran Associates Inc.,
Red Hook, NY, USA.

Luo, H., Specia, L., 2024. From understanding to utilization: A survey on ex-
plainability for large language models. URL: https://arxiv.org/abs/
2401.12874, arXiv:2401.12874.

de Marneffe, M.C., Manning, C.D., Nivre, J., Zeman, D., 2021.
Universal Dependencies. Computational Linguistics 47, 255–

308. URL: https://aclanthology.org/2021.cl-2.11/,
doi:10.1162/coli_a_00402.

Mavromatis, C., Ioannidis, V.N., Wang, S., Zheng, D., Adeshina, S., Ma, J., Zhao,
H., Faloutsos, C., Karypis, G., 2023. Train your own gnn teacher: Graph-aware
distillation on textual graphs. URL: https://arxiv.org/abs/2304.
10668 , arXiv:2304.10668.

Mills, E., Su, S., Russell, S., Emmons, S., 2025. Almanacs: A simulatabil-
ity benchmark for language model explainability. URL: https://arxiv.
org/abs/2312.12747, arXiv:2312.12747.


Modarressi, A., Fayyaz, M., Aghazadeh, E., Yaghoobzadeh, Y., Pilehvar, M.T.,

2023. DecompX: Explaining transformers decisions by propagating token de-
composition, in: Rogers, A., Boyd-Graber, J., Okazaki, N. (Eds.), Proceedings
of the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), Association for Computational Linguistics, Toronto,
Canada. pp. 2649–2664. URL: https://aclanthology.org/2023.
acl-long.149, doi:10.18653/v1/2023.acl-long.149.

Nonkes, N., Agaronian, S., Kanoulas, E., Petcu, R., 2024. Leveraging graph
structures to detect hallucinations in large language models, in: Ustalov, D.,
Gao, Y., Panchenko, A., Tutubalina, E., Nikishina, I., Ramesh, A., Sakhovskiy,
A., Usbeck, R., Penn, G., Valentino, M. (Eds.), Proceedings of TextGraphs-
17: Graph-based Methods for Natural Language Processing, Association for
Computational Linguistics, Bangkok, Thailand. pp. 93–104. URL: https:
//aclanthology.org/2024.textgraphs-1.7/.

Palikhe, A., Yu, Z., Wang, Z., Zhang, W., 2025. Towards transparent ai: A survey
on explainable large language models. URL: https://arxiv.org/abs/
2506.21812, arXiv:2506.21812.

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., Manning, C.D., 2020. Stanza: A python
natural language processing toolkit for many human languages, in: Proceedings
of the 58th Annual Meeting of the Association for Computational Linguistics:
System Demonstrations, Association for Computational Linguistics. pp. 101–

108. URL: https://aclanthology.org/2020.acl-demo.14.

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C.D., Ng, A., Potts, C.,

2013. Recursive deep models for semantic compositionality over a sentiment
treebank, in: Yarowsky, D., Baldwin, T., Korhonen, A., Livescu, K., Bethard,
S. (Eds.), Proceedings of the 2013 Conference on Empirical Methods in Nat-
ural Language Processing, Association for Computational Linguistics, Seat-
tle, Washington, USA. pp. 1631–1642. URL: https://aclanthology.
org/D13-1170.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix,
T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin,
A., Grave, E., Lample, G., 2023. Llama: Open and efficient founda-
tion language models. URL: https://arxiv.org/abs/2302.13971,
arXiv:2302.13971.


Xue, R., Deng, H., He, F., Wang, M., Zhang, Z., 2025. Trustworthy gnns with
llms: A systematic review and taxonomy. URL: https://arxiv.org/
abs/2502.08353, arXiv:2502.08353.

Yang, K., Deng, J., 2020. Strongly incremental constituency parsing with
graph neural networks. URL: https://arxiv.org/abs/2010.14568,
arXiv:2010.14568.

Yang, S., Zhang, R., Erfani, S., 2020. GraphDialog: Integrating graph
knowledge into end-to-end task-oriented dialogue systems, in: Webber, B.,
Cohn, T., He, Y., Liu, Y. (Eds.), Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP), Association
for Computational Linguistics, Online. pp. 1878–1888. URL: https://
aclanthology.org/2020.emnlp-main.147/, doi:10.18653/v1/
2020.emnlp-main.147.

Yuan, H., Yu, H., Wang, J., Li, K., Ji, S., 2021. On explainability of graph neural
networks via subgraph explorations. URL: https://arxiv.org/abs/
2102.05152, arXiv:2102.05152.

Yuan, Z., Chamoun, E., Aly, R., Whitehouse, C., Vlachos, A., 2025. Pro-
belm: Plausibility ranking evaluation for language models. URL: https:
//arxiv.org/abs/2404.03818, arXiv:2404.03818.

Zhang, X., Zhao, J., LeCun, Y., 2015. Character-level convolutional networks
for text classification, in: Proceedings of the 28th International Conference on
Neural Information Processing Systems - Volume 1, MIT Press, Cambridge,
MA, USA. p. 649–657.

Zhao, H., Chen, H., Yang, F., Liu, N., Deng, H., Cai, H., Wang, S., Yin, D.,
Du, M., 2023. Explainability for large language models: A survey. URL:
https://arxiv.org/abs/2309.01029, arXiv:2309.01029.


Appendix A. Fair-Comparison Methodology

```
Dimension General Strategy Module Configuration
Budget (B) Fixed number of forward passes across modules(N ). BBTSGSV X: N coalition evaluations: N node samples
BSX: rollout × samples = N
Sampling Ratio (R) Rmodule= N
total search space
```
```
RTS= 2 nN− 1
RGSVX= 0. 25
RSX≈expand atoms|V|×avg steps
Sparsity (S) Fixed based on the input elements for every node S = 0. 2 at word/node level
```
```
Context (C) Equivalent dependency coverage ensuringmodality independence CGSV XCTS: two-hop neighborhood: full self-attention
CSX: two-hop receptive field
Granularity (G) Consistent perturbation resolution perexploration unit GGGSV XTS: single token sampling: single token sampling
GSX: single node expansion
```
Table A.6: Fair comparison dimensions across explainability modules. All explainer configura-
tions share equivalent evaluation depth and perturbation resolution to isolate differences in algo-
rithmic behaviour.

To enable rigorous cross-architecture comparison, we standardise core termi-
nology across GNN and LLM modules according to Table A.7.

```
Table A.7: Standardized Terminology and Notation for GNN and LLM Explainability
```
```
Term Definition (General) Instantiation
Feature Atomic importance unit GNNs: nodes/edges; LLMs: tokens
Importance
Ranking Score-based ordering
```
```
GraphSVX/SubgraphX scores; TokenSHAP
values
Masking Feature removal operation GNNs: remove nodes/edges;LLMs: mask tokens
Insertion/Deletion Adding/Removing features to baseline GNNs: add nodes/edges to empty graph; LLMs:add tokens
Model
Confidence Prediction probability for class P (ˆ |y x) for predicted class ˆy
Confidence Drop Change in confidence after perturbation ∆ = P (ˆ |y x)− P (ˆ |y x′) where x′is
perturbed input
```

Appendix B. Multidimensional Analysis Metrics Considered

Table B.8: Multi-Dimensional Explainability Evaluation Framework: Complete Metrics Overview

Metric Definition and Operational Role

Dimension 1: Error Detection Capacity—Insertion-Deletion AUC

Deletion AUC AUC of confidence drops as top-ranked features are progressively removed: AUCR 1 del =
0 conf(f (x−featuresk)) dk. Low values indicate feature necessity; high values indicate features
are not critical.
Insertion AUC AUC of confidence recovery when reintroducing features from neutral baseline: AUCR 1 ins =
0 conf(f (xneutral+ featuresk)) dk. High values indicate collective feature sufficiency for model
behaviour.

Dimension 2: Feature Ranking Stability

Maskout Drop Progres-
sion

Cumulative confidence drops{∆ 1 , ∆ 2 ,... , ∆K} as features are progressively masked: ∆k=
conf(f (x))− conf(f (x− features1:k)). Indicates monotonic degradation consistency in impor-
tance rankings.
Sufficiency Drop Terminal confidence drop after masking all top-K features: ∆K= conf(f (x))− conf(f (x−
features1:K)). Quantifies explanation completeness via the fraction of confidence attributable to
identified features.
Importance Concentra-
tion

```
Cumulative importance fraction in top-N features: C(N ) =
```
```
PN
PiK=1importancei
i=1importancei. Reveals hierarchi-
cal sharpness versus distributed importance in explanations.
```
Dimension 3: Explanation Consistency Analysis

Origin Contrastivity Margin on unmodified input: conf(ˆy)− conf(ˆ 2 y) where ˆy is predicted class and ˆy 2 is runner-up.
Establishes baseline discrimination, stratified by correctness and class.
Masked Contrastivity Margin after zeroing all features except explainer-identified important ones: confmasked(ˆy) −
confmasked(ˆ 2 y). A high margin indicates that the identified features sufficiently support the predic-
tion.
Maskout Contrastivity Margin after zeroing top-ranked features while keeping others: confmaskout(ˆy)− confmaskout(ˆ 2 y).
Margin collapse indicates necessity; comparison across correct/incorrect predictions reveals dif-
ferential feature function.

Dimension 4: Explanation Faithfulness via Fidelity Asymmetry

Fidelity+ (Insertion) Confidence increase when adding top-ranked features to neutral baseline: M+= conf(f (x +
features))− conf(f (xneutral)). Measures collective decision reproduction.
Fidelity- (Deletion) Confidence decrease when removing top-ranked features: M−= conf(f (x)) − conf(f (x −
features)). Measures performance degradation from feature absence.
Asymmetry Comparative balance metric: A = (M−− M+)/(|M−| +|M+|). Range [− 1 , +1] quantifies
deletion versus insertion effect magnitude without prescriptive interpretation.


Appendix C. Dimension-Level Coefficient Analysis


Table C.9: Dimension importance by graph type and explainability method. Weight (%) shows
average dimension importance estimated from bootstrap logistic regression. Top Feature shows
the most important feature with average coefficient (β ̄).

```
(a) AG News Dataset
Graph Type Dimension Weight (%) Top Feature β ̄
```
```
Constituency
```
```
AUC 41.4 Insertion AUC 4.87
Progression 10.1 Maskout Drop (k1) 4.94
Consistency 22.8 Baseline Margin 3.44
Fidelity 25.7 Fidelity- 2.50
```
```
Syntactic
```
```
AUC 41.1 Deletion AUC 5.92
Progression 10.8 Sufficiency Drop (k1) 6.09
Consistency 34.2 Preservation Sufficiency 4.62
Fidelity 13.9 Fidelity- 2.74
```
```
Skipgrams
```
```
AUC 35.5 Deletion AUC 5.93
Progression 12.8 Sufficiency Drop (k1) 8.35
Consistency 35.7 Baseline Margin 4.59
Fidelity 15.9 Fidelity+ 4.60
```
```
Window
```
```
AUC 39.6 Deletion AUC 6.97
Progression 9.7 Sufficiency Drop (k1) 7.65
Consistency 40.8 Baseline Margin 6.62
Fidelity 10.0 Fidelity- 5.99
```
```
Tokens
```
```
AUC 32.7 Deletion AUC 6.05
Progression 11.8 Sufficiency Drop (k1) 3.82
Consistency 42.5 Preservation Necessity 3.27
Fidelity 13.0 Fidelity- 2.26
(b) SST-2 Dataset
Graph Type Dimension Weight (%) Top Feature β ̄
```
```
Constituency
```
```
AUC 35.6 Insertion AUC 2.51
Progression 14.7 Maskout Drop (k10) 2.40
Consistency 32.0 Preservation Sufficienty 2.41
Fidelity 17.7 Fidelity+ 1.32
```
```
Syntactic
```
```
AUC 32.9 Insertion AUC 2.06
Progression 11.5 Maskout Drop (k3) 1.70
Consistency 34.0 Preservation Sufficiency 3.15
Fidelity 21.7 Fidelity- 2.09
```
```
Skipgrams
```
```
AUC 36.3 Insertion AUC 2.83
Progression 26.9 Sufficiency Drop (k1) 5.81
Consistency 34.5 Preservation Sufficiency 1.54
Fidelity 2.3 Fidelity+ 1.48
```
```
Window
```
```
AUC 35.5 Deletion AUC 2.27
Progression 26.1 Sufficiency Drop (k1) 3.26
Consistency 33.8 Preservation Sufficiency 1.44
Fidelity 4.6 Fidelity- 1.15
```
```
Tokens
```
```
AUC 38.4 Deletion AUC 1.90
Progression 27.5 Sufficiency Drop (k1) 5.10
Consistency 31.9 Baseline Margin 1.65
Fidelity 2.2 Fidelity- 0.39
```

