## **HalluZig: Hallucination Detection using Zigzag Persistence**

**Shreyas N. Samaga** **Gilberto Gonzalez Arroyo** **Tamal K. Dey**
Department of Computer Science
Purdue University
West Lafayette, IN
{ssamaga,gonza982,tamaldey}@purdue.edu


**Abstract**



The factual reliability of Large Language Models (LLMs) remains a critical barrier to their
adoption in high-stakes domains due to their
propensity to hallucinate. Current detection
methods often rely on surface-level signals
from the model’s output, overlooking the failures that occur within the model’s internal reasoning process. In this paper, we introduce a
new paradigm for hallucination detection by
analyzing the dynamic topology of the _evolu-_
_tion of model’s layer-wise attention_ . We model
the sequence of attention matrices as a _zigzag_
_graph filtration_ and use _zigzag persistence_, a
tool from Topological Data Analysis, to extract
a topological signature. Our core hypothesis
is that factual and hallucinated generations exhibit distinct topological signatures. We validate our framework, HalluZig, on multiple
benchmarks, demonstrating that it outperforms
strong baselines. Furthermore, our analysis reveals that these topological signatures are generalizable across different models and hallucination detection is possible only using structural signatures from partial network depth.


**1** **Introduction**


Large Language Models (LLMs) form the foundation of modern natural language processing, powering systems for search, question answering, decision support for various domains such as healthcare, law and finance. Despite their impressive
fluency, LLMs are prone to _hallucination_ - the
generation of confident yet factually incorrect or
unsupported content (Huang et al., 2025; Sahoo
et al., 2024). This undermines trust and remains a
critical barrier in the adoption of LLMs in safetycritical applications. A growing body of work has
sought to address this (Manakul et al., 2023; Sriramanan et al., 2024; Fadeeva et al., 2024; Farquhar et al., 2024; Azaria and Mitchell, 2023; Zhou
et al., 2025; Bazarova et al., 2025; Orgad et al.,



Figure 1: Attention matrices modeled as graphs show
distinct topological patterns as they evolve through a
model’s layers. We leverage zigzag persistence in topological data analysis to quantify these evolving attention
structures for hallucination detection.


2025; Binkowski et al., 2025), yet most of the existing methods share a common limitation: they
primarily operate on the final output text or shallow
token-level statistics. They inspect the result of the
model’s reasoning and not the reasoning pathway.

To address this, we shift the focus from _what_ the
model generates to _how_ it arrives at its conclusion.
We hypothesize that a faithful generation relies on
coherent flow of information, where tokens consistently attend to the relevant evidence across layers.
Contrastingly, hallucination may arise when this
flow breaks down and attention patterns diverge
towards spurious contexts, leading to fabricated
answers. Capturing these dynamics requires tools
that can characterize how the structure of attention
evolves across the model’s layers.

Characterizing this evolving structure is a nontrivial task. While aggregate metrics of the attention matrix, such as eigenvalues (Binkowski
et al., 2025) or the determinant (Sriramanan et al.,
2024), provide valuable insights, they do not capture higher-order structural properties. They cannot, for example, describe how distinct groups of
tokens form conceptual clusters or how reason

Figure 2: **The HalluZig framework: capturing attention evolution for hallucination detection.** We model the
layer-wise attention matrices from an LLM as a sequence of attention graphs ( _G_ 1 _, . . ., GL_ ). Zigzag persistence is
applied to this sequence to capture the evolution of topological features resulting in a Persistence Diagram. The
Persistence Diagram is vectorized into a topological signature, which is used by a classifier to detect hallucinations.



ing loops are formed. Topological Data Analysis
(TDA) provides a principled way to bridge this gap.


TDA (Edelsbrunner and Harer, 2010; Dey and
Wang, 2022) provides a mathematical language to
describe the ‘shape’ of this evolving structure. Persistent Homology, a flagship concept of TDA, is a
method for identifying the most robust structural
features of a system by analyzing it at all scales simultaneously. This is achieved through a filtration,
a process analogous to gradually lowering a threshold on attention weights to see which conceptual
clusters and reasoning loops are fundamental (i.e.,
they persist for a long time) versus those that are
ephemeral artifacts of noise. However, standard
persistent homology is limited to analyzing systems that only grow, following a nested sequence
( _G_ 0 _⊆_ _G_ 1 _⊆_ _. . ._ ).


This model is insufficient for capturing evolution of attention, where the structure is not merely
augmented but is completely transformed between
layers, with connections being both formed and
broken. To capture this complex evolution, we
leverage _zigzag persistence_ (Carlsson and de Silva,
2010; Maria and Oudot, 2016; Carlsson et al.,
2019; Dey and Hou, 2022). Zigzag persistence
is an extension of standard persistence designed to
track topological changes through a series of inclusions (additions) and deletions (subtractions). Thus,
viewing through the lens of zigzag persistence enables us to move beyond quantifying individual
connections, to characterizing the topology of the
evolving attention matrices.


In this paper, we introduce HalluZig - a novel



framework that captures the _evolution of attention_
in an LLM using _zigzag persistence_ . We model the
attention matrix at each layer as a graph ( _attention_
_graph)_ and connect successive graphs through a
_zigzag filtration_ (Dey and Wang, 2022, Chapter 4).
By computing the zigzag persistence of this filtration, we obtain a topological summary quantifying
births and deaths of connected components and cycles as information propagates through the model.
This topological summary is distilled into a numerical vector to give a _topological signature_ using
established methods (Adams et al., 2017; Atienza
et al., 2020). Our core hypothesis is that factual and
hallucinated responses leave distinct topological
signatures in the evolution of the model’s attention.
To evaluate this hypothesis, we experiment with
diverse datasets, which include human annotated
datasets and generic QA based datasets which we
annotate using LLM-as-a-judge paradigm (Zheng
et al., 2023). Our experiments confirm the hypothesis, showing that the topological signatures can
distinguish between factual and hallucinated responses. Further, we demonstrate the applicability
of HalluZig for hallucination detection using partial network depth. We show that it can reliably
detect hallucinations by analyzing only the first
70% of the model’s layers with minimal degradation in accuracy compared to a full-model analysis.
Furthermore, we demonstrate that these topological signatures are not model-specific, exhibiting
remarkable zero-shot generalization when transferred between different LLM architectures.


In summary, our main contributions are as fol

lows: (1) We propose a new framework for hallucination detection by modeling the evolution of
attention through an LLM. (2) To the best of our
knowledge, this is the first application of zigzag persistence for capturing the layer-to-layer structural
transformations of attention graphs for this task. (3)
We demonstrate that HalluZig outperforms strong
baselines on multiple datasets annotated for hallucination detection. (4) We empirically demonstrate that HalluZig achieves near-maximum performance when restricted to topological signatures
from the first 70% of model layers.


**2** **Related Work**


Research in hallucination detection has attracted
a lot of attention in the recent times (Huang et al.,
2025; Zhang et al., 2023; Wang et al., 2024) can
be broadly classified into _black-box_ methods and
_white-box_ methods.


**Black-box methods.** These methods operate
only on the model’s final text output. Consistency based techniques such as (Manakul et al.,
2023; Chen et al., 2024; Kuhn et al., 2023; Qiu
and Miikkulainen, 2024; Nikitin et al., 2024) evaluate agreement among multiple generations. These
methods rely on multiple model runs which imposes significant computational overhead. Surfacelevel confidence measures, such as perplexity,
logit entropy, or predictive uncertainty (Fadeeva
et al., 2024; Malinin and Gales, 2021) provide
lightweight alternatives but with limited discrimating power, as they do not consider the model’s
internal reasoning process.


**White-box methods.** These methods leverage
internal representations such as hidden states, attention maps, or logits. One of the early works
in this direction was (Azaria and Mitchell, 2023)
which had a linear probe into these states to determine factuality. Subsequent studies such as (Farquhar et al., 2024; Chen et al., 2024; CH-Wang
et al., 2024) quantified internal uncertainties by
comparing hidden states across multiple generations. (Du et al., 2024) reduced annotation requirements while (Kossen et al., 2024) learned to approximate expensive self-consistency scores. A
more recent and less explored white-box direction
involves analyzing _attention maps_ . Chuang et al.
(2024) introduced lookback ratio which measures
how strongly a model attends to relevant input tokens when generating context-dependent answers.



Sriramanan et al. (2024) introduced simple attention statistics to flag a response as hallucinated in
an unsupervised manner. Binkowski et al. (2025)
use the eigen values of the attention matrix and
the eigen values of the Laplacian of the attention
matrix (modelled as a graph) to classify whether
a response is hallucinated. Bazarova et al. (2025)
introduce a topology-based hallucination detection
technique which leverages a topological divergence
metric between the prompt and the response subgraphs. These techniques typically exploit only
local or scalar attention features, overlooking the
rich, global structure of the evolution of attention
graph that HalluZig aims to capture.


**3** **Method**


**3.1** **Attention Mechanism**


The _self-attention_ (Vaswani et al., 2017) is the core
component of the transformer architecture, which
allows the LLM to dynamically weigh the importance of different tokens in a sequence when producing a representation for a given token.
At a high level, self-attention operates using
three learned vector representations for each token:
(1) Query( _Q_ ), (2) Key( _K_ ) and (3) Value( _V_ ). Given
a generated sequence of tokens _S_ = _{t_ 1 _, . . ., tT }_,
let _X ∈_ R _[T]_ _[×][d]_ denote the matrix of _T_ tokens,
each having a _d_ -dimensional representation. Let
_WQ, WK, WV ∈_ R _[d][×][d]_ denote the trainable projection matrices. Then, the three vector representations _Q, W_ and _V_ are _XWQ, XWK_ and _XWV_
respectively. Note that _Q, K, V ∈_ R _[T]_ _[×][d]_ . The
attention mechanism is defined as follows:



**3.2** **Attention Graph Construction**


For each layer _l ∈{_ 1 _,_ 2 _, . . ., L}_, we average the
attention matrices across all heads to get a mean
attention matrix _A_ [(] _[l]_ [)] _∈_ R _[T]_ _[×][T]_ . We model this
as a weighted, graph _Gl_ = ( _V, El, wl_ ), where _V_
is the set of tokens, and _wl_ ( _ti, tj_ ) is the attention




        - _QKT_
_Attn_ ( _Q, K, V_ ) = softmax ~~_√_~~



~~_√_~~



_d_




_V,_



where _d_ denotes the dimension of the token embed
           - _QK_ _[T]_            ding. The matrix _A_ = softmax ~~_√_~~ is called




           - _QK_ _[T]_            ding. The matrix _A_ = softmax ~~_√_~~ is called

_d_

the _attention matrix_ .
Modern LLMs use multi-head attention, which
performs this process multiple times in parallel
with different, learned projections for _Q, K_ and _V_ .
We denote the attention matrix at head _h_ of layer _l_
as _A_ [(] _[l,h]_ [)] .



~~_√_~~



_d_


weight from token _ti_ to _tj_ . We choose the top _k_ percentile of attention weights to form the edges of the
graph _Gl_ in order to focus on the most significant
structural connections within the layer, ensuring
our analysis is robust to the noise from low-value
attention weights. We refer to these graphs as _at-_
_tention graphs_ . This gives us a sequence of graphs,
which captures the state of information flow at a
specific depth in the model. Refer to Figure 3 for
an illustration.



Figure 4: **Zigzag filtration.** The figure shows a zigzag
filtration where _G_ 2 = _G_ 1 _∪_ _G_ 3 and _G_ 4 = _G_ 3 _∪_ _G_ 5.


were to be reversed, we get a _zigzag filtration Z_


_Z_ : _G_ 0 _⊆_ _G_ 1 _⊇_ _G_ 2 _⊆_ _. . . ⊇_ _Gn._


Refer to Figure 4. Unlike a standard filtration, which only allows the addition of new vertices/edges, a zigzag filtration also allows their removals. Attention values between tokens, as a sentence passes through various layers, keep evolving.
To capture the evolution of attention graphs across
layers, it is crucial to allow the deletion of edges.
This is because two tokens that exhibit strong mutual attention in layer _li_ may no longer do so in
layer _lj_, requiring the corresponding edge to be
removed from the attention graph at that layer.
We obtain a zigzag persistence module _MZ_ from
such a zigzag filtration of graphs by computing
the homology groups in dimensions 0 (# of connected components) and 1 (# of independent cycles) of each graph _Gi_ appearing in the filtration.
In other words, for each stage _Gi_, we record the
_p_ -dimensional topological features (such as connected components for _p_ = 0, loops for _p_ = 1).
These homology groups are then connected by linear maps that follow the direction of the inclusion
arrows in _Z_ .


_MZ_ : _Hp_ ( _G_ 0) _→_ _Hp_ ( _G_ 1) _←_ _. . . ←_ _Hp_ ( _Gn_ )


The persistence module _MZ_ thus encodes how
topological features appear, disappear, or reappear
as the graphs evolve. We refer the reader to Appendix A for formal definitions of (simplicial) homology groups.
A key result in zigzag persistence is that any
zigzag persistence module _MZ_ decomposes into
a direct sum of indecomposable interval modules (Carlsson and de Silva, 2010). Each interval
module corresponds to a contiguous range of indices in the zigzag filtration over which a homology
class exists. Intuitively, each interval corresponds
to the “lifetime” of an individual topological feature. This decomposition is unique up to reordering.
This collection of intervals provides a complete and



(a) **A single-head attention**
**matrix from a decoder-only**
**model.** The matrix is lowertriangular due to the causal
mask, which prevents tokens
from attending to future positions in the sequence.



(b) **An attention graph de-**
**rived from the matrix in Fig-**
**ure 3a.** The nodes represent
tokens, while the edges visualize the of attention. The thickness of each edge corresponds
to its attention weight, highlighting the key inter-token relationships that form the basis
for our topological analysis.



Figure 3: **Visualizing the Attention Mechanism.** (a) A
causally masked attention matrix. (b) The corresponding attention graph where nodes are tokens and thick
edges represent high-attention links. This structural representation is the input to our topological pipeline.


**3.3** **Topological Signatures**


To analyze the structure of attention dynamics, we
employ tools from TDA. TDA provides a mathematical framework for characterizing the "shape"
of complex data. Here, we introduce the core concepts, building intuition from the familiar structure
of an attention graph.
Our starting point is an attention graph, where tokens are vertices and attention scores are weighted
edges. TDA provides a formal language to describe
the structure of such graphs that keep on changing.
A _filtration F_ of graphs is a nested sequence of
graphs indexed by natural numbers/integers


_F_ : _G_ 0 _⊆_ _G_ 1 _⊆_ _G_ 2 _. . . ⊆_ _Gn._


Notice that the inclusions here are all in the forward
direction, i.e., the sequence is a non-decreasing
sequence of graphs. Now, if some of the inclusions


concise summary of the filtration’s topological dynamics and can be visualized in two equivalent
ways:


  - Barcodes: A collection of horizontal line segments, where the length and the position of
each line segment represents the lifetime of a
feature.


  - Persistence Diagrams: A 2D scatter plot of
(birth, death) coordinates for each feature.







Here, _Gl ∪_ _Gl_ +1 is an intermediate graph containing the union of vertices and edges in _Gl_ and _Gl_ +1.
Since the vertex set is the same for graphs from
each layer, _Gl ∪_ _Gl_ +1 is the union of the edges of
_Gl_ and _Gl_ +1. Refer to Figure 4 for an illustration.
The weight of the edge in _Gl ∪_ _Gl_ +1 is taken as
the maximum of the weights of the edge in _Gl_ and
_Gl_ +1 if it exists in both graphs. This explicitly
models how the structure of _Gl_ is transformed to
produce the structure of _Gl_ +1. The pairwise connections weave together all graphs from layer 1
to layer L to model the full sequence of attention
dynamics.
In our case, the vertex set of the graphs do not
change and thus we focus on the evolution of loops
(representing cyclic attention on tokens). Algebraically, we track the evolution of the number
of independent cycles or loops in _Gl_, which is the
rank of the 1-dimensional homology group _H_ 1( _Gl_ ).
Such loops can be interpreted in two ways: as stable, resonant structures that reinforce a coherent
semantic concept, or as flawed, circular reasoning
pathways. We hypothesize that these two cases
can be distinguished by their topological persistence. Cycles in factual statements are likely to
be formed by strong, stable attention weights that
persist across multiple layers, representing the successful consolidation of a concept. Conversely, we
posit that hallucinations may manifest as numerous, short-lived, and structurally unstable cycles,
indicative of spurious thought loops. By analyzing
the 1-dimensional persistence diagrams, we aim to
capture these signatures of flawed reasoning that
are invisible to simpler connectivity measures.
In this paper, we use three vectorization schemes
to vectorize the zigzag persistence diagrams and
capture the topological information present in
the barcodes. Different vectorization techniques
present different representations of the underlying
topological information:
**(1) Persistence Images (PersImg) (Adams**
**et al., 2017):** PersImg treats the persistence diagram as a 2D distribution. It projects each persistence point ( _b, d_ ) onto a grid using a Gaussian
kernel, effectively creating a heatmap or "image"
that captures the geometric density and spatial relationships of topological features.
**(2)** **Persistence** **Entropy** **(PersEntropy)**
(Atienza et al., 2020): PersEntropy provides a
statistical summary of the barcode. It calculates
the Shannon entropy (Shannon, 1948) of the
distribution of feature lifetimes (the lengths of the






|20<br>Death<br>15<br>10<br>5<br>0<br>0 5 10 15<br>Birt|0<br>5<br>0<br>5|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|~~0~~<br>~~5~~<br>~~10~~<br>~~15~~<br>Birt<br>0<br>5<br>10<br>15<br>20<br>Death|0<br>|||||~~y = x (diag~~|~~   onal)~~|
|~~0~~<br>~~5~~<br>~~10~~<br>~~15~~<br>Birt<br>0<br>5<br>10<br>15<br>20<br>Death|0<br>||||~~20~~<br>~~25~~<br>~~30~~<br>h|~~20~~<br>~~25~~<br>~~30~~<br>h|~~20~~<br>~~25~~<br>~~30~~<br>h|



(a) **Persistence** **Diagram.**
Each point ( _b, d_ ) corresponds
to a topological feature born
at filtration value _b_ and dying
at _d_ . The vertical distance
from the diagonal _y_ = _x_,
known as the feature’s persistence.



(b) **Persistence** **Barcode.**
Each horizontal bar represents a single topological
feature, starting at its birth
layer (x-axis) and ending at
its death layer. The length
of the bar directly visualizes
the feature’s persistence. For
readability, this plot displays
the first 100 bars, sorted
vertically by birth time.



Figure 5: **Two Views of a Topological Summary.** (a)
The 2D persistence diagram plots each feature’s birth
vs. death layers. (b) The 1D barcode represents each
feature’s lifetime as a horizontal bar.


**3.4** **Featurization and Classification**


A persistence diagram is not a suitable input for
most machine learning models. We need to transform each persistence diagram into a vector, which
would be representative of the topological information captured in the persistence diagram. This
vector can be used as input to a machine learning classifier. There are multiple known vectorization techniques such as persistence images (Adams
et al., 2017), persistence entropy (Atienza et al.,
2020), persistence landscapes (Bubenik, 2015),
Betti curves (Umeda, 2017).


**3.5** **Zigzag Persistence of Attention Dynamics**


To analyze the information flow between layers, we
focus on consecutive pairs of graphs _Gl_ and _Gl_ +1.
We construct a short zigzag filtration between these
two graphs as follows:


_Gl �→_ _Gl ∪_ _Gl_ +1 _←�Gl_ +1 _._


Model Measure AUCROC Accuracy TPR @ 5% FPR F1 Score


Self-Prompt 50.30 50.30                 - 66.53
FAVA Model 53.29 53.29              - 43.88
SelfCheckGPT-Prompt 50.08 54.19             - 67.24
INSIDE 59.03 57.98 13.17 39.66
Llama-2-7b LLM-Check (Attn Score) 72.34 67.96 14.97 69.27
PersImg **82.09** **75.00** **26.79** **80.67**
PersEntropy 75.67 72.82 21.43 78.26
Betti Curve 74.75 68.47 23.21 77.86


LLM-Check (Attn Score) 68.19 65.87 15.57 70.53
PersImg **82.64** **73.91** **46.43** **80.33**
Llama-3-8b PersEntropy 74.09 71.73 19.29 78.69
Betti Curve 74.33 70.65 21.43 77.69


LLM-Check (Attn Score) 71.69 66.47 24.55 62.00
PersImg **83.28** **77.17** **35.71** **82.35**
Vicuna-7b PersEntropy 75.47 68.47 17.86 76.03
Betti Curve 76.71 70.65 24.59 80.29


Table 1: **Hallucination detection results on the FAVA Annotated Dataset.** The LLM-Check (Attn Score),
Self-Prompt, FAVA Model, SelfCheckGPT-Prompt and INSIDE results are according to the numbers in Sriramanan
et al. (2024).



bars or equivalently, ( _d −_ _b_ ) for a point ( _b, d_ ) in
the persistence diagram). A single entropy value
quantifies the overall complexity of the topological
signature, with higher values indicating more
uniform persistence across features.
**(3) Betti Curve (Umeda, 2017)** : This method
produces a 1D vector by plotting the Betti number,
the count of currently active bars, as a function
of the filtration value. The resulting curve, which
tracks the feature counts, is then sampled at discrete
points to form the feature vector.
For simplicity, we refer to our overall approach
as HalluZig, regardless of the specific vectorization scheme employed.


**4** **Experiments**


**4.1** **Experimental Setup**


**Datasets.** To ensure comprehensive evaluation,
we test HalluZig on a diverse suite of benchmarks
covering different domains and annotation styles:
_Generative Benchmarks:_ We use two benchmarks with explicit human provided hallucination
labels. The FAVA Annotated Dataset (Mishra
et al., 2024) provides passage-level binary labels
for Wikipedia abstract generation. The RAGTruth
Summarization Dataset (Niu et al., 2024) offers
span-level annotations, which we normalize to
passage-level task: a summary is considered if it
contains one or more annotated spans.
_QA-based Benchmarks:_ We assess the performance on TruthfulQA (Lin et al., 2022) and NQOpen (Kwiatkowski et al., 2019) datasets. As these



datasets do not contain explicit hallucination labels,
we employ the LLM-as-a-judge paradigm (Zheng
et al., 2023). We use GPT-4o-mini (Hurst et al.,
2024), a closed-source LLM, to automatically annotate whether a generated answer is a hallucination or not.


**Models and metrics.** For our experiments, we
use a diverse set of open-source LLMs, including models from the Llama family (Llama-2-7b,
Llama-2-13b (Touvron et al., 2023), Llama-3-8b,
Llama-3.1-8b, Llama-3.2-3b (Grattafiori et al.,
2024)), Vicuna-7b (Chiang et al., 2023) and
Mistral-7b (Jiang et al., 2023), all accessed via
the Hugging Face transformers library (Wolf et al.,
2020). Our topological pipeline is implemented
using the FastZigzag library (Dey and Hou, 2022)
for persistence computation and the Gudhi library
(Carrière et al., 2025) for vectorization. We employ
a Random Forest Classifier for the final binary classification and evaluate its performance using Accuracy, F1-Score, AUC-ROC, and TPR@5%FPR.
For clarity in our main results, we report the performance of the best run across multiple random seeds.
To ensure full reproducibility, we provide the mean
and standard deviation for all experiments, along
with a detailed list of all hyperparameters, in Appendix B. Moreover, all the results reported in
the paper are the result of a single LLM run of
the respective models. The code is available at
[https://github.com/TDA-Jyamiti/halluzig](https://github.com/TDA-Jyamiti/halluzig)


**Baselines.** Since our method leverages structural
signals embedded in attention matrices, we eval

uate our approach against baselines that capture
analogous information. On generative benchmarks
(FAVA Annotated Dataset and RAGTruth Summarization Dataset), we compare against the Attention Score metric from the LLM-Check (Sriramanan et al., 2024). To provide additional context
on where HalluZig stands with respect to other
uncertainty quantification methods, we add SelfPrompt (Kadavath et al., 2022), INSIDE (Chen
et al., 2024), SelfCheckGPT-Prompt (Manakul
et al., 2023) and FAVA Model (Mishra et al., 2024)
to the comparisoin. For the QA benchmarks, we
add LapEigVals (Binkowski et al., 2025) to the
comparison as it extracts spectral properties from
attention matrices.


**4.2** **Main Results**


HalluZig outperforms the baselines across models
on most datasets. The key empirical findings are
summarized below.



RAGTruth Summarization dataset. While performance varies by vectorization, both PersImg and
Betti Curve significantly outperform the baseline in both scenarios. This demonstrates that
the dynamic structural information captured by
HalluZig is a robust signal for hallucination, even
when direct model access is limited. Refer to Figure 6 for an illustration.



|LLM|Feature|NQOpen TruthfulQA|
|---|---|---|
|Llama-3.1-8b|AttentionScore<br>AttnEigvals<br>LapEigvals<br>PersImg<br>PersEntropy<br>Betti Curve|0.556<br>0.541<br>0.732<br>0.587<br>**0.748**<br>0.589<br>0.730<br>**0.733**<br>0.682<br>0.664<br>0.715<br>0.684|


Llama-3.2-3b



AttentionScore 0.546 0.581
AttnEigvals 0.694 0.535
LapEigvals 0.693 0.539
PersImg **0.712** 0.641
PersEntropy 0.685 0.656
Betti Curve 0.688 **0.667**






|Method Metric White-box Black-box<br>Llama-2-7b Llama-2-13b Mistral-7b|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**Method**<br>**Metric**<br>**White-box**<br>**Black-box**<br>Llama-2-7b<br>Llama-2-13b<br>Mistral-7b|**Method**<br>**Metric**<br>**White-box**<br>**Black-box**<br>Llama-2-7b<br>Llama-2-13b<br>Mistral-7b|**Method**<br>**Metric**<br>**White-box**<br>**Black-box**<br>Llama-2-7b<br>Llama-2-13b<br>Mistral-7b|Llama-2-13b|Llama-2-13b|
|Attention Score|AUCROC<br>Accuracy<br>TPR @ 5% FPR<br>F1 Score|54.19<br>54.52<br>5.88<br>54.50|60.05<br>59.66<br>14.48<br>55.97|55.37<br>56.99<br>5.18<br>57.72|
|PersImg|AUCROC<br>Accuracy<br>TPR @ 5% FPR<br>F1 Score|**73.37**<br>**62.44**<br>19.03<br>**69.10**|**72.90**<br>**61.63**<br>**18.39**<br>**68.63**|**74.45**<br>**63.45**<br>**26.22**<br>**70.84**|
|PersEntropy|AUCROC<br>Accuracy<br>TPR @ 5% FPR<br>F1 Score|51.46<br>52.00<br>7.96<br>60.15|52.78<br>53.95<br>2.69<br>61.22|52.97<br>52.02<br>3.11<br>58.37|
|Betti Curve|AUCROC<br>Accuracy<br>TPR @ 5% FPR<br>F1 Score|69.26<br>57.78<br>**23.89**<br>65.95|67.93<br>60.05<br>14.80<br>67.99|71.03<br>63.00<br>19.11<br>70.69|



Table 2: **Hallucination detection performance on the**
**RAGTruth Summarization Dataset.** We compare
HalluZig (PersImg, PersEntropy, Betti Curve)
with Attention Score from (Sriramanan et al., 2024).
Results are reported for both a white-box setting (using
the Llama-2-7b generator) and a black-box setting (using Llama-2-13b and Mistral-7b as substitutes).


**Performance on Generative Benchmarks.** On
the FAVA Annotated (Mishra et al., 2024) and
RAGTruth Summarization Datasets (Niu et al.,
2024), HalluZig achieves superior performance
(Table 1 and Table 2). A key takeaway is the robustness of the underlying topological signal: all three
vectorization schemes (persistence images, Betti
curves, and persistent entropy) yield competitive
results, confirming that the structural information
is the dominant feature. Furthermore, HalluZig
excels in both black-box and white-box settings on



Table 3: **Hallucination detection performance on QA**
**Benchmark.** Test AUROC scores for baseline and
HalluZig. The results for AttentionScore, AttnEigVals
and LapEigvals are based on the experiments we performed with their methods.


**Performance on QA Benchmarks.** HalluZig’s
effectiveness extends to the QA domain (Table 3). HalluZig shows consistent improvement
on both, TruthfulQA (Lin et al., 2022) and NQOpen (Kwiatkowski et al., 2019) datasets, demonstrating that topological features generalize from
controlled generative tasks to open-domain question answering.
We note that there is a discrepancy in the
LapEigVal score reported in Table 3 and that reported in their original paper (Binkowski et al.,
2025). One of the main reasons is that using
GPT-4o-mini as annotator produced a heavily class
imbalanced dataset, (711/91/15) in our case versus (~500/~250/~80) reported in (Binkowski et al.,
2025). Moreover, we used the temperature values
as 0.7 for all our experiments while the results reported in (Binkowski et al., 2025) are with 0.1 and
1.0. Our aim for these experiments was only to
ensure a fair comparison between HalluZig and
other baselines under similar labeling conditions.
We can see from the results that, generally, PersImg has a higher performance than
BettiCurve and PersEntropy. Persistence image is a 2D heatmap (distribution) that captures the


Figure 6: **Visualization of persistence diagrams for hallucinated versus non-hallucinated responses for FAVA**
**dataset.** The top row depicts the persistence diagrams for three randomly selected non-hallucinated responses while
the bottom row depicts the persistence diagrams for three randomly selected hallucinated responses from FAVA
dataset. The persistence diagrams are colored by multiplicity, i.e., the multiplicity of each point in the diagram
is depicted by its color. We can see that the persistence diagrams look visually different for hallucinated versus
non-hallucinated responses which gets reflected in the HalluZig performance.



spatial relationship between birth and death layers
in a persistence diagram. Betti Curve generally
performs the second best which is a 1-D vector of
counts of the number of features at different points
over the filtration. Persistence Entropy is a single
value summary of the persistence diagram. While
still useful, quantifying the entire 2D plot (persistence diagram) by a single number inherently loses
information, explaining its lower but competitive
performance.


**5** **Analysis and Discussion**


The results demonstrate the efficacy of our method
and also provide deeper insights into the nature
of hallucinations and the potential for new model
safety mechanisms. We analyze the implications
of our key experimental findings.


**5.1** **The Critical Role of Dynamics: Zigzag vs.**
**Static TDA**


To validate our central hypothesis that the dynamics of attention evolution are more informative than
static snapshots, we conducted an ablation study
comparing our full model against a Static TDA
baseline. This baseline computes standard persistent homology on each layer individually, deliberately ignoring the layer-to-layer transformations.
As shown in Figure 7, the Static TDA baseline
underperforms HalluZig by a significant margin.
This provides compelling evidence that a substan

|80<br>60<br>40<br>20<br>0|7<br>61.95%|Col3|3.91% 7<br>69.23%|Col5|5.63%|69.57|Col8|%|Col10|
|---|---|---|---|---|---|---|---|---|---|
|0<br>20<br>40<br>60<br>0|||||||||25.42%|
|0<br>20<br>40<br>60<br>0||||||||<br> <br>12.96%|<br>|



Figure 7: **Role of Dynamics: Comparing HalluZig**
**against a baseline using static, layer-wise persistent**
**homology.** Our full model demonstrates improved
performance across all metrics, notably achieving a
+12.78% F1-score. The results underscore the importance of modeling the dynamic evolution of attention
graphs over static analysis.


tial amount of discriminative information is lost
when inter-layer dynamics are disregarded. The
key takeaway is that the crucial signal for hallucination lies not in the static topology of individual
layers, but in the dynamics of their evolution, a
property that zigzag persistence is equipped to capture, which supports our core hypothesis.


**5.2** **Universality of Signatures: Cross-Model**
**Generalization**


A critical question is whether the learned topological signatures are universal or model-specific. We
















|Col1|Col2|Col3|Col4|Col5|82.35|
|---|---|---|---|---|---|
|||||||
|8<br>9<br>0<br>1|||81.|6||
|8<br>9<br>0<br>1|||~~79.~~|~~4~~|~~80.33~~|
|8<br>9<br>0<br>1|76.9|2<br>~~78.~~<br>77.|~~41~~<br>97<br>|||
|6<br>7|77.0|3||~~Vicun~~<br>Llam|~~a-7B~~<br>a~~-~~3~~-~~8B|


















|80<br>60<br>40<br>20<br>0|Baseline<br>Generali<br>HalluZig|Col3|zation<br>80.95|Col5|Col6|Col7|82.35|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|0<br>20<br>40<br>60<br>80|62.00<br>|62.00<br>|65.10||70.53|70.53|77.61|||
|0<br>20<br>40<br>60<br>80||||||||||
|0<br>20<br>40<br>60<br>80||||||||||



Figure 8: **Zero-Shot Cross-Model Generalization Per-**
**formance.** The figure evaluates the ability of HalluZig
to generalize across different LLM architectures without retraining. (Left) Performance of the classifier
on Llama-3-8b trained exclusively on topological signatures extracted from Vicuna-7b. The generalization model outperforms the baseline (Attn-Score (Sriramanan et al., 2024)) by 3%. (Right) The reverse scenario, showing performance on Vicuna-7b of the classifier trained on Llama-3-8b. The generalization model
surpasses the baseline by 7% in this case.


investigated this via a zero-shot cross-model experiment, training a classifier on topological signatures extracted from Llama-3-8b and testing it
on the topological signatures from Vicuna-7b and
the vice-versa, without retraining. We use FAVA
Annotated Dataset (Mishra et al., 2024) for this
analysis. The results reveal a remarkable degree of
transferability. This result indicates that different
LLMs may exhibit similar topological dynamics
when hallucinating.


**5.3** **Practical Implications: Hallucination**
**Detection from Partial Network Depth**


Our final analysis investigates whether hallucinations can be detected reliably before the
final layer. We computed topological features (PersImg) up to varying model depths of
Vicuna-7b and Llama-3-8b on the FAVA Annotated dataset (Mishra et al., 2024) and evaluated performance at each stage (Figure 9). We observe from
the performance curves that HalluZig achieves an
F1-score nearly identical to that of the full-model
analysis just at 70% of the model’s depth for both
models.
This result suggests that hallucination may not
be a last-minute failure but is encoded and stabilized relatively early in the model’s reasoning pathway.
Together, these analyses reinforce that HalluZig



Figure 9: **Hallucination detection performance as**
**a function of model depth.** The plot shows the F1scores achieved when using topological features from
an increasing percentage of the model’s total layers. We
observe that HalluZig performance achieves over 98%
of the final score by the 70% depth mark. This indicates
that the structural signatures of hallucination are formed
in the middle layers.


is not only as a diagnostic tool but can also be a
safety monitoring mechanism in LLMs.


**6** **Conclusion**


In this paper, we introduced HalluZig, a novel approach for detecting hallucination by analyzing the
topological information hidden in the evolution of
attention in an LLM. By leveraging zigzag persistence, we demonstrated that the structural evolution of attention provides a powerful and robust
signal for hallucination detection. Our experiments
show that this method not only outperforms strong
baselines but also exhibits remarkable cross-model
generalization and enables reliable early detection.
This work establishes the viability of structural interpretability, offering a new lens to understand and
improve the trustworthiness of LLMs.


**Acknowledgements**


This work is partially supported by NSF grants
DMS-2301360 and CCF-2437030.


**Limitations**


While our work establishes the viability of topological analysis of evolving attention for hallucination
detection, we point out some limitations that can
be in agenda for future research.


**Computational Complexity**


The primary limitation of our current approach is
the computational cost of computing persistent homology, particularly for long sequences (sentences)


which result in large attention graphs. However, we
note that this computational challenge is not unique
to our framework, but is a well-known limitation
within the applied TDA community.


**Scope of Analysis**


Our framework focuses exclusively on the attention
mechanism. However, other model components,
particularly the MLP layers, are known to store
and manipulate factual knowledge. Our approach
is currently blind to procedural failures that may
originate solely within these components. A more
holistic approach could integrate topological signals from both attention and MLP activations.


**Attention-Head Averaging**


A methodological choice of our framework is the
mean-pooling of attention heads to create a single graph per layer. While this provides a stable,
holistic view of information flow and ensures computational tractability, it is an important limitation.
It is known that attention heads can specialize in
distinct functions, and our averaging approach may
dilute or obscure a strong, localized signal from
a single “rogue” head whose aberrant behavior is
the primary cause of a hallucination. Future work
could pursue more fine-grained, head-specific topological analysis to gain deeper diagnostic insights,
though this would entail a significant increase in
computational cost.


**Scope of Model Scale**


Our experimental validation is conducted on opensource LLMs with up to 13 billion parameters. This
leads to the natural question of whether our findings
on the topological dynamics of hallucination generalize to much larger, state-of-the-art foundation
models (e.g., 70B+ parameters). We hypothesize
that the observed structural patterns are a fundamental property of the Transformer architecture
and will therefore apply to larger models. However,
empirically verifying this scalability is a direction
for future research.


**Ethical Considerations**


To the best of our knowledge, we did not violate any
ethical code while conducting the research work described in this paper. We report the technical details
needed for reproducing the results and will release
the code upon acceptance. All results are from a
machine learning model and should be interpreted



as such. The LLMs used to generate attention matrices for this paper are publicly available and are
allowed for scientific research.


**References**


Henry Adams, Tegan Emerson, Michael Kirby, Rachel
Neville, Chris Peterson, Patrick Shipman, Sofya Chepushtanova, Eric Hanson, Francis Motta, and Lori
Ziegelmeier. 2017. Persistence images: A stable vector representation of persistent homology. _Journal of_
_Machine Learning Research_, 18(8):1–35.


Josh Alman, Ran Duan, Virginia Vassilevska Williams,
Yinzhan Xu, Zixuan Xu, and Renfei Zhou. 2025.
_[More Asymmetry Yields Faster Matrix Multiplication](https://doi.org/10.1137/1.9781611978322.63)_,
pages 2005–2039.


Nieves Atienza, Rocio Gonzalez-DÃaz, and Manuel
[Soriano-Trigueros. 2020. On the stability of persis-](https://doi.org/10.1016/j.patcog.2020.107509)
[tent entropy and new summary functions for topolog-](https://doi.org/10.1016/j.patcog.2020.107509)
[ical data analysis.](https://doi.org/10.1016/j.patcog.2020.107509) _Pattern Recognition_, 107:107509.


Amos Azaria and Tom Mitchell. 2023. The internal
state of an LLM knows when it’s lying. In _Find-_
_ings of the Association for Computational Linguistics:_
_EMNLP 2023_, pages 967–976, Singapore. Association for Computational Linguistics.


Alexandra Bazarova, Aleksandr Yugay, Andrey Shulga,
Alina Ermilova, Andrei Volodichev, Konstantin
Polev, Julia Belikova, Rauf Parchiev, Dmitry
Simakov, Maxim Savchenko, Andrey Savchenko,
Serguei Barannikov, and Alexey Zaytsev. 2025.
[Hallucination detection in llms with topologi-](https://arxiv.org/abs/2504.10063)
[cal divergence on attention graphs.](https://arxiv.org/abs/2504.10063) _Preprint_,
arXiv:2504.10063.


Jakub Binkowski, Denis Janiak, Albert Sawczyn, Bog[dan Gabrys, and Tomasz Kajdanowicz. 2025. Hallu-](https://arxiv.org/abs/2502.17598)
[cination detection in llms using spectral features of](https://arxiv.org/abs/2502.17598)
[attention maps.](https://arxiv.org/abs/2502.17598) _Preprint_, arXiv:2502.17598.


[Peter Bubenik. 2015. Statistical topological data anal-](https://doi.org/10.5555/2789272.2789275)
[ysis using persistence landscapes.](https://doi.org/10.5555/2789272.2789275) _J. Mach. Learn._
_Res._, 16:77–102.


Gunnar Carlsson and Vin de Silva. 2010. Zigzag persistence. _Foundations of computational mathematics_,
10(4):367–405.


Gunnar Carlsson, Vin De Silva, Sara Kališnik, and
Dmitriy Morozov. 2019. Parametrized homology via
zigzag persistence. _Algebraic & Geometric Topology_,
19(2):657–700.


Mathieu Carrière, Gard Spreemann, and Wojciech Reise.
[2025. Persistence representations scikit-learn like](https://gudhi.inria.fr/python/3.11.0/representations.html)
[interface. In](https://gudhi.inria.fr/python/3.11.0/representations.html) _GUDHI User and Reference Manual_,
3.11.0 edition. GUDHI Editorial Board.


Sky CH-Wang, Benjamin Van Durme, Jason Eisner, and
[Chris Kedzie. 2024. Do androids know they’re only](https://doi.org/10.18653/v1/2024.findings-acl.260)
[dreaming of electric sheep? In](https://doi.org/10.18653/v1/2024.findings-acl.260) _Findings of the As-_
_sociation for Computational Linguistics: ACL 2024_,


pages 4401–4420, Bangkok, Thailand. Association
for Computational Linguistics.


Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu,
Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024.
[INSIDE: LLMs’ Internal States Retain the Power](https://openreview.net/forum?id=Zj12nzlQbz)
[of Hallucination Detection. In](https://openreview.net/forum?id=Zj12nzlQbz) _The Twelfth Interna-_
_tional Conference on Learning Representations_ .


Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng,
Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan
Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion
[Stoica, and Eric P. Xing. 2023. Vicuna: An open-](https://lmsys.org/blog/2023-03-30-vicuna/)
[source chatbot impressing gpt-4 with 90%* chatgpt](https://lmsys.org/blog/2023-03-30-vicuna/)
[quality.](https://lmsys.org/blog/2023-03-30-vicuna/)


Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, and James R. Glass. 2024.
[Lookback Lens: Detecting and Mitigating Contex-](https://doi.org/10.18653/v1/2024.emnlp-main.84)
[tual Hallucinations in Large Language Models Using](https://doi.org/10.18653/v1/2024.emnlp-main.84)
[Only Attention Maps. In](https://doi.org/10.18653/v1/2024.emnlp-main.84) _Proceedings of the 2024_
_Conference on Empirical Methods in Natural Lan-_
_guage Processing_, pages 1419–1436, Miami, Florida,
USA. Association for Computational Linguistics.


[Tamal K. Dey and Tao Hou. 2021. Computing zigzag](https://doi.org/10.4230/LIPICS.SOCG.2021.30)
[persistence on graphs in near-linear time. In](https://doi.org/10.4230/LIPICS.SOCG.2021.30) _37th_
_International Symposium on Computational Geome-_
_try, SoCG 2021, June 7-11, 2021, Buffalo, NY, USA_
_(Virtual Conference)_, volume 189 of _LIPIcs_, pages
30:1–30:15. Schloss Dagstuhl - Leibniz-Zentrum für
Informatik.


[Tamal K. Dey and Tao Hou. 2022. Fast Computation](https://doi.org/10.4230/LIPIcs.ESA.2022.43)
[of Zigzag Persistence. In](https://doi.org/10.4230/LIPIcs.ESA.2022.43) _30th Annual European_
_Symposium on Algorithms (ESA 2022)_, volume 244
of _Leibniz International Proceedings in Informat-_
_ics (LIPIcs)_, pages 43:1–43:15, Dagstuhl, Germany.
Schloss Dagstuhl – Leibniz-Zentrum für Informatik.


Tamal K. Dey and Yusu Wang. 2022. _[Computational](https://doi.org/10.1017/9781009099950)_
_[Topology for Data Analysis](https://doi.org/10.1017/9781009099950)_ . Cambridge University
Press.


[Xuefeng Du, Chaowei Xiao, and Yixuan Li. 2024. Halo-](https://arxiv.org/abs/2409.17504v1)
[Scope: Harnessing Unlabeled LLM Generations for](https://arxiv.org/abs/2409.17504v1)
[Hallucination Detection.](https://arxiv.org/abs/2409.17504v1)


Herbert Edelsbrunner and John Harer. 2010. _Computa-_
_tional Topology: An Introduction_ . Applied Mathematics. American Mathematical Society.


Ekaterina Fadeeva, Aleksandr Rubashevskii, Artem
Shelmanov, Sergey Petrakov, Haonan Li, Hamdy
Mubarak, Evgenii Tsymbalov, Gleb Kuzmin, Alexander Panchenko, Timothy Baldwin, and 1 others. 2024.
Fact-checking the output of large language models
via token-level uncertainty quantification. In _Find-_
_ings of the Association for Computational Linguistics_
_ACL 2024_, pages 9367–9385.


Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and
Yarin Gal. 2024. Detecting hallucinations in large
language models using semantic entropy. _Nature_,
630(8017):625–630.



Peter Gabriel. 1972. [Unzerlegbare darstellungen.](https://doi.org/10.1007/BF01265300)
_Manuscripta Mathematica_, 6(1):71–103.


Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur
[Hinsvark, and 542 others. 2024. The llama 3 herd of](https://arxiv.org/abs/2407.21783)
[models.](https://arxiv.org/abs/2407.21783) _Preprint_, arXiv:2407.21783.


Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
[Liu. 2025. A survey on hallucination in large lan-](https://doi.org/10.1145/3703155)
[guage models: Principles, taxonomy, challenges, and](https://doi.org/10.1145/3703155)
[open questions.](https://doi.org/10.1145/3703155) _ACM Trans. Inf. Syst._, 43(2).


Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, Aleksander M ˛adry, Alex Baker-Whitcomb, Alex Beutel,
Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, and 399 others. 2024.
[Gpt-4o system card.](https://arxiv.org/abs/2410.21276) _Preprint_, arXiv:2410.21276.


Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
[and William El Sayed. 2023. Mistral 7b.](https://arxiv.org/abs/2310.06825) _Preprint_,
arXiv:2310.06825.


Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, Scott Johnston, Sheer El-Showk,
Andy Jones, Nelson Elhage, Tristan Hume, Anna
Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, and
[17 others. 2022. Language models (mostly) know](https://arxiv.org/abs/2207.05221)
[what they know.](https://arxiv.org/abs/2207.05221) _Preprint_, arXiv:2207.05221.


Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa
[Schut, Shreshth Malik, and Yarin Gal. 2024. Se-](https://doi.org/10.48550/arXiv.2406.15927)
[mantic Entropy Probes: Robust and Cheap Hal-](https://doi.org/10.48550/arXiv.2406.15927)
[lucination Detection in LLMs.](https://doi.org/10.48550/arXiv.2406.15927) _arXiv preprint_ .
ArXiv:2406.15927 [cs].


Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023.
Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation.
In _The Eleventh International Conference on Learn-_
_ing Representations_ .


Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
[Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-](https://doi.org/10.1162/tacl_a_00276)
[ral Questions: A Benchmark for Question Answer-](https://doi.org/10.1162/tacl_a_00276)
[ing Research.](https://doi.org/10.1162/tacl_a_00276) _Transactions of the Association for_


_Computational Linguistics_, 7:452–466. Place: Cambridge, MA Publisher: MIT Press.


Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
TruthfulQA: Measuring how models mimic human
falsehoods. In _Proceedings of the 60th Annual Meet-_
_ing of the Association for Computational Linguistics_
_(Volume 1: Long Papers)_, pages 3214–3252.


[Andrey Malinin and Mark Gales. 2021. Uncertainty](https://openreview.net/forum?id=jN5y-zb5Q7m)
[estimation in autoregressive structured prediction. In](https://openreview.net/forum?id=jN5y-zb5Q7m)
_International Conference on Learning Representa-_
_tions_ .


Potsawee Manakul, Adian Liusie, and Mark Gales. 2023.

[SelfcheckGPT: Zero-resource black-box hallucina-](https://openreview.net/forum?id=RwzFNbJ3Ez)
[tion detection for generative large language models.](https://openreview.net/forum?id=RwzFNbJ3Ez)
In _The 2023 Conference on Empirical Methods in_
_Natural Language Processing_ .


Clément Maria and Steve Oudot. 2016. Computing zigzag persistent cohomology. _arXiv preprint_
_arXiv:1608.06039_ .


Abhika Mishra, Akari Asai, Vidhisha Balachandran,
Yizhong Wang, Graham Neubig, Yulia Tsvetkov, and
[Hannaneh Hajishirzi. 2024. Fine-grained hallucina-](https://openreview.net/forum?id=dJMTn3QOWO)
[tion detection and editing for language models. In](https://openreview.net/forum?id=dJMTn3QOWO)
_First Conference on Language Modeling_ .


Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka
Marttinen. 2024. Kernel language entropy: Finegrained uncertainty quantification for llms from semantic similarities. _Advances in Neural Information_
_Processing Systems_, 37:8901–8929.


Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
2024. Ragtruth: A hallucination corpus for developing trustworthy retrieval-augmented language models.
In _Proceedings of the 62nd Annual Meeting of the_
_Association for Computational Linguistics (Volume_
_1: Long Papers)_, pages 10862–10878.


Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Be[linkov. 2025. LLMs Know More Than They Show:](https://openreview.net/forum?id=KRnsX5Em3W)
[On the Intrinsic Representation of LLM Hallucina-](https://openreview.net/forum?id=KRnsX5Em3W)
[tions. In](https://openreview.net/forum?id=KRnsX5Em3W) _The Thirteenth International Conference on_
_Learning Representations_ .


Xin Qiu and Risto Miikkulainen. 2024. Semantic density: Uncertainty quantification for large language
models through confidence measurement in semantic
space. In _The Thirty-eighth Annual Conference on_
_Neural Information Processing Systems_ .


Pranab Sahoo, Prabhash Meharia, Akash Ghosh, Sriparna Saha, Vinija Jain, and Aman Chadha. 2024. A
comprehensive survey of hallucination in large language, image, video and audio foundation models.
In _Findings of the Association for Computational_
_Linguistics: EMNLP 2024_, pages 11709–11724.


[Claude Elwood Shannon. 1948. A mathematical the-](http://plan9.bell-labs.com/cm/ms/what/shannonday/shannon1948.pdf)
[ory of communication.](http://plan9.bell-labs.com/cm/ms/what/shannonday/shannon1948.pdf) _The Bell System Technical_
_Journal_, 27:379–423.



Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar
Sadasivan, Shoumik Saha, Priyatham Kattakinda,
[and Soheil Feizi. 2024. LLM-check: Investigating](https://openreview.net/forum?id=LYx4w3CAgy)
[detection of hallucinations in large language models.](https://openreview.net/forum?id=LYx4w3CAgy)
In _The Thirty-eighth Annual Conference on Neural_
_Information Processing Systems_ .


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint_
_arXiv:2307.09288_ .


[Yuhei Umeda. 2017. Time series classification via topo-](https://doi.org/10.1527/tjsai.D-G72)
[logical data analysis.](https://doi.org/10.1527/tjsai.D-G72) _Transactions of the Japanese_
_Society for Artificial Intelligence_, 32:D–G72_1.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. _Advances in neural information processing_
_systems_, 30.


Yanling Wang, Haoyang Li, Hao Zou, Jing Zhang, Xinlei He, Qi Li, and Ke Xu. 2024. [Hidden Ques-](https://doi.org/10.48550/arXiv.2406.05328)
[tion Representations Tell Non-Factuality Within and](https://doi.org/10.48550/arXiv.2406.05328)
[Across Large Language Models.](https://doi.org/10.48550/arXiv.2406.05328) _arXiv preprint_ .
ArXiv:2406.05328 [cs].


Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven
[Le Scao, Sylvain Gugger, and 3 others. 2020. Trans-](https://doi.org/10.18653/v1/2020.emnlp-demos.6)
[formers: State-of-the-Art Natural Language Process-](https://doi.org/10.18653/v1/2020.emnlp-demos.6)
[ing. In](https://doi.org/10.18653/v1/2020.emnlp-demos.6) _Proceedings of the 2020 Conference on Em-_
_pirical Methods in Natural Language Processing:_
_System Demonstrations_, pages 38–45, Online. Association for Computational Linguistics.


Tianhang Zhang, Lin Qiu, Qipeng Guo, Cheng Deng,
Yue Zhang, Zheng Zhang, Chenghu Zhou, Xinbing
[Wang, and Luoyi Fu. 2023. Enhancing Uncertainty-](https://doi.org/10.18653/v1/2023.emnlp-main.58)
[Based Hallucination Detection with Stronger Focus.](https://doi.org/10.18653/v1/2023.emnlp-main.58)
In _Proceedings of the 2023 Conference on Empiri-_
_cal Methods in Natural Language Processing_, pages
915–932, Singapore. Association for Computational
Linguistics.


Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-bench and Chatbot Arena.
In _Proceedings of the 37th International Conference_
_on Neural Information Processing Systems_, NIPS
’23, Red Hook, NY, USA. Curran Associates Inc.
Event-place: New Orleans, LA, USA.


Xiaoling Zhou, Mingjie Zhang, Zhemg Lee, Wei Ye,
[and Shikun Zhang. 2025. Hademif: Hallucination](https://openreview.net/forum?id=VwOYxPScxB)
[detection and mitigation in large language models. In](https://openreview.net/forum?id=VwOYxPScxB)
_The Thirteenth International Conference on Learning_
_Representations_ .


[Afra Zomorodian and Gunnar Carlsson. 2004. Com-](https://doi.org/10.1145/997817.997870)
[puting persistent homology. In](https://doi.org/10.1145/997817.997870) _Proceedings of the_
_Twentieth Annual Symposium on Computational Ge-_
_ometry_, SCG ’04, page 347–356, New York, NY,
USA. Association for Computing Machinery.


**A** **Topological Data Analysis**
**Preliminaries**


This appendix provides the necessary background
in topological data analysis, with particular emphasis on concepts relevant to zigzag persistence and
its application to graphs.


**A.1** **Simplicial Homology**


We begin by introducing simplicial complexes.
Simplicial complexes are spaces built with smaller
geometric objects (simplices), such as vertices,
edges, filled triangles, and so on. More formally:
An _abstract simplicial complex K_ is a family of
non-empty subsets of an underlying finite set _V_ ( _K_ )
that is closed under the operation of taking subsets.
In other words, if _σ ∈_ _K_ and _τ ⊂_ _σ_ then _τ ∈_ _K_ .
Each such element _σ ∈_ _K_, is called a _p_ -simplex if
_|σ|_ = _p_ + 1. A _τ ⊂_ _σ_ is called a face of _σ_ .
A graph _G_ = ( _V, E_ ) is easily seen as abstract
simplicial complex, where the underlying set is
the set of vertices _V_ = _V_ ( _K_ ), and the family of
subsets _K_ is given by the union of set of vertices
and the set of edges. Then the vertices correspond
to the 0-simplices and the set of edges to the 1simplices.
For an abstract simplicial complex _K_, define for
each _p ≥_ 0 a _chain group Cp_ ( _K,_ Z2), which is the
vector space over Z2 generated by the _p_ -simplices
of _K_ . We usually omit the field Z2 in our notation. Elements of _Cp_ ( _K_ ) are called _p-chains_, and
they are formal sums of _p_ -simplices with Z2 coefficients. One can connect _Cp_ ( _K_ ) and _Cp−_ 1( _K_ )
by a boundary operator _∂p_ : _Cp_ ( _K_ ) _→_ _Cp−_ 1( _K_ ),
which maps each _p_ -simplex to the sum of its ( _p−_ 1)dimensional faces. These boundary operators satisfy the key property _∂p−_ 1 _◦_ _∂p_ = 0.
Using these boundary maps, define:


  - The group of _p-cycles_ :


_Zp_ ( _K_ ) = ker( _∂p_ ) _,_


which consists of _p_ -chains with zero boundary
(closed cycles).


  - The group of _p-boundaries_ :


_Bp_ ( _K_ ) = im( _∂p_ +1) _,_



which consists of _p_ -chains that are themselves
boundaries of ( _p_ + 1)-chains.


The _p-th simplicial homology group_ is then defined as the quotient


_Hp_ ( _K_ ) = _Zp_ ( _K_ ) _/Bp_ ( _K_ ) _,_


which measures the _p_ -cycles modulo those that
bound higher-dimensional simplices. Intuitively,
elements of _Hp_ ( _K_ ) correspond to _p_ -dimensional
“holes” in the space. Since Z2 is a field, each
homology group forms a vector space.


**A.2** **Zigzag Persistence**


Simplicial homology provides a powerful tool for
analyzing the topology of a single, static simplicial
complex. However, real-world data is often not a
single static object but a sequence of related data
sampled at different moments or corresponding to
different parameters of evolution. In order to study
these evolving objects, we define a _zigzag filtration_ .
A zigzag filtration _Z_ is a sequence of simplicial
complexes


_Z_ : _K_ 0 _⊆_ _K_ 1 _⊇_ _K_ 2 _⊆_ _. . . ⊇_ _Kn._


where the sequence of inclusions could be both in
forward and backward directions.
The homology group of _Ki_ is a vector space
_Hp_ ( _Ki_ ) (under a field coefficient like Z2) for each
_i ∈{_ 1 _, . . ., n}_ . The inclusion map between _Ki_
and _Kj_, induces a natural linear map between
_Hp_ ( _Ki_ ) and _Hp_ ( _Kj_ ). By assembling these vector spaces and linear maps together, we obtain the
_zigzag persistence module MZ_ defined by


_MZ_ : _Hp_ ( _K_ 0) _→_ _Hp_ ( _K_ 1) _←· · · ←_ _Hp_ ( _Kn_ )


The Interval Decomposition Theorem for zigzag
persistence modules (Carlsson and de Silva,
2010; Gabriel, 1972), states that _MZ_ decomposes
uniquely (up to reordering and isomorphism) into
interval modules:


   _MZ_ = _[∼]_ _I_ [ _bk, dk_ ]


_k_


where each _I_ [ _bk, dk_ ] is an interval module with
the support on the interval [ _bk, dk_ ], called a bar.
The collection of these bars yields the _zigzag bar-_
_code_, defined as the multiset of pairs _{_ [ _bk, dk_ ] _}k_ .
This barcode carries the topological information


present in the zigzag persistence module, which we
leverage for hallucination detection.
One of the important properties of zigzag persistence is its stability under small perturbations
(Carlsson and de Silva, 2010), implying that minor
changes in attention weights between layers result
in only small changes in the barcodes. This stability property ensures that the topological signatures
we extract are robust to noise.


**A.3** **Time Complexity**


We use the algorithm proposed in (Dey and Hou,

2022) to compute zigzag persistence barcodes. The
authors show that the time complexity for computing zigzag persistent homology is the same as the
time complexity for computing standard persistent
homology. The time complexity for computing
standard persistent homology is _O_ ( _n_ _[ω]_ ), where _n_ is
the number of simplices (Zomorodian and Carlsson,
2004) and _ω <_ 2 _._ 371339 is the matrix multiplication exponent (Alman et al., 2025). However, for
the special case of graphs there exists a near linear time algorithm (Dey and Hou, 2021) that could
potentially be used.


**B** **Additional Experimental Details**


To ensure the robustness of our method, we conducted several experiments to select the optimal
hyperparameters for our main results. We use an
NVIDIA A100 (40GB) GPU with 16 cores for all
the experiments.


**B.1** **Hyperparameter Tuning**


**Minimum Persistence Filtering.** As in wellknown in the applied TDA community, shorter bars
are considered to be noise. Hence, we filter out
shorter bars and retain the longer ones. In order
to choose the optimal threshold, we evaluate the
performance after filtering out bars below various
persistence thresholds (5, 7, 9 or 11 layers) at two
different levels of graph sparsity (edge selection)

- 30% (Table 5) and 10% (Table 6). Based on our
results, we find that a moderate level of persistence
filtering is optimal. Therefore, to balance the benefit of removing noise against the risk of discarding
valuable signals, we select a conservative minimum
persistence threshold of 5 for all main experiments.


**Edge Selection Threshold.** The construction of
our attention graphs depends on a sparsity parameter. We investigated the impact of this by varying
the percentage of top attention weights retained



(5%, 10%, 20%, and 30%) for different minimum
persistence values - 5 (Table 7) and 9 (Table 8). The
results indicate that retaining the top 10% of edges
provides a strong balance between detection performance and computational efficiency. We therefore
use this 10% threshold for all main experiments
reported in the paper.


**B.2** **Implementation Details**


**Model Temperature.** We used a temperature
value of 0.7 for all the models while generating
answers. For GPT-4o-mini, we used a temperature
of 0 while generating the annotation labels on QA
Benchmarks.


**Topological Feature Vectorization.** We use
GUDHI (Carrière et al., 2025) for all vectorization
schemes. The specific parameters are as follows:


  - **Persistence Images (PersImg):** We use a resolution of 32 x 32.


  - **Betti Curves:** We use a sampling resolution
of 32 points.


  - **Persistence Entropy (PersEntropy):** We
use the default implementation from the library.


**Classifier.** For all our experiments, we classify
using a random forest classifier. For experiments on
RAGTruth Dataset, we use n_estimators = 500.
For all other experiments, we use n_estimators =
100. The max_depth parameter was tuned for each
configuration. The values are reported in Table 4.


**Baseline Reproduction.** To ensure a fair comparison with LapEigVals baseline from (Binkowski
et al., 2025), we followed their experimental protocol precisely. As specified in their work, we retained the top _k_ = 50 eigenvalues across all heads
and layers and trained a logistic regression classifier with their reported parameters.


**B.3** **Detailed Performance Metrics and**
**Statistical Uncertainty**


To ensure reproducibility of our results, this section presents the detailed performance metrics for
HalluZig across all datasets. The main results
reported in the body of the paper correspond to
the single best run for clarity. Here, we report the
mean and standard deviation calculated over five
independent runs with different random seeds. Refer to Table 10 and Table 9. For the RAGTruth


dataset, we use the official train-test split provided
in the dataset. Consequently, all results reported in
the main text are based on this single, pre-defined
partition. For all other datasets, we use a 80-20
train-test split.











|Dataset|Model|PersImg PersEntropy BettiCurve|
|---|---|---|
|**FAVA Annotated**|Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b|20<br>10<br>10<br>20<br>10<br>10<br>20<br>10<br>10|
|**RAGTruth**|Llama-2-7b<br>Llama-2-13b<br>Mistral-7b|25<br>25<br>25<br>25<br>25<br>25<br>25<br>25<br>25|
|**NQOpen**|Llama-3.1-8b<br>Llama-3.2-3b|6<br>4<br>5<br>5<br>4<br>4|
|**TruthfulQA**|Llama-3.1-8b<br>Llama-3.2-3b|6<br>5<br>5<br>6<br>4<br>5|


Table 4: Maximum tree depth (max_depth) used in Random Forest Classifier.


|Method Metric Top % of edges (attn weights) selected<br>30% 20% 10% 5%|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**Method**<br>**Metric**<br>**Top % of edges (attn weights) selected**<br>30%<br>20%<br>10%<br>5%|**Method**<br>**Metric**<br>**Top % of edges (attn weights) selected**<br>30%<br>20%<br>10%<br>5%|**Method**<br>**Metric**<br>**Top % of edges (attn weights) selected**<br>30%<br>20%<br>10%<br>5%|20%|10%|10%|
|PersImg|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|73.46<br>65.22<br>7.41<br>72.88|72.83<br>66.30<br>20.37<br>73.50|73.42<br>66.30<br>12.96<br>73.95|71.47<br>65.22<br>9.26<br>72.88|
|PersEntropy|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|75.41<br>66.30<br>33.33<br>71.56|74.39<br>66.30<br>25.93<br>71.56|66.96<br>65.22<br>7.41<br>70.91|74.78<br>67.39<br>22.22<br>74.14|
|Betti Curve|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|75.78<br>69.57<br>27.78<br>75.86|72.78<br>66.30<br>31.48<br>73.95|69.96<br>66.30<br>12.96<br>73.50|74.17<br>65.22<br>38.89<br>71.93|



Table 7: This table shows the performance of our three
vectorization schemes while varying the percentage of
top attention weights used to construct the attention
graphs (minimum bar persistence is held constant at 5).










|Method Metric Min. Persistence of Bars<br>⩾5 ⩾7 ⩾9 ⩾11|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**Method**<br>**Metric**<br>**Min. Persistence of Bars**<br>⩾5<br>⩾7<br>⩾9<br>⩾11|**Method**<br>**Metric**<br>**Min. Persistence of Bars**<br>⩾5<br>⩾7<br>⩾9<br>⩾11|**Method**<br>**Metric**<br>**Min. Persistence of Bars**<br>⩾5<br>⩾7<br>⩾9<br>⩾11|⩾7|⩾9|⩾9|
|PersImg|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|73.46<br>65.22<br>7.41<br>72.88|73.27<br>**68.48**<br>7.41<br>**75.63**|72.93<br>65.22<br>7.41<br>**72.88**|**73.76**<br>64.13<br>7.41<br>72.27|
|PersEntropy<br>AUC-ROC<br>75.41<br>**77.75**<br>**74.44**<br>69.98<br>Accuracy<br>66.30<br>68.48<br>65.22<br>**65.22**<br>TPR at 5% FPR<br>**33.33**<br>25.93<br>24.07<br>22.22<br>F1 Score<br>71.56<br>74.34<br>70.91<br>72.41|PersEntropy<br>AUC-ROC<br>75.41<br>**77.75**<br>**74.44**<br>69.98<br>Accuracy<br>66.30<br>68.48<br>65.22<br>**65.22**<br>TPR at 5% FPR<br>**33.33**<br>25.93<br>24.07<br>22.22<br>F1 Score<br>71.56<br>74.34<br>70.91<br>72.41|PersEntropy<br>AUC-ROC<br>75.41<br>**77.75**<br>**74.44**<br>69.98<br>Accuracy<br>66.30<br>68.48<br>65.22<br>**65.22**<br>TPR at 5% FPR<br>**33.33**<br>25.93<br>24.07<br>22.22<br>F1 Score<br>71.56<br>74.34<br>70.91<br>72.41|PersEntropy<br>AUC-ROC<br>75.41<br>**77.75**<br>**74.44**<br>69.98<br>Accuracy<br>66.30<br>68.48<br>65.22<br>**65.22**<br>TPR at 5% FPR<br>**33.33**<br>25.93<br>24.07<br>22.22<br>F1 Score<br>71.56<br>74.34<br>70.91<br>72.41|PersEntropy<br>AUC-ROC<br>75.41<br>**77.75**<br>**74.44**<br>69.98<br>Accuracy<br>66.30<br>68.48<br>65.22<br>**65.22**<br>TPR at 5% FPR<br>**33.33**<br>25.93<br>24.07<br>22.22<br>F1 Score<br>71.56<br>74.34<br>70.91<br>72.41|PersEntropy<br>AUC-ROC<br>75.41<br>**77.75**<br>**74.44**<br>69.98<br>Accuracy<br>66.30<br>68.48<br>65.22<br>**65.22**<br>TPR at 5% FPR<br>**33.33**<br>25.93<br>24.07<br>22.22<br>F1 Score<br>71.56<br>74.34<br>70.91<br>72.41|
|Betti Curve|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|**75.78**<br>**69.57**<br>27.78<br>**75.86**|73.73<br>67.39<br>**29.63**<br>74.14|72.59<br>64.13<br>**29.63**<br>71.30|**74.29**<br>**65.22**<br>**37.04**<br>**73.77**|



Table 5: This table evaluates our three vectorization
schemes when filtering out topological features (bars)
with low persistence. We select the top 30% of the edges
for this experiment.




|Method Metric Top % of edges (attn weights) selected<br>30% 20% 10% 5%|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**Method**<br>**Metric**<br>**Top % of edges (attn weights) selected**<br>30%<br>20%<br>10%<br>5%|**Method**<br>**Metric**<br>**Top % of edges (attn weights) selected**<br>30%<br>20%<br>10%<br>5%|**Method**<br>**Metric**<br>**Top % of edges (attn weights) selected**<br>30%<br>20%<br>10%<br>5%|20%|10%|10%|
|PersImg|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|72.34<br>64.13<br>14.81<br>71.79|72.93<br>65.22<br>7.41<br>72.88|72.39<br>66.30<br>16.67<br>73.95|72.00<br>65.22<br>18.52<br>72.88|
|PersEntropy|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|74.44<br>65.22<br>24.07<br>70.91|71.08<br>64.13<br>14.81<br>71.30|59.23<br>60.87<br>14.81<br>67.27|74.98<br>67.39<br>25.93<br>73.21|
|Betti Curve|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|72.59<br>64.13<br>29.63<br>71.30|72.39<br>66.30<br>22.22<br>73.95|71.03<br>64.13<br>20.37<br>71.79|72.64<br>66.30<br>33.33<br>72.07|






|Method Metric Min. Persistence of Bars<br>⩾5 ⩾7 ⩾9 ⩾11|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**Method**<br>**Metric**<br>**Min. Persistence of Bars**<br>⩾5<br>⩾7<br>⩾9<br>⩾11|**Method**<br>**Metric**<br>**Min. Persistence of Bars**<br>⩾5<br>⩾7<br>⩾9<br>⩾11|**Method**<br>**Metric**<br>**Min. Persistence of Bars**<br>⩾5<br>⩾7<br>⩾9<br>⩾11|⩾7|⩾9|⩾9|
|PersImg|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|73.42<br>66.30<br>12.96<br>73.95|73.08<br>64.13<br>25.93<br>71.79|72.34<br>64.13<br>14.81<br>71.79|74.10<br>68.48<br>20.37<br>76.03|
|PersEntropy|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|66.96<br>65.22<br>7.41<br>70.91|57.36<br>64.13<br>7.41<br>70.80|59.23<br>60.87<br>14.81<br>67.27|64.91<br>65.22<br>14.81<br>71.93|
|Betti Curve|AUC-ROC<br>Accuracy<br>TPR at 5% FPR<br>F1 Score|69.96<br>66.30<br>12.96<br>73.50|70.49<br>65.22<br>14.81<br>72.41|71.03<br>64.13<br>20.37<br>71.79|72.69<br>65.22<br>18.52<br>72.88|


|Method|TruthfulQA (AUC-ROC)<br>Llama-3.1-8b Llama-3.2-3b|NQOpen (AUC-ROC)|
|---|---|---|
|**Method**|**TruthfulQA (AUC-ROC)**<br>Llama-3.1-8b<br>Llama-3.2-3b|Llama-3.1-8b<br>Llama-3.2-3b|
|PersImg<br>PersEntropy<br>BettiCurve|65_._92_ ±_ 4_._05<br>62_._08_ ±_ 2_._32<br>63_._62_ ±_ 3_._07<br>58_._34_ ±_ 6_._61<br>59_._31_ ±_ 6_._11<br>58_._22_ ±_ 4_._74|68_._29_ ±_ 2_._58<br>67_._27_ ±_ 2_._16<br>66_._11_ ±_ 1_._36<br>65_._53_ ±_ 1_._68<br>67_._97_ ±_ 2_._92<br>65_._86_ ±_ 2_._07|



Table 6: This table evaluates our three vectorization
schemes when filtering out topological features (bars)
with low persistence. We select the top 10% of the edges
for this experiment.



Table 8: This table shows the performance of our three
vectorization schemes while varying the percentage of
top attention weights used to construct the attention
graphs (minimum bar persistence is held constant at 9).


Table 9: Each entry reports mean _±_ standard deviation
across five random seeds for Test AUC-ROC scores.


|PersImg PersEntropy Betti Curve<br>Metric<br>Llama-2-7b Llama-3-8b Vicuna-7b Llama-2-7b Llama-3-8b Vicuna-7b Llama-2-7b Llama-3-8b Vicuna-7b|Col2|PersEntropy|Col4|
|---|---|---|---|
|**Metric**<br>**PersImg**<br>**PersEntropy**<br>**Betti Curve**<br>Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b<br>Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b<br>Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b|**Metric**<br>**PersImg**<br>**PersEntropy**<br>**Betti Curve**<br>Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b<br>Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b<br>Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b|Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b|Llama-2-7b<br>Llama-3-8b<br>Vicuna-7b|
|**AUC-ROC**<br>**Accuracy**<br>**TPR @ 5% FPR**<br>**F1 Score**|78_._72_ ±_ 3_._41<br>78_._89_ ±_ 2_._74<br>79_._20_ ±_ 3_._38<br>73_._04_ ±_ 2_._70<br>74_._35_ ±_ 2_._24<br>74_._13_ ±_ 2_._70<br>30_._02_ ±_ 12_._17<br>37_._70_ ±_ 11_._57<br>36_._62_ ±_ 12_._25<br>80_._21_ ±_ 1_._66<br>81_._42_ ±_ 1_._77<br>81_._16_ ±_ 1_._70|70_._35_ ±_ 4_._53<br>72_._29_ ±_ 4_._34<br>71_._31_ ±_ 3_._63<br>67_._83_ ±_ 2_._54<br>71_._52_ ±_ 2_._52<br>69_._57_ ±_ 3_._30<br>19_._86_ ±_ 3_._70<br>16_._97_ ±_ 4_._61<br>25_._55_ ±_ 8_._08<br>76_._77_ ±_ 2_._16<br>78_._15_ ±_ 2_._33<br>76_._72_ ±_ 2_._53|71_._64_ ±_ 3_._75<br>72_._71_ ±_ 4_._90<br>72_._95_ ±_ 2_._48<br>67_._83_ ±_ 3_._12<br>71_._30_ ±_ 2_._54<br>70_._65_ ±_ 2_._75<br>21_._41_ ±_ 3_._95<br>16_._45_ ±_ 6_._69<br>22_._76_ ±_ 8_._28<br>77_._05_ ±_ 2_._54<br>79_._45_ ±_ 1_._60<br>78_._96_ ±_ 2_._30|


Table 10: Each entry reports mean _±_ standard deviation across five random seeds for all the metrics on the FAVA
Annotated Dataset.


**C** **QA prompt**


We used OpenAI’s API and ChatGPT-4o-mini model as judge for labeling the generated response of each
LLM in the whitebox setting. The prompt we used is the same for NQopen, and TruthfullQA.


**Prompt used for LLM-as-judge with Open AI’s ChatGPT4o-mini**


You are an evaluator for QA tasks. Given the question, the reference best answer, and a model’s
answer, classify the model’s answer as:


   - CORRECT if it matches or is equivalent to the best answer.


   - HALLUCINATION if it introduces facts not in the best answer.


   - N/A if the model expresses uncertainty or says it cannot answer.
Respond with only one label: CORRECT, HALLUCINATION, or N/A.


Question: {question}
Reference best answer: {best_answer}
Model answer: {model_answer}


