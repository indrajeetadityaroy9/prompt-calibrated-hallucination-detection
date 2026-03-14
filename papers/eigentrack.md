# EigenTrack: Temporal Spectral Analysis of Hidden Activations for Hallucination and OOD Detection



Davide Ettori
_University of Illinois Chicago_
Chicago, IL, USA
detto3@uic.edu



Nastaran Darabi
_University of Illinois Chicago_
Chicago, IL, USA
ndarab2@uic.edu



Ranganath Krishnan
_Capital One AI Labs_
USA
ranganath.krishnan@capitalone.com



Mahesh Subedar
_Intel Labs_
USA
mahesh.subedar@intel.com



Omesh Tickoo
_Intel Labs_
USA
omesh.tickoo@intel.com



Sina Tayebati
_University of Illinois Chicago_
Chicago, IL, USA
stayeb3@uic.edu


Amit Ranjan Trivedi
_University of Illinois Chicago_
Chicago, IL, USA
amitrt@uic.edu



_**Abstract**_ **—Large language and vision-language models (LLMs**
**and VLMs) offer broad utility but remain prone to hallucination**
**and out-of-distribution (OOD) errors. We propose** _**EigenTrack**_ **,**
**an interpretable real-time detector that characterizes model**
**dynamics through the spectral geometry of hidden activations.**
**EigenTrack constructs sliding-window activation matrices, ex-**
**tracts covariance spectrum statistics including leading eigen-**
**values, spectral gaps, entropy, and Random Matrix Theory**
**features based on divergence from the Marchenko-Pastur law,**
**and streams these signals into a lightweight recurrent classifier**
**to model temporal evolution. This design detects shifts toward**
**noise-like representation regimes, enabling early identification**
**of hallucination and OOD behavior before surface-level errors**
**appear. Unlike black-box and grey-box methods that require**
**multiple generations or log-probability access, EigenTrack op-**
**erates with a single forward pass and no resampling. Unlike**
**prior white-box approaches based on static snapshots, it preserves**
**temporal context and aggregates global spectral signals across**
**layers. We evaluate EigenTrack on LLaMa, Qwen, Mistral, and**
**LLaVa models for hallucination detection on HaluEval and OOD**
**detection using WebQuestions and Eurlex, achieving consistent**
**AUROC gains with favorable accuracy-latency tradeoffs. Early**
**detection enables efficient termination of failing generations, re-**
**ducing computation while improving LLMs and VLMs reliability.**


_**Index Terms**_ **—Hallucination Detection, OOD Detection, Large**
**Language Models, Spectral Analysis, Interpretability**


I. INTRODUCTION AND PRIOR WORKS


Large language and vision-language models (LLMs and
VLMs) are increasingly deployed in high-stakes domains
such as healthcare, law, and finance, yet remain unreliable
due to hallucination and severe degradation under out-ofdistribution (OOD) inputs [1], [2]. Early detection approaches
relied on surface-level signals such as softmax confidence [3]
or semantic-entropy surrogates [4], but ignored internal model
dynamics and often failed under domain shift.
More recent methods differ by access level. _Black-box_ approaches, including SelfCheckGPT [5], CoNLI [6], and CostEffective HD [7], rely on ensembles or multiple generations
to capture uncertainty, at the cost of high latency. _Grey-_
_box_ methods such as DetectGPT [8], Fast-DetectGPT [9],



Fig. 1. **EigenTrack.** Spectral signatures, including entropy, spectral gaps, and
divergence from a random matrix baseline, are extracted from intermediate
feed-forward layers and streamed into a recurrent spectral discrepancy detector. Tracking their temporal evolution enables early detection of failure.


and Glimpse [10] exploit log-probability curvature or partial
logits, but remain snapshot-based and lack temporal context.
_White-box_ detectors probe hidden states directly, but often
lack generality or temporal modeling. For example, MIND

[11] streams activations without capturing evolution over time,
LapEigvals [12] analyzes spectra at individual steps, ReDeEP

[13] relies on retrieval signals, and [14] tracks attention shifts.
Conformal prediction at the output layer is leveraged for
uncertainty and lack of reliability in [15], [16].

We argue that spectral signatures provide a principled
foundation for hallucination and OOD detection by compressing high-dimensional hidden activations into compact
descriptors of representation geometry. Eigenvalue distributions, entropy, and spectral gaps are sensitive to low-rank
correlations and instabilities that arise under distribution shift
or hallucination. Unlike token-level probabilities, which reflect
only output-layer uncertainty, spectral statistics integrate information across hidden layers and capture global uncertainty
dynamics. Moreover, spectral analysis enables the use of
Random Matrix Theory by comparing empirical activation
spectra to the Marchenko-Pastur (MP) law [17] using KL
divergence and Wasserstein distance. Deviations from this
noise baseline provide a compact and interpretable indicator
of structural breakdown for hallucination or OOD behavior.

Building on this insight, EigenTrack computes covariance


spectra over a sliding window of hidden activations and
streams the resulting spectral statistics into a lightweight
recurrent classifier (Fig. 1). Prior spectral methods such as
RankFeat [18], SpectralGap [19], and SNoJoE [20] discriminate OOD from in-distribution samples using singular values
or spectral gaps, but operate on static snapshots and ignore
temporal evolution. In contrast, EigenTrack transforms streaming spectral features into low-dimensional temporal trajectories
that reveal how uncertainty accumulates during generation.
The key novelty lies in combining temporal modeling with
RMT-grounded spectral features computed from global hidden
activations, yielding an interpretable and effective detector for
both hallucination and OOD. Our contributions are:


_•_ We introduce _EigenTrack_, a real-time detector that models
the temporal evolution of multi-layer spectral features to
identify hallucination and OOD behavior.

_•_ We propose an interpretable set of spectral indicators that
link eigenvalue dynamics to latent representation shifts.

_•_ We develop a data-generation pipeline for hallucination
detection based on multi-LLM interaction.

EigenTrack achieves AUROC values of 0.82 to 0.94 for
hallucination detection and 0.85 to 0.96 for OOD detection
across LLMs and VLMs. On LLaMa-7B, it reaches 0.89
AUROC for hallucination and 0.92 for OOD, outperforming
baselines such as HaloScope, LapEigvals, and SelfCheckGPT
while using only a lightweight recurrent head. By enabling
early termination of failing generations, EigenTrack reduces
generation cost while increasing reliability.


II. BACKGROUND


Random Matrix Theory (RMT) provides asymptotic laws
for the spectra of large random matrices and serves as a
principled null model for high-dimensional statistics [17], [21].
When empirical spectra deviate from these laws, the deviation
signals structure beyond isotropic noise. This property has
made RMT a useful analytical tool for probing signal versus
noise in high-dimensional systems. A central object in RMT
is the _spiked covariance model_ [22], which describes data
composed of a low-rank signal embedded in isotropic noise.
The population covariance takes the form



triangle and _Wij_ = _Wji_ . As _n →∞_, the empirical eigenvalue
density converges almost surely to the Wigner semicircle law,



1      _ρ_ W( _λ_ ) =
2 _πσ_ [2]



4 _σ_ [2] _−_ _λ_ [2] _,_ _|λ| ≤_ 2 _σ,_



and is zero outside this interval. This law provides a canonical noise-only baseline for symmetric random matrices and
motivates the use of spectral bulk behavior as a proxy for
unstructured noise.


_B. Sample Covariance and the Marchenko-Pastur Law_

For data matrices _X ∈_ R _[d][×][N]_ with independent rows or
columns and covariance _σ_ [2] _I_, the sample covariance _C_ =
1
_N_ _[XX]_ _[⊤]_ [has eigenvalues that converge to the Marchenko-]
Pastur (MP) distribution,



1
_ρ_ MP( _λ_ ) =
2 _πσ_ [2] _qλ_







( _λ_ + _−_ _λ_ )( _λ −_ _λ−_ ) _,_ _λ±_ = _σ_ [2] (1 _±_ _[√]_ ~~_q_~~ ~~)~~ [2] _,_



where _q_ = _d/N_ is the aspect ratio. The MP law serves
as a null baseline for covariance spectra generated by highdimensional isotropic noise. To quantify deviations from this
baseline, we compare the empirical spectral density _p_ ( _λ_ ) to
the MP reference _q_ ( _λ_ ) using KL divergence,


          _D_ KL( _p∥q_ ) = _p_ ( _λ_ ) log _[p]_ [(] _[λ]_ [)]

_q_ ( _λ_ ) _[dλ,]_


and the 1-Wasserstein distance,




     - 1
_W_ 1( _p, q_ ) =

0



�� _F −p_ 1( _u_ ) _−_ _Fq_ _[−]_ [1] ( _u_ )�� _du._



**Σ** = _σ_ [2] **I** _p_ +



_k_

- _θiuiu_ _[T]_ _i_ _[,]_

_i_ =1



where _θi_ denote spike strengths and _ui_ are orthonormal signal
directions. In this model, meaningful structure manifests as
isolated spectral components that separate from the noise bulk.
In this work, we apply this framework to covariance matrices
constructed from LLM hidden activations, using RMT as a
reference to distinguish structured representation dynamics
from noise-like behavior.


_A. Wigner Matrices and the Semicircle Law_


A Wigner matrix is a symmetric matrix _W ∈_ R _[n][×][n]_ with
independent, zero-mean entries of variance _σ_ [2] _/n_ in the upper



KL divergence measures how unlikely the observed spectrum
is under the MP baseline, while Wasserstein distance captures
the average mass displacement required to transform one
distribution into the other. Persistent eigenvalues outside MP
support [ _λ−, λ_ +] indicate low-rank correlations and structure.


_C. BBP Phase Transition_


The Baik-Ben Arous-Peche (BBP) phase transition [23]
characterizes when a low-rank signal becomes detectable
in high-dimensional noise. If the spike strength is below
a critical threshold, the corresponding sample eigenvalues
remain embedded within the MP bulk and are statistically
indistinguishable from noise. Only when

_θ > σ_ [2] (1 + _[√]_ _c_ ) _,_


equivalently when population eigenvalues exceed _λ_ + =
_σ_ [2] (1 + _[√]_ ~~_c_~~ ) [2], do sample eigenvalues separate from the bulk
and emerge as isolated outliers. These detached eigenvalues
carry signal, while eigenvalues inside the bulk represent noisedominated directions. This transition provides a concrete criterion for distinguishing structured from unstructured regimes.
RMT has been widely applied to analyze deep learning systems by relating spectra of weights, activations, or Hessians to
training dynamics, implicit regularization, and generalization
behavior [24], [25]. These results motivate the use of spectral
statistics as compact probes of representation geometry in
modern neural networks.


_D. Why These Results Apply to LLM Activations_

Modern decoder-only LLMs apply LayerNorm at every
transformer block [26], centering activations and normalizing
variance at each layer. Combined with the near-orthogonality
induced by weight decay and Adam-style optimization [27],
per-token hidden activations are approximately mean-zero and
weakly correlated across dimensions. In addition, hidden layer
widths are large, placing activation matrices firmly in the highdimensional regime where RMT predictions are most accurate.
Under these conditions, the eigenvalue distribution of a
sliding-window activation matrix _Ht ∈_ R _[N]_ _[×][d]_ is well approximated by the Marchenko-Pastur law and serves as a
principled null baseline. Our RMT-grounded hypothesis is that
during anomalous behavior, including hallucination and OOD
generation, representation structure weakens and activation
spectra drift toward this noise baseline. In contrast, factual and
in-distribution generation exhibits stronger low-rank structure,
reflected by spectral spikes that exceed the BBP threshold [23].
This behavior has been observed in related spectral analyses
of VLM robustness [28], [29] and is confirmed empirically
by EigenTrack. In our experiments, anomalous sequences
move closer to the MP baseline, while factual sequences
display clearer spectral separation, and the most predictive
signals are precisely divergence-to-baseline and eigenvaluebased statistics (Fig. 2, Table I, Fig. 3).


III. EIGENTRACK: METHODOLOGY

EigenTrack detects hallucination and out-of-distribution
(OOD) behavior by transforming hidden activations of LLMs
and VLMs into compact spectral descriptors and modeling
their temporal evolution. It is designed to identify when
representation structure degrades toward noise-like regimes
that precede observable generation errors. The pipeline comprises three stages: (i) sliding-window aggregation of hidden
activations, (ii) extraction of spectral features that capture lowrank structure and stability, and (iii) temporal classification
using a lightweight recurrent model. This design rests on
two principles: spectral statistics provide global, token-robust
indicators of representation geometry, and temporal modeling
enables early detection by tracking how uncertainty accumulates across layers and decoding steps. Consistent with our
RMT-based hypothesis, hallucination/OOD exhibit spectra that
move toward noise baselines and lack clear low-rank spikes.


_A. Sliding-Window Representation_

We monitor a set of _m_ transformer layers _L_ = _{ℓ_ 1 _, . . ., ℓm}_,
each producing a hidden activation _h_ _[ℓ]_ _t_ _[∈]_ [R] _[d]_ [ at generation step]
_t_ . Activations from the selected layers are concatenated to form

_vt_ = [ _h_ _[ℓ]_ _t_ [1] _[∥· · · ∥]_ _[h]_ _t_ _[ℓ][m]_ []] _[ ∈]_ [R] _[md][.]_


To capture local temporal context, we stack the most recent
_N_ token representations into a sliding-window matrix

_Ht_ = [ _vt−N_ +1 _, . . ., vt_ ] _[⊤]_ _∈_ R _[N]_ _[×][md]_ _,_


which is updated at each decoding step. This construction
aggregates cross-layer and short-horizon temporal information
while remaining compatible with streaming inference.



_B. Spectral Feature Extraction_

Rather than forming the covariance matrix _Ct_ = _N_ [1] _[H]_ _t_ _[⊤][H][t]_ [,]

we compute a truncated singular value decomposition


_Ht_ = _Ut_ Σ _tVt_ _[⊤][,]_


and recover the corresponding covariance eigenvalues as
_λt,i_ = _σt,i_ [2] _[/N]_ [. From these eigenvalues we construct a] _[ k]_ [-]
dimensional spectral feature vector _Ft_ (default _k_ = 22 discussed in the interpretability section) capturing signal strength
and noise alignment, including: (i) leading eigenvalues, (ii)
spectral gaps such as _λt,_ 1 _/λt,_ 2, (iii) spectral entropy



_St_ = _−_ 


_pt,i_ log _pt,i,_ _pt,i_ = _λt,i/_  
_i_ _j_



_λt,j,_

_j_



(iv) spectral variance and (v) divergence from the MP reference law via KL divergence and Wasserstein distance. These
statistics summarize low-rank structure, dispersion, and proximity to noise-dominated regimes, providing a compact and
interpretable representation of activation dynamics.


_C. Recurrent Classification_


The sequence of spectral feature vectors ( _F_ 1 _, . . ., FT_ ) is
treated as a multivariate time series and processed by a
lightweight recurrent classifier, instantiated as an RNN, GRU,
or LSTM. At each step, _Ft_ is input to the recurrent cell, whose
hidden state propagates temporal context across decoding
steps. A feed-forward output head produces a binary logit
corresponding to factual or in-distribution versus hallucinated
or anomalous behavior. Because weights are shared across
time, the number of parameters is independent of sequence
length, enabling efficient modeling of spectral trajectories
rather than static snapshots.


_D. Hyperparameters and Extensions_


EigenTrack exposes several tunable parameters that control
the accuracy-latency tradeoff, including the monitored layer
set _L_, sliding-window length _N_, number of spectral features
_k_, and recurrent hidden size. The framework naturally extends to multimodal architectures by constructing _Ht_ from
cross-modal fusion layers or vision encoder blocks, enabling
consistent treatment of LLMs and VLMs. Prior analyses
indicate that later transformer layers tend to be more taskspecific and informative for downstream probing [30], while
combining representations from multiple layers can further
improve robustness [31]. In our experiments, the final layers
are most predictive when monitored alone, but incorporating
earlier layers improves stability across prompts and domains.
Accordingly, EigenTrack monitors all layers when feasible,
or samples layers at fixed intervals from early to late blocks,
which is our default configuration.


_E. Computational Overhead_


Let _D_ = _md_ denote the concatenated feature dimension
and _N_ the sliding-window length. Computing the SVD of
an _N × D_ matrix requires _O_ ( _ND_ min _{N, D}_ ) operations. In


Fig. 2. Temporal evolution of spectral features over 80 generation steps on
LLaMa-3B for factual and hallucinated sequences.


practice, _D > N_, so the dominant cost is _O_ ( _N_ [2] _D_ ). The perwindow computation therefore scales quadratically with _N_ and
linearly with the number of monitored layers _m_ and hidden
dimension _d_ . For a fixed response length, smaller windows
yield more updates and higher cost, while larger windows
reduce the number of SVDs at the expense of responsiveness.
An ablation study in Sec. VI quantifies this tradeoff and reports
wall-clock latency in milliseconds across window sizes.

Additionally, EigenTrack processes activations in a streaming manner and retains only the compact spectral feature
vector for each window, rather than storing full activation
histories or complete eigenspectra. As a result, memory usage
scales linearly with the number of windows and is independent
of model size beyond the selected feature dimension.


_F. Spectral Dynamics_


Fig. 2 illustrates the temporal behavior of representative
spectral features for factual and hallucinated generations. Clear
and consistent patterns emerge. Hallucinated sequences concentrate variance along a small number of dominant directions,
reflected in larger sums of leading eigenvalues. At the same
time, their spectra are flatter and more dispersed, leading to
higher entropy. From an RMT perspective, purely unstructured
activations would follow the Marchenko-Pastur distribution

[17]. Relative to this baseline, hallucinated sequences remain
closer to the noise regime, exhibiting lower KL divergence and
weaker spectral separation, while factual sequences diverge
more strongly, indicating structured representation dynamics.
The median eigenvalue is also higher and more stable for
hallucinations, whereas factual generations show lower and
more variable medians that track model confidence. These
spectral signatures emerge early in generation and provide
reliable indicators of hallucination and OOD behavior before
errors become visible in the output.



IV. EXPERIMENTAL SETUP


**Models and Inference Protocol:** We evaluate EigenTrack on
open-source HuggingFace LLMs and VLMs ranging from 1B
to 7B parameters, including models from the LLaMa, Qwen,
Mistral, and LLaVa families. Both base and instruction-tuned
variants are considered to assess robustness across training
regimes. Each model generates up to 128 tokens per prompt,
and hidden activations are streamed from selected layers.


**Hallucination Detection:** Hallucination detection is evaluated
on HaluEval, a HotpotQA-based benchmark designed to distinguish factual from hallucinated responses. For each passage,
we construct two queries: the ground-truth question associated
with the passage, and an unrelated question generated by
LLaMa-8B. Responses to the former are treated as factual,
while responses to the latter are labeled as hallucinated,
following an LLM-as-a-Judge approach. This procedure defines an automatic data-generation pipeline involving three
LLMs: (i) a _Main Model_, corresponding to the evaluated
model, (ii) a _Question Generator_, which produces unrelated
questions, and (iii) an _Answer Judge_, implemented using a
larger LLM (LLaMa-8B) that verifies answer consistency. This
setup enables scalable generation of labeled hallucinated and
non-hallucinated responses without manual annotation. For
VLMs such as LLaVa, textual passages are augmented with
images from Flickr8k to construct multimodal inputs.


**Out-of-Distribution Detection:** For OOD detection, we treat
WebQuestions as the in-distribution dataset and Eurlex, a
legal-domain question dataset, as the OOD set. Eurlex queries
are not encountered during pretraining of the evaluated models
and therefore induce a clear domain shift. For VLM experiments, the same questions are paired with contextual images
from Flickr8k to maintain a consistent multimodal setting.


**EigenTrack Configuration:** EigenTrack converts streamed
hidden activations into sequences of spectral feature vectors,
later processed by recurrent classifiers. We evaluate three
recurrent architectures: RNN, GRU, and LSTM. Each classifier consists of a linear projection, a single recurrent layer,
and a binary output head for in-distribution or factual versus
anomalous behavior.


**Baselines and Evaluation Metrics:** We compare EigenTrack against standard confidence-based and distance-based
baselines, including Max Softmax Probability, Energy Score,
and cosine-distance variants, as well as recent state-of-the-art
methods summarized in Table II. All methods are evaluated
using the Area Under the Receiver Operating Characteristic
curve (AUROC), which ranges from 0.5 for random guessing
to 1.0 for perfect discrimination. To ensure fair comparison, all methods use identical train and test splits and the
same preprocessing. For score-based baselines, AUROC is
computed by sweeping thresholds over in-distribution scores
without supervision. In contrast, EigenTrack trains the recurrent classifier in a supervised manner using labeled data. All
hyperparameters, including sliding-window length, spectral
feature dimensionality, and recurrent hidden size, are selected


TABLE I
FULL RESULTS OF HALLUCINATION AND OUT-OF-DISTRIBUTION (OOD) DETECTION ACROSS LLMS AND VLMS OF VARYING SIZES USING DIFFERENT
RECURRENT CLASSIFIERS (RNN, GRU, LSTM). WE REPORT AUROC AND F1 SCORES OBTAINED WITH THE FULL SPECTRAL FEATURE SET, AS WELL
AS AUROC VALUES FOR THE BEST AND WORST TRIPLETS OF SPECTRAL FEATURES IDENTIFIED THROUGH ABLATION.


**AUROC Full** **F1 Full** **AUROC Best Triplet** **AUROC Worst Triplet**


**Model** RNN GRU LSTM RNN GRU LSTM RNN GRU LSTM RNN GRU LSTM


LLaMa 1B 0.799 0.842 0.831 0.750 0.790 0.783 0.725 0.774 0.758 0.597 0.644 0.630
LLaMa 3B 0.832 0.861 0.844 0.779 0.808 0.794 0.760 0.779 0.772 0.632 0.662 0.645
LLaMa 7B 0.853 0.894 0.872 0.805 0.851 0.825 0.771 0.789 0.778 0.656 0.698 0.668
Qwen 1.8B 0.724 0.824 0.821 0.672 0.798 0.774 0.657 0.780 0.750 0.525 0.649 0.622
Qwen 7B 0.842 0.931 0.922 0.794 0.881 0.870 0.773 0.813 0.810 0.646 0.726 0.718
Mistral 7B 0.864 0.888 0.871 0.812 0.839 0.819 0.791 0.817 0.799 0.662 0.687 0.675
LLaVa 7B 0.902 0.941 0.934 0.853 0.892 0.887 0.813 0.828 0.815 0.699 0.740 0.735


LLaMa 1B 0.825 0.855 0.852 0.776 0.814 0.802 0.753 0.793 0.781 0.626 0.666 0.652
LLaMa 3B 0.858 0.892 0.871 0.810 0.841 0.821 0.787 0.788 0.779 0.660 0.693 0.671
LLaMa 7B 0.879 0.924 0.897 0.829 0.874 0.847 0.801 0.807 0.805 0.680 0.724 0.692
Qwen 1.8B 0.762 0.872 0.846 0.713 0.821 0.796 0.690 0.800 0.775 0.563 0.673 0.647
Qwen 7B 0.867 0.948 0.936 0.817 0.898 0.885 0.796 0.815 0.814 0.669 0.747 0.736
Mistral 7B 0.883 0.906 0.892 0.832 0.855 0.842 0.788 0.791 0.790 0.683 0.707 0.692
LLaVa 7B 0.923 0.958 0.946 0.873 0.906 0.897 0.822 0.826 0.814 0.724 0.757 0.746



on a validation set and fixed prior to testing to prevent leakage.


V. RESULTS


We evaluate EigenTrack using RNN, GRU, and LSTM
recurrent heads and report all results. Across all settings, GRUs
consistently achieve the strongest performance. Unless stated
otherwise, reported EigenTrack results therefore refer to the
GRU variant. For the same reason, ablation studies primarily
use GRUs, while differences across recurrent architectures are
analyzed explicitly in the interpretability section.


**Hallucination Detection:** Table I shows that EigenTrack
achieves strong hallucination detection performance across
all evaluated LLMs, with AUROC ranging from 0.82 to
0.94. GRUs consistently outperform LSTMs and standard
RNNs, highlighting the importance of gated temporal memory
for tracking spectral dynamics over decoding steps. Performance improves with model scale: moving from LLaMa1B to LLaMa-7B yields an average gain of approximately
0.05 AUROC. The strongest results are on 7B-scale models,
with Qwen-7B and LLaVa-7B exceeding 0.93 AUROC. Larger
models produce sharper and more stable spectral signatures,
which GRUs exploit more effectively than simpler recurrent
cells. Compared to prior hallucination detectors evaluated on
HaluEval, EigenTrack achieves a clear margin. On LLaMa7B, it reaches 0.89 AUROC, surpassing HaloScope (0.86),
LapEigvals (0.87), INSIDE (0.81), and even SelfCheckGPT
applied to the much larger LLaMa-30B (0.84). These results
indicate that temporal spectral modeling can compensate for
smaller backbone size by extracting more informative internal
dynamics.


**OOD Detection:** A similar pattern emerges for OOD detection, as shown in Table I. EigenTrack achieves AUROC values
between 0.85 and 0.96 across models and architectures. GRU
performance improves from 0.855 on LLaMa-1B to 0.924 on
LLaMa-7B, and reaches 0.948 on Qwen-7B and 0.958 on



LLaVa-7B. The gap between GRUs and RNNs or LSTMs
is larger for smaller models, suggesting that gated memory
is particularly valuable when spectral signals are weaker or
noisier. Across all models, OOD detection performance is
typically higher than hallucination detection. This suggests
that domain shifts induce more pronounced and consistent
spectral deviations than factual drift within the same domain.
By combining global spectral features with temporal modeling, EigenTrack achieves state-of-the-art performance on both
hallucination and OOD detection while relying only on a
lightweight recurrent head.


**Best vs. Worst Feature Triplets:** Table I also reports the
best and worst performing triplets of spectral features identified through ablation. The large gap between these extremes
across model sizes and recurrent architectures indicates that
detection performance is driven by a compact subset of
informative statistics rather than diffuse contributions. GRUs
benefit most from well-chosen triplets, while RNNs degrade
more sharply under suboptimal feature selection, consistent
with their weaker ability to integrate temporal context.

Larger models exhibit smaller gaps between best and worst
triplets, suggesting that their activation spectra provide more
stable signals even when the feature subset is not optimal.
Consistent with the RMT grounding, triplets that include divergence to the Marchenko-Pastur baseline, leading eigenvalues,
and spectral gaps are consistently most predictive, whereas
triplets dominated by central-tendency measures contribute
less. This pattern aligns with the interpretability analysis in
Sec. V, which shows that predictive power concentrates on a
small set of theoretically motivated spectral features.


**Comparison to State of the Art:** Table II compares EigenTrack to representative state-of-the-art detectors on the LLaMa
model family. Across both hallucination and OOD settings,
EigenTrack consistently achieves the highest AUROC scores.
These gains stem from modeling temporal spectral statistics


TABLE II
AUROC COMPARISON ON LLAMA FAMILY MODELS (1B/3B/7B) OF
EIGENTRACK WITH GRU HEAD CLASSIFIER AND OTHER SOTA METHODS.


Method LLaMa-1B LLaMa-3B LLaMa-7B


**EigenTrack** **0.842** **0.861** **0.894**
LapEigvals 0.785 0.819 0.871
INSIDE 0.753 0.831 0.810
SelfCheckGPT 0.739 0.804 0.809
HaloScope 0.820 0.827 0.861


**EigenTrack** **0.855** **0.892** **0.924**
Cosine Distance 0.819 0.877 0.920
Energy Score 83.2 0.852 0.890
Max Softmax Prob 0.701 0.710 0.720
ODIN 0.801 0.842 0.921


of hidden activations, which capture global representation
dynamics that surface-level confidence methods such as Max
Softmax Probability and ODIN, as well as snapshot spectral analyses such as LapEigvals, fail to capture. Baseline
OOD methods are score-based and operate without OOD
supervision. Their AUROC values are obtained by sweeping
thresholds over in-distribution scores, ensuring a fair and
threshold-independent comparison. Overall, GRUs outperform
LSTMs and RNNs, confirming the value of gated memory for
modeling spectral evolution, with a lightweight architecture.
EigenTrack outperforms simpler baselines because it jointly
models multi-layer and temporal dynamics. RMT-based distances, such as KL divergence and Wasserstein distance to
the Marchenko-Pastur baseline, provide a principled reference
for quantifying deviation from isotropic noise. The recurrent
architecture further captures sequential patterns that static
metrics and single-step distances miss, enabling early detection
when spectra drift toward noise-like regimes. We also observe
that VLMs generally exhibit stronger detection performance
than text-only LLMs. A plausible explanation is that crossmodal alignment and redundant visual cues reduce ambiguity,
making distribution shifts more detectable in the joint representation. A deeper multimodal analysis is left to future work.


**Interpretability Analysis:** An important question is which
spectral statistics are most critical for detection performance.
To address this, we train GRU classifiers on all triplets of
spectral features and report results in Table I. Discriminative
power is concentrated in a small subset of features rather than
distributed uniformly. In particular, spectral power measures,
leading eigenvalues, KL divergence from the MarchenkoPastur baseline, and intermediate spectral gaps consistently
appear in the most informative combinations, yielding AUROC
values up to 0.79 on LLaMa-7B. Excluding these features
reduces performance to approximately 0.69, indicating that a
compact and interpretable feature set is sufficient.
To further validate feature importance, we apply SHAP to
trained recurrent classifiers. SHAP assigns contribution values
to each feature, enabling fine-grained attribution of model
decisions. Fig. 3 shows the five most influential features
per architecture. GRUs emphasize KL divergence, leading



Fig. 3. Heatmap (red: high, blue: low) shows the most important features for
each classifier on LLaMa 3B and Hallucination dataset, computed by SHAP
and normalized. It highlights how different RNNs focus on spectral statistics.


eigenvalue gaps, and entropy, reflecting sensitivity to structural
instabilities. LSTMs place more weight on top- _k_ eigenvalues
and entropy, consistent with their ability to model gradual
drifts. Standard RNNs rely more heavily on central-tendency
measures alongside entropy and the maximum eigenvalue,
consistent with their limited temporal memory.
Fig. 4(a) summarizes the cumulative SHAP attribution mass
as features are added in rank order. GRUs concentrate nearly
80% of total attribution within the top ten features, while
RNNs distribute importance more evenly. LSTMs lie between
these extremes, both in feature concentration and overall
performance (Table I). These results confirm that EigenTrack
relies on a small set of theoretically grounded spectral descriptors and that gated temporal modeling enables effective
exploitation of this structure.


VI. ABLATION STUDIES


We analyze two key hyperparameters that govern the
accuracy-latency tradeoff of EigenTrack. The first is the
sliding-window length, which determines how much temporal
context is aggregated into each spectral snapshot. Short windows respond quickly but are more sensitive to noise, while
longer windows smooth fluctuations at the cost of delayed
detection. The second is the response length, which controls
how many generated tokens are observed before making a
decision. Early decisions reduce latency but may miss lateemerging cues, whereas longer responses improve reliability at
higher cost. All ablation studies are conducted using LLaMa3B with a GRU classifier on the hallucination detection task,
which serves as a representative and challenging setting.


_A. Sliding-Window Length_


Fig. 4(b) reports AUROC and inference latency as functions
of the sliding-window length. Shorter windows capture finergrained temporal dynamics and yield higher AUROC, but incur
greater latency because more windows must be processed per
sequence. As the window length increases, performance slowly
declines while latency decreases.
A clear knee emerges in the range of 25 to 50 tokens, which
provides a strong operating point across models. Below this
range, spectral estimates become noisy, and performance stabilizes while computational cost increases. Beyond it, additional


Fig. 4. (a) Cumulative SHAP attribution mass as a function of the number of top-ranked spectral features for different recurrent architectures on LLaMa-3B
(Hallucination dataset). (b) AUROC and inference latency versus sliding-window length for GRU-based hallucination detection on LLaMa-3B (EigenTrack
head latency, independent of LLM inference speed). (c) AUROC versus observed response length for GRU-based hallucination detection on LLaMa-3B.



temporal context degrades and delays detection capabilities,
adding unnecessary smoothing in the sequence. This behavior
indicates that the spectral dynamics relevant for hallucination
detection are captured within a moderate temporal horizon.


_B. Response Length_


Fig. 4(c) plots AUROC as a function of the number of
generated tokens observed before classification. Performance
starts near chance and rises sharply within the first few tokens
before saturating. This indicates that spectral cues associated
with hallucination emerge early in generation, well before
errors become explicit in the output. The rapid AUROC
increase is consistent with uncertainty accumulating quickly as
representations drift toward noise-like regimes, as predicted by
the RMT-based analysis, while the plateau reflects diminishing
marginal information from later tokens. These results support
EigenTrack’s suitability for early stopping, allowing practitioners to trade accuracy for latency by adjusting response length
and terminating failing generations without full output.


VII. CONCLUSION


We presented EigenTrack, a lightweight and interpretable
framework for detecting hallucination and out-of-distribution
behavior in LLMs and VLMs via temporal modeling of
spectral features. By combining global eigenvalue-based descriptors with RMT-grounded divergence to the MarchenkoPastur baseline and recurrent temporal modeling, EigenTrack
captures representation dynamics that static or output-layer
heuristics miss, enabling early detection and efficient termination. Experiments demonstrate state-of-the-art performance
across LLM and VLM families with favorable accuracylatency tradeoffs. Future work will focus on tighter theoretical
guarantees, deeper multimodal analysis, adaptive correction
strategies, and automatic selection of spectral features, window
size, and monitored layers.


REFERENCES


[1] S. Tayebati, N. Darabi, D. Kumar, D. K. K. Dona, T. Tulabandhula,
R. Krishnan, and A. R. Trivedi, “Cap: Conformalized abstention policies
for context-adaptive risk management for llms and vlms,” in _Proceedings_
_of the Asian Conference on Machine Learning (ACML)_, 2025.

[2] D. Jayasuriya, S. Tayebati, D. Ettori, R. Krishnan, and A. R. Trivedi,
“Sparc: Subspace-aware prompt adaptation for robust continual learning
in llms,” in _Proceedings of the 2025 International Joint Conference on_
_Neural Networks_, 2025.




[3] D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified
and out-of-distribution examples in neural networks,” in _International_
_Conference on Learning Representations_, 2017.

[4] S. Farquhar and Y. Gal, “Semantic entropy: Language models detect
and explain their failures,” _Transactions on Machine Learning Research_,
2024, arXiv:2305.17409.

[5] P. Manakul, A. Liusie, and M. J. F. Gales, “Selfcheckgpt: Zero-resource
black-box hallucination detection for generative large language models,”
in _Proceedings of EMNLP 2023_, 2023, pp. 9004–9017.

[6] D. Lei, Y. Li, M. Hu, M. Wang, V. Yun, E. Ching, and E. Kamal,
“Chain of natural language inference for reducing large language model
ungrounded hallucinations,” _arXiv preprint arXiv:2310.03951_, 2023.

[7] S. Valentin, J. Fu, G. Detommaso, S. Xu, G. Zappella, and
B. Wang, “Cost-effective hallucination detection for llms,” _arXiv_
_preprint arXiv:2407.21424_, 2024.

[8] E. Mitchell, Y. Lee, A. Khazatsky, C. D. Manning, and C. Finn,
“Detectgpt: Zero-shot machine-generated text detection using probability
curvature,” in _Proceedings of ICML 2023_, 2023, pp. 24 950–24 962.

[9] G. Bao, Y. Zhao, Z. Teng, L. Yang, and Y. Zhang, “Fast-detectgpt:
Efficient zero-shot detection of machine-generated text via conditional
probability curvature,” _arXiv preprint arXiv:2310.05130_, 2023.

[10] G. Bao, Y. Zhao, J. He, and Y. Zhang, “Glimpse: Enabling whitebox methods to use proprietary models for zero-shot llm-generated text
detection,” in _Proceedings of ICLR 2025_, 2025.

[11] W. Su, C. Wang, Q. Ai, Y. Hu, Z. Wu, Y. Zhou, and Y. Liu,
“Unsupervised real-time hallucination detection based on the internal
states of large language models,” in _Findings of the Association for_
_Computational Linguistics: ACL 2024_, 2024, pp. 14 379–14 391.

[12] J. Binkowski, D. Janiak, A. Sawczyn, B. Gabrys, and T. Kajdanowicz,
“Hallucination detection in llms using spectral features of attention
maps,” in _Proceedings of ICML 2025_, 2025.

[13] Z. Sun, X. Zang, K. Zheng, Y. Song, J. Xu, X. Zhang, W. Yu, and H. Li,
“Redeep: Detecting hallucination in retrieval-augmented generation via
mechanistic interpretability,” in _Proceedings of ACL 2025_, 2025.

[14] A. Sriramanan, S. Shen, T. Dao _et al._, “Attentionscore: Faithfulness
detection from attention patterns in llms,” in _Proceedings of ICML 2024_,
2024, arXiv:2311.09516.

[15] A. C. Stutts, D. Erricolo, S. Ravi, T. Tulabandhula, and A. R. Trivedi,
“Mutual information-calibrated conformal feature fusion for uncertaintyaware multimodal 3d object detection at the edge,” in _2024 IEEE_
_international conference on robotics and automation (ICRA)_ . IEEE,
2024, pp. 2029–2035.

[16] A. C. Stutts, D. Erricolo, T. Tulabandhula, and A. R. Trivedi,
“Lightweight, uncertainty-aware conformalized visual odometry,” in
_2023 IEEE/RSJ International Conference on Intelligent Robots and_
_Systems (IROS)_ . IEEE, 2023, pp. 7742–7749.

[17] V. A. Marˇcenko and L. A. Pastur, “Distribution of eigenvalues for some
sets of random matrices,” _Mathematics of the USSR-Sbornik_, 1967.

[18] Y. Song, Y. Huang, C. Yang, Y. Shi, D. Wei, L. Sun, and L. Chen,
“Rankfeat: Rank-1 feature removal for out-of-distribution detection,” in
_Advances in Neural Information Processing Systems (NeurIPS)_, 2022.

[19] J. Gu, Y. Qiao, and P. Li, “Spectralgap: Graph-level out-ofdistribution detection via laplacian eigenvalue gaps,” _arXiv preprint_
_arXiv:2505.15177_, 2025.

[20] Y. Mei, Z. Li, Y. Li, and J. Zou, “Spectral normalized joint
energy for multi-label out-of-distribution detection,” _arXiv preprint_
_arXiv:2405.04759_, 2024.


[21] E. P. Wigner, “On the distribution of the roots of certain symmetric
matrices,” _Annals of Mathematics_, vol. 67, no. 2, pp. 325–327, 1958.

[22] D. Paul, “Asymptotics of sample eigenstructure for a large dimensional
spiked covariance model,” _Statistica Sinica_, 2007.

[23] J. Baik, G. Ben Arous, and S. P´ech´e, “Phase transition of the largest
eigenvalue for nonnull complex sample covariance matrices,” _Annals of_
_Probability_, vol. 33, no. 5, pp. 1643–1697, 2005.

[24] J. Pennington and P. Worah, “Nonlinear random matrix theory for deep
learning,” in _Advances in Neural Information Processing Systems_, 2017.

[25] C. H. Martin and M. W. Mahoney, “Implicit self-regularization in deep
neural networks: Evidence from random matrix theory and implications
for learning,” _arXiv preprint arXiv:1710.09553_, 2017.

[26] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever,
“Language models are unsupervised multitask learners,” _OpenAI Blog_,
vol. 1, no. 8, p. 9, 2019.

[27] S. Kobayashi, Y. Akram, and J. von Oswald, “Weight decay induces
low-rank attention layers,” _arXiv preprint arXiv:2410.23819_, 2024.

[28] N. Darabi, D. Naik, S. Tayebati, D. Jayasuriya, R. Krishnan, and
A. R. Trivedi, “Eigenshield: Causal subspace filtering via random matrix
theory for adversarially robust vision-language models,” _arXiv preprint_
_arXiv:2502.14976_, 2025.

[29] D. Ettori, N. Darabi, S. Senthilkumar, and A. R. Trivedi, “Rmt-kd:
Random matrix theoretic causal knowledge distillation,” _arXiv preprint_
_arXiv:2509.15724_, 2025.

[30] B. van Aken, B. Winter, A. Loeser, and F. A. Gers, “How
does bert answer questions? a layer-wise analysis of transformer
representations,” in _Proceedings_ _of_ _the_ _28th_ _ACM_ _International_
_Conference on Information and Knowledge Management (CIKM)_, 2019,
arXiv:1909.04925. [Online]. Available: https://arxiv.org/abs/1909.04925

[31] M. Hosseini, M. Munia, and L. Khan, “Bert has more to offer: Bert layers combination yields better sentence embeddings,” in _Findings of the_
_Association for Computational Linguistics: EMNLP 2023_ . Singapore:
Association for Computational Linguistics, 2023, pp. 15 419–15 431.


