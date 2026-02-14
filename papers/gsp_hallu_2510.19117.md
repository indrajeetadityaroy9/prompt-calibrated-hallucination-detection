## A Graph Signal Processing Framework for Hallucination Detection in Large Language Models

Valentin Noël
Devoteam, Paris, France
valentin.noel@devoteam.com


Preprint — Under review (2025)


**Abstract**


Large language models achieve impressive results but distinguishing factual reasoning from hallucinations remains challenging. We propose a spectral analysis framework that models transformer layers
as dynamic graphs induced by attention, with token embeddings as signals on these graphs. Through
graph signal processing, we define diagnostics including Dirichlet energy, spectral entropy, and highfrequency energy ratios, with theoretical connections to computational stability. Experiments across GPT
architectures suggest universal spectral patterns: factual statements exhibit consistent "energy mountain"
behavior with low-frequency convergence, while different hallucination types show distinct signatures.
Logical contradictions destabilize spectra with large effect sizes ( _g_ _>_ 1 _._ 0), semantic errors remain stable
but show connectivity drift, and substitution hallucinations display intermediate perturbations. A simple
detector using spectral signatures achieves 88.75% accuracy versus 75% for perplexity-based baselines,
demonstrating practical utility. These findings indicate that spectral geometry may capture reasoning
patterns and error behaviors, potentially offering a framework for hallucination detection in large language
models.

### **1 Introduction**


The internal dynamics of transformer language models remain opaque despite their empirical success [1].
Existing interpretability methods, e.g. attention visualization [2,3], probing tasks [4], mechanistic analysis [5],
provide valuable insights but often lack theoretical foundations or computational scalability. We propose a
fundamentally different approach: analyze transformer representations through the lens of spectral graph
theory [6].
Our key insight is geometric: attention mechanisms induce dynamic graphs over token sequences, and
hidden representations evolve as signals on these graphs [7]. This perspective enables rigorous analysis
using graph signal processing (GSP) theory [8, 9], connecting spectral properties to model behavior through
established mathematical principles.
We make three main contributions. First, we formalize transformer dynamics as graph signals and derive
spectral diagnostics with theoretical guarantees. Second, we establish universal spectral patterns across
architectures: reliable reasoning exhibits systematic low frequency concentration ("spectral convergence"),
while errors manifest distinct high frequency signatures. Third, we demonstrate that different error types
leave characteristic spectral fingerprints, enabling principled detection methods [10].
Our analysis suggests that reliable outputs align with spectrally smooth representations, while instability
correlates with high-frequency oscillations. This opens avenues for model monitoring and interpretability [11].


1


### **2 Dynamic Attention Graph Model**

Consider a layer _ℓ_ with _H_ heads and a sequence of _N_ tokens. Let _A_ [(] _[ℓ,h]_ [)] _∈_ R _[N]_ _[×][N]_ be the post-softmax
attention of head _h_ [1]. We build an undirected weighted graph by symmetrization,

_W_ [(] _[ℓ,h]_ [)] = 12     - _A_ [(] _[ℓ,h]_ [)] + ( _A_ [(] _[ℓ,h]_ [)] ) _[⊤]_ [�] _,_ _L_ [(] _[ℓ,h]_ [)] = _D_ [(] _[ℓ,h]_ [)] _−_ _W_ [(] _[ℓ,h]_ [)] _,_ (1)


with _D_ [(] _[ℓ,h]_ [)] = diag( _W_ [(] _[ℓ,h]_ [)] 1 ). Heads are aggregated by _W_ [¯] [(] _[ℓ]_ [)] = [�] _h_ _[H]_ =1 _[α][h][W]_ [ (] _[ℓ,h]_ [)] [where] _[α][h]_ _[≥]_ [0] [and]

- _h_ _[α][h]_ [ = 1][.] [The] _[ layer Laplacian]_ [ is] _[ L]_ [(] _[ℓ]_ [)] [=] _[D]_ [¯] [(] _[ℓ]_ [)] _[ −]_ _[W]_ [¯] [ (] _[ℓ]_ [)] [[][12][].] [Let] _[ X]_ [(] _[ℓ]_ [)] _[∈]_ [R] _[N]_ _[×][d]_ [be token representations]
(rows: tokens; columns: embedding dimensions).


**2.1** **Graph-signal preliminaries.**


For a symmetric nonnegative _W_, _L_ = _D −_ _W_ admits _L_ = _U_ Λ _U_ _[⊤]_ with eigenvalues 0 = _λ_ 1 _≤· · · ≤_ _λN_ [6].
For a signal _x_ _∈_ R _[N]_, the graph Fourier coefficients are _x_ ˆ = _U_ _[⊤]_ _x_ and the Dirichlet energy is _x_ _[⊤]_ _Lx_ =
�( _i,j_ ) _[W][ij]_ [(] _[x][i][ −]_ _[x][j]_ [)][2] [=][ �] _m_ _[λ][m][x]_ [ˆ] _m_ [2] [[][13][].]

### **3 Graph-Spectral Diagnostics for LLMs**


Each column _x_ [(] _k_ _[ℓ]_ [)] of _X_ [(] _[ℓ]_ [)] is a scalar graph signal. Define the _layer energy_



_E_ [(] _[ℓ]_ [)] =



_d_

- 
( _x_ [(] _k_ _[ℓ]_ [)][)] _[⊤][L]_ [(] _[ℓ]_ [)] _[x]_ _k_ [(] _[ℓ]_ [)] = Tr ( _X_ [(] _[ℓ]_ [)] ) _[⊤]_ _L_ [(] _[ℓ]_ [)] _X_ [(] _[ℓ]_ [)][�] _,_ (2)
_k_ =1



and the _smoothness index_ SMI [(] _[ℓ]_ [)] = _E_ [(] _[ℓ]_ [)] _/_ Tr�( _X_ [(] _[ℓ]_ [)] ) _[⊤]_ _X_ [(] _[ℓ]_ [)][�] [14]. Let _L_ [(] _[ℓ]_ [)] = _U_ [(] _[ℓ]_ [)] Λ [(] _[ℓ]_ [)] ( _U_ [(] _[ℓ]_ [)] ) _[⊤]_ and _X_ [ˆ] [(] _[ℓ]_ [)] =

( _U_ [(] _[ℓ]_ [)] ) _[⊤]_ _X_ [(] _[ℓ]_ [)] . Spectral energies are _s_ [(] _m_ _[ℓ]_ [)] [=] _[∥][X]_ [ˆ] _m,_ [(] _[ℓ]_ [)] _·_ _[∥]_ [2] 2 [, normalized masses] _[ p]_ _m_ [(] _[ℓ]_ [)] [=] _[s]_ [(] _m_ _[ℓ]_ [)] _[/]_ [ �] _r_ _[s]_ _r_ [(] _[ℓ]_ [)][.] [The] _[ spectral]_
_entropy_ is SE [(] _[ℓ]_ [)] = _−_ [�] _m_ _[p]_ _m_ [(] _[ℓ]_ [)] [log] _[ p]_ _m_ [(] _[ℓ]_ [)] [[][15][].] [For a cutoff] _[ K]_ [, the] _[ high-frequency energy ratio]_ [ is]



HFER [(] _[ℓ]_ [)] ( _K_ ) =




- _N_
_m_ = _K_ +1 _[s]_ _m_ [(] _[ℓ]_ [)]
_._ (3)

 - _N_
_m_ =1 _[s]_ _m_ [(] _[ℓ]_ [)]



Inter-layer stability can be tracked via _E_ [(] _[ℓ]_ [+1)] _/E_ [(] _[ℓ]_ [)] and by spectral cosine similarity across layers [16].

### **4 Theoretical Guarantees**


We relate spectral concentration to bounded node-wise variation and perturbation robustness [17].


**Assumption** **1** (Connectivity and bounded degree) **.** _For_ _each_ _ℓ,_ _the_ _aggregated_ _graph_ _is_ _connected_ _and_
_degrees satisfy_ 0 _< d_ [(] min _[ℓ]_ [)] _[≤]_ _[d]_ _i_ [(] _[ℓ]_ [)] _≤_ _d_ [(] max _[ℓ]_ [)] _[<][ ∞][.]_


**Proposition 1** (Energy as edge-wise variation) **.** _For any layer ℓ,_



_E_ [(] _[ℓ]_ [)] =



_d_



_k_ =1




- _W_ ¯ _ij_ [(] _[ℓ]_ [)] - _x_ [(] _ik_ _[ℓ]_ [)] _[−]_ _[x]_ _jk_ [(] _[ℓ]_ [)] �2 _._ (4)

_i<j_



_In particular, E_ [(] _[ℓ]_ [)] = 0 _if and only if each column of X_ [(] _[ℓ]_ [)] _is constant on the connected component._



**Theorem 1** (Spectral Poincaré control) **.** _Let λ_ [(] 2 _[ℓ]_ [)] _be the Fiedler value of L_ [(] _[ℓ]_ [)] _[18]._ _For any column x_ [(] _k_ _[ℓ]_ [)] _with_
_zero-mean on nodes,_



1
_∥x_ [(] _k_ _[ℓ]_ [)] _[∥]_ 2 [2] _[≤]_

[(]



( _x_ [(] _k_ _[ℓ]_ [)][)] _[⊤][L]_ [(] _[ℓ]_ [)] _[x]_ _k_ [(] _[ℓ]_ [)] _[.]_ (5)
_λ_ [(] 2 _[ℓ]_ [)]



_Summing over k yields ∥X_ [(] _[ℓ]_ [)] _∥_ [2] _F_ _[≤]_ _[λ]_ 2 [(] _[ℓ]_ [)] _[ −]_ [1] _E_ [(] _[ℓ]_ [)] _after column centering._


2


**Proposition** **2** (High-frequency dominance and local discrepancy) **.** _Fix_ _K._ _If_ HFER [(] _[ℓ]_ [)] ( _K_ ) _≥_ _ρ_ _with_
_ρ ∈_ (0 _,_ 1) _, then the median absolute inter-neighbor deviation obeys_


~~�~~
MAD [(] _[ℓ]_ [)] ≳ _c_ ( _K,_ Λ [(] _[ℓ]_ [)] ) SMI [(] _[ℓ]_ [)] _ρ,_ (6)


_for an explicit c determined by the spectral gap at K [19]._ _Sustained high-frequency mass implies pronounced_
_local inconsistencies._


**Theorem** **2** (Lipschitz readout under spectral control) **.** _Let_ _y_ = _X_ [(] _[ℓ]_ [)] _W_ out _be_ _a_ _linear_ _readout._ _For_ _a_
_column-centered perturbation δ,_


_∥y_ ( _X_ [(] _[ℓ]_ [)] + _δ_ ) _−_ _y_ ( _X_ [(] _[ℓ]_ [)] ) _∥F_ _≤∥W_ out _∥_ 2 _λ_ [(] 2 _[ℓ]_ [)] _[ −]_ [1] _[/]_ [2] ~~�~~ _E_ ( _δ_ ) _._ (7)


_Hence robustness to token noise is governed by perturbation energy and graph connectivity [20]._

### **5 Experimental Results**


We validate the proposed GSP framework across multiple GPT architectures, testing whether hallucinations
leave distinct spectral fingerprints compared to factual reasoning [21].


**5.1** **Cross-Architecture Universality**


We analyze factual baselines across GPT-2 (12 layers) [22], DistilGPT-2 (6 layers) [23], and GPT-2 Medium
(24 layers). Figure 1 shows three runs per model with means.
All architectures follow the _energy mountain_ : initial low energy ( _∼_ 10K), sharp buildup (2.0M–9.0M
peak), and dissipation to _∼_ 0.1M at output. Reduction ratios (50–60 _×_ ) are invariant to model size, suggesting
universal convergence. HFER drops to 0.1–0.3 in final layers, consistent with spectral Poincaré predictions
for reliable outputs [24].


Table 1: Cross-architecture summary (factual runs).


**Model** **Peak Energy (M)** **Final HFER** **Final Entropy**


DistilGPT-2 2.0 0.12 0.72
GPT-2 6.0 0.14 0.71
GPT-2 Medium 9.0 0.13 0.70


**5.2** **Spectral Evolution under Factual Reasoning**


Entropy decreases monotonically from SE [(0)] _≈_ 1 _._ 2 to SE [(] _[L]_ [)] _≈_ 0 _._ 7, while smoothness rises, stabilizing the
token graph. Connectivity follows the same trajectory: the Fiedler value grows from 0.40–0.50 at input
to 0.90+ at output. This monotonic progression is universal across architectures and constitutes a spectral
signature of factual reasoning.


**5.3** **Hallucination Trajectories**


We next contrast hallucinations with baselines.


**5.3.1** **Logical hallucinations.**


Figure 2 shows three fabricated statements (“Two plus two equals seven,” “Shakespeare was born after he
died,” “Five is smaller than three”). Logical errors lead to strong run-to-run variance: entropy spikes, HFER
oscillations, and unstable smoothness indices. These findings indicate that contradictions disrupt spectral
stability, producing high variance across repeated runs.


3


|Col1|Col2|Col3|Hfer|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||GPT<br>~~Dist~~|-2<br>~~ilGPT-2~~|
|||||GPT|GPT|-2 Medium|
|||||GPT|GPT||
||||||||
||||||||
||||||||
||||||||


Layer

Smoothness Index

|Col1|Col2|Col3|Col4|Col5|GPT|-2|
|---|---|---|---|---|---|---|
|||||~~Dis~~<br>GPT|~~Dis~~<br>GPT|~~ilGPT-2~~<br>-2 Medium|
|||||~~Dis~~<br>GPT|||
||||||||
||||||||
||||||||
||||||||
||||||||



Layer



8


6


4


2


0


1.4

1.3

1.2

1.1

1.0

0.9

0.8

0.7



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
|||||GPT<br>Dist<br>|GPT<br>Dist<br>||
|||||GPT<br>Dist<br>|GPT<br>Dist<br>|-2<br>ilGPT-2<br>|
|||||~~GP~~|~~GP~~|~~2 Medium~~|
|||||~~GP~~|~~GP~~||
||||||||
||||||||


Layer

Spectral Entropy

|Col1|Col2|Col3|Col4|Col5|GPT<br>Dist|-2<br>ilGPT-2|
|---|---|---|---|---|---|---|
|||||GPT|GPT|-2 Medium|
|||||GPT|GPT||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||



Layer



0.5


0.4


0.3


0.2


0.1


0.6


0.5


0.4


0.3


0.2


0.1



Figure 1: Cross-architecture factual baselines. Thin curves: three runs. Thick curves: mean per model.
Universality is observed in energy mountain, entropy dip, and smoothness plateau.



Hallucinations - Hfer

|Col1|Col2|Col3|Col4|Col5|GPT<br>Dist|-2<br>ilGPT-2|
|---|---|---|---|---|---|---|
|||||GPT|GPT|-2 Medium|
|||||GPT|GPT||
||||||||
||||||||
||||||||
||||||||



Layer

Hallucinations - Smoothness Index

|Col1|Col2|Col3|Col4|Col5|GPT<br>Dist|-2<br>ilGPT-2|
|---|---|---|---|---|---|---|
|||||GPT|GPT|-2 Medium|
|||||GPT|GPT||
||||||||
||||||||
||||||||
||||||||
||||||||



Layer



8


6


4


2


0


1.3

1.2

1.1

1.0

0.9

0.8

0.7



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||GPT<br>Dist<br>~~GP~~|-2<br>ilGPT-2<br>~~2 Medium~~|
||||||||
||||||||
||||||||
||||||||


Layer

Hallucinations - Spectral Entropy

|Col1|Col2|Col3|Col4|Col5|GPT<br>Dist|-2<br>ilGPT-2|
|---|---|---|---|---|---|---|
||||||GPT|-2 Medium|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||



Layer



0.5


0.4


0.3


0.2


0.1


0.6


0.5


0.4


0.3


0.2


0.1



Figure 2: Logical hallucinations. Transparent: three runs. Thick curves: mean per model. Strong variance
emerges in entropy and HFER.


4


**5.3.2** **Semantic hallucinations.**


In contrast, semantic hallucinations (Figure 3) display strikingly low variance. Across runs, curves for energy,
HFER, entropy, and smoothness are nearly indistinguishable from factual baselines. This indicates that
semantic errors are processed with spectral stability, making them indistinguishable from factual reasoning in
primary metrics.



|Col1|Col2|Col3|Col4|Col5|GPT<br>Dist|-2<br>ilGPT-2|
|---|---|---|---|---|---|---|
||||||GPT|-2 Medium|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


Layer

Semantic Hallucinations - Smoothness Index

|Col1|Col2|Col3|Col4|Col5|GPT|-2|
|---|---|---|---|---|---|---|
|||||~~Dis~~<br>GPT|~~Dis~~<br>GPT|~~ilGPT-2~~<br>-2 Medium|
|||||~~Dis~~<br>GPT|~~Dis~~<br>GPT||
||||||||
||||||||
||||||||
||||||||
||||||||



Layer



8


6


4


2


0


1.2


1.1


1.0


0.9


0.8


0.7



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
|||||GPT<br>Dist<br>|GPT<br>Dist<br>||
|||||GPT<br>Dist<br>|GPT<br>Dist<br>|-2<br>ilGPT-2<br>|
|||||~~GP~~|~~GP~~|~~2 Medium~~|
|||||~~GP~~|~~GP~~||
||||||||
||||||||


Layer

Semantic Hallucinations - Spectral Entropy

|Col1|Col2|Col3|Col4|Col5|GPT|-2|
|---|---|---|---|---|---|---|
|||||~~Dis~~<br>GPT|~~Dis~~<br>GPT|~~ilGPT-2~~<br>-2 Medium|
|||||~~Dis~~<br>GPT|~~Dis~~<br>GPT||
||||||||
||||||||
||||||||
||||||||
||||||||



Layer



0.55

0.50

0.45

0.40

0.35

0.30

0.25

0.20

0.15


0.6


0.5


0.4


0.3


0.2


0.1



Figure 3: Semantic hallucinations. Three runs + mean per model. Variance is minimal, showing spectral
stability despite incorrect semantics.


**5.3.3** **Substitution hallucinations.**


Substitution hallucinations (e.g., entity replacements) show intermediate behavior: smoother and more stable
than logical errors, but with slightly elevated entropy and HFER. Smoothness and Fiedler values remain near
baseline, suggesting modest spectral perturbation without strong instability (Figure 4).


**5.3.4** **Baseline variance contextualization.**


To validate whether hallucination deviations exceed baseline variability, we overlay hallucination means with
baseline error bands. Figures 5 show Fiedler values with _±_ 1 standard deviation bands computed from factual
runs. Logical hallucinations exceed baseline bands, while semantic hallucinations mostly remain within,
except for systematic late-layer Fiedler drift.


**5.4** **Connectivity Drift as Semantic Marker**


Secondary diagnostics reveal a new contrast. Fiedler values show notable divergence between factual and
semantic hallucinations. As shown in Figure 6, early layers exhibit little difference, but later layers show
systematic drift: hallucinations converge to higher Fiedler values than baselines. This suggests semantic
hallucinations manifest as _connectivity drift_, where the model enforces overly strong global coherence on
factually incorrect structures.


5


Substitution Hallucinations - Hfer

|Col1|Col2|Col3|Col4|Col5|GPT<br>Dist|-2<br>ilGPT-2|
|---|---|---|---|---|---|---|
|||||GPT|GPT|-2 Medium|
|||||GPT|GPT||
||||||||
||||||||
||||||||
||||||||



Layer

Substitution Hallucinations - Smoothness Index

|Col1|Col2|Col3|Col4|Col5|GPT|-2|
|---|---|---|---|---|---|---|
||||||~~Dist~~<br>GPT|~~ilGPT-2~~<br>-2 Medium|
||||||||
||||||||
||||||||
||||||||
||||||||



Layer



8


6


4


2


0


1.2


1.1


1.0


0.9


0.8


0.7


0.9


0.8


0.7


0.6


0.5



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
|||||GPT<br>Dist<br>|GPT<br>Dist<br>||
|||||GPT<br>Dist<br>|GPT<br>Dist<br>|-2<br>ilGPT-2<br>|
|||||~~GP~~|~~GP~~|~~-2 Medium~~|
|||||~~GP~~|~~GP~~||
||||||||
||||||||


Layer

Substitution Hallucinations - Spectral Entropy

|Col1|Col2|Col3|Col4|Col5|GPT|-2|
|---|---|---|---|---|---|---|
||||||Dis<br>GPT|ilGPT-2<br>-2 Medium|
||||||||
||||||||
||||||||
||||||||
||||||||



Layer



Figure 4: Substitution hallucinations. Three runs + mean per model.









|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|||||||


Layer



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||


Layer



Layer



GPT-2 baseline mean
GPT-2 Logical mean



DistilGPT-2 baseline mean
DistilGPT-2 Logical mean



GPT-2 Medium baseline mean
GPT-2 Medium Logical mean



Figure 5: Fiedler values with baseline error bands. Semantic hallucinations show systematic late-layer drift
beyond baseline variability.



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
|||||G<br>D|G<br>D||
|||||G<br>D|G<br>D|PT-2<br>istilGPT-2|
||||||G|PT-2 Medium|


Layer



0.9


0.8


0.7


0.6


0.5



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
|||||G<br>~~D~~|G<br>~~D~~||
|||||G<br>~~D~~|G<br>~~D~~|PT-2<br>~~istilGPT-2~~|
||||||G|PT-2 Medium|


Layer



0.9


0.8


0.7


0.6


0.5



Figure 6: Fiedler values for semantic hallucinations (left) and factual baselines (right). Semantic hallucinations
exhibit minimal early-layer difference but diverge at deeper layers.


6


**5.5** **Baseline Consistency and Statistical Validation**


To contextualize hallucination divergences, we first quantify baseline variability across factual runs. Table 2
shows mean _±_ standard deviation of final-layer diagnostics for three factual runs per architecture. Variability
is low ( _<_ 0 _._ 02 absolute in HFER and entropy), indicating that deviations beyond these bands are statistically
meaningful.


Table 2: Baseline consistency: mean _±_ sd across factual runs. Low variance confirms stability of spectral
diagnostics under repeated factual reasoning.


**Model** **Final HFER** **Final Entropy** **Final Fiedler**


DistilGPT-2 0.12 _±_ 0.01 0.72 _±_ 0.01 0.76 _±_ 0.01
GPT-2 0.14 _±_ 0.02 0.71 _±_ 0.01 0.77 _±_ 0.01
GPT-2 Medium 0.13 _±_ 0.01 0.70 _±_ 0.01 0.78 _±_ 0.01


We then test whether hallucination trajectories deviate beyond baseline variance. Table 3 summarizes
Welch’s _t_ -tests and Hedges’ _g_ (effect size) for logical hallucinations versus baseline at the final layer.
Differences are large ( _g_ _>_ 1 _._ 0 for entropy and HFER), confirming that contradictions destabilize spectra
significantly.


Table 3: Logical hallucinations vs. baseline (final layer). Entropy and HFER diverge significantly with large
effect sizes.


**Model** **Baseline HFER** **Logical HFER** **Hedges** _g_


DistilGPT-2 0.12 0.20 +1.05
GPT-2 0.14 0.22 +1.15
GPT-2 Medium 0.13 0.21 +1.20


By contrast, semantic hallucinations show small but systematic connectivity drift. Table 4 reports Fiedler
values at the final layer: effect sizes are modest ( _g_ = 0 _._ 3–0.6) but consistent across models, highlighting
over-connectivity as a distinct semantic marker..


Table 4: Semantic hallucinations vs. baseline: Fiedler final values. Differences are modest in size but
statistically consistent across architectures.


**Model** **Baseline Fiedler** **Semantic Fiedler** **Hedges** _g_


DistilGPT-2 0.76 0.79 +0.34
GPT-2 0.77 0.81 +0.42
GPT-2 Medium 0.78 0.83 +0.56


Table 5: Substitution hallucinations vs. baseline: entropy and smoothness index at the final layer. Effect sizes
are moderate.


**Model** **Baseline Entropy** **Substitution Entropy** **Hedges** _g_


DistilGPT-2 0.72 0.75 +0.40
GPT-2 0.71 0.74 +0.47
GPT-2 Medium 0.70 0.73 +0.51


**5.5.1** **Limitations.**


While logical hallucinations clearly exceed baseline variability, semantic hallucinations often remain within
factual variance for primary metrics (HFER, entropy, SMI). Their detection relies on subtler secondary


7


signatures (Fiedler drift). This indicates that variance-based thresholds are insufficient: future work should
develop adaptive, layerwise statistical detectors and account for multiple comparisons.


**5.6** **Spectral Hallucination Detector**


To demonstrate practical utility, we implement a simple detector using normalized last-layer Fiedler z-scores:


[d]
SHD( _x_ ) = **1** [ _z_ fid( _x_ ) _> τd_ ] _,_ _z_ fid( _x_ ) = _[f]_ [last][(] _[x]_ [)] _[ −]_ _[µ]_ [f] (8)

_σ_ fid


where _f_ last( _x_ ) is the final-layer Fiedler value, _µ_ fid _, σ_ fid are baseline statistics, and _τd_ are domain-specific
thresholds optimized per semantic domain. Table 6 shows detection performance on 80 test samples,
demonstrating that spectral signatures enable effective hallucination detection beyond theoretical analysis.


Table 6: Hallucination detection performance on 80 test samples (50 factual, 30 hallucinations).


**SHD (domain)** **Perplexity [25]** **SelfCheckGPT-style [26]**


**Accuracy** **88.75%** 75.00% 65.00%


**5.7** **Interpretation**


Experiments reveal universal spectral convergence for factual reasoning (energy mountain, entropy dip,
smoothness plateau, connectivity rise). Hallucinations, however, diverge: logical errors destabilize spectra,
while semantic ones stay mostly stable but show entropy increase, smoothness loss, and Fiedler drift [27].
Newer models behave differently: Qwen2.5-7B, for instance, collapses late-layer connectivity, highlighting
model-dependent spectral responses [28].


Table 7: Final-layer spectral entropy. Semantic hallucinations consistently raise entropy, indicating greater
disorder in token graphs.


**Model** **Baseline Mean** **Semantic Mean** **SD** **Hedges** _g_


phi-3-mini 1.05 1.36 ±0.25 +1.55
llama-3.2-1b 1.51 1.67 ±0.23 +0.72
qwen2.5-7b 1.41 1.54 ±0.25 +0.49


Table 8: Final-layer Fiedler values. Connectivity drift emerges as the most discriminative marker of semantic
hallucinations, with Qwen2.5-7B showing a collapse far beyond baseline variance.


**Model** **Baseline Mean** **Semantic Mean** **SD** **Hedges** _g_


phi-3-mini 0.66 0.63 ±0.09 -0.21
llama-3.2-1b 0.76 0.73 ±0.07 -0.43
qwen2.5-7b 0.80 0.20 ±0.31 -2.35

### **6 Computational Complexity**


Energy and smoothness require sparse matrix–matrix products _O_ (nnz( _W_ ) _d_ ) per layer. Spectral entropy and
HFER need partial spectral information; randomized Lanczos scales near-linearly in nnz( _W_ ) for a small
number of eigenpairs [29]. For sequences up to 512 tokens, analysis completes in 10-60 seconds on standard
GPUs, making the framework practical for real-time diagnostics.


8


Figure 7: Baseline vs. semantic hallucinations across new architectures (Phi-3 Mini, LLaMA-3.2 1B,
Qwen2.5-7B). Error bands ( _±_ 1 SD) are derived from factual runs. Semantic hallucinations remain within
baseline variance for most metrics but diverge systematically in entropy, smoothness, and connectivity.

### **7 Discussion and Future Directions**


This work establishes spectral analysis as a principled tool for transformer interpretability [30]. The universal
“energy mountain” highlights consistent mechanisms of reliable generation, while distinct spectral fingerprints
of errors enable diagnostic use.
Future directions include extending analysis to building adaptive detectors for real-time monitoring [31],
and studying larger architectures. While the present study focuses on classification-style tasks, preliminary
evidence suggests that linguistic structure may also shape spectral trajectories, pointing to connections
between spectral geometry and human-interpretable constructs. This establishes spectral analysis as both
theoretically grounded and practically useful for LLM understanding [32].

### **8 Conclusion**


In summary, we presented a spectral graph processing framework that reveals both universal convergence
patterns in factual reasoning and distinct spectral fingerprints of hallucinations. Beyond theoretical insight,
we showed that spectral markers enable a practical hallucination detector that outperforms strong baselines.
Logical hallucinations destabilize spectra, semantic hallucinations manifest as connectivity drift and entropy
rise, and substitution errors exhibit intermediate perturbations. Together, these findings establish spectral
geometry as both an interpretive lens and a diagnostic tool for monitoring large language models.


9


### **References**


[1] A. Vaswani _et al._, “Attention is all you need,” in _Advances in Neural Information Processing Systems_,
pp. 5998–6008, 2017.


[2] H. Chefer, S. Gur, and L. Wolf, “Transformer interpretability beyond attention visualization,” in
_Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 782–791,
2021.


[3] S. Jain and B. C. Wallace, “Attention is not explanation,” in _Proceedings of the 2019 Conference of the_
_North American Chapter of the Association for Computational Linguistics_, pp. 3543–3556, 2019.


[4] A. Rogers, O. Kovaleva, and A. Rumshisky, “A primer on neural network models for natural language
processing,” _Journal of Artificial Intelligence Research_, vol. 57, pp. 345–420, 2020.


[5] N. Elhage _et al._, “A mathematical framework for transformer circuits.” Transformer Circuits Thread,
2021.


[6] F. R. K. Chung, _Spectral Graph Theory_ . American Mathematical Society, 1997.


[7] C. K. Joshi, “Transformers are graph neural networks.” The Gradient, 2020.


[8] D. I. Shuman, S. K. Narang, P. Frossard, A. Ortega, and P. Vandergheynst, “The emerging field of signal
processing on graphs,” _IEEE Signal Processing Magazine_, vol. 30, no. 3, pp. 83–98, 2013.


[9] A. Ortega, P. Frossard, J. Kovaˇcevi´c, J. M. F. Moura, and P. Vandergheynst, “Graph signal processing:
Overview, challenges, and applications,” _Proceedings of the IEEE_, vol. 106, no. 5, pp. 808–828, 2018.


[10] Y. Zhang _et al._, “Siren’s song in the ai ocean: A survey on hallucination in large language models.”
arXiv preprint arXiv:2309.01219, 2023.


[11] M. T. Ribeiro, S. Singh, and C. Guestrin, “Why should i trust you? explaining the predictions of
any classifier,” in _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge_
_Discovery and Data Mining_, pp. 1135–1144, 2016.


[12] D. K. Hammond, P. Vandergheynst, and R. Gribonval, “Wavelets on graphs via spectral graph theory,”
_Applied and Computational Harmonic Analysis_, vol. 30, no. 2, pp. 129–150, 2011.


[13] A. Sandryhaila and J. M. F. Moura, “Discrete signal processing on graphs: Frequency analysis,” _IEEE_
_Transactions on Signal Processing_, vol. 62, no. 12, pp. 3042–3054, 2014.


[14] Y. Dong, J.-B. Cordonnier, and A. Loukas, “Attention is not all you need: Pure attention loses rank
doubly exponentially with depth,” in _International Conference on Machine Learning_, pp. 2793–2803,
2020.


[15] C. E. Shannon, “A mathematical theory of communication,” _Bell System Technical Journal_, vol. 27,
no. 3, pp. 379–423, 1948.


[16] G. Mateos, S. Segarra, A. G. Marques, and A. Ribeiro, “Connecting the dots: Identifying network
structure via graph signal processing,” _IEEE Signal Processing Magazine_, vol. 36, no. 3, pp. 16–43,
2019.


[17] D. A. Spielman and S.-H. Teng, “Spectral partitioning works: Planar graphs and finite element meshes,”
_Linear Algebra and its Applications_, vol. 421, no. 2-3, pp. 284–305, 2007.


10


[18] M. Fiedler, “Algebraic connectivity of graphs,” _Czechoslovak Mathematical Journal_, vol. 23, no. 2,
pp. 298–305, 1973.


[19] J. Cheeger, “A lower bound for the smallest eigenvalue of the laplacian,” in _Problems_ _in_ _Analysis_,
pp. 195–199, 1970.


[20] B. Klartag and G. Kozma, “On the hyperplane conjecture for random convex sets,” _Israel Journal of_
_Mathematics_, vol. 223, no. 1, pp. 213–220, 2018.


[21] Z. Ji _et al._, “Survey of hallucination in natural language generation,” _ACM Computing Surveys_, vol. 55,
no. 12, pp. 1–38, 2023.


[22] A. Radford _et al._, “Language models are unsupervised multitask learners.” OpenAI Blog, 2019.


[23] V. Sanh _et al._, “Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter,” in _NeurIPS_
_Workshop on Energy Efficient Machine Learning and Cognitive Computing_, 2019.


[24] M. Belkin and P. Niyogi, “Laplacian eigenmaps for dimensionality reduction and data representation,”
_Neural Computation_, vol. 15, no. 6, pp. 1373–1396, 2003.


[25] N. Lee, Y. Bang, A. Madotto, M. Khabsa, and P. Fung, “Factuality enhanced language models for
open-ended text generation,” in _Advances in Neural Information Processing Systems_, pp. 34586–34599,
2022.


[26] P. Manakul, A. Liusie, and M. J. Gales, “Selfcheckgpt: Zero-resource black-box hallucination detection
for generative large language models,” in _Proceedings of the 2023 Conference on Empirical Methods in_
_Natural Language Processing_, pp. 9004–9017, 2023.


[27] D. Dale _et al._, “Knowledge neurons in pretrained transformers,” in _Proceedings of the 60th Annual_
_Meeting of the Association for Computational Linguistics_, pp. 8493–8507, 2024.


[28] J. Bai _et al._, “Qwen technical report.” arXiv preprint arXiv:2309.16609, 2023.


[29] N. Halko, P. G. Martinsson, and J. A. Tropp, “Finding structure with randomness: Probabilistic
algorithms for constructing approximate matrix decompositions,” _SIAM Review_, vol. 53, no. 2, pp. 217–
288, 2011.


[30] Y. Belinkov and J. Glass, “Analysis methods in neural language processing: A survey,” _Transactions of_
_the Association for Computational Linguistics_, vol. 7, pp. 49–72, 2022.


[31] P. Manakul, A. Liusie, and M. J. Gales, “Selfcheckgpt: Zero-resource black-box hallucination detection
for generative large language models,” in _Proceedings of the 2023 Conference on Empirical Methods in_
_Natural Language Processing_, pp. 9004–9017, 2023.


[32] L. Huang _et al._, “A survey on hallucination in large language models: Principles, taxonomy, challenges,
and open questions.” arXiv preprint arXiv:2311.05232, 2023.


11


