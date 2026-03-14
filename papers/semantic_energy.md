Semantic Energy: Detecting LLM Hallucination Beyond Entropy

## SEMANTIC ENERGY: DETECTING LLM HALLUCINA### TION BEYOND ENTROPY


**Huan Ma** [1] _[,]_ [2] **, Jiadong Pan** _[∗]_ [2] _[,]_ [5] **, Jing Liu** [2] **, Yan Chen** _[†]_ [2] **, Joey Tianyi Zhou** [3] **, Guangyu Wang** [4]

**Qinghua Hu** [1] **, Hua Wu** [2] **, Changqing Zhang** [1] _[‡]_ **, Haifeng Wang** [2]

1Tianjin University 2Baidu Inc.
3A*STAR Centre for Frontier AI Research (CFAR), Singapore
4Beijing University of Posts and Telecommunications
5University of Chinese Academy of Sciences


ABSTRACT


Large Language Models (LLMs) are being increasingly deployed in real-world
applications, but they remain susceptible to hallucinations, which produce fluent yet incorrect responses and lead to erroneous decision-making. Uncertainty
estimation is a feasible approach to detect such hallucinations. For example, semantic entropy estimates uncertainty by considering the semantic diversity across
multiple sampled responses, thus identifying hallucinations. However, semantic entropy relies on post-softmax probabilities and fails to capture the model’s
inherent uncertainty, causing it to be ineffective in certain scenarios. To address this issue, we introduce Semantic Energy, a novel uncertainty estimation
framework that leverages the inherent confidence of LLMs by operating directly on logits of penultimate layer. By combining semantic clustering with a
Boltzmann-inspired energy distribution, our method better captures uncertainty
in cases where semantic entropy fails. Experiments across multiple benchmarks
show that Semantic Energy significantly improves hallucination detection and uncertainty estimation, offering more reliable signals for downstream applications
such as hallucination detection. The code and intermediate data are available at
[https://github.com/SemanticEnergy.](https://github.com/MaHuanAAA/SemanticEnergy/tree/main)


1 INTRODUCTION


Large Language Models (LLMs) have been widely deployed in various aspects of production and
daily life, demonstrating strong capabilities in different fields (Schlegel et al., 2025; Xiang et al.,
2025). However, LLMs are still prone to being influenced by hallucinations and are prone to generate incorrect answers in situations where they lack knowledge, thus misleading users into making
errors (Zhou et al., 2024; Farquhar et al., 2024). Recently, uncertainty estimation has been shown
to be a reliable indicator for detecting hallucinations, reflecting the tendency of an LLM to generate
hallucinations (Xiao & Wang, 2021; Huang et al., 2024). When the uncertainty of an LLM response
is high, it often suggests a greater likelihood that the response is a hallucination, prompting further actions such as self-reflection (Renze & Guven, 2024; Kirchhof et al., 2025), regenerating of
answers (Xu et al., 2025), or intervention by human experts (Liu et al., 2025; Hopkins et al., 2025).


Entropy is a commonly used metric for estimating uncertainty in LLM (Cheng et al., 2025; Duan
et al., 2024). Similarly to traditional discriminative models, high entropy indicates high uncertainty
because it means that the model cannot confidently select a particular outcome. However, due to the
nature of natural language, the entropy of a single response cannot accurately reflect the reliability
of LLMs. Specifically, even though LLMs may not confidently generate the next token, the semantic
meaning of any generated token can still be the same. In such cases, we cannot identify an unreliable
response simply attributing to its low probability of being generated. To accurately describe the
uncertainty of responses composed of natural language, semantics must be considered.


_∗_ This work was completed during an internship at Baidu.

_†_ Project leader.

_‡_ Corresponding to zhangchangqing@tju.edu.cn.


1


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


Semantic entropy (Farquhar et al., 2024) is a typical method to characterize the semantic uncertainty of responses, effectively representing the probability that an LLM generates hallucinations.
Given a question, semantic entropy involves sampling multiple responses, clustering them based
on their semantic meaning, and then replacing individual responses with clusters to calculate entropy, thus achieving semantic-aware uncertainty characterization. Based on this method, a wide
range of downstream applications have been developed, such as guiding Chain-of-Thought (CoT)
reasoning (Ye et al., 2025) and parallel thinking (Xu et al., 2025). However, semantic entropy has
significant drawbacks stemming from entropy itself: it fails to capture the model’s inherent uncertainty, leading to its ineffectiveness in some scenarios.


A representative case occurs when the model produces identical responses in multiple sampling
instances for a given question, as illustrated in Fig. 1. According to semantic entropy, the resulting
value is 0, which is considered a reliable response. However, even answering incorrectly, LLMs
might also provide responses with the same semantics. Among samples with consistently semantic
responses across multiple responses, the proportion of incorrect responses (like Question3 in
Fig. 1) approaches 50% in some datasets. In such cases, it is necessary to leverage the model’s
inherent uncertainty for differentiation: even if the LLM provides multiple responses with the same
semantics for two different questions, their corresponding reliability still differs. In scenarios with a
higher inherent uncertainty in the model, the likelihood of the LLM making mistakes is greater.


Several previous studies have shown that logits exhibit stronger inherent capabilities to characterize
uncertainty compared to probabilities, and the magnitude of logits can indicate whether the model
has undergone adequate training in a given scenario (Liu et al., 2020; Fu et al., 2025; Zhang et al.,
2024). For example, in out-of-distribution (OOD) detection, studies have highlighted that the logit
values for in-distribution (InD) samples are significantly higher than those for OOD samples (Liu
et al., 2020). Recent work named LogToKU (Ma et al., 2025) points out that probabilities lose the intensity information of logits during normalization, thus limiting their ability to represent the inherent
uncertainty of LLM. From this insight, we propose a new method to improve the failure cases of Semantic Entropy, termed _Semantic Energy_ . Specifically, for a given prompt, we first perform multiple
response samplings, followed by semantic sampling. When calculating the final uncertainty, rather
than relying on probability as in Semantic Entropy, we estimate the response uncertainty based on
logits, enabling the estimated uncertainty to reflect the model’s inherent uncertainty. Our proposed
metric significantly outperforms Semantic Entropy in evaluating the reliability of LLM responses,
particularly in scenarios where Semantic Entropy fails. The main contributions are as follows:


    - We expose the limitations of current uncertainty estimation methods based on probability
and identify the failure cases in Semantic Entropy.

   - We introduce **Semantic Energy**, a novel framework to evaluate the uncertainty of LLM
responses, which indicates potential errors in the responses.

   - We instantiate Semantic Energy using the Boltzmann formulation, and in the hallucination detection task, it achieves an average performance improvement of more than 13%
compared in terms of AUROC to Semantic Entropy in cases where the latter is confident.


2 PRELIMINARIES


2.1 ESTIMATING LLM UNCERTAINTY WITH TOKEN-LEVEL ENTROPY


Let _**q**_ denote a natural language query provided as input to the LLM. Given the prompt _**q**_, a single
response sequence is generated in an auto-regressive manner. This response can be represented as:


_**x**_ = [ _x_ 1 _, x_ 2 _, . . ., xT_ ] _,_ (1)


where _**x**_ denotes a complete token sequence of variable length _T_ . At each decoding step _t_, the model
computes a probability distribution over its entire vocabulary _V_, assigning a conditional probability
_p_ ( _xt | x<t,_ _**q**_ ) to each candidate token given the preceding context and the original query. To quantify the local uncertainty inherent in the model’s predictions at each generation step, we compute the
token-level entropy for position _t_, defined formally as:


_Ht_ = _−_           - _p_ ( _x | x<t,_ _**q**_ ) log _p_ ( _x | x<t,_ _**q**_ ) _,_ (2)

_x∈V_


2


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


**Response Sampling & Semantic Cluster**


**High uncertainty cases in Semantic Entropy**


**Question1: What was the last US state to reintroduce alcohol**
**after prohibition?**



**Performance**












|0.3<br>0.2|0.5|
|---|---|
|0.2<br>|0.4|
|0.2<br>0.1|0.1|



**Ground truth: Utah**


**Low uncertainty cases in Semantic Entropy**


**Question2 : How many seconds are there in an hour?**





1.0


|0.2<br>0.2<br>0.2<br>0.2<br>0.2|Col2|
|---|---|
|0.2<br>0.2<br>0.2<br>0.2<br>0.2||



**Ground truth: 3600**










|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|~~**Question 3**~~|~~**Q**~~|~~**uestion**~~|~~** 2**~~|





for a large proportion.



















Figure 1: An intuitive comparison between Semantic Entropy and Semantic Energy in their ability to
characterize uncertainty. Both approaches first sample and perform semantic clustering over distinct
clusters of activity. The difference lies in the computation: Semantic Entropy is calculated based
on normalized probabilities, while Semantic Energy is derived from logits. This enables Semantic
Energy to distinguish cases when Semantic Entropy fails.


where _V_ is the model vocabulary. A higher value of _Ht_ indicates a more uniform probability distribution and thus greater uncertainty in the model’s choice for the _t_ -th token.


To estimate the overall uncertainty associated with the entire generated response _**x**_, a common and
straightforward strategy is to aggregate these local token-level entropy values. The most prevalent
aggregation method is to compute the arithmetic mean across all tokens in the sequence:



_H_ avg( _**x**_ ) = [1]

_T_



_T_

- _Ht,_ (3)


_t_ =1



where _H_ avg( _**x**_ ) serves as a proxy for the total uncertainty of the response, with a higher average
entropy suggesting a more uncertain generation process. However, this approach implicitly assumes
that each token contributes equally to the overall uncertainty, which may not hold in practice. Recognizing that different tokens can carry varying levels of importance for the meaning and correctness
of the final response, some recent studies (Duan et al., 2024) have proposed a refinement by employing a weighted average. This method aims to amplify the contribution of critical or pivotal tokens
(e.g., those conveying key facts or decisive information) to the final uncertainty score:



_T_

- _wt_ = 1 _,_ (4)


_t_ =1



_H_ wavg( _**x**_ ) =



_T_

- _wtHt,_ where


_t_ =1



where weights _wt_ can be determined based on heuristic rules or learned mechanisms designed to
identify semantically important tokens.


Although token-level entropy offers a fine-grained, local perspective on the uncertainty during the
auto-regressive generation process, it possesses an inherent limitation: it operates purely on a syntactic or surface level. Quantifies the model’s hesitation in choosing the next token, but does not


3


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


necessarily reflect uncertainty over the underlying meaning or intent of the full response. This is
because vastly different token sequences can express the same semantic content, while highly similar token-level probability distributions might lead to responses with divergent meanings. Consequently, token-level metrics may not fully capture the diversity in the semantic content of different
possible responses. This critical shortcoming motivates the need for a more holistic, higher-level
notion of uncertainty that operates on the distribution of semantically distinct outputs, leading to the
concept of _semantic entropy_ (Kuhn et al., 2024).


2.2 SEMANTIC ENTROPY AND RESPONSE CLUSTERING


To capture semantic-level uncertainty, semantic entropy samples a set of _n_ candidate responses to
the query _**q**_ from LLM:


X = _{_ _**x**_ [(1)] _,_ _**x**_ [(2)] _, . . .,_ _**x**_ [(] _[n]_ [)] _},_ _**x**_ [(] _[i]_ [)] = [ _x_ [(] 1 _[i]_ [)] _[, x]_ 2 [(] _[i]_ [)] _[, . . ., x]_ _T_ [(] _[i]_ _i_ [)][]] _[,]_ (5)


where _**x**_ [(] _[i]_ [)] indicates the _i_ -th sampled response and _Ti_ indicates the number of tokens in this response. Each response _**x**_ [(] _[i]_ [)] has an associated likelihood in the model:



_p_ ( _**x**_ [(] _[i]_ [)] _|_ _**q**_ ) =



_t_ - _T_ =1 _i_ _p_ ( _x_ [(] _t_ _[i]_ [)] _| x_ [(] _<t_ _[i]_ [)] _[,]_ _**[ q]**_ [)] _[,]_ _p_ ¯( _**x**_ [(] _[i]_ [)] ) = ~~�~~ ~~_n_~~ _jp_ =1( _**x**_ _[p]_ [(][(] _[i]_ [)] _**[x]**_ _|_ [(] _**q**_ _[j]_ [)] ) _[|]_ _**[ q]**_ [)] _[.]_ (6)



where ¯ _p_ ( _**x**_ [(] _[i]_ [)] ) indicates a normalized distribution over the sampled responses. Due to surface-level
variability in language, semantically similar responses may have different forms. Therefore, semantic entropy clusters the responses into _K_ semantically coherent groups:


C = _{_ C1 _,_ C2 _, . . .,_ C _K},_ C _k ⊆_ X _,_ (7)


where each cluster C _k_ contains responses that are semantically equivalent. The probability mass
assigned to each cluster is defined as the sum over its members:


       _p_ (C _k_ ) = _p_ ¯( _**x**_ [(] _[i]_ [)] ) _._ (8)


_**x**_ [(] _[i]_ [)] _∈_ C _k_


Finally, _semantic entropy_ ( _H_ SE) is computed by applying the standard Shannon entropy formula to
this distribution over semantic clusters:



_H_ SE = _−_



_K_


_p_ (C _k_ ) log _p_ (C _k_ ) _,_ (9)

_k_ =1



which quantifies the model’s uncertainty over distinct meanings conveyed by its responses. However, semantic entropy fails in some scenarios due to the limitations of entropy-based uncertainty
estimation.


3 MODELING UNCERTAINTY VIA SEMANTIC ENERGY


3.1 LIMITATIONS OF ENTROPY-BASED UNCERTAINTY ESTIMATION


While _H_ SE captures semantic variability, it only reflects _aleatoric uncertainty_ —uncertainty arising
from intrinsic randomness in the generation process. However, it fails to capture _epistemic uncer-_
_tainty_ —uncertainty stemming from the model’s lack of knowledge. For example, as illustrated by
the pair of instances in _Low uncertainty cases in Semantic Entropy_ (see Fig. 1):


(1) Consider two queries _**q**_ 2 and _**q**_ 3, where the model has been extensively trained on data
related to _**q**_ 2 (thus confident), but has limited exposure to _**q**_ 3 (thus uncertain).

(2) Assume that each query is sampled to obtain 5 responses, which were subsequently grouped
into a certain cluster based on their semantic similarity ( _K_ = 1), respectively.


(3) In this case, _H_ SE = 0 for both, despite the LLM outputs 5 incorrect answers on _**q**_ 3 with the
same semantic.


4


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


This means that if an LLM gives a wrong answer, and the semantics of multiple sampled results
are all aligned with that wrong answer, then semantic entropy will mistakenly identify it as reliable.
Unfortunately, LLMs are very good at “steadfastly” repeating the same wrong response. In some
datasets, nearly half of the samples with repeated identical semantics are actually incorrect answers,
making this limitation impossible to ignore.


The reason behind this is that the entropy calculated based on probabilities only reflects the relative likelihood of a particular LLM response compared to other possible responses generated by the
model, rather than the actual probability of that response to the question in the real world. However,
when computing the probability of the next token, the model approximates the partition function
as the sum of probabilities over its vocabulary. Therefore, for entropy to accurately represent uncertainty, two assumptions must hold: (1) the model has seen all possible responses (that is, the
training distribution matches the real-world distribution perfectly), and (2) the model has fit the
training distribution without bias (that is, the model output distribution matches the training distribution exactly). Clearly, neither of these assumptions holds. Worse still, the LLM response is
based on the joint probability prediction of many tokens, rather than the single word classification
of traditional discriminative models, which causes the error of the approximated partition function
to accumulate to a degree that can no longer be ignored.


3.2 ENERGY-BASED CONFIDENCE ESTIMATION


To address this limitation, we introduce an energy-based formulation that complements semantic entropy and captures epistemic uncertainty. In thermodynamics of physics, lower energy corresponds
to a more stable and less random state. Drawing on thermodynamic analogies, we treat lower-energy
states as higher-confidence predictions, following the intuition that physical systems evolve toward
minimal-energy configurations.


3.2.1 BOLTZMANN DISTRIBUTION


The classical Boltzmann distribution defines the probability that a system occupying a state is capable of generating _x_ [(] _t_ _[i]_ [)] as:

_t /kτ_
_p_ ( _x_ [(] _t_ _[i]_ [)][) =] _[e][−][E]_ [(] _[i]_ [)] _,_ (10)
_Zt_

where _k_ is the Boltzmann constant, _T_ is the temperature, and _Et_ [(] _[i]_ [)] is the token energy _x_ [(] _t_ _[i]_ [)][, and] _[ Z][t]_ [is]
the partition function. Specifically, for LLMs, _Zt_ = [�] _x∈_ V _[e][−][E][t]_ [(] _[x]_ [)][ is the normalization value across]

the entire vocabulary V (the difference between _V_ and V is that _V_ is the vocabulary in the predefined
tokenizer of a specific LLM, while V is the space of possible next tokens in the real world, which is
infinite and intractable). For simplicity, we assume that _Zt_ is constant across _t_ . The probability of a
complete sequence and the average sequence-level energy can be represented as:



_p_ ( _**x**_ [(] _[i]_ [)] ) =



_Ti_

_t_ =1 _[E]_ _t_ [(] _[i]_ [)]

- _p_ ( _x_ [(] _t_ _[i]_ [)][) =] _[e][−]_ ~~�~~ [�] _T_ _[Ti]_ _i_ _,_ _E_ ( _**x**_ [(] _[i]_ [)] ) = _T_ [1] _i_

_t_ =1 _t_ =1 _[Z][t]_



_Ti_




_Ti_



_Ti_

- _Et_ [(] _[i]_ [)] _[.]_ (11)

_t_ =1



Suppose that we want to evaluate the total energy of a set C. According to the Boltzmann equation,
the total energy of C is the sum of its different states:


       _E_ Bolt(C) = _E_ ( _**x**_ [(] _[i]_ [)] ) _._ (12)


_**x**_ [(] _[i]_ [)] _∈_ C


Lower energy indicates that the cluster containing this response is more stable, i.e., it has lower
uncertainty and thus higher reliability.


3.2.2 SPECIFIC IMPLEMENTATION IN LLMS


For a model parameterized by _**θ**_, we approximate _E_ ( _x_ [(] _t_ _[i]_ [)][)][ as the negative value of the logit, that is:]

_E_ ˜( _x_ [(] _t_ _[i]_ [)][) =] _[ −][z]_ _**[θ]**_ [(] _[x]_ _t_ [(] _[i]_ [)][)] _[,]_ (13)


5


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


where _z_ _**θ**_ ( _x_ [(] _t_ _[i]_ [)][)][ is the logits value of] _[ x]_ _t_ [(] _[i]_ [)][, and][ ˜] _[E]_ [(] _[x]_ _t_ [(] _[i]_ [)][)][ is the approximation of] _[ E]_ [(] _[x]_ _t_ [(] _[i]_ [)][)][. The final]
uncertainty is defined as:



1
_U_ ( _**x**_ [(] _[i]_ [)] ) =
_nTi_






_**x**_ [(] _[i]_ [)] _∈_ C _k_



_Ti_

- _−z_ _**θ**_ ( _x_ [(] _t_ _[i]_ [)][)] _[,]_ (14)

_t_ =1



which means that the lower the energy, the lower the uncertainty. By enhancing semantic entropy
with LogTokU, we can better characterize the intrinsic uncertainty of LLMs when answering questions, significantly improving the performance of semantic entropy.


4 EXPERIMENTS


4.1 SETUP


**Model & Baseline.** We conduct experiments using the Qwen3-8B model (Yang et al., 2025), and
the ERNIE-21B-A3B model (MOE architecture) (Baidu, 2025). Our primary goal is to highlight the
differences between probability-based methods and energy-based approaches. Therefore, we use
the semantic entropy (Farquhar et al., 2024) as a baseline.


**Datasets & Metrics.** Experiments are performed on standard open-domain QA datasets in both
Chinese and English: the Chinese dataset _CSQA_ (He et al., 2024) and the English dataset _Triv-_
_iaQA_ (Joshi et al., 2017). To assess whether the estimated uncertainty can capture the risk that
the model makes errors, we estimate the AUROC between uncertainty scores and correctness (i.e.,
whether the answer is correct).


4.2 MAIN RESULTS


Table 1: Uncertainty estimation performance on OpenQA Datasets.


**Semantic Entropy** **Semantic Energy**
**Model** **Dataset**

AUROC AUPR FPR95 AUROC( _↑_ ) AUPR( _↑_ ) FPR95( _↓_ )


Table 1 summarizes the performance of uncertainty estimation methods on the CSQA and TriviaQA
datasets. We evaluate models using standard metrics: AUROC, AUPR, and FPR@95, based on
whether the uncertainty score can discriminate correct from incorrect responses.


In both models and datasets, _semantic energy_ consistently outperforms _semantic entropy_ . On CSQA,
the Boltzmann energy improves AUROC from 71 _._ 6% to 76 _._ 1% on Qwen3-8B and from 77 _._ 4% to
80 _._ 2% on ERNIE-21B-A3B. Similar trends are observed on TriviaQA, where Boltzmann energy
yields AUROC gains of more than 5% compared to the semantic entropy. Improvements are also
reflected in AUPR and FPR@95, indicating better calibration and reduced false positive rates.


These results highlight the robustness of energy-based uncertainty estimation, particularly in lowdiversity scenarios where entropy becomes degenerate (details in Table 2). By incorporating internal
model states via logits, semantic energy captures a richer signal for uncertainty estimation beyond
probability-based entropy.


4.3 ABLATION STUDIES


4.3.1 RESULTS ON QUESTIONS WITH SINGLE CLUSTER


In Table 2, we present the case where all responses share the same semantics, that is, all responses are
clustered into a single group as described in Sec. 3.1. In this scenario, semantic entropy completely
fails, whereas semantic energy is still able to provide a certain level of distinction, resulting in


6


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


Table 2: Uncertainty estimation performance on questions with **single** cluster.


**Semantic Entropy** **Semantic Energy**
**Model** **Dataset**

AUROC AUPR FPR95 AUROC( _↑_ ) AUPR( _↑_ ) FPR95( _↓_ )


semantic energy achieving an average performance improvement of more than 13% compared in
terms of AUROC to semantic entropy in cases where the latter is confident. It is important to note
that the value of semantic entropy in such cases is always zero, meaning that its performance reflects
the expected performance when the uncertainty indicator is meaningless, for example, the AUPR
corresponds to the number of positive samples (i.e. correct responses).


4.3.2 ADVANTAGES OF SEMANTIC CLUSTER


Inspired by semantic entropy, we incorporate semantic influence when computing energy. If semantics are not considered and the energy of a single response is directly used to characterize the
reliability of an LLM’s reply, such as in LogTokU (Ma et al., 2025), a clear problem arises: a single
response having high energy does not necessarily mean that the entire cluster of responses sharing
the same semantics also has high energy. This is because different responses belonging to the same
semantic cluster can still have varying energy values. Therefore, we represent the energy of a response by the energy of the cluster to which it belongs. As shown in Fig. 2, we conduct an ablation
study on whether to include semantics. The experimental results demonstrate that incorporating
semantics significantly improves the accuracy of uncertainty estimation.






|Correct<br>Incorrect|AUROC =|
|---|---|
|||


|ct|AUROC = 76.1|
|---|---|
|||



(d) CSimpleQA (with semantic)





(c) TriviaQA (with semantic)











(a) TriviaQA (w/o semantic)



(b) CSimpleQA (w/o semantic)













Figure 2: Comparison of semantic vs. non-semantic uncertainty modeling on TriviaQA and CSimpleQA datasets.


7


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


4.3.3 RESULTS ON THINK MODE


To validate the performance of the proposed method on the thinking model, we conduct experiments
on Qwen-8B using the CSQA dataset and explore the case where the think mode is enabled. Specifically, we activate the think mode but discard the content within <think>...<think> during
evaluation, considering only the response portion. The final results, shown in Fig. 3, are consistent
with the observations in Table 1. This indicates that even when the LLM undergoes a lengthy thinking process during output generation, its final results can still accurately capture the uncertainty of
the model’s responses through logits, thereby reflecting the reliability of the answers. Additionally,
we observe that both semantic entropy and semantic energy demonstrate significantly improved uncertainty characterization capabilities in the think mode compared to the performance reported in
Table 1. This suggests that the context during the deep thinking process may positively contribute
to characterizing the distributional uncertainty of the final responses.


(a) Semantic Entropy (b) Semantic Energy


Figure 3: Comparison of semantic entropy vs. semantic energy on CSQA datasets with think mode
on.


5 RELATED WORK


**Uncertainty estimation methods.** Recently, numerous uncertainty estimation methods for LLMs
have been proposed. These include methods that utilize natural language for uncertainty feedback,
including heuristically designed and trained approaches (Tao et al., 2025; Xiong et al., 2023; Lin
et al., 2023); methods that estimate uncertainty based on model states, including those leveraging
prior knowledge or statistical observations of model states (Kostenok et al., 2023; Li et al., 2025;
Liu et al., 2024), or observing changes under perturbations (Zhang et al., 2025b; Gao et al., 2024);
and methods that take into account the semantics of the response, including consistency-based uncertainty characterizations (Lyu et al., 2025; Bartsch et al., 2023; Xiao et al., 2025) and approaches
that integrate semantics with model states (Kuhn et al., 2024; Grewal et al., 2024).


**Uncertainty-guided applications.** The utilization of uncertainty estimation is widely applied in
both the post-train and inference phases of LLMs. For example, minimizing entropy during the
reinforcement learning process helps reduce uncertainty (Zhang et al., 2025a; Agarwal et al., 2025)
and encourages exploration of critical positions with higher uncertainty (Zheng et al., 2025; Cheng
et al., 2025). In the inference phase, uncertainty can guide the model’s reasoning path (Ye et al.,
2025; Wang et al., 2022), guiding RAG (Guo et al., 2025; Chen & Varoquaux, 2025), collaboration (Dey et al., 2025; Kruse et al., 2025; Cui et al., 2025), and indicate when the model should stop
or skip thinking (Xu et al., 2025; Zhu et al., 2025).


6 DISCUSSION AND CONCLUSION


In this paper, we introduce the concept of semantic energy as an enhancement to semantic entropy,
an uncertainty modeling method that substitutes entropy with energy (derived from logits). Semantic energy can effectively compensate for the shortcomings of semantic entropy and better depict


8


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


the inherent uncertainty within models. We elucidate the limitations imposed by probability normalization and demonstrate the potential to overcome these constraints. However, it is important to
note that semantic energy is not a flawless, ultimate solution. Given that the cross-entropy loss employed in current LLM training is invariant to the scale of the logits, logits are not strictly equivalent
to energy. Instead, logits exhibit energy-like characteristics only due to implicit constraints arising
from network initialization and regularization during training. Therefore, to enable models to more
precisely capture their own uncertainty and acknowledge their unknowns, it may be necessary to
tackle the limitations introduced by cross-entropy loss during the training process.


REFERENCES


Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable effectiveness of entropy minimization in llm reasoning. _arXiv preprint arXiv:2505.15134_, 2025.


[Baidu. Ernie 4.5 technical report. Technical report, Baidu, 2025. URL https://yiyan.baidu.](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)
[com/blog/publication/ERNIE_Technical_Report.pdf. Accessed: 2025-07-30.](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)


Henning Bartsch, Ole Jorgensen, Domenic Rosati, Jason Hoelscher-Obermaier, and Jacob Pfau.
Self-consistency of large language models under ambiguity. _arXiv preprint arXiv:2310.13439_,
2023.


Lihu Chen and Ga¨el Varoquaux. Query-level uncertainty in large language models. _arXiv preprint_
_arXiv:2506.09669_, 2025.


Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and
Furu Wei. Reasoning with exploration: An entropy perspective. _arXiv preprint arXiv:2506.14758_,
2025.


Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen
Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for
reasoning language models. _arXiv preprint arXiv:2505.22617_, 2025.


Prasenjit Dey, Srujana Merugu, and Sivaramakrishnan Kaveri. Uncertainty-aware fusion: An ensemble framework for mitigating hallucinations in large language models. In _Companion Proceedings_
_of the ACM on Web Conference 2025_, pp. 947–951, 2025.


Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura,
and Kaidi Xu. Shifting attention to relevance: Towards the predictive uncertainty quantification of
free-form large language models. In _Proceedings of the 62nd Annual Meeting of the Association_
_for Computational Linguistics (Volume 1: Long Papers)_, pp. 5050–5063, 2024.


Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large
language models using semantic entropy. _Nature_, 630(8017):625–630, 2024.


Tianyu Fu, Yi Ge, Yichen You, Enshu Liu, Zhihang Yuan, Guohao Dai, Shengen Yan, Huazhong
Yang, and Yu Wang. R2r: Efficiently navigating divergent reasoning paths with small-large model
token routing. _arXiv preprint arXiv:2505.21600_, 2025.


Xiang Gao, Jiaxin Zhang, Lalla Mouatadid, and Kamalika Das. Spuq: Perturbation-based uncertainty quantification for large language models. _arXiv preprint arXiv:2403.02509_, 2024.


Yashvir S Grewal, Edwin V Bonilla, and Thang D Bui. Improving uncertainty quantification in large
language models via semantic embeddings. _arXiv preprint arXiv:2410.22685_, 2024.


Kai Guo, Harry Shomer, Shenglai Zeng, Haoyu Han, Yu Wang, and Jiliang Tang. Empowering
graphrag with knowledge filtering and integration. _arXiv preprint arXiv:2503.13804_, 2025.


Yancheng He, Shilong Li, Jiaheng Liu, Yingshui Tan, Weixun Wang, Hui Huang, Xingyuan Bu,
Hangyu Guo, Chengwei Hu, Boren Zheng, et al. Chinese simpleqa: A chinese factuality evaluation for large language models. _arXiv preprint arXiv:2411.07140_, 2024.


Aspen Hopkins, Angie Boggust, and Harini Suresh. Chatbot evaluation is (sometimes) ill-posed:
Contextualization errors in the human-interface-model pipeline. 2025.


9


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


Hsiu-Yuan Huang, Yutong Yang, Zhaoxi Zhang, Sanwoo Lee, and Yunfang Wu. A survey of uncertainty estimation in llms: Theory meets practice. _arXiv preprint arXiv:2410.15326_, 2024.


Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehension. _arXiv preprint arXiv:1705.03551_, 2017.


Michael Kirchhof, Luca F¨uger, Adam Goli´nski, Eeshan Gunesh Dhekane, Arno Blaas, and Sinead
Williamson. Self-reflective uncertainties: Do llms know their internal answer distribution? _arXiv_
_preprint arXiv:2505.20295_, 2025.


Elizaveta Kostenok, Daniil Cherniavskii, and Alexey Zaytsev. Uncertainty estimation of transformers’ predictions via topological analysis of the attention matrices. _arXiv preprint_
_arXiv:2308.11295_, 2023.


Maya Kruse, Majid Afshar, Saksham Khatwani, Anoop Mayampurath, Guanhua Chen, and Yanjun
Gao. An information-theoretic perspective on multi-llm uncertainty estimation. _medRxiv_, pp.
2025–07, 2025.


Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for
uncertainty estimation in natural language generation. _Nature_, 2024.


Yinghao Li, Rushi Qiang, Lama Moukheiber, and Chao Zhang. Language model uncertainty quantification with attention chain. _arXiv preprint arXiv:2503.19168_, 2025.


Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. Generating with confidence: Uncertainty quantification for black-box large language models. _arXiv preprint arXiv:2305.19187_, 2023.


Jingyu Liu, Jingquan Peng, Xubin Li, Tiezheng Ge, Bo Zheng, Yong Liu, et al. Do not abstain!
identify and solve the uncertainty. _arXiv preprint arXiv:2506.00780_, 2025.


Linyu Liu, Yu Pan, Xiaocheng Li, and Guanting Chen. Uncertainty estimation and quantification
for llms: A simple supervised approach, 2024. _URL https://arxiv. org/abs/2404.15993_, 2024.


Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. _Advances in neural information processing systems_, 33:21464–21475, 2020.


Qing Lyu, Kumar Shridhar, Chaitanya Malaviya, Li Zhang, Yanai Elazar, Niket Tandon, Marianna
Apidianaki, Mrinmaya Sachan, and Chris Callison-Burch. Calibrating large language models with
sample consistency. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 39,
pp. 19260–19268, 2025.


Huan Ma, Jingdong Chen, Joey Tianyi Zhou, Guangyu Wang, and Changqing Zhang. Estimating
llm uncertainty with evidence. _arXiv preprint arXiv:2502.00290_, 2025.


Matthew Renze and Erhan Guven. Self-reflection in llm agents: Effects on problem-solving performance. _arXiv preprint arXiv:2405.06682_, 2024.


Katja Schlegel, Nils R Sommer, and Marcello Mortillaro. Large language models are proficient in
solving and creating emotional intelligence tests. _Communications Psychology_, 3(1):80, 2025.


Linwei Tao, Yi-Fan Yeh, Minjing Dong, Tao Huang, Philip Torr, and Chang Xu. Revisiting uncertainty estimation and calibration of large language models. _arXiv preprint arXiv:2505.23854_,
2025.


Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models.
_arXiv preprint arXiv:2203.11171_, 2022.


Jinxi Xiang, Xiyue Wang, Xiaoming Zhang, Yinghua Xi, Feyisope Eweje, Yijiang Chen, Yuchen
Li, Colin Bergstrom, Matthew Gopaulchan, Ted Kim, et al. A vision–language foundation model
for precision oncology. _Nature_, 638(8051):769–778, 2025.


Quan Xiao, Debarun Bhattacharjya, Balaji Ganesan, Radu Marinescu, Katsiaryna Mirylenka,
Nhan H Pham, Michael Glass, and Junkyu Lee. The consistency hypothesis in uncertainty quantification for large language models. _arXiv preprint arXiv:2506.21849_, 2025.


10


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


Yijun Xiao and William Yang Wang. On hallucination and predictive uncertainty in conditional
[language generation, 2021. URL https://arxiv.org/abs/2103.15025.](https://arxiv.org/abs/2103.15025)


Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms
express their uncertainty? an empirical evaluation of confidence elicitation in llms. _arXiv preprint_
_arXiv:2306.13063_, 2023.


Zenan Xu, Zexuan Qiu, Guanhua Huang, Kun Li, Siheng Li, Chenchen Zhang, Kejiao Li, Qi Yi,
Yuhao Jiang, Bo Zhou, et al. Adaptive termination for multi-round parallel reasoning: An universal semantic entropy-guided framework. _arXiv preprint arXiv:2507.06829_, 2025.


An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv preprint_
_arXiv:2505.09388_, 2025.


Zihuiwen Ye, Luckeciano Carvalho Melo, Younesse Kaddar, Phil Blunsom, Sam Staton, and Yarin
Gal. Uncertainty-aware step-wise verification with generative reward models. _arXiv preprint_
_arXiv:2502.11250_, 2025.


Jianyi Zhang, Da-Cheng Juan, Cyrus Rashtchian, Chun-Sung Ferng, Heinrich Jiang, and Yiran
Chen. Sled: Self logits evolution decoding for improving factuality in large language models.
_Advances in Neural Information Processing Systems_, 37:5188–5209, 2024.


Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question
is already half the answer: Fully unsupervised llm reasoning incentivization. _arXiv preprint_
_arXiv:2504.05812_, 2025a.


Tunyu Zhang, Haizhou Shi, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen,
Ligong Han, Kai Xu, Huan Zhang, et al. Token-level uncertainty estimation for large language
model reasoning. _arXiv preprint arXiv:2505.11737_, 2025b.


Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, et al. First return, entropy-eliciting explore. _arXiv_
_preprint arXiv:2507.07017_, 2025.


Lexin Zhou, Wout Schellaert, Fernando Mart´ınez-Plumed, Yael Moros-Daval, C`esar Ferri, and Jos´e
Hern´andez-Orallo. Larger and more instructable language models become less reliable. _Nature_,
634(8032):61–68, 2024.


Yuqi Zhu, Ge Li, Xue Jiang, Jia Li, Hong Mei, Zhi Jin, and Yihong Dong. Uncertainty-guided
chain-of-thought for code generation with llms. _arXiv preprint arXiv:2503.15341_, 2025.


11


Semantic Energy: Detecting LLM Hallucination Beyond Entropy


A DETAILS FOR COMPARISON METHODS


In this paper, we primarily compare our method with the well-known approach, semantic entropy.
Since both methods require response sampling and semantic clustering, we use identical data for
these parts. That is, the only difference lies in the final uncertainty calculation process—the responses and clusters are generated from the same set of data. For the TriviaQA dataset, due to its
large number of entries, we only estimate results for the first 5000 samples. Note that this is a commonly adopted practice. Additionally, the sampling temperature is set to 0.6, as recommended in
the official documentation, and the random seed is set to values from 1 to 10 to sample ten distinct
responses.


It is important to note that the value of semantic entropy in table 2 is always zero, meaning that its
performance reflects the **expected** result when the uncertainty indicator is meaningless; for example,
the AUPR corresponds to the number of positive samples (i.e. correct responses).


B PROMPTS FOR DIFFERENT EXPERIMENTS







12


Semantic Energy: Detecting LLM Hallucination Beyond Entropy





13


