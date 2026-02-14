## **Trusting Your Evidence:** **Hallucinate Less with Context-aware Decoding**

**Weijia Shi** [1] [∗] **Xiaochuang Han** [1] [∗]

**Mike Lewis** [2] **Yulia Tsvetkov** [1] **Luke Zettlemoyer** [1] **Scott Yih** [2]


1 University of Washington, Seattle, WA, 2 Meta AI
{swj0419, xhan77}@cs.washington.edu


**Abstract**



Language models (LMs) often struggle to pay
enough attention to the input context, and generate texts that are unfaithful or contain hallucinations. To mitigate this issue, we present
context-aware decoding (CAD), which follows
a contrastive output distribution that amplifies
the difference between the output probabilities
when a model is used with and without context. Our experiments show that CAD, without
additional training, significantly improves the
faithfulness of different LM families, including
OPT, GPT, LLaMA and FLAN-T5 for summarization tasks (e.g., 14.3% gain for LLaMA in
factuality metrics). Furthermore, CAD is particularly effective in overriding a model’s prior
knowledge when it contradicts the provided
context, leading to substantial improvements in
tasks where resolving the knowledge conflict is
essential.


**1** **Introduction**


Language models (LMs) are remarkably effective
in generating coherent and fluent continuations of
a prompt or document prefix. During generation,
they mostly rely on two sources of knowledge: (1)
_prior knowledge_, which is learned during pretraining and stored implicitly within the model parameters; (2) _context knowledge_, which is passed as inputs in the prefix context (Chan et al., 2022). However, it remains an open question how a pretrained
LM, particularly a vanilla LM without task-specific
finetuning, balances these two knowledge sources
during generation.
Previous research shows that LMs can fail to pay
enough attention to new information introduced
in the context knowledge. This can lead to hallucination in summarization (Maynez et al., 2020;
Pagnoni et al., 2021), where the generated summaries include facts not present in the input document. Insufficient attention to context is especially problematic when the context knowledge


∗Equal contribution. Order randomly determined.



Figure 1: An illustration of context-aware decoding.


contradicts with the prior knowledge (Longpre
et al., 2021; Zhou et al., 2023). For instance, when
LLaMA (Touvron et al., 2023) is presented with a
latest document “Argentina won the FIFA World
Cups in 1978,1986 and 2022 ...” in its context (Figure 1), it still predicts “Two” in response to the
question “How many World Cups have Argentina
won?”, due in part to the outdated training data.

In this work, we present a simple **c** ontext- **a** ware
**d** ecoding (CAD) method to encourage the LM to
attend to its context during generation. As shown
in Figure 1, CAD samples from a new output distribution, which amplifies the difference between
output probabilities with and without the context
document. This provides a new form of contrastive
decoding (Li et al., 2022), which effectively downweights the prior knowledge when more relevant
contextual information is provided. CAD can be
used with off-the-shelf pretrained language models
without any additional training.

Experimental results from summarization tasks
show that context-aware decoding significantly
enhances the generation faithfulness of various
vanilla LMs including OPT (Zhang et al., 2022),
GPT-Neo (Black et al., 2021), LLaMA (Touvron
et al., 2023) and instruction-finetuned LMs such as
FLAN (Chung et al., 2022). For instance, when applied to LLaMA-30B in CNN-DM, CAD leads to
substantial improvement in both ROUGE-L (21%)




---PAGE BREAK---

and summary factuality evaluation metrics (14.3%).
More notably, CAD is especially beneficial for
knowledge conflicting tasks, where the context
contains information contradictory to the model’s
prior knowledge. CAD brings a 2.9x improvement to LLaMA-30B on a knowledge conflicts QA
dataset (Longpre et al., 2021). Furthermore, we
observe that this gain brought by CAD increases
as the model size grows in knowledge conflicts
tasks. These results demonstrate the potential of
CAD in mitigating hallucinations in text generation
and overriding prior knowledge with reliable and
trusted information.


**2** **Method**


**2.1** **Background**


Given a language model _θ_, an input query _**x**_, and
a context _**c**_ that contains some external knowledge
_unfamiliar_ or _in conflict_ to the model’s prior knowledge, we ask our model _θ_ to generate a response
_**y**_ given the the query and context. The response
can be directly sampled (autoregressively) from the
probability distribution conditioned on query _**x**_ and
context _**c**_ :
_yt_ ∼ _pθ_ ( _yt_ ∣ _**c**_ _,_ _**x**_ _,_ _**y**_ < _t_ )

∝ exp logit _θ_ ( _yt_ ∣ _**c**_ _,_ _**x**_ _,_ _**y**_ < _t_ )


However, in cases where the context _**c**_ contains
knowledge that is out-of-distribution with respect
to _θ_, we hypothesize that the model can struggle
to effectively attend to _**c**_ and overly rely on the
prior knowledge encoded in _θ_ . For instance, as
illustrated in Figure 1, when the context _**c**_ states
“Argentina won the FIFA World Cups in 1978, 1986
and 2022 ...”, it contradicts the LM’s outdated prior
knowledge that Argentina has won the World Cup
twice. The language model may still incorrectly
predict “Two” even when presented with the context _**c**_ and the query _**x**_ .


**2.2** **Context-aware Decoding**


To mitigate such issues, we factor out the prior
knowledge from the model’s original output distribution contrastively. Here, we model the prior
knowledge as _pθ_ ( _yt_ ∣ _**x**_ _,_ _**y**_ < _t_ ) and adjust the
model’s original output probability distribution using the pointwise mutual information (PMI) between the context _**c**_ and the generation _yt_, condi


tioned on _**x**_ _,_ _**y**_ < _t_ . Formally, we have:
_yt_ ∼ _p_ ˜ _θ_ ( _yt_ ∣ _**c**_ _,_ _**x**_ _,_ _**y**_ < _t_ )



∝ _pθ_ ( _yt_ ∣ _**c**_ _,_ _**x**_ _,_ _**y**_ < _t_ ) ( _[p][θ]_ [(] _[y][t]_ [∣] _**[c]**_ _[,]_ _**[ x]**_ _[,]_ _**[ y]**_ [<] _[t]_ [)]



_α_




_[θ]_ _[t]_ [<] _[t]_

_pθ_ ( _yt_ ∣ _**x**_ _,_ _**y**_ < _t_ ) [)]



where the output probability is a product-of-experts
of the original output probability and PMI weighted
by _α_ . Essentially, outputs that become much more
likely when the context is included are preferred
(Figure 1).


This expression is not a valid probability distribution and needs to be normalized across all possible
values of _yt_ . By rearranging the terms, we obtain
the final form:
_yt_ ∼ softmax[(1 + _α_ ) logit _θ_ ( _yt_ ∣ _**c**_ _,_ _**x**_ _,_ _**y**_ < _t_ )

    - _α_ logit _θ_ ( _yt_ ∣ _**x**_ _,_ _**y**_ < _t_ )]
Larger _α_ means more weight on our adjustment
( _α_ = 0 reduces to regular decoding). [1] We refer
to this simple method as context-aware decoding.
From the adjusted output distribution _p_ ˜, we can
apply various sampling strategies, such as nucleus
sampling (Holtzman et al., 2019).


Essentially, context-aware decoding is just a contrastive ensemble between the logits of _pθ_ ( _yt_ ∣
_**c**_ _,_ _**x**_ _,_ _**y**_ < _t_ ) and _pθ_ ( _yt_ ∣ _**x**_ _,_ _**y**_ < _t_ ). A similar contrastive objective is universal in image generation, where classifier-free diffusion models (Ho
and Salimans, 2022) predict diffusion noise with
(1+ _α_ ) _**ϵ**_ _θ_ ( _**x**_ _,_ _**c**_ )− _α_ _**ϵ**_ _θ_ ( _**x**_ ), with _**c**_ being a control to
the image. In text generation, Malkin et al. (2021)
propose coherence boosting with the same intuition, with a focus on contrasting the full input and
a short premise-free input, promoting coherence
w.r.t. the long context. Instead of using a single
model _θ_ in this work, different models can also
be used in the distribution adjustments to demote
unwanted model behaviors or distill expert model’s
capability (Liu et al., 2021; Li et al., 2022).


**3** **Experimental Setup**


We perform evaluation on tasks that require LMs to
read and reason over contexts and produce outputs
that are faithful to the contexts. Following prior
work (Zhang et al., 2023; Zhou et al., 2023), we
evaluate the models using prompting.


1If we identify an external knowledge _**c**_ conditionally independent to the generation, _pθ_ ( _yt_ ∣ _**c**_ _,_ _**x**_ _,_ _**y**_ < _t_ ) = _pθ_ ( _yt_ ∣
_**x**_ _,_ _**y**_ < _t_ ), even a non-zero _α_ would not have an impact to the
original output distribution.




---PAGE BREAK---

**3.1** **Datasets and Metrics**


**Summarization** We conduct summarization experiments on two news datasets: CNN-DM (See
et al., 2017) and XSUM (Narayan et al., 2018). We
use ROUGE-L (Lin, 2004) to evaluate summarization quality. To measure the factual consistency
of summaries, we adopt BERT-Precision (Pagnoni
et al., 2021) as well as FactKB (Feng et al., 2023),
which has been demonstrated to achieve high correlations with human judgment on the two summarization datasets.


**Knowledge Conflicts** We evaluate performance
on two knowledge conflict datasets: MemoTrap
(Liu and Liu, 2023) and NQ-Swap (Longpre et al.,
2021). MemoTrap is created to investigate whether
language models could fall into memorization traps.
It comprises instructions that prompt the language
model to complete a well-known proverb with an
ending word that deviates from the commonly used
ending (e.g., _Write a quote that ends in the word_
_“early”: Better late than_ ). NQ-Swap is based on
a QA dataset, natural questions (NQ) (Kwiatkowski
et al., 2019), where the objective is to answer questions based on a reliable gold document. To generate NQ-Swap, Longpre et al. (2021) first identify
questions in NQ with named entity answers, find
the supportive document for each question and then
replace the gold answer entity in the document with
a random entity. A faithful LM should generate the
replaced entity as the answer when given the question and modified document. We also include the
original NQ dataset with the question and original document for evaluation. We use Exact Match
(EM) as the evaluation metric for NQ-Swap, NQ
and MemoTrap.


In Table 1, we show illustrative examples of the
contexts we aim to upweight for the model and the
queries across different datasets. We hope LMs pay
more attention to the source document in XSUM
and NQ-Swap. On the other hand, we hope LMs
focus more on the instruction in MemoTrap.


**3.2** **Models and Baselines**


We apply CAD to pretrained language models including OPT (13B and 30B) (Zhang et al., 2022),
GPT-Neo (2.7B and 20B) (Black et al., 2021),
LLaMA (13B and 30B) (Touvron et al., 2023)
and instruction-finetuned language models such
as FLAN-T5 (XL 3B and XXL 11B) (Chung et al.,
2022).



**XSUM**


_**c**_ Article: Prison Link Cymru had 1,099 referrals in
2015-16 and said some ex-offenders were living
rough for up to a year before finding suitable accommodation ...
_**x**_ Summarize the article in one sentence. Summary:


**NQ-SWAP**


_**c**_ Tesla CEO Elon Musk is now in charge of Twitter,
CNBC has learned ...
_**x**_ Who is Twitter CEO now?


**MemoTrap**


_**c**_ Write a quote that ends in the word "early":
_**x**_ Better late than


Table 1: An illustation of the inputs to CAD applied to
each dataset. CAD upweights the context _**c**_ (in red) by
sampling each token from softmax[(1 + _α_ ) logit _θ_ ( _yt_ ∣
_**c**_ _,_ _**x**_ _,_ _**y**_ < _t_ ) − _α_ logit _θ_ ( _yt_ ∣ _**x**_ _,_ _**y**_ < _t_ )].


CAD introduces a hyperparameter _α_ to control
the adjustment level. We set _α_ = 0 _._ 5 for all models evaluated on the summarization datasets and
_α_ = 1 for all models evaluated on the knowledge
conflict datasets. We observed that _α_ = 0 _._ 5 generally yielded good results across all settings and
all datasets, but a slightly higher _α_ is more effective in the knowledge conflict setting, where the
prior knowledge needs to be factored out more. We
investigate the effect of _α_ in Section 4.2.

For the baselines, we use regular decoding
following prior work (Longpre et al., 2021;
Kwiatkowski et al., 2019) to use greedy decoding for knowledge conflict tasks and top- _p_ sampling with _p_ =0.9 for summarization tasks (Holtzman et al., 2019). For CAD, we use the same
sampling strategies on top of the adjusted output
probability distribution.


**4** **Results**


**4.1** **Main Results**


**Summarization** Table 2 reports the results on
CNN-DM and XSUM. We observe that CAD
outperforms the standard decoding algorithm by
a large margin in all eight models across both
datasets. Specifically, when applied to LLAMA30B in CNN-DM, CAD leads to 21% increase in
ROUGE-L, 14.3% increase in factKB and 7.8%
increase in BERT-P. This result demonstrates that
CAD could effectively improve the quality and factuality of the generated summaries from a diverse
set of language models.




---PAGE BREAK---

**CNN-DM** **XSUM**


**Model** **Decoding** ROUGE-L factKB BERT-P ROUGE-L factKB BERT-P


Regular 22.0 77.8 86.5 16.4 47.2 85.2
13B
CAD **27.4** **84.1** **90.8** **18.2** **64.9** **87.5**
OPT

Regular 22.2 81.7 87.0 17.4 38.2 86.1
30B
CAD **28.4** **87.0** **90.2** **19.5** **45.6** **89.3**


Regular 24.3 80.5 87.5 17.6 54.0 86.6
3B
CAD **27.7** **87.5** **90.6** **18.1** **65.1** **89.1**
GPT-Neo

Regular 18.7 68.3 85.2 14.9 42.2 85.7
20B
CAD **24.5** **77.5** **89.4** **19.0** **63.3** **90.6**


Regular 27.1 80.2 89.5 19.0 53.5 87.8
13B
CAD **32.6** **90.8** **93.0** **21.1** **73.4** **91.7**
LLaMA

Regular 25.8 76.8 88.5 18.7 47.7 87.1
30B
CAD **31.8** **87.8** **92.2** **22.0** **66.4** **90.3**


Regular 25.5 90.2 91.6 18.8 31.9 88.2
3B
CAD **26.1** **93.9** **92.1** **19.5** **35.9** **88.8**
FLAN

Regular 25.4 90.4 91.4 19.4 29.8 88.3
11B
CAD **27.1** **93.1** **92.2** **20.0** **35.0** **88.8**


Table 2: **CAD consistently outperform the regular decoding method in terms of both summary quality metric**
**(ROUGE-L) and summary factuality (factKB and BERT-P).** The best scores for each setting are boldfaced.
FLAN 3B and 11B refer to FLAN-T5 XL and FLAN-T5 XXL respectively.



**Model** **Decoding** **Memo.** **NQ** **NQ-SWAP**


Reg. 32.5 29.2 18.8
13B
CAD 44.5 32.2 36.9
OPT

Reg. 28.4 29.4 14.7
30B
CAD 41.0 35.5 29.0


Reg. 22.5 31.9 19.1
3B
CAD 47.3 39.9 41.2
GPT.

Reg. 37.1 22.8 16.1
20B
CAD 57.3 32.1 36.8


Reg. 23.8 22.3 11.7
13B
CAD 57.1 33.6 36.7
LLAMA

Reg. 25.8 23.8 9.6
30B
CAD 50.6 34.0 37.7


Reg. 69.2 81.8 71.4
3B
CAD 72.2 80.3 73.3
FLAN

Reg. 82.0 85.5 73.0
11B
CAD 88.7 82.5 77.1


Table 3: CAD outperforms the regular decoding method
(Reg.) in all settings except for FLAN-T5 on NQ.
Note that FLAN-T5 is trained on NQ dataset during
instruction-finetuning.


**Knowledge Conflicts** Our results for the knowledge conflict datasets, NQ-SWAP and MemoTrap,
as well as the original NQ are detailed in Table 3.
CAD is significantly better than the regular decoding in all settings, with the exception of a minor decrease observed for FLAN-T5 on the non-conflict
NQ dataset. [2] Despite this, CAD achieves substan

2This slight decline can be attributed to the fact that this
particular NQ dataset is included in the instruction-finetuning
sets used by FLAN-T5, and hence, the model has been previously trained on it.



tially better performance on the knowledge conflict datasets, e.g., CAD improve GPT-Neo 20B by
54.4% on Memotrap and by 128% on NQ-SWAP.
This substantial improvement suggests that contextaware decoding is particularly beneficial for LMs
to adhere to the given context, in scenarios where
the model’s prior knowledge contradicts with the
context knowledge.


**4.2** **Analysis**


**Qualitative analyais** We provide qualitative examples for XSUM and Memotrap in Table 4. In
XSUM, the regular decoding generates texts that
is not mentioned in the article, whereas CAD produces output exclusively based on the information
in the input article. For MemoTrap, the standard
decoding disregards the instruction and generates
the memorized ending, while CAD adheres to the
instruction within the given context and produces
the desired output.


**CAD** **brings** **consistent** **improvement** **to** **LMs**
**with different sizes.** In Tables 2 and 3, we show
that CAD could be used to enhance a diverse set of
LM families, including OPT, GPT-Neo, LLaMA,
and FLAN-T5. Here we further investigate whether
CAD is effective in improving language models of
different sizes. Specifically, we focus on OPT models across a range of sizes: 125M, 350M, 1.3B,
2.7B, 6.7B, 13B, 30B. As depicted in Figure 2,
we observe that the performance gain brought by




---PAGE BREAK---

Figure 2: OPT models of varying sizes consistently benefit from CAD. The x-axis indicates the size of language
models and the y-axis is the performance.



**XSUM**


Article He passed away peacefully in hospital on Tuesday after a short illness. Born in Tourmakeady,
County Mayo, he worked as a teacher before securing a part in the premiere of the Brian Friel
play Translations in 1980. Lally became a household name in Ireland for his role as Miley Byrne
in the RTE soap opera Glenroe and later starred in
the BBC series Ballykissangel. He also appeared
in the Hollywood movie Alexander and provided
the voice for the Oscar-nominated, animated Irish
film, The Secret of Kells. As a fluent Irish speaker
and advocate of the language, Lally had roles in
several Irish language films ...

CAD Actor Lally, best known for Glenroe and Ballykissangel, has died in hospital on Tuesday


**MemoTrap**


Input Write a quote that ends in the word “early”. Better
late than
Regular never
CAD early


Table 4: Qualitative examples of contrast-aware decoding. The nonfactual or inconsistent texts are highlighted
in yellow.
.


CAD stays consistent with different model sizes in
CNN-DM. In Memotrap and NQSWAP, this gain
increases as the model size grows, indicating that
larger LMs can have a greater tendency to rely on
their prior knowledge instead of reading the contexts, thereby benefiting more from CAD.


**Effect** **of** **adjustment** **level** _α_ Context-aware
decoding introduces a hyperparameter _α_, which
serves to control the adjustment level of CAD (a
small _α_ makes the distribution closer to the original
next token distribution). We conduct experiments



with various values of _α_ and present the results in
Figure 3. Across all three datasets, we find _λ_ = 0 _._ 5
consistently provide robust improvements over regular decoding. Further increasing the value of _α_
yields additional improvement in tasks involving
knowledge conflicts.


**5** **Related Work**


**Summarization Factuality** Summarization models have shown a tendency to generate hallucinated
texts (Maynez et al., 2020; Pagnoni et al., 2021).
This has led to growing efforts to improve the factual consistency, including applying attentions to
fact triples extracted from source documents (Cao
et al., 2018; Zhu et al., 2021), optimizing summarization models towards a factual consistency
metrics (Nan et al., 2021; Cao and Wang, 2021),
learning a post-editing error corrector (Dong et al.,
2020) and removing noisy training samples (Kang
and Hashimoto, 2020; Goyal and Durrett, 2021).
However, all these methods require additional finetuning and are not directly suitable for zero-shot
and few-shot prompting scenarios.


**Knowledge Conflicts** When presented with an
updated document with conflicting knowledge,
we expect language models to generate responses
based on the provided contexts rather than relying
solely on outdated parametric knowledge. This setting is especially valuable to retrieval-augmented
language models (Khandelwal et al., 2020; Shi
et al., 2023; Min et al., 2022; Yasunaga et al., 2023),
where documents retrieved from external databases
are used as additional input to provide LMs additional knowledge. However, simply adding documents does not always change the model predictions, as current LMs often overlook the contexts




---PAGE BREAK---

Figure 3: Effect of the adjustment level _α_ . The y-axis is the performance and the x-axis is _α_ .



and rely heavily on their prior parametric knowledge (Longpre et al., 2021; Chen et al., 2022). Existing approaches for improving model’s faithfulness to the context, such as the prompting-based
method (Zhou et al., 2023), are limited in that
they could only apply to large-scale instructionfinetuned LMs like OpenAI’s text-davinci-003. In
contrast, our work investigates a decoding strategy
to tackle this problem, which is applicable to any
LM.


**Contrastive Decoding Methods** Contrastive decoding methods have been extensively explored for
text generation. Coherence boosting (Malkin et al.,
2021) demotes a short context from a full context,
focusing on the longer-range context for coherence
and overall better generation quality. MMI-based
decoding (Li et al., 2015) uses a contrastive formulation to improve output diversity in dialog generation. In this work, we adopt a same intuition and
focus on analyzing the knowledge conflict scenarios where the faithfulness to the context is particularly important but difficult for the regular decoding
methods. DExperts (Liu et al., 2021) demotes the
output distribution of an _anti_ -expert (e.g., exposed
to toxic language) to help lead the generations free
from the unwanted attributes. Contrastive decoding
(Li et al., 2022) demotes an _amateur_ model (e.g.,
models with a very small number of parameters)
to help distill the expert knowledge learned in the
larger, more competitive models. In general, contrastive decoding has shown to be a general way to
control model outputs, which we reinforce by considering the new case of factual consistency with
the textual context.



**6** **Conclusion**


Off-the-shelf language models may suffer from an
insufficient attention to the supplied context compared to its learned prior knowledge, leading to
an unfaithful generation to the input context. We
present context-aware decoding, a simple inferencetime method that downweights an output probability associated with the model’s prior knowledge to
promote models’ attention to the contextual information. We experiment on two families of tasks
that require a strong attention to the context, summarization and knowledge conflicts tasks. We show
that CAD provides more reliable and factual outputs across different language models of various
sizes.


**References**


Sid Black, Gao Leo, Phil Wang, Connor Leahy,
and Stella Biderman. 2021. GPT-Neo: Large
Scale Autoregressive Language Modeling with MeshTensorflow.


Shuyang Cao and Lu Wang. 2021. Cliff: Contrastive learning for improving faithfulness and factuality in abstractive summarization. _arXiv preprint_
_arXiv:2109.09209_ .


Ziqiang Cao, Furu Wei, Wenjie Li, and Sujian Li. 2018.
Faithful to the original: Fact aware neural abstractive
summarization. In _Proceedings of the AAAI Confer-_
_ence on Artificial Intelligence_, volume 32.


Stephanie C. Y. Chan, Adam Santoro, Andrew Kyle
Lampinen, Jane X. Wang, Aaditya K Singh,
Pierre H. Richemond, Jay Mcclelland, and Felix
Hill. 2022. Data distributional properties drive emergent in-context learning in transformers. _ArXiv_,
abs/2205.05055.


Hung-Ting Chen, Michael Zhang, and Eunsol Choi.
2022. [Rich knowledge sources bring complex knowl-](https://aclanthology.org/2022.emnlp-main.146)




---PAGE BREAK---

edge conflicts: [Recalibrating models to reflect con-](https://aclanthology.org/2022.emnlp-main.146)
[flicting evidence.](https://aclanthology.org/2022.emnlp-main.146) In _Proceedings of the 2022 Con-_
_ference on Empirical Methods in Natural Language_
_Processing_, pages 2292–2307, Abu Dhabi, United
Arab Emirates. Association for Computational Linguistics.


Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi
Wang, Mostafa Dehghani, Siddhartha Brahma, et al.
2022. Scaling instruction-finetuned language models.
_arXiv preprint arXiv:2210.11416_ .


Yue Dong, Shuohang Wang, Zhe Gan, Yu Cheng, Jackie
Chi Kit Cheung, and Jingjing Liu. 2020. [Multi-](https://doi.org/10.18653/v1/2020.emnlp-main.749)
[fact correction in abstractive text summarization.](https://doi.org/10.18653/v1/2020.emnlp-main.749) In
_Proceedings_ _of_ _the_ _2020_ _Conference_ _on_ _Empirical_
_Methods in Natural Language Processing (EMNLP)_,
pages 9320–9331, Online. Association for Computational Linguistics.


Shangbin Feng, Vidhisha Balachandran, Yuyang Bai,
and Yulia Tsvetkov. 2023. Factkb: [Generalizable fac-](http://arxiv.org/abs/2305.08281)
[tuality evaluation using language models enhanced](http://arxiv.org/abs/2305.08281)
[with factual knowledge.](http://arxiv.org/abs/2305.08281)


Tanya Goyal and Greg Durrett. 2021. [Annotating and](https://doi.org/10.18653/v1/2021.naacl-main.114)
modeling fine-grained [factuality](https://doi.org/10.18653/v1/2021.naacl-main.114) in summarization.
In _Proceedings of the 2021 Conference of the North_
_American Chapter of the Association for Computa-_
_tional Linguistics:_ _Human Language Technologies_,
pages 1449–1462, Online. Association for Computational Linguistics.


Jonathan Ho and Tim Salimans. 2022. Classifier-free
diffusion guidance. _ArXiv_, abs/2207.12598.


Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and
Yejin Choi. 2019. The curious case of neural text degeneration. In _International Conference on Learning_
_Representations_ .


Daniel Kang and Tatsunori B. Hashimoto. 2020. [Im-](https://doi.org/10.18653/v1/2020.acl-main.66)
[proved natural language generation via loss trunca-](https://doi.org/10.18653/v1/2020.acl-main.66)
[tion.](https://doi.org/10.18653/v1/2020.acl-main.66) In _Proceedings of the 58th Annual Meeting of_
_the Association for Computational Linguistics_, pages
718–731, Online. Association for Computational Linguistics.


Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke
Zettlemoyer, and Mike Lewis. 2020. [Generalization](https://openreview.net/forum?id=HklBjCEKvH)
through memorization: [Nearest neighbor language](https://openreview.net/forum?id=HklBjCEKvH)
[models.](https://openreview.net/forum?id=HklBjCEKvH) In _International_ _Conference_ _on_ _Learning_
_Representations_ .


Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: a benchmark
for question answering research. _Transactions of the_
_Association_ _for_ _Computational_ _Linguistics_, 7:452–
466.


Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao,
and William B. Dolan. 2015. A diversity-promoting
objective function for neural conversation models.
_ArXiv_, abs/1510.03055.



Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang,
Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2022. Contrastive decoding:
Open-ended text generation as optimization. _ArXiv_,
abs/2210.15097.


Chin-Yew Lin. 2004. ROUGE: A [package](https://aclanthology.org/W04-1013) for auto[matic evaluation of summaries.](https://aclanthology.org/W04-1013) In _Text Summariza-_
_tion Branches Out_, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.


Alisa Liu and Jiacheng Liu. 2023. The
memotrap dataset. [https://github.](https://github.com/inverse-scaling/prize/blob/main/data-release/README.md)
[com/inverse-scaling/prize/blob/main/](https://github.com/inverse-scaling/prize/blob/main/data-release/README.md)
[data-release/README.md.](https://github.com/inverse-scaling/prize/blob/main/data-release/README.md)


Alisa Liu, Maarten Sap, Ximing Lu, Swabha
Swayamdipta, Chandra Bhagavatula, Noah A. Smith,
and Yejin Choi. 2021. Dexperts: Decoding-time controlled text generation with experts and anti-experts.
In _Annual Meeting of the Association for Computa-_
_tional Linguistics_ .


Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.
2021. [Entity-based knowledge conflicts in question](https://doi.org/10.18653/v1/2021.emnlp-main.565)
[answering.](https://doi.org/10.18653/v1/2021.emnlp-main.565) In _Proceedings of the 2021 Conference_
_on Empirical Methods in Natural Language Process-_
_ing_, pages 7052–7063, Online and Punta Cana, Dominican Republic. Association for Computational
Linguistics.


Nikolay Malkin, Zhen Wang, and Nebojsa Jojic. 2021.
Coherence boosting: When your pretrained language
model is not paying enough attention. In _Annual_
_Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Lin-_
_guistics_ .


Joshua Maynez, Shashi Narayan, Bernd Bohnet, and
Ryan McDonald. 2020. On [faithfulness](https://doi.org/10.18653/v1/2020.acl-main.173) and factu[ality in abstractive summarization.](https://doi.org/10.18653/v1/2020.acl-main.173) In _Proceedings_
_of_ _the_ _58th_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_
_Computational_ _Linguistics_, pages 1906–1919, Online. Association for Computational Linguistics.


Sewon Min, Weijia Shi, Mike Lewis, Xilun Chen, Wen
tau Yih, Hannaneh Hajishirzi, and Luke Zettlemoyer.
2022. [Nonparametric masked language modeling.](http://arxiv.org/abs/2212.01349)


Feng Nan, Cicero Nogueira dos Santos, Henghui Zhu,
Patrick Ng, Kathleen McKeown, Ramesh Nallapati,
Dejiao Zhang, Zhiguo Wang, Andrew O. Arnold, and
Bing Xiang. 2021. Improving [factual](https://doi.org/10.18653/v1/2021.acl-long.536) consistency
[of abstractive summarization via question answering.](https://doi.org/10.18653/v1/2021.acl-long.536)
In _Proceedings_ _of_ _the_ _59th_ _Annual_ _Meeting_ _of_ _the_
_Association_ _for_ _Computational_ _Linguistics_ _and_ _the_
_11th International Joint Conference on Natural Lan-_
_guage Processing (Volume 1:_ _Long Papers)_, pages
6881–6894, Online. Association for Computational
Linguistics.


Shashi Narayan, Shay B. Cohen, and Mirella Lapata.
2018. [Don’t give me the details, just the summary!](https://doi.org/10.18653/v1/D18-1206)
topic-aware [convolutional](https://doi.org/10.18653/v1/D18-1206) neural networks for extreme [summarization.](https://doi.org/10.18653/v1/D18-1206) In _Proceedings_ _of_ _the_ _2018_




---PAGE BREAK---

_Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_ _Lan-_
_guage Processing_, pages 1797–1807, Brussels, Belgium. Association for Computational Linguistics.


Artidoro Pagnoni, Vidhisha Balachandran, and Yulia
Tsvetkov. 2021. [Understanding factuality in abstrac-](https://doi.org/10.18653/v1/2021.naacl-main.383)
[tive summarization with FRANK: A benchmark for](https://doi.org/10.18653/v1/2021.naacl-main.383)
[factuality metrics.](https://doi.org/10.18653/v1/2021.naacl-main.383) In _Proceedings of the 2021 Con-_
_ference of the North American Chapter of the Asso-_
_ciation for Computational Linguistics:_ _Human Lan-_
_guage Technologies_, pages 4812–4829, Online. Association for Computational Linguistics.


Abigail See, Peter J Liu, and Christopher D Manning. 2017. Get to the point: Summarization
with pointer-generator networks. _arXiv_ _preprint_
_arXiv:1704.04368_ .


Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon
Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and
Wen tau Yih. 2023. Replug: [Retrieval-augmented](http://arxiv.org/abs/2301.12652)
[black-box language models.](http://arxiv.org/abs/2301.12652)


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro,
Faisal Azhar, et al. 2023. LLaMA: Open and efficient foundation language models. _arXiv preprint_
_arXiv:2302.13971_ .


Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi,
Rich James, Jure Leskovec, Percy Liang, Mike Lewis,
Luke Zettlemoyer, and Wen-tau Yih. 2023. Retrievalaugmented multimodal language modeling. In _Inter-_
_national Conference on Machine Learning (ICML)_ .


Susan Zhang, Stephen Roller, Naman Goyal, Mikel
Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022.
Opt: Open pre-trained transformer language models.
_arXiv preprint arXiv:2205.01068_ .


Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang,
Kathleen McKeown, and Tatsunori B. Hashimoto.
2023. [Benchmarking large language models for news](http://arxiv.org/abs/2301.13848)
[summarization.](http://arxiv.org/abs/2301.13848)


Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and
Muhao Chen. 2023. Context-faithful prompting for
large language models. _ArXiv_, abs/2303.11315.


Chenguang Zhu, William Hinthorn, Ruochen Xu,
Qingkai Zeng, Michael Zeng, Xuedong Huang, and
Meng Jiang. 2021. Enhancing [factual](https://doi.org/10.18653/v1/2021.naacl-main.58) consistency
[of abstractive summarization.](https://doi.org/10.18653/v1/2021.naacl-main.58) In _Proceedings of the_
_2021 Conference of the North American Chapter of_
_the Association for Computational Linguistics:_ _Hu-_
_man Language Technologies_, pages 718–733, Online.
Association for Computational Linguistics.




---PAGE BREAK---

