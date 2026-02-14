DISCOVERING LATENT KNOWLEDGE IN LANGUAGE
MODELS WITHOUT SUPERVISION


**Collin Burns** _[∗]_ **Haotian Ye** _[∗]_ **Dan Klein** **Jacob Steinhardt**
UC Berkeley Peking University UC Berkeley UC Berkeley


ABSTRACT


Existing techniques for training language models can be misaligned with the truth:
if we train models with imitation learning, they may reproduce errors that humans
make; if we train them to generate text that humans rate highly, they may output
errors that human evaluators can’t detect. We propose circumventing this issue by
directly finding latent knowledge inside the internal activations of a language model
in a purely unsupervised way. Specifically, we introduce a method for accurately
answering yes-no questions given only unlabeled model activations. It works by
finding a direction in activation space that satisfies logical consistency properties,
such as that a statement and its negation have opposite truth values. We show that
despite using no supervision and no model outputs, our method can recover diverse
knowledge represented in large language models: across 6 models and 10 questionanswering datasets, it outperforms zero-shot accuracy by 4% on average. We also
find that it cuts prompt sensitivity in half and continues to maintain high accuracy
even when models are prompted to generate incorrect answers. Our results provide
an initial step toward discovering what language models know, distinct from what
they say, even when we don’t have access to explicit ground truth labels.


1 INTRODUCTION


The increasing deployment of language models in real-world applications opens up exciting possibilities, but it also raises the stakes of AI research and presents new risks (Bommasani et al., 2021;
Weidinger et al., 2021; Bender et al., 2021). One of these risks is that language models do not always
output text that is true (Evans et al., 2021; Hendrycks et al., 2021; Kenton et al., 2021).


Common training objectives can cause models to learn internal representations related to truth, since
truth is a useful feature for many tasks. However, these objectives can also cause language models to
output text that is false, at least in some circumstances. For example, if we train a model to imitate
human-generated text, it may learn to output common misconceptions (Lin et al., 2022). Or if we train
a chat bot to optimize a reward such as engagement, it may learn to generate text that is compelling
but false (Roller et al., 2021). If we try to reward model outputs that look true, a model may still learn
to output false text if human raters can’t evaluate the correctness of that text (Kenton et al., 2021).


In each case, this is an issue that stems from the misalignment between a training objective and the
truth. As models are applied to more complex domains, human supervision may become less effective
at mitigating this misalignment. Moreover, because this is a problem with the training objective rather
than a model’s capabilities, it likely won’t be solved by scaling up models alone.


We propose a different approach for addressing this misalignment: using models to answer questions
in a purely _unsupervised_ way. Intuitively, instead of trying to explicitly, externally specify truth,
we search for implicit, internal “beliefs” or “knowledge” learned by a model. We approach this
problem by leveraging the fact that a model’s representation of truth must satisfy logical consistency
properties, which are unlikely to be satisfied by many other features.


We implement this idea by introducing Contrast-Consistent Search (CCS), a method that learns a
linear projection of the hidden states that is consistent across negations, as illustrated in Figure 1.
We find that despite its simplicity and despite not having access to any labels or model outputs,


*Equal contribution.


1




---PAGE BREAK---

|Col1|Col2|Col3|lab|lab|el<br>True<br>False|
|---|---|---|---|---|---|
||||lab|||


Given a set of Yes-No questions, answer Extract internal Map each activation to a Optimize unsupervised loss to make For each question, estimate the
each question with both "Yes" and "No" model activations probability of being true probabilities consistent and confident probability that the answer is "Yes"


Figure 1: An illustration of our method, Contrast-Consistent Search (CCS). For each yes-no question
_qi_, we let _x_ [+] _i_ [and] _[x][−]_ _i_ [be] [the] [natural] [language statements] [where] [we] [answer] _[q][i]_ [as] [“Yes”] [and] [“No”]
respectively. Answering the question _qi_ then amounts to determining which of _x_ [+] _i_ [or] _[x][−]_ _i_ [is] [true.]
We compute probabilities _p_ [+] _i_ [and] _[ p]_ _i_ _[−]_ [that] _[ x]_ _i_ [+] [and] _[ x]_ _i_ _[−]_ [are true respectively using a learned mapping]
from the hidden states to a number between 0 and 1. We search for a mapping such that that the
probabilities are both confident and consistent. On the right, we show a histogram of the “Yes”
probabilities, _p_ ˜ _i_ = 0 _._ 5 _·_ ( _p_ [+] _i_ [+ (1] _[ −]_ _[p]_ _i_ _[−]_ [))][,] [learned] [by] [our] [method] [on] [the] [unlabeled] [train] [split] [of]
the COPA dataset (Roemmele et al., 2011) with the UnifiedQA model (Khashabi et al., 2020). Our
method uses no labels and no model outputs, but still learns to accurately answers questions.


CCS can accurately recover knowledge from model representations: evaluated across 6 models and
10 question-answering datasets, CCS outperforms the accuracy of strong zero-shot baselines by
4% on average (Section 3.2.1). The resulting classifier is also less sensitive to different prompts
than zero-shot, cutting the standard deviation in accuracy in half. Additionally, we try deliberately
prompting models to make incorrect outputs, which should intuitively change what models say but
which shouldn’t affect their latent knowledge. We find that this causes zero-shot accuracy to drop by
up to 9.5% (Section 3.2.2) without decreasing the accuracy of CCS.


We systematically analyze CCS to understand the features it discovers. We show that it transfers
across unrelated tasks, suggesting that models may have a task-agnostic representation of the truth
and that CCS is able to approximately discover it (Section 3.3.1). Moreover, CCS sometimes works
best using the hidden states in the _middle_ layers of a network and can work even when model outputs
aren’t very informative, suggesting that it can leverage different features from those used by the
outputs (Section 3.3.2). Finally, we show that representations of truth tend to be salient in models:
they can often be found without much data, and they can often be found by taking the top principal
component of a slightly modified representation space (Section 3.3.3).


Most existing techniques for making models truthful use human supervision to explicitly specify
what is correct. However, it is not feasible to provide supervision in some settings. Our work serves
as a proof of concept that an external source of ground truth may not actually be necessary: we may
instead be able to find a model’s latent representation of truth, independent of what a model says,
without using any supervision in the first place.


2 PROBLEM STATEMENT AND FRAMEWORK


In this section we describe our problem setup in more detail and introduce Contrast-Consistent Search
(CCS), a method for discovering latent knowledge in language models without supervision.


2.1 PROBLEM: DISCOVERING LATENT KNOWLEDGE


Given a pre-trained neural language model and a set _q_ 1 _, . . ., qn_ of yes-no questions [1], our goal is
to answer each _qi_ correctly. Here, _qi_ can be any question with a well-defined answer, including
procedural questions like “Is 22+59 = 237?”, for which the answer is “No”, and factual questions like
“Are cats mammals?”, for which the answer is “Yes”.


1Technically, we only require that there are two mutually exclusive answers. For example, we can also use
the labels “positive” and “negative” for sentiment classification. Moreover, our setup can easily extend to the
case where we want to evaluate the truth of a set of statements instead of answering a set of questions.


2




---PAGE BREAK---

Importantly, we want methods that do not rely on the model generating correct outputs and that do
not rely on external supervision. Instead, we turn to the model’s unlabeled hidden representations.
Specifically, let _ϕ_ ( _x_ ) _∈_ R _[d]_ denote some feature representation on a natural langauge input _x_, such
as the hidden states of a Transformer-based language model. Our goal is to answer the questions
_q_ 1 _, . . ., qn_ only given access to _ϕ_ ( _·_ ). In Section 2.2 we introduce a method for this problem that
attains high accuracy (Section 3), demonstrating that this task is tractable.


2.2 METHOD: CONTRAST-CONSISTENT SEARCH


To make progress on the goal described above, we exploit the fact that truth has special structure: it
satisfies consistency properties that few other features in a language model are likely to satisfy. Our
method, Contrast-Consistent Search (CCS), leverages this idea by finding a direction in activation
space that is consistent across negations. As we illustrate in Figure 1, CCS works by (1) answering
each question _qi_ as both “Yes” ( _x_ [+] _i_ [) and “No” (] _[x]_ _i_ _[−]_ [), (2) computing the representations] _[ ϕ]_ [(] _[x]_ _i_ [+][)][ and]
_ϕ_ ( _x_ _[−]_ _i_ [)][ of each answer, (3) mapping the answer representations to probabilities] _[ p]_ _i_ [+] [and] _[ p]_ _i_ _[−]_ [of being]
true, then (4) optimizing that mapping so that the probabilities are both consistent and confident.


Concretely, the input to CCS is a set of Yes-No questions, _q_ 1 _, . . ., qn_, and access to a pretrained
model’s representations, _ϕ_ ( _·_ ); the output of CCS is a lightweight probe on top of _ϕ_ ( _·_ ) that can
answer new questions. Here, _ϕ_ ( _·_ ) is fixed but should contain useful information about the answers
to _q_ 1 _, . . ., qn_, in the sense that if one _did_ (hypothetically) have access to the ground-truth labels for
_q_ 1 _, . . ., qn_, one would be able to train a small supervised probe on _ϕ_ ( _·_ ) that attains high accuracy.
Importantly, CCS does not modify the weights of the pretrained model and it does not use labels.


**Constructing contrast pairs.** An important property that truth satisfies is negation consistency: the
answer to a clear-cut question cannot be both “Yes” and “No” at the same time, as these are negations
of each other. Probabilistically, for each question _qi_, the probability that the answer to _qi_ is “Yes”
should be one minus the probability that the answer to _qi_ is “No”. To use this property, we begin by
constructing contrast pairs: for each question _qi_, we answer _qi_ both as “Yes”, resulting in the new
natural language statement _x_ [+] _i_ [, and as “No”, resulting in the natural language statement] _[ x]_ _i_ _[−]_ [.] [We]
illustrate this in Figure 1 (left). We will then learn to classify _x_ [+] _i_ [and] _[ x]_ _i_ _[−]_ [as true or false; if] _[ x]_ _i_ [+] [is true,]
then the answer to _qi_ should be “Yes”, and if _x_ _[−]_ _i_ [is true, then the answer to] _[ q][i]_ [should be “No”.]


In practice, we convert each task into a question-answering task with two possible labels, then we use
task-specific zero-shot prompts to format questions and answers as strings to construct each contrast
pair. The opposite labels we use to construct contrast pairs can be “Yes” and “No” for a generic task,
or they can be other tasks-specific labels, such as “Positive” and “Negative” in the case of sentiment
classification. We describe the exact prompts we use to for each task in Appendix B.

**Feature extraction and normalization.** Given a contrast pair ( _x_ [+] _i_ _[, x]_ _i_ _[−]_ [)][, CCS first computes the]
representations _ϕ_ ( _x_ [+] _i_ [)][ and] _[ ϕ]_ [(] _[x]_ _i_ _[−]_ [)][ using the feature extractor] _[ ϕ]_ [(] _[·]_ [)][.] [Intuitively, there are two salient]
differences between _ϕ_ ( _x_ [+] _i_ [)][ and] _[ ϕ]_ [(] _[x]_ _i_ _[−]_ [)][:] [(1)] _[ x]_ [+] _i_ [ends with “Yes” while] _[ x]_ _i_ _[−]_ [ends with “No”, and (2)]
one of _x_ [+] _i_ [or] _[ x]_ _i_ _[−]_ [is true while the other is false.] [We want to find (2) rather than (1), so we first try]
to remove the effect of (1) by normalizing _{ϕ_ ( _x_ [+] _i_ [)] _[}]_ [ and] _[ {][ϕ]_ [(] _[x]_ _i_ _[−]_ [)] _[}]_ [ independently.] [In particular, we]
construct normalized representations _ϕ_ [˜] ( _x_ ) as follows:
_ϕ_ ˜( _x_ [+] _i_ [) :=] _[ ϕ]_ [(] _[x]_ _i_ [+][)] _[ −]_ _[µ]_ [+][ ;] _ϕ_ ˜( _x_ _[−]_ _i_ [) :=] _[ ϕ]_ [(] _[x]_ _i_ _[−]_ [)] _[ −]_ _[µ][−][,]_

where _µ_ [+] _, µ_ _[−]_ _∈_ R _[d]_ are the means of _{ϕ_ ( _x_ [+] _i_ [)] _[}]_ _i_ _[n]_ =1 [and] _[ {][ϕ]_ [(] _[x]_ _i_ _[−]_ [)] _[}]_ _i_ _[n]_ =1 [.] [This normalization ensures that]
_{ϕ_ [˜] ( _x_ [+] _i_ [)] _[}]_ [ and] _[ {][ϕ]_ [˜][(] _[x]_ _i_ _[−]_ [)] _[}]_ [ no longer form two separate clusters.] [In practice we also normalize the scale]
of the features, but this isn’t essential for the method to work; see Appendix G.1 for details.


**Mapping activations to probabilities.** Next, we learn a probe _pθ,b_ ( _ϕ_ [˜] ) that maps a (normalized)
hidden state _ϕ_ [˜] ( _x_ ) to a number between 0 and 1 representing the probability that the statement _x_ is
true. We use a linear projection followed by a sigmoid _σ_ ( _·_ ), i.e. _pθ,b_ ( _ϕ_ [˜] ) = _σ_ ( _θ_ _[T]_ [ ˜] _ϕ_ + _b_ ), but nonlinear
projections can also work. For simplicity, we sometimes omit the _θ, b_ subscript in _p_ .


**Training objective.** To find features that represent the truth, we leverage the consistency structure of
truth. First, we use the fact that a statement and its negation should have probabilities that add up to
1. This motivates the consistency loss:

_L_ consistency( _θ, b_ ; _qi_ ) :=        - _pθ,b_ ( _x_ [+] _i_ [)] _[ −]_ [(1] _[ −]_ _[p][θ,b]_ [(] _[x]_ _i_ _[−]_ [))] �2


3




---PAGE BREAK---

However, this objective alone has a degenerate solution: _p_ ( _x_ [+] ) = _p_ ( _x_ _[−]_ ) = 0 _._ 5. To avoid this
problem, we encourage the model to also be confident with the following confidence loss:

_L_ confidence( _θ, b_ ; _qi_ ) := min _{pθ,b_ ( _x_ [+] _i_ [)] _[, p][θ,b]_ [(] _[x]_ _i_ _[−]_ [)] _[}]_ [2]

We can equivalently interpret _L_ confidence as imposing a second consistency property on the probabilities:
the law of excluded middle (every statement must be either true or false). The final unsupervised loss
is the sum of these two losses, averaged across all contrast pairs:



_L_ CCS( _θ, b_ ) := [1]

_n_



_n_

- _L_ consistency( _θ, b_ ; _qi_ ) + _L_ confidence( _θ, b_ ; _qi_ )


_i_ =1



Note that both losses are necessary; _L_ confidence alone also has a degenerate solution.

**Inference.** Both _p_ ( _x_ [+] _i_ [)][ and][ 1] _[ −]_ _[p]_ [(] _[x]_ _i_ _[−]_ [)][ should represent the probability that the answer to] _[ q][i]_ [is “Yes”.]
However, because we use a soft consistency constraint, these may not be exactly equal. To make a
prediction on an example _xi_ after training, we consequently take the average of these:

_p_ ˜( _qi_ ) := [1] _i_ [) + (1] _[ −]_ _[p]_ [(] _[x]_ _i_ _[−]_ [))]

2 [(] _[p]_ [(] _[x]_ [+]

We then predict that the answer to _qi_ is “Yes” based on whether _p_ ˜( _qi_ ) is greater than 0 _._ 5. Technically,
we also need to determine whether _p_ ˜( _qi_ ) _>_ 0 _._ 5 corresponds to “Yes” or “No,” as this isn’t specified
by _L_ CCS. For simplicity in our evaluations we take the maximum accuracy over the two possible
ways of labeling the predictions of a given test set. However, in Appendix A we describe how one
can identify the two clusters without any supervision in principle by leveraging conjunctions.


3 RESULTS


3.1 EXPERIMENTAL SETUP


Here we give an overview of our experimental setup; see Appendix G for full details. We provide code
[at https://www.github.com/collin-burns/discovering_latent_knowledge.](https://www.github.com/collin-burns/discovering_latent_knowledge)


**Models.** We test six models: encoder-decoder models (T5 (Raffel et al., 2020), UnifiedQA (Khashabi
et al., 2020), T0 (Sanh et al., 2021)), autoregressive models (GPT-J (Wang & Komatsuzaki, 2021)),
and encoder-only models (RoBERTa (Liu et al., 2019), DeBERTa (He et al., 2021)).


**Data.** We test models on 10 datasets: sentiment classification (IMDB (Maas et al., 2011) and Amazon
(McAuley & Leskovec, 2013)), topic classification (AG-News (Zhang et al., 2015) and DBpedia-14
(Lehmann et al., 2015)), NLI (RTE (Wang et al., 2018) and QNLI (Rajpurkar et al., 2016)), story
completion (COPA (Roemmele et al., 2011) and Story-Cloze (Mostafazadeh et al., 2017)), question
answering (BoolQ (Clark et al., 2019)), and common sense reasoning (PIQA (Bisk et al., 2020)).


We convert each dataset to a yes-no question-answering task or a binary classification task, as
described in Appendix G. We balance the labels and randomly subsample 1000 examples from each
dataset (except for COPA, which has only 500 examples total), then randomly split each dataset into
an unsupervised training set (60% of the data) and test set (40%). We subsample each dataset for
computational efficiency reasons; because we aggregate over 9 prompts per dataset, 10 datasets, and
6 models, 1000 datapoints per dataset actually corresponds to approximately 180k examples in total.


**Methods.** We test four main methods: zero-shot, calibrated zero-shot, Contrast-Consistent Search
(CCS), and Logistic Regression (LR). Zero-shot works by predicting the answer with the highest
log probability according to the language model, averaged across the tokens that make up that label.
Calibrated zero-shot works by balancing zero-shot predictions to be 50 _/_ 50 for each answer, as
we describe in more detail below, similar to Zhao et al. (2021). For Logistic Regression we train
on the training split for each dataset using ( _ϕ_ [˜] ( _x_ [+] ) _,_ _ϕ_ [˜] ( _x_ _[−]_ )) as the covariates, then evaluate on the
corresponding test split. We treat LR as a ceiling since it uses labeled data.


When testing CCS, we optimize it 10 times using AdamW (Loshchilov & Hutter, 2017) with learning
rate 0 _._ 01, then take the run with the lowest unsupervised loss. Unless otherwise specified, we train
CCS using all prompts for a single training set (normalized independently for each prompt), then
evaluate it on the corresponding test split.


**Zero-shot baselines.** Zero-shot outputs sometimes suffer from miscalibration (Zhao et al., 2021), in
which models are biased towards predicting specific answers. Calibrating the outputs to be uniform


4




---PAGE BREAK---

Method RoBERTa DeBERTa GPT-J T5 UQA T0 _[∗]_ Mean _[∗]_

0-shot 60.1(5.7) 68.6(8.2) 53.2(5.2) 55.4(5.7) 76.8(9.6) 87.9(4.8) 62.8(6.9)
Calibrated 0-shot **64.3(6.2)** 76.3(6.0) 56.0(5.2) 58.8(6.1) 80.4(7.1) 90.5(2.7) 67.2(6.1)
CCS 62.1(4.1) **78.5(3.8)** 61.7(2.5) 71.5(3.0) 82.1(2.7) 77.6(3.3) 71.2(3.2)
CCS (All Data) 60.1(3.7) 77.1(4.1) **62.1(2.3)** **72.7(6.0)** **84.8(2.6)** 84.8(3.7) **71.5(3.7)**
LR (Ceiling) 79.8(2.5) 86.1(2.2) 78.1(2.3) 84.6(3.1) 89.8(1.9) 90.5(2.1) 83.7(2.4)


Table 1: Accuracy of each method and model averaged across all prompts and dataset, with the
average standard deviation of accuracy across different prompts shown in parentheses. For most
models, CCS outperforms zero-shot accuracy and exhibits lower sensitivity to prompts, even though
this was not our goal. This shows that we can recover knowledge from language model activations
without supervision, and can do so in a way that is competitive with strong baseline methods that use
model outputs. _[∗]_ T0 was trained on 9 out of 10 of the datasets we evaluate on, including some of the
data in our test splits, so we ignore it when averaging over models.


over different answers can mitigate this problem. We use a variant of the calibration method presented
in Zhao et al. (2021) by balancing predictions to be 50/50 across the two output labels. Specifically,
if _l_ + and _l−_ are the logits for the positive and negative label respectively, then instead of classifying
an example as positive if _l_ + _>_ _l−_, we classify it as positive if _l_ + _>_ _l−_ + _γ_, where we select the
threshold _γ_ _∈_ R so that the predictions are balanced. We find this increases accuracy by about 5% on
average. Unless otherwise specified, we always report zero-shot accuracy after calibration.


Encoder-only models (e.g. RoBERTa and DeBERTa) cannot be easily used to do zero-shot classification out of the box, so to evaluate them we follow the method of Yin et al. (2020): we finetune both
models on an NLI dataset (MNLI, which we do not evaluate on) and treat the difference between
the entailment and contradiction probabilities as the effective logit. This provides a strong zero-shot
baseline for encoder-only models that works even for non-NLI tasks (Yin et al., 2020). This finetuning
isn’t necessary for CCS to work on encoder-only models (see Appendix C), but we test CCS using
the same MNLI-finetuned models for ease of comparison.


**Hidden states.** We extract the hidden states corresponding to the last token in the last layer of each
model for simplicity, unless otherwise specified. For encoder-decoder models, we evaluate CCS on
the last layer hidden states of both the encoder and decoder, and use whichever one generally achieves
a lower unsupervised loss; for T0 this is the decoder hidden states, while for T5 and UnifiedQA this is
the encoder hidden states. See Appendix G.2 for further implementation details, such as tokenization.


**Prompts.** To reduce prompt sensitivity, we use between 8 and 13 prompts for each dataset (9 on
average), derived or slightly modified from Sanh et al. (2021). Unless otherwise specified, we average
across all prompts when showing results. To construct contrast pairs, we let _x_ [+] _i_ [be the zero-shot]
prompt using _qi_ and the first label (e.g. “Positive” for sentiment classification datasets) and let _x_ _[−]_ _i_ [be]
the prompt using _qi_ and the second label (e.g. “Negative”). We describe all prompts in Appendix I.


3.2 EVALUATING CCS


3.2.1 CCS OUTPERFORMS ZERO-SHOT


We evaluate CCS on all 6 models and compute the average accuracy across all datasets and prompts.
T0 was trained on 9 out of 10 of the datasets we evaluate on, including some of the data in our test
splits, so we ignore it when averaging over models to avoid unfair comparisons. We display the
results in Table 1. To assess prompt sensitivity, for each model and dataset we compute the standard
deviation (s.d.) of accuracy across different prompts, then average the resulting standard deviations
across all datasets, which we show in parentheses in Table 1. For comparison, we also include results
when training CCS on all datasets simultaneously, which we refer to as CCS (All Data).


CCS attains an accuracy of 71 _._ 2% on average, compared to 67 _._ 2% for calibrated zero-shot. It
outperforms zero-shot accuracy for every model, except for RoBERTa (where it does 2% worse) and
T0 (for which zero-shot accuracy is inflated). Training on all datasets improves accuracy by only an
insignificant amount on average (0 _._ 3%), but with large gains for T0 in particular (77 _._ 6% _→_ 84 _._ 8%).


These results show that CCS can _exceed_ the performance of strong baseline methods that access the
model outputs, even though this wasn’t our main goal. This indicates that it is indeed possible to
classify examples with high accuracy using only unlabeled model representations.


5




---PAGE BREAK---

3.2.2 CCS IS ROBUST TO MISLEADING PROMPTS


Recall our goal: to discover latent knowledge in a language model even when the model outputs false
text. In particular, language models are typically trained to imitate text whether or not it is correct, so
if a model sees false text it should intuitively be more likely to predict that subsequent text will also be
false. Based on this idea, we provide an initial proof of concept that CCS can make progress toward
our goal by constructing prompts that aim to mislead the outputs of language models. Specifically, we
add a prefix to the beginning of our zero-shot prompts that consists of questions answered incorrectly
(Figure 5). The hope is that such a prefix will decrease zero-shot accuracy because the model will
imitate its context and answer subsequent questions incorrectly even if it internally “knows” better. [2]
We found that while most models are robust to this type of prefix (see Appendix B), it significantly
drops calibrated zero-shot performance in UnifiedQA, decreasing accuracy from 80 _._ 4% to 70 _._ 9%.


We evaluate CCS on these examples and show the results in Figure 4 of the Appendix. We find that
despite the 9 _._ 5% drop in zero-shot accuracy, CCS maintains high accuracy (82 _._ 1% _→_ 83 _._ 8%). This
provides evidence that our method can still work well even when model outputs are unreliable.


3.3 ANALYZING CCS


We have shown that CCS attains strong classification performance in standard settings and when
we deliberately mislead models. Moreover, we have described our motivation as discovering latent
representations of truth in language models, but in practice CCS just finds a direction in representation
space that attains high accuracy. This raises the question: in what sense is CCS actually finding “truth”
features? We now provide a preliminary investigation of this question.


3.3.1 CCS FINDS A TASK-AGNOSTIC REPRESENTATION OF TRUTH


From the results described so far, it may be possible that the classifier we find is capturing datasetspecific properties, such as which of two labels (e.g. “Yes” vs. “No”) is more likely to be correct. We
rule this out by showing that it generalizes across completely different tasks, including ones with
different label spaces (such as from “Yes” and “No” for a generic task to “Positive” and “Negative”
for sentiment classification). In particular, we train and test CCS on every pair of datasets, and show
the resulting transfer for several models in Figure 2. (See Appendix F for results with other models.)


We find that CCS indeed transfer wells: in the majority of datasets, the transfer accuracy of CCS is
competitive with training and testing on the same dataset (the last row in Figure 2, “No transfer”).
Transfer performance can even outperform the no-transfer setting, especially when we train on simpler
tasks, such as sentiment classification. For example, training CCS on the Amazon sentiment dataset
achieves an average transfer accuracy of 71 _._ 8%, which is 0 _._ 6% higher than CCS without transfer. We
speculate that this performs so well because the difference in representations between correct and
incorrect answers is especially pronounced for easier tasks like Amazon.


Additionally, transfer accuracy tends to be similar for many datasets. For example, training on IMDB
(row 1 of Figure 2) has similar accuracy as training on DBPedia (row 4). This provides evidence
that CCS can find a functionally similar direction across many different types of training datasets.
Overall, these results suggest that (1) models may have a task-agnostic representation related to what
is true, and that (2) CCS may approximately find this representation even without diverse data.


3.3.2 CCS DOES NOT JUST RECOVER MODEL OUTPUTS


One possibility is that CCS can only recover knowledge already contained in a model’s outputs. We
have shown that CCS can outperform zero-shot accuracy (Section 3.2.1), especially when model
outputs are misled (Section 3.2.2), which already provides evidence against this possibility. We now
provide additional evidence against this concern.


First, if CCS were just recovering knowledge in the model outputs, using the last layer of a network
(right before the outputs) should presumably outperform intermediate layers (which are more causally
distant from the outputs). However, for T5 and UnifiedQA, we find that using hidden states in the


2In practice we found that model behavior was qualitatively similar between using this prefix with correct
and incorrect answers, in agreement with the findings of Min et al. (2022b) that using incorrect labels in
demonstrations often does not significantly degrade the performance of a few-shot prompt. Consequently, the
prefix may instead actually be reducing accuracy because it is out-of-distribution. See also Kim et al. (2022) for
more experiments on the subtleties of correct vs incorrect labels for few-shot prompts.


6




---PAGE BREAK---

100


90


80


70


60


50


|94|95|92|99|73|74|88|64|61|82|
|---|---|---|---|---|---|---|---|---|---|
|93|96|92|99|74|73|89|63|63|77|
|92|94|92|99|70|74|89|61|60|73|
|92|94|92|99|76|74|88|62|61|75|
|92|93|92|99|88|74|88|63|63|84|
|92|92|91|99|74|64|84|63|61|76|
|93|95|92|99|77|75|89|62|61|82|
|67|62|69|64|52|52|56|56|53|59|
|78|82|71|66|58|54|63|57|54|62|
|94|95|92|98|84|71|87|63|63|89|
|94|96|92|99|88|64|89|56|54|89|


|95|93|91|99|74|72|78|59|59|78|
|---|---|---|---|---|---|---|---|---|---|
|93|97|91|99|76|71|77|58|61|75|
|91|92|91|99|69|69|75|57|60|71|
|91|95|91|99|63|64|74|58|61|71|
|90|93|92|98|85|55|64|53|63|85|
|75|73|70|76|55|53|59|55|54|61|
|78|86|82|86|58|54|58|55|56|70|
|81|89|85|94|55|55|58|52|56|68|
|85|85|91|99|70|60|63|54|58|71|
|77|86|89|98|66|62|64|58|63|76|
|95|97|91|99|85|53|58|52|58|76|


|92|94|80|87|54|62|72|63|55|66|
|---|---|---|---|---|---|---|---|---|---|
|91|94|81|92|62|69|74|67|55|71|
|86|86|88|96|54|65|69|62|57|72|
|84|87|88|97|60|72|68|59|54|70|
|70|75|73|81|61|61|57|54|55|72|
|59|69|66|68|56|82|65|62|54|57|
|71|72|70|81|56|79|67|68|55|68|
|77|79|68|73|54|62|64|68|54|65|
|62|62|66|72|54|64|58|52|52|58|
|88|88|79|91|67|60|69|63|56|86|
|92|94|88|97|61|82|67|68|52|86|



Test


Figure 2: Transfer Performance using CCS on UnifiedQA, T0 and DeBERTa. The y-axis corresponds
to the training dataset, and the x-axis corresponds to the test dataset. The final row (“No Transfer”)
corresponds to training and testing on the same dataset, which is the same as the diagonal. On most
datasets, CCS transfers well to other datasets (relative to no transfer), including to different tasks
with completely different labels. In some cases transfer even outperforms the no-transfer setting. See
Appendix E for results with other models.


middle of the network outperform hidden states at the end of the network when using CCS (see
Figure 10). This is especially true for UnifiedQA on misleading prefixes; we find that using the
encoder hidden states is robust to misleading prefixes (Section 3.2.2), but that accuracy using the
decoder hidden states drops from 81 _._ 0% to 73 _._ 5%, a similar amount to zero-shot accuracy. This
suggests that compared to the later layers of a model, intermediate layers are more robust and less
correlated with the model outputs, and that CCS can take advantage of this.


Finally, if CCS were just recovering knowledge in the model outputs, we would only expect it to work
in cases where model outputs are informative. However, we show in Appendix C that CCS still works
with masked language models when their outputs are uninformative: when we don’t [MASK] any
input tokens, and when we prompt models so that the labels used to construct contrast pairs appear in
the _middle_ of a prompt rather than at the end. These results show that CCS can sometimes recover
latent knowledge in a model that is distinct from—and more useful than—what the model outputs.


3.3.3 TRUTH IS A SALIENT FEATURE


From the results we have presented so far, it is possible that the direction learned by CCS is difficult
to find and requires using a large amount of unsupervised data. We provide evidence against this
possibility by showing that finding such a direction can both (1) often be done with only a small
amount of data, and can also (2) often be done by essentially taking the top principal component of a
slightly modified representation space.


**CCS doesn’t require much data.** We now evaluate how well CCS performs with different amounts
of data. Whereas before we trained CCS using the full training set and all prompts, here we use
limited data and a single prompt. Specifically, we train using only _k_ unlabeled contrast pairs, using
the single prompt for each model and dataset that achieves the highest zero-shot accuracy. We still
test on all prompts for each dataset. We resample _k_ points 32 times for each of _k_ = 1 _,_ 2 _,_ 4 _, · · ·_, and
take the average accuracy across those 32 samples. Finally, we plot the average such accuracy across
all datasets and prompts for several models in Figure 3.


We find that while CCS benefits from more data, it can often do well with very limited data. In fact, it
can sometimes even do well with only a single contrast pair, though we find high variance across
individual datasets; see Appendix D for more details. This suggests that the strong performance of
CCS does not primarily come from using a large amount of unsupervised data, and indicates that the
direction learned by CCS may be relatively easy to find.


**Contrastive Representation Clustering.** We now show that directions correlated with the truth
may be “salient” in a different way: by showing that we can also find such directions using either
(1) PCA or (2) clustering. Specifically, suppose we construct contrast pairs ( _x_ [+] _i_ [,] _[x][−]_ _i_ [)] [as] [before.]
Intuitively, these two examples are qualitatively almost identical except that one is true and the other


7




---PAGE BREAK---

80


78


76


74


72



|U<br>T0<br>D|QA<br>eBERTa|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||


1 2 4 8 16 32 64 128 256 512
#Samples



Figure 3: Accuracy when we train CCS on _k_ samples for different values of _k_ (each time averaged
across 32 trials). We use the single prompt with the highest zero-shot accuracy for each dataset and
model. While CCS benefits from more examples, it can often work well with limited data.


is false, so the main difference between the representations _ϕ_ [˜] ( _x_ [+] _i_ [)][ and] _[ϕ]_ [˜][(] _[x][−]_ _i_ [)][ should relate to truth.]
Consequently, we can take the differences in (normalized) hidden states, _{ϕ_ [˜] ( _x_ [+] _i_ [)] _[ −]_ _[ϕ]_ [˜][(] _[x]_ _i_ _[−]_ [)] _[}]_ _i_ _[n]_ =1 [, and]
cluster them. We call this method Contrastive Representation Clustering (CRC). Clustering can be
achieved by taking the top principal component (TPC) and thresholding at 0, or by doing a “bimodal
salience search” (BSS) to find a direction that looks bimodal; see Appendix G.4 for further details.


We compare CCS and these two variants of Contrastive Representation Clustering in Table 2, using
the same setting as in Table 1. While CCS performs best, all methods attain high accuracy and
are competitive with zero-shot performance. This indicates both that (1) representations of truth
often lie in a high-variance direction in the contrastive representation space _{ϕ_ [˜] ( _x_ [+] _i_ [)] _[ −]_ _[ϕ]_ [˜][(] _[x]_ _i_ _[−]_ [)] _[}]_ _i_ _[n]_ =1 [,]
and also that (2) true and false examples are often well-clustered in this same contrastive space.
This strengthens the idea that representations of truth may be salient features inside models that
are relatively easy to find. This may help explain why CCS can perform well without using any
supervision, and how it can do so even with only a limited amount of unlabeled data.


4 RELATED WORK


**Zero-Shot Prompting.** Since the release of GPT-3 (Brown et al., 2020), one of the main paradigms
for eliciting what models know has been zero-shot prompting (Liu et al., 2022; Beltagy et al., 2022).
Zero-shot exploits how language models are trained to predict diverse data from the internet, which
incidentally includes tasks such as question-answering. If prompted appropriately, this can be used to
solve various useful tasks with reasonable performance (Brown et al., 2020). However, these models
are trained to imitate human-generated data, which bounds the quality of their outputs.


Many methods improve upon vanilla zero-shot prompting (Liu et al., 2022; Zhao et al., 2021; Lu et al.,
2022; Wei et al., 2022b; Min et al., 2022a). While our goal is not to improve zero-shot performance,
some of the ideas underlying these methods are similar to CCS. Particularly relevant are methods
that also leverage unsupervised consistency properties, such as Jung et al. (2022); Zhou et al. (2022).
However, these methods still bootstrap from language model outputs trained via imitation learning,
which limits their applicability to our main goals.


Method RoBERTa DeBERTa GPT-J T5 UQA T0 _[∗]_ Mean _[∗]_

Calibrated 0-shot 64.3(6.2) 76.3(6.0) 56.0(5.2) 58.8(6.1) 80.4(7.1) 90.5(2.7) 67.2(6.1)
CCS 62.1(4.1) **78.5(3.8)** 61.7(2.5) **71.5(3.0)** **82.1(2.7)** 77.6(3.3) **71.2(3.2)**
CRC (TPC) **65.7(4.9)** 77.0(4.9) 60.8(3.2) 68.3(4.6) 78.8(3.0) 65.3(6.8) 70.1(4.1)
CRC (BSS) 63.6(5.5) 77.9(4.9) **61.9(2.3)** 67.4(2.2) 80.0(3.4) 79.0(5.1) 70.2(3.7)


Table 2: We compare CCS to two variants of Contrastive Representation Clustering: TPC, which
clusters by projecting onto the top principal component, and BSS, which clusters by finding a direction
that looks bimodal. We show accuracy and standard deviation of each model averaged across all
prompts and datasets, in the same setting as Table 1. We find that CCS generally performs the best,
but that all methods are competitive with zero-shot.


8




---PAGE BREAK---

