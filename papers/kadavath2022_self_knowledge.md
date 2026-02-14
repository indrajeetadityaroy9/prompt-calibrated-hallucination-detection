## **Language Models (Mostly) Know What They Know**

**Saurav Kadavath** _[∗]_ **, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez,**


**Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston,**


**Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai,**


**Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson,**


**Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson,**


**Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph,**


**Ben Mann, Sam McCandlish, Chris Olah, Jared Kaplan** _[∗]_

### Anthropic


**Abstract**


We study whether language models can evaluate the validity of their own claims and predict
which questions they will be able to answer correctly. We first show that larger models are
well-calibrated on diverse multiple choice and true/false questions when they are provided
in the right format. Thus we can approach self-evaluation on open-ended sampling tasks
by asking models to first propose answers, and then to evaluate the probability "P(True)"
that their answers are correct. We find encouraging performance, calibration, and scaling
for P(True) on a diverse array of tasks. Performance at self-evaluation further improves
when we allow models to consider many of their own samples before predicting the validity of one specific possibility. Next, we investigate whether models can be trained to
predict "P(IK)", the probability that "I know" the answer to a question, without reference
to any particular proposed answer. Models perform well at predicting P(IK) and partially
generalize across tasks, though they struggle with calibration of P(IK) on new tasks. The
predicted P(IK) probabilities also increase appropriately in the presence of relevant source
materials in the context, and in the presence of hints towards the solution of mathematical
word problems. We hope these observations lay the groundwork for training more honest
models, and for investigating how honesty generalizes to cases where models are trained
on objectives other than the imitation of human writing.


_∗_ Correspondence to: {saurav, jared}@anthropic.com
Author contributions are listed at the end of the paper.




---PAGE BREAK---

**Contents**


**1** **Introduction** **3**


1.1 Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4


1.2 Models and Evaluation Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5


1.3 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6


**2** **Larger Models are Calibrated on Diverse Multiple Choice Questions** **7**


**3** **From Calibration to Knowing What You Know** **8**


3.1 Replacing an Option with ‘None of the Above’ Harms Performance and Calibration . . . . . 8


3.2 Models are Well-Calibrated on True/False Tasks . . . . . . . . . . . . . . . . . . . . . . . . 10


3.3 RLHF Policy Miscalibration Can Be Remediated with a Temperature Tuning . . . . . . . . 11


**4** **Ask the AI: Is your proposed answer True or False?** **11**


4.1 Basic Self-Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11


4.2 Showing Many _T_ = 1 Samples Improves Self-Evaluation . . . . . . . . . . . . . . . . . . . 12


**5** **Training Models to Predict Whether They Can Answer Questions Correctly** **14**


5.1 Evaluating P(IK) Training and Model Size Trends . . . . . . . . . . . . . . . . . . . . . . . 14


5.2 Out of Distribution Generalization of P(IK) . . . . . . . . . . . . . . . . . . . . . . . . . . 14


5.3 P(IK) Generalizes to Account for Source Materials . . . . . . . . . . . . . . . . . . . . . . 18


5.4 P(IK) Generalizes to Account for Hints Towards GSM8k Solutions . . . . . . . . . . . . . . 19


5.5 Comparing Models Trained with Distinct Pretraining Distributions . . . . . . . . . . . . . . 20


**6** **Discussion** **22**


6.1 Limitations and Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22


6.2 Broader Impacts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23


**7** **Contribution Statement** **23**


**A** **Metrics, Formatting Details, and P(IK) Training** **24**


**B** **Discriminating What Models Know with Entropy or Loss** **28**


**C** **More P(True) Evaluation Results and Details** **30**


**D** **Mixed-Arithmetic and Function Synthesis Dataset Descriptions** **30**


2




---PAGE BREAK---

0.04


0.03


0.02


0.01


0.00

0.0 0.2 0.4 0.6 0.8 1.0
P(True) of Sampled Answers



0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0



10 [10]

Parameters



**Figure 1** **(left)** We show the overall ability of a 52B language model to evaluate its own proposed answers
(sampled at unit temperature) to questions from TriviaQA, Lambada, Arithmetic, GSM8k, and Codex HumanEval. We have weighted the overall contribution from each of these five datasets equally. We evaluate
20-shot using the method of section 4, where we show the model several of its own samples and then ask for
the probability P(True) that a specific sample is correct. **(right)** We show the improvement in the accuracy
on each sampling task when only including questions where a randomly sampled (unit temperature) response
achieved P(True) > 0.5.


**1** **Introduction**


We would eventually like to train AI systems that are honest, which requires that these systems accurately
and faithfully evaluate their level of confidence in their own knowledge and reasoning. So AI systems must
be able to recognize what they do and do not know, as a prerequisite. In this work, we study the extent to
which Language Models (LMs) possess this ability and how it can be elicited and imparted.


As a starting point, we examine calibration: do the probabilistic predictions from language models match
up with frequencies of occurrence? Language models can produce well-calibrated predictions for token
probabilities on-distribution [Guo et al., 2017]. We show that large language models are also well-calibrated
on a diverse array of multiple choice questions, as long as the questions are formatted appropriately. In
particular, calibration improves with model size and few-shot prompting.


Good calibration opens up the possibility for using models to evaluate the accuracy of their own outputs
(“self-evaluation”). For example, given any open-ended query, we can sample an answer from the model
and then have the model evaluate P(True), the probability that its answer is correct. We may expect selfevaluation to be challenging, because the model may be overconfident that its own samples [1] are correct. Our
self-evaluation procedure nevertheless distinguishes correct and incorrect samples, as summarized in Figure
1. Furthermore, as model size and capabilities increase, models improve at self-evaluation, which suggests
that verification improves faster than generation quality in this context.


We also show that self-evaluation can be improved if we provide a model with _many_ of its own samples,
before asking it to evaluate any single sample. That is, ‘brainstorming other possibilities’ helps large models
to evaluate the validity of a given answer option.


These techniques address a question about the world, as they ask models to evaluate “according to accepted
truth in the wider world (i.e. according to humans), is a particular answer to a question correct?” In the case
of self-evaluation, the proposed answer was provided by the model, but its validity is nevertheless an external
fact.


But we are also interested in having language models attempt to directly evaluate their own state of knowledge. To this purpose, we investigate what happens when we train models to predict whether or not they can
correctly answer questions themselves. This is really a question about the model [2] since we are training the
model to learn what sorts of questions _it_, in particular, can answer.


1Another more subtle issue is that in our setup, models have no way to distinguish the tokens they’ve written from
tokens they were given by a third party.
2This claim is subtle, since the model could be learning “how hard is this question in general?” rather than “do I know
the answer to this?” We partially address this concern in section 5.5.


3




---PAGE BREAK---

0.07


0.06


0.05


0.04


0.03


0.02


0.01


0.00



P(IK) on Out-of-Distribution Evals (52B Model)

|Col1|Col2|Col3|Col4|Col5|Grou<br>Grou|nd Truth P<br>nd Truth P|(IK) > 0.5<br>(IK) < 0.5|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||



0.1 0.2 0.3 0.4 0.5 0.6 0.7
Predicted P(IK)



0.16

0.14

0.12

0.10

0.08

0.06

0.04

0.02

0.00



P(IK) on In-Distribution Evals (52B Model)

|Col1|Col2|Col3|Col4|Col5|Ground Truth|P(IK) > 0.5|
|---|---|---|---|---|---|---|
||||||<br>Ground Truth|<br> P(IK) < 0.5|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||



0.0 0.2 0.4 0.6 0.8 1.0
Predicted P(IK)



**Figure 2** **Left** : We train a P(IK) classifier to predict whether or not a model knows the answer to TriviaQA
questions, and then evaluate on Arithmetic, Python Function Synthesis, and Lambada questions. This histogram shows P(IK) scores exclusively from OOD questions. **Right** : We train a P(IK) classifier on TriviaQA,
Arithmetic, Python Function Synthesis, and Lambada and histogram P(IK) scores for their test sets.


We find that language models can easily learn to perform well at evaluating P(IK), the probability that they
know the answer to a question, on a given distribution (see Figure 3 for an illustration). More intriguingly,
we also find some generalization across tasks, for example from trivia to story completion, math, and code,
though models struggle with calibration OOD. We also observe some other types of generalization: although
P(IK) was only trained on bare questions, it generalizes in such a way that it increases when we provide
source materials that address these questions (for trivia) and when we provide hints for math word problems.


**1.1** **Contributions**


**Calibration:** **Multiple Choice, None of the Above, and True/False**


    - We show that when we use a format with visible lettered answer options, large models are very
well calibrated on diverse multiple choice questions (e.g. from BIG Bench [Srivastava et al., 2022],
MMLU [Hendrycks et al., 2021], and many other evaluations); see Figures 4, 5, and 6.

    - Calibration improves with model size, and it also improves when we pass from zero-shot to few-shot
evaluation; see Figure 4.

    - Replacing an option with ‘none of the above’ reduces accuracy and calibration significantly with our
models (see Figure 7). However, our models are also well calibrated on True/False distinctions (see
Figure 8), with accuracy and calibration also increasing with model capability.

    - We also show that RLHF policies [Bai et al., 2022] naively seem miscalibrated, but with a simple
temperature adjustment they become fairly well-calibrated on several evaluations (Figure 9).


**Self-Evaluation of Model-Generated** _**Samples**_ **, without Finetuning**


We evaluate on the generative tasks TriviaQA [Joshi et al., 2017], Lambada [Paperno et al., 2016], GSM8k

[Cobbe et al., 2021], the Codex HumanEval [Chen et al., 2021], arithmetic problems, and natural function
synthesis problems scraped from GitHub. See Figure 1 for a brief overview of results.


    - Models can self-evaluate whether their own samples are True or False, though this tends to be a more
challenging task (since models tend to find their own samples more plausible). Self-evaluations are
well-calibrated few-shot, though models aren’t as well-calibrated zero-shot. In particular, larger _k_
for _k_ -shot self-evaluation seems to primarily help by improving calibration, rather than by improving
the AUROC for separating correct and incorrect responses. See Figures 10 and 11 in the main text
for representative results, and Figures 28, 29, and 30 in the appendix for complete results.

    - Showing models many of their own _T_ = 1 samples, along with a single sample to evaluate as True/False, can significantly improve their performance (this is somewhat reminiscent of self-consistency
prompting [Wang et al., 2022]).


We conclude that language models can perform well at this task few-shot, with most measures of performance
improving with model size, even though models are being asked to evaluate their own samples.


4




---PAGE BREAK---

**Finetuning to Identify the** _**Questions**_ **Models Can Correctly Answer**


We are also interested in whether models know, or can be taught, to specifically identify _questions_ that they
can and cannot answer, rather than simply evaluating whether answers to questions are in fact correct. Figure
2 provides a brief overview of results.


    - We train models with a value head to predict the probability that they can answer a question correctly, which we refer to as P(IK). We find that models trained on TriviaQA have significant power to
differentiate between math, lambada, and code questions that they can answer; see Figure 12. However, P(IK) tends to be poorly calibrated on these other distributions (see Figure 13). Generalization
is perhaps most clearly illustrated in Figures 16 and 17.


    - We study generalization of P(IK) to the inclusion of source materials and mathematical hints, see
Figures 18 and 19. We find that P(IK) responds appropriately to sources, correct hints, and incorrect
or distracting hints.


    - We compare two models with roughly equal capability, but different pretraining data, in Figure 20.
These models are somewhat better at predicting their own P(IK) rather than that for another model.
We also try "crossing" the P(IK) training data for these models, and find a small effect suggesting
that models generalize better when trained on what they know, rather than on what the other model
knew.


Thus we conclude that we only find partial generalization on this task.


**Glossary:** **Observables and Metrics**


    - **P(True)**     - The probability a model assigns to the proposition that a specific sample is the correct
answer to a question.


    - **P(IK)**     - The probability a model assigns to "I know", i.e. the proposition that it will answer a given
question correctly when samples are generated at unit temperature. In this work, P(IK) is usually
computed using a binary classification head on top of a language model.


    - **Ground Truth P(IK)**     - The fraction of unit temperature samples to a question that are correct.


    - **Calibration Charts**     - We often plot prediction probability vs frequency that a prediction was correct,
see Figure 4 as an example. We use all predictions (not just predictions for the correct answer) and
put the same number of predictions in each bin (rather than using equally spaced bins).


    - **Expected** **or** **RMS** **Calibration** **Error**     - To obtain a single number summarizing calibration, we
compute the ECE as the mean of the absolute difference between probabilistic predictions and frequencies. For ECE we always use 10 bins, with an equal number of predictions in each. What we
call the ECE has also been called the mean absolute deviation calibration error [Lin et al., 2022]. To
more closely match other works, we only use the most likely predictions (for multiple choice) when
computing the ECE, instead of using all predictions. We also include the RMS calibration error,
which is better motivated theoretically, but seems to be less widely used in the literature.


    - **AUROC**     - We sometimes share the area under the receiver operating characteristic (AUROC) discriminating between questions models do or do not know the answer to, or discriminating between
samples a model does or does not identify as correct. This captures discriminative power but is
indifferent to calibration (note chance AUROC is 0.5, and larger scores are better).


    - **(Conditional) Accuracy**     - When models self-evaluate the probability P(True) that their own samples
are valid, we are very interested in the accuracy of the samples that models label as ‘True’ (i.e.
correct), and how this compares to the accuracy on the full distribution of problems in the task.


    - **Brier Score**     - In some cases we observe tradeoffs between discrimination (e.g. best AUROC) and
calibration. Brier scores combine the discriminative power of self-evaluation with calibration (note
the chance Brier score on binary choice tasks is 0.25, and smaller scores are better).


**1.2** **Models and Evaluation Tasks**


Our goal in this study is to evaluate calibration and generalization on a diverse range of tasks. As
such we include all of the multiple choice evaluations in BIG Bench [Srivastava et al., 2022], the MMLU


5




---PAGE BREAK---

**Figure** **3** Examples of P(IK) scores from a 52B model. Token sequences that ask harder questions have
lower P(IK) scores on the last token. To evaluate P(IK) on a specific full sequence, we simply take the P(IK)
score at the last token. Note that we only train P(IK) on final tokens (and not on partial questions).


evaluation [Hendrycks et al., 2021], TruthfulQA [Lin et al., 2021], LogiQA [Liu et al., 2020], and QuALITY [Pang et al., 2021]. We are most interested in open-ended generation, so we study the samplingbased evaluations TriviaQA [Joshi et al., 2017], Lambada [Paperno et al., 2016], the Codex HumanEval

[Chen et al., 2021], GSM8k [Cobbe et al., 2021], some basic arithmetic problems, and a dataset of additional
web-scraped Python function synthesis problems. See Appendix D for more information on our arithmetic
and function synthesis datasets. Ultimately, we would like to train models to express calibrated confidence
levels when generating long-form responses in natural language dialogue.


When we perform few-shot evaluation, we simply stuff the context with randomly chosen examples from
the (test) evaluation itself. For BIG Bench this means all few-shot examples come from within the specific
subtask we are evaluating, though in the case of MMLU we randomize across the entire evaluation, without
respecting subject boundaries.


We study a series of language models with 800M, 3B, 12B, 52B parameters. We do not include smaller
models because they perform poorly on many of the evaluations we consider. The architecture and training
setup for these models is identical to that in [Bai et al., 2022], except that the models we consider here were
pretrained for 850B tokens, rather than the 400B tokens used in that work. For simplicity, we do not study
models that have been finetuned on python code, though our pretraining distribution includes about 10%
code. In section 3.3 we briefly study helpful and harmless RLHF policies finetuned (via the process in

[Bai et al., 2022]) from these language models, but otherwise we only study pure language models. We show
accuracies for our models on a few of the multiple choice tasks we study in Figure 37 in the appendix.


**1.3** **Related Work**


Calibration for general ML predictions, and interventions to improve calibration, have been studied [Nguyen and O’Connor, 2015, Hendrycks and Gimpel, 2016, Nalisnick et al., 2019, Guo et al., 2017,
Hendrycks et al., 2018, Ovadia et al., 2019, Minderer et al., 2021] for some time. Calibration for language
models and QA has also been studied [Desai and Durrett, 2020, Jiang et al., 2021], but typically it has been
found that to achieve good calibration predictions must be adjusted. Selective prediction, where models abstain from answering certain questions, has been studied as well [Varshney et al., 2022]. Recently, the calibration of a wide range of models was analyzed on the diverse BIG Bench suite of tasks [Srivastava et al., 2022],
where it was shown that language model calibration improves with model size. We are indebted to the
BIG Bench collaboration for providing a convenient, huge, and diverse evaluation set. The authors of Gopher

[Rae et al., 2021] briefly studied calibration on MMLU [Hendrycks et al., 2021] and found promising results,
which led us to experiment with a variety of multiple choice formats.


6




---PAGE BREAK---

1.0


0.8


0.6


0.4


0.2


0.0



Calibration: BIG Bench Multiple Choice (5-shot)


0.0 0.2 0.4 0.6 0.8 1.0
Probabilities



BIG Bench Calibration Trends


10 [9] 10 [10]

Parameters



10 [10]


10 [9]



10 1


10 2



**Figure** **4** **(left)** We show calibration curves for various model sizes on all of the multiple choice tasks in
BIG Bench, in the format described in section 2. We include a dashed line indicating perfect calibration.
**(right)** Here we show trends in the expected calibration error on BIG Bench, for both multiple choice and a
separate True/False format (see Section 3.2). We show the RMS calibration error in Figure 21 in the appendix.


Truthfulness [Evans et al., 2021] has been a recent focus of various works, including benchmarks

[Lin et al., 2021] and the incorporation of web search and citation [Nakano et al., 2021, Menick et al., 2022]
into language models. That said, truthfulness focuses primarily on factual accuracy "in the world",
rather than on self-knowledge, or eliciting latent knowledge [Christiano et al., 2021]. We use "honesty"

[Askell et al., 2021] as an umbrella term for ideas including truthfulness, calibration, self-knowledge,
explainability, and non-deceptiveness. Language models finetuned to perform non-language tasks

[Ahn et al., 2022, Dinh et al., 2022] might provide an interesting test-bed for honesty in the future.


Perhaps the work most similar to ours is [Mielke et al., 2020], which is a very interesting application of
metacognition/self-evaluation to improve natural language calibration. Another quite similar work is the very
recent [Lin et al., 2022], where the authors train language models to express their calibration on arithmetic in
words, and also study a signal analogous to P(True).


**2** **Larger Models are Calibrated on Diverse Multiple Choice Questions**


A model makes calibrated predictions if the probability it assigns to outcomes coincides with the frequency
with which these outcomes actually occur. Language models are known to produce calibrated token-level
probabilities. In this section we will see that language models can produce well-calibrated probabilities when
they are asked to choose the correct answer from among several explicit options. We believe calibration
is interesting on its own, but it is especially relevant to honesty, since a model that can produce calibrated
answers to meta-questions like ‘do you know the answer to X?’ must know something about what it knows.
Generally we can use calibration as a way to bootstrap towards self-knowledge.


We find that when multiple choice problems are formatted in this way (as used by e.g. [Rae et al., 2021]):


Question: Who was the first president of the United States?
Choices:
(A) Barack Obama
(B) George Washington
(C) Michael Jackson
Answer:


and we identify the answers only by their labels, as e.g. ‘ (B)’, our largest models tend to produce a wellcalibrated probability distribution among the available options. We show the calibration chart for all multiple
choice BIG Bench tasks, in this format, in Figure 4. As can be seen in Figure 6, models are well-calibrated
(in this format) even for somewhat adversarial datasets like TruthfulQA [3] [Lin et al., 2021], as well as for
QuALITY [Pang et al., 2021] and LogiQA [Liu et al., 2020].


3Though note that in this format, where the model sees its options before making a choice, we do not observe antiscaling with model size.


7




---PAGE BREAK---

BIG Bench Calibration vs Normalized Acccuracy (52B, 5-shot)



1.0





BIG Bench Calibration Variations (52B)


0.0 0.2 0.4 0.6 0.8 1.0
Probability



0.8


0.6


0.4


0.2


0.0


0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 0.200
Expected Calibration Error



0.8


0.6


0.4


0.2


0.0



**Figure** **5** **(left)** We show expected calibration error versus normalized accuracy for all BIG Bench tasks;
the number of problems in each task is represented by the marker sizes. We do not find any noticeable
correlation between accuracy and calibration within BIG Bench. To normalize accuracies we linearly map
chance accuracy to 0, keeping perfect accuracy at 1. **(right)** We compare calibration for several variations
on BIG Bench evaluations: we vary between 0-shot and 5-shot, replace an answer option with "none of the
above", and compare our format with letter-labeled choices to the default BIG Bench formatting.


It is crucial that the model gets to see the answer choices explicitly before choosing amongst them; without
this, we would not expect a calibrated response, due to ambiguities and degeneracies among possible paraphrases and specializations of any given answer option (e.g. "Washington" vs "George Washington, the first
US president"). As can be seen in figure 5, task formatting is important for achieving excellent calibration,
and calibration improves as we pass from 0-shot to 5-shot evaluation. We expect calibration is also easier to
achieve with this format because each answer option corresponds to a single token (this isn’t the case in BIG
Bench by default, see appendix A.4).


To simplify the interpretation of results, we reduce calibration curves to a single number by computing the
expected calibration error (ECE), after binning the predictions in 10 equally-represented bins. On the right
of Figure 4 we show scaling trends for calibration on BIG Bench. We typically find good calibration 0-shot,
without any other prompt, though calibration improves as we pass from 0-shot to few-shot. We find that in
almost all cases, calibration improves with model size and capability. Accuracy on the tasks also improves
with model size, but as can be seen in Figure 5, we do not observe any obvious causal connection between
accuracy and calibration. For details on how we obtain calibration charts and ECE see Appendix A.


In what follows we will work to leverage these calibration results to ask language models to evaluate what
they do and do not know.


**3** **From Calibration to Knowing What You Know**


If language models can answer multiple choice questions in a calibrated way, then one might hope that they
can apply this ability to evaluate their own outputs. That is, we can simply ask models to generate potential
answers to questions, and then have the model evaluate whether any of them are correct. In this section we
will begin to explore this idea by reformulating existing tasks. Then in section 4 we will study self-evaluation.


**3.1** **Replacing an Option with ‘None of the Above’ Harms Performance and Calibration**


We have seen that language models can produce calibrated probabilities for multiple choice questions, at least
when the questions and choices are provided in the right format. However, to achieve this feat the model only
needs to determine the relative weight for several concrete options, when they are compared to each other.


We are more interested in whether the model actually knows whether each of the answer options is correct,
when judged independently. To probe this question, we modified our multiple choice evaluations by replacing
their final option with “none of the above”. This procedure can make questions that do not actually have a
unique correct answer ambiguous or impossible, but for many tasks it should result in a well-defined new
evaluation. In particular this procedure appears to be sensible for the vast majority of questions in MMLU.
Concretely this means that we took questions such as the example in section 2 and replaced them with:


8




---PAGE BREAK---

