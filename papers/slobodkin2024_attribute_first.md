## **Towards verifiable text generation with symbolic references**



**Lucas Torroba Hennigen** _[⋄]_ [,] _[∗]_ **Shannon Zejiang Shen** _[⋄]_ [,] _[∗]_ **Aniruddha Nrusimha** _[⋄]_



**Bernhard Gapp** [†] **David Sontag** _[⋄]_ **Yoon Kim** _[⋄]_



_⋄_ Massachusetts Institute of Technology †Good Data Initiative
[lucastor@mit.edu](mailto:lucastor@mit.edu) [zjshen@mit.edu](mailto:zjshen@mit.edu) [anin@mit.edu](mailto:anin@mit.edu)
[bernhard.gapp@gooddatainitiative.com](mailto:bernhard.gapp@gooddatainitiative.com) [dsontag@mit.edu](mailto:dsontag@mit.edu) [yoonkim@mit.edu](mailto:yoonkim@mit.edu)



**Abstract**


LLMs are vulnerable to hallucinations, and thus their outputs generally
require laborious human verification for high-stakes applications. To this
end, we propose _symbolically grounded generation_ (SymGen) as a simple approach for enabling easier manual validation of an LLM’s output. SymGen
prompts an LLM to interleave its regular output text with explicit symbolic
references to fields present in some conditioning data (e.g., a table in JSON
format). The references can be used to display the provenance of different
spans of text in the generation, reducing the effort required for manual
verification. Across a range of data-to-text and question-answering experiments, we find that LLMs are able to directly output text that makes use
of accurate symbolic references while maintaining fluency and factuality.
In a human study we further find that such annotations can streamline
human verification of machine-generated text.


       - [https://symgen.github.io](https://symgen.github.io)


**1** **Introduction**


Many applications of generative AI involve generating text based on structured data (e.g,
tabular data, surveys, API outputs, etc.) that is known (or assumed) to be trustworthy. For
example, newspapers may be interested in generating summaries of sports games based
on official match statistics, and search providers would want generative search engines to
ground its output in search results. These applications require _conditional_ _text_ _generation_
that is fluent, accurate, and verifiable with respect to the conditioning information. Large
language models (LLMs) have advanced to the point where they can sometimes be relied
upon to generate fluent- and faithful-enough summaries of text and other structured data.
However they remain far from perfect (Liu et al., 2023; Yue et al., 2023), and insofar as
high-stakes applications will almost certainly require a human to manually _verify_ that an
LLM’s generation is accurate with respect to its conditioning information, there is a need to
develop frameworks that maintain the fluent and accurate few-shot generation capabilities
of LLMs while enabling streamlined human verification of the model’s output.


This paper proposes _symbolically_ _grounded_ _generation_ (SymGen) as a step towards verifiable text generation with LLMs. Given a string representation of data in a structured but
human-readable format (e.g., JSON, YAML, etc.), we prompt the LLM to generate the output with explicit symbolic references into the provided data structure instead of generating
the text directly. The references are then rendered using a parser, which can faithfully represent values from the original data, and can moreover be used to include visual cues that
enable a user to readily verify the provenance of a particular span of text (see Fig. 2 for an
overview). In contrast to classic templated approaches for text generation (Kukich, 1983;
McKeown, 1992; McRoy et al., 2000), SymGen instead offloads the template specification
process to an LLM. We exploit the fact that LLMs have likely been sufficiently exposed to
such kind of templated text during pretraining that they are able perform zero- and fewshot text generation with symbolic references across multiple domains.


_∗_
Equal Contribution.


1




---PAGE BREAK---

Experiments on zero- and few-shot data-to-text generation and question answering indicate that LLMs can be prompted to generate fluent text that is interleaved with symbolic
references which explicitly refer to data fields. Further, in a human study, we find that
the annotations enabled by SymGen simplify the process of post-hoc verification, both as
perceived by the user and by reducing the average verification time by 20%.


**2** **Text generation with symbolic references**


**Task** **formulation.** Let _V_ be some vocabulary, and let _F_ be a set of fields in a data
structure. One special case of conditional
generation consists of taking as input some
structured data encoded as field–value tuples _d_ = _{_ ( _fi_, _vi_ ) _}i_ _[n]_ =1 [with] [field] _[f][i]_ _[∈F]_ [,]
value _vi_ _∈V_ _[∗]_, and some instruction _x_ _∈_
_V_ _[∗]_, and then producing an appropriate response _y_ = [ _w_ 1, . . ., _wn_ ] _∈V_ _[∗]_ . One simple technique for this task is to encode the
field–value tuples as regular text (e.g., as a
JSON dictionary as in Fig. 2 (1)) and provide it to an LLM alongside an instruction
(e.g., “Summarize a basketball match given
its statistics in JSON.”).


**Symbolically** **grounded** **generation.** In
this paper, we propose a simple _sym-_
_bolically_ _grounded_ _generation_ (SymGen) ap
Figure 1: Compare a standard LLM
proach for such tasks. The idea is to

generated (A) with a SymGen (B, ours)

first generate a response in _symbolic_ _form_

description of a basketball game, based on

_y_ ˜ _∈_ ( _V_ _∪F_ ) _[∗]_, which interleaves regular

match statistics. SymGen imbues spans of

text and references to fields from the data.

generated text (highlighted in blue) with

For example, in Fig. 2, the response in (2)

symbolic references to the source data,

would be:

enabling easier verification: e.g., when hov
_y_ ˜ = [The, visitor.city, . . ., ).].

ering over a span, the number “30” displays

We can subsequently _render_ _y_ ˜ by substi- a tooltip and link (highlighted in yellow)
tuting each field _fj_ _∈F_ with its corre- indicating the value it is referencing.
sponding value _vj_ _∈V_ _[∗]_ in the data to obtain the final response. For example, in Fig. 2 the symbolic form (2) is rendered into
_y_ = [The, Portland, . . ., ).], which is shown in (3). Importantly, the inclusion of symbolic
references enables fine-grained attribution of portions of the text to their generating fields
in the data. In this sense, SymGen is similar in spirit to citation-enabled LLMs (Liu et al.,
2023; Yue et al., 2023; Gao et al., 2023), except that our approach focuses on providing very
precise and easy-to-validate citations.


**Operationalization.** While there are many ways of operationalizing the approach above
in practice we have found that prompted LLMs are highly effective at this task. Specifically,
we consider two ways of prompting LLMs for symbolically grounded generation:

1. **Direct SymGen** : In this approach, we encode the structured data as a JSON dictionary, and then prompt the LLM to generate its output using Jinja-like syntax, i.e.,
to use {{ field }} to refer to a field in the data.
2. **Indirect** **SymGen** : We also explore a variant of the above, where we prompt an
LLM to generate the response directly without any symbolic references, and afterwards prompt it to convert this response into one with a Jinja-like syntax. Effectively, this consists of first tasking the LLM with building a plan, and then conditioning on this plan to obtain the symbolically grounded response.


We detail the used prompts in App. H.2. Note that in the event that an LLM a Jinja expression that cannot be rendered, one can resolve it to a default value, e.g., “undefined.”


2




---PAGE BREAK---

Figure 2: Overview of the proposed symbolically grounded generation (SymGen) pipeline,
on an example generated by GPT-4. Given the structured data input (1), we use a prompted
LLM to generate a response (2) that contains symbolic references into the data (e.g., the
variable visitor.city marked by A ). A parser then substitutes the symbolic references
with their corresponding values in the structured data (e.g., Portland for visitor.city) to
obtain the final rendered text (3). Our SymGen pipeline can implicitly leverage the input
data to generate prose without any symbolic references (e.g., the span marked by B ). It
can also transform existing variables (e.g., adding up two numbers, marked by C ) to create
unavailable data in the source, while preserving references to the original variables.


The advantage of the indirect approach is that it should maintain the same text quality as
regular (non-symbolic) generation as long as converting regular prose into symbolicallyaugmented text is no harder than generating the regular text in the first place; the caveat
is that this approach can be more costly with APIs that charge on a per-token basis, as it
requires roughly twice the amount of tokens to be generated. We also found this strategy
to be unreliable with weaker LLMs such as GPT-3.5, [1] so we only employ this with GPT-4.


**Evaluation.** To evaluate the effectiveness of SymGen, we study its performance along
four axes. The first is whether the _textual_ _quality_ of the final, rendered text is comparable
to the quality of the text generated by a standard, prompted LLM (§3). Then we evaluate
the _citation_ _accuracy_, i.e., whether the symbolic references introduced by SymGen are not
only correct but also non-trivial in the sense that they cannot be added post-hoc by a naïve
baseline (§4.1), and we further assess whether these links actually enable better verifiability
in terms of speed and accuracy (§4.2) via human evaluation. Finally, we explore some extensions of SymGen to question–answering and mathematical reasoning (§5). Collectively
across all studies, our analysis is conducted on five datasets, which are shown in Tab. 8 of
App. H.1. Unless otherwise stated, in all cases we compare against a baseline that consists
of prompting the LLM to generate the response directly, without any symbolic references.


**3** **Textual quality study**


The first empirical study in this work aims to understand whether SymGen generations
exhibit the same quality as a (prompted) LLM baseline; our main goal is not to improve
upon this baseline (although in some cases we observe improvements) as our overarching
goal is to improve generation verifiability without sacrificing performance. For this set
of experiments, we evaluate on two traditional data-to-text datasets (§3.1), and on a new
counterfactual obituary generation (§3.2) dataset. For the former, our focus is in analyzing
the fluency of the resulting text, whereas in the latter we focus on the factuality of the
generated text. We consider both zero- and few-shot settings for each of those tasks.


Depending on the nature of the task (e.g., complexity, size of the structured data, etc.),
we evaluate our approach on a mix of GPT-3.5 (with 4K or 16K context windows) and
GPT-4 (with 8K or 32K context windows). The full experimental setups, including the
prompts, are given in App. H. Anecdotally, we found that extensive experiments with
different prompts were not necessary to get LLMs to perform symbolic generation; we
expect that further improvements are possible with more prompt engineering.


1Specifically, we found that GPT-3.5 would often behave as in the direct SymGen strategy, despite
being prompted and given examples of the desired, indirect SymGen behavior.


3




---PAGE BREAK---

BLEU BERT F1 ER GER


Baseline 31.15 53.43 0.00 0.00
Direct 26.64 47.88 0.28 0.00


Baseline 30.98 54.66 0.00 0.00
Direct 32.43 53.32 3.30 0.09


Baseline 30.08 54.27 0.00 0.00
Direct 33.31 54.07 0.00 0.00
Indirect 35.02 55.58 0.00 0.00


Baseline 32.43 56.39 0.00 0.00
Direct 36.83 57.71 2.11 0.00
Indirect 37.56 57.88 3.21 0.00


(a) Results on SynthBio (Yuan et al., 2021).



BLEU ROU.-L ER GER


Baseline 5.87 19.52 0.00 0.00
Direct 5.07 19.14 4.00 0.00


Baseline 12.26 22.66 0.00 0.00
Direct 9.10 20.65 0.00 0.00


Baseline 4.94 19.91 0.00 0.00
Direct 2.24 17.88 0.00 0.00
Indirect 4.66 19.61 0.00 0.00


Baseline 9.11 22.16 0.00 0.00
Direct 7.02 21.78 0.00 0.00
Indirect 8.29 21.33 2.00 2.00


(b) Results on Rotowire (Wiseman et al., 2017).



Table 1: For SynthBio, we report BLEU (Papineni et al., 2002) and BERTScore F1 (BERT F1;
Zhang et al., 2020) against the reference biographies. For Rotowire, we report the BLEU and
ROUGE (ROU.-L; Lin, 2004) of the generated answers against the reference generations in
the dataset. We also report the (general) error rate (ER) and the global error rate (GER).


3.1 Data-to-text


SymGen is similar in spirit to data-to-text methods, where the goal is to generate some
text based on structured data. Classical approaches to this problem consist of designing a
template based on the schema of the data, and populating it with the values of a datapoint
at runtime (Kukich, 1983; McKeown, 1992; McRoy et al., 2000). However, relying solely
on the schema of the data tends to yield formulaic text. Neural approaches to data-totext generation (Wiseman et al., 2017; 2018; Wang, 2019; Yin & Wan, 2022) improve on this
by generating text that is datapoint-dependent, but achieve this by training on data-to-text
datasets. To this end, we revisit this setting in the context of zero- and few-shot data-to-text
generation with LLMs.


**Datasets.** We first consider SynthBio (Yuan et al., 2021), a collection of synthetically constructed _fictional_ entities, described in terms of a collection of key–value pairs, which has
been used in prior work on templated generation with finetuned models (Zhang et al.,
2022). The task is to generate a textual description of the entity. We also consider the Rotowire (Wiseman et al., 2017) dataset, where the objective is to generate a summary of a
basketball game given its box (individual player statistics) and line (aggregate team statistics) scores. The Rotowire dataset presents a challenging testbed for our approach, since
its data structure is substantially more complex than the previous task (i.e., there are many
more fields in each datapoint) and it relies on more specific in-domain knowledge (i.e., understanding how to read basketball box scores charts). To keep LLM API costs manageable,
we sample 100 examples from the test set to evaluate on. Refer to App. H for more details.


**Experimental details.** On SynthBio, we generate a short biography for each entity in the
test set using GPT-3.5-4K and GPT-4-8K, and we evaluate against reference texts using
BLEU (Papineni et al., 2002) and BERTScore F1 (Zhang et al., 2020). We consider both
zero- and 2-shot learning. See App. H.2.1 for the prompts and example generations. On
Rotowire, we explore both GPT-3.5-16K and GPT-4-32K (the longer context windows are
needed due to the length of the JSON), and evaluate against reference texts using using
BLEU and ROUGE (Lin, 2004). See App. H.2.2 for the Rotowire prompts and examples.


**Results.** Our results are shown in the respective tables for each dataset (SynthBio, Tab. 1a;
Rotowire, Tab. 1b). The results for data-to-text generation are generally positive. For SynthBio, in the few-shot case, we find that both symbolic generation strategies we considered


4




---PAGE BREAK---

EM (%) ROU.-1 ROU.-2 ROU.-L UR ER GER


Baseline 69.05 75.25 53.76 75.23 14.10 0.00 0.00
0-shot
Direct 71.83 76.51 53.89 76.50 14.60 0.00 0.00


Baseline 79.34 83.97 59.87 83.97 5.53 0.00 0.00
2-shot
Direct 78.61 82.90 58.15 82.88 6.76 1.39 0.00


Table 2: GPT-3.5-4K results of the automated question–answering evaluation the obituary
dataset, which includes counterfactually generated entities. We report the exact match
accuracy (EM (%)) and ROUGE (ROU.-1, ROU.-2, ROU.-L; Lin, 2004) of the inferred answers
from the generated text against the true answer given by the source JSON. We also report
the percentage of answers that the QA model did not find an answer for (the unknown
rate; UR), and the regular (ER) and global (GER) error rates. Refer to §3.2 for a discussion.

yield comparable or superior performance to the baseline. [2] Interestingly, in the zero-shot
GPT-3.5-4K case, direct symbolic generation underperforms the baseline, though this gap
is bridged via in-context learning. This could suggest that biographical generation in symbolic space is slightly harder than regular (non-symbolic) generation for GPT-3.5-4K.


For Rotowire, we find that the quality of symbolic generation generally trails the baseline,
though this difference is smaller when using GPT-4-32K and indirect generation. Interestingly, GPT-3.5-16K seems to outperform GPT-4-32K in this task, as evidenced by comparing the baseline results under each model, which suggests that better performance may be
obtained by modifying the indirect strategy so that GPT-3.5-16K is used to generate summaries in regular prose and GPT-4-32K is used to rewrite them using symbolic references.


In many cases, we find that poor symbolic generations arise from rendering errors. To
this end, we also report the percentage of outputs whose symbolic form has at least one
error (a specific reference in a response failed to render locally and was instead rendered
as “undefined”; ER) and with at least one global error (where the Jinja parser simply failed
to run, causing the whole response to fail and be replaced with “The text is not available.”;
GER). We find that (i) GPT-3.5 tends to commit more errors than GPT-4 and (ii) providing
few-shot examples tends to reduce errors, both of which are expected, though we also
find that the indirect SymGen strategy leads to more errors than the direct strategy. We
believe this to be because adapting regular, non-symbolic text to have symbolic references
can sometimes be hard if the JSON is incomplete or the original text is not written in a way
that is amenable to the insertion of references.


3.2 Counterfactual text generation


There is evidence that LLMs are capable of memorizing their training data (Carlini et al.,
2019; 2023) and that this leads to their struggling to generate _counterfactual_ data that goes
against their learnt priors (Hernandez et al., 2023). In this section, we explore whether
SymGen retains the same performance as the baseline when some of the conditioning data
is counterfactual in nature.


**Dataset.** We collect a dataset comprised of 167 famous scientists who lived between 1800
CE and 2000 CE (e.g., Albert Einstein, Carl Sagan, Louis Pasteur, etc.) and further generate
_counterfactual_ variants of each entity. The counterfactual variant is designed to test for the
extent to which an LLM may ignore data that contradicts the information an LLM has seen
during training, which has been noted to be a common failure mode of smaller language
models (Hernandez et al., 2023). For more details refer to App. H.


**Experimental** **setup.** We use GPT-3.5-4K to generate obituaries for each of these (possibly counterfactual) entities. For the few-shot experiments, we provide two examples (see
App. H.2.3 for our prompts and some example generations). Unlike SynthBio, we have
no reference biographies for these entities, so we devise a new evaluation procedure that
tries to measure the factuality of the summaries. Specifically, we wrote questions for each
property in the schema of the data, and then prompted GPT-3.5-4K to answer them using
only the rendered text (see App. A for details). If the answer was not present in the text,


2For comparison, TempLM (Zhang et al., 2022)—which finetunes a pretrained LLM on the full
training set—attains 40.30 BLEU and 54.30 in BERTScore, which are comparable to our GPT-4 results.


5




---PAGE BREAK---

we asked the model to answer with “Unknown.” (the percentage of questions answered in
this way is reported by the unknown rate; UR). We then computed both exact match accuracy and ROUGE of the provided responses against the response specified in the JSON; we
include the latter as it is provides more leeway in the exact phrasing of the response than
the exact match metric.


**Results.** The counterfactual obituary results are shown in Tab. 2. We find that in the zeroshot case, SymGen slightly outperforms our baseline, whereas in the few-shot case we find
that it slightly underperforms the baseline. However, when we condition on whether the
model gave a response versus stating it did not know the answer (Tab. 6), we find that
performance is further improved in the zero-shot case and comparable in the few-shot
setting. One possible explanation for this is that symbolic generation includes slightly less
information than regular generation (especially in the few-shot case, as seen by the higher
unknown rate), which in turn leads to more incorrect answers, since more questions were
left unanswered. That is, SymGen seems to favor precision at the expense of recall, which
may be desirable in some applications.


**4** **Verifiability Study**


4.1 Assessing symbolic reference accuracy


We first aim to determine whether the fields attributed to different spans of text are correct
given their context. This is important since, in principle, symbolic references may be incorrect even when the final text is accurate; for example if two fields in our data have the same
value, but a symbolic reference is made to the wrong field, then the rendered text would
be correct but not the reference.


**Metric.** For a given generation _y_ ˜, we measure the _accuracy_ of the generated symbolic references. The accuracy _P_ is defined by comparing whether a generated field _f_ [ˆ] _j_ _∈F_ is the
same as the intended _fj_ _∈F_ given the context: _m_ [1] [∑] _[j]_ [ 1] [[][ ˆ] _[f][j]_ [=] _[f][j]_ []][, where] _[ m]_ [ is the number of]

symbolic references in a given generation.


**Regex baseline.** Besides SymGen, we also use a simple regular expression-based baseline
to generate symbolic references. In short, for each field _fi_ in the data, we search for its
corresponding value _vi_ in the text and attribute any of its occurrences to _fi_ .


**Evaluation.** We sample 20 test examples from the Rotowire (Wiseman et al., 2017) dataset,
which is a challenging testbed for this study due to the lengthy data associated with every generation. For each example, we compare the direct SymGen generation to the regex
baseline, using either GPT-3.5-16K or GPT-4-32K as the base LLM. Three of the authors of
this paper independently annotated the symbolic references in each example according to
whether it was judged to be contextually correct or not, resulting in in 9068 annotations
in total, with an inter-annotator agreement of 97.8%. Overall, we find that when the accuracy of SymGen is 99.77% and 99.52% with GPT-3.5-16K and GPT-4-32K as the base LLMs,
respectively, whereas the one the baseline is 35.40% and 46.10%. This highlights that not
only SymGen is extremely precise when generating symbolic references, but also that such
accuracy cannot be obtained with simple, post-hoc methods.


4.2 Human evaluation of improved verifiability


Having established that textual quality is mostly unaffected by SymGen (§3) and that the
symbolic references are correct and non-trivial (§4.1), we now turn to whether our symbolically grounded generations actually aid in verifiability. To do so, we conduct a human
study to ascertain whether the annotations enabled by SymGen actually aid users in verifying LLM generations. We once again focus on Rotowire since it is a challenging testbed
for verification: summaries contain many numbers referring to a wide range of fields, and
slight errors might be hard to spot.


**Annotation data.** We picked five SymGen-generated (Indirect, GPT-4-32K) summaries of
different games in the Rotowire dataset and manually verified them for correctness. We
then generated three version of each document that were inconsistent with the data by


6




---PAGE BREAK---

picking a random symbolic reference and replacing it with another symbolic reference with
a different value that was within 2 units of the original value of the field. We prevent the
resulting value from being nonsensical, e.g., we restrict percentages to be within 0 and 100.
For each of these four documents, we created two versions: a version with annotations,
and one without any annotations (i.e., as if it were generated by a regular LLM).


**Survey** **overview.** We ask annotators to annotate four summaries: two of them contain
SymGen annotations, allowing them to inspect the provenance of different numbers in the
text when hovering over a number using their cursor, and two containing no annotations
(i.e., the output of a regular LLM). Each summary is equally likely to come from a pool of
correct or incorrect summaries, and the first step of each annotation is for the participant
to determine whether the summary is correct (i.e., there are no inconsistencies between
the summary and the source table) or incorrect (i.e., there is an inconsistency between the
summary and the table). Crucially, this first step is timed. Answering this question reveals
two more questions, asking the annotators to rate their confidence in their answer, and
how easy it was for them to reach a decision, on a Likert scale. We ask these questions in
order to measure whether SymGen annotations meaningfully impact (i) human accuracy
at finding errors and (ii) how easy it is to verify a summary, where both _objective_ ease (i.e.,
how long did it take to reach a decision) and _subjective_ ease (i.e., how hard did it _feel_ to
go through and verify the text). At the end of the study, annotators are provided an exit
survey were they are asked to rate whether they prefer annotations (over no annotations)
with respect to both (i) the confidence in their decisions and (ii) their ease in reaching a
decision. We also reward annotators for each question answered correctly to ensure that
they are spending adequate effort on the task. More details on the study design, exact
questions, and interface are given in App. E.


**Enrollment.** We recruited 60 annotators for the study via Prolific. See App. E for more
details, including selection criteria, quality filters, etc.


**Results.** We find that annotators were equally likely to identify errors regardless of
whether SymGen annotations were shown. However, when annotations were displayed,
they took on average 20% less time to reach a decision about whether an error was present
or not. This makes sense, since the annotations do not change whether the text is correct or
not, they should only facilitate the process of verifying it. We further find that annotators
feel 5% more confident of their final answers when using our approach and perceive the
verification task to be 14% easier. Finally, from exit survey, we have that 71.67% of participants agree that annotations made them more confident in their answers, and 83.33%
agree that annotations made the verification task easier. In all, the results suggest that
SymGen annotations enable a faster and more pleasant verification experience, with some
annotators reporting “The annotations help tremendously”, “[t]he annotations were nice
and definitely helped”, and “[...] I could still compare the data without annotations it just
took longer to find each piece of data”. For more details, refer to Tab. 7 in App. G.


**5** **Extensions to other use-cases**


In the preceding sections, we established that the textual quality of SymGen is comparable
to a regular prompted LLM (§3), that its annotations are accurate and non-trivial (§4.1)
and that SymGen indeed aids in verifiability (§4.2). Now, we turn to some extensions of
SymGen to question–answering (QA) and mathematical reasoning use-cases.


5.1 Question answering over structured data


Another possible application of symbolically grounded generation is to enhance verifiability in QA over structured data. To evaluate this setting, we construct a dataset of 32
finance-related questions about particular companies (e.g., “How does the book value of
NFLX compare to that of ASML?”), coupled with structured company information. The
results in Tab. 3a suggest that with GPT-4, all models offer reasonably comparable text
quality across generation strategies, though SymGen underperforms when using GPT-3.5.
Refer to App. B for more details.


7




---PAGE BREAK---

0-shot 3-shot


Baseline 93.75 90.63
GPT-3.5-16K
Direct 65.63 68.75


Baseline 93.75 87.50
GPT-4-32K Direct 87.50 87.50
Indirect 90.63 93.75


(a) Results on the financial QA setting.



GSM8K GSM-hard


GPT-4 GPT-3.5 GPT-4 GPT-3.5


CoT 95.0 81.0 64.0 53.5
PAL 95.0 82.0 79.5 73.0


SymGen 95.0 79.0 75.0 60.5


(b) Results on GSM8K.



Table 3: For financial QA, we report the acceptability of the answers in the zero- and 3-shot
settings. Refer to App. B for details. For GSM8K, we compare SymGen reasoning with
other reasoning methods on GSM8K and GSM-hard; all approaches were evaluated using
GPT-4-8K. Refer to §5.2 for a discussion.


5.2 Mathematical reasoning


Besides providing symbolic references to
fields in the source data, SymGen can also
be used to express _symbolic_ _computations_
over the variables, e.g., computing the halftime points of a basketball game based
on the first two quarter scores (Fig. 2).
By chaining a series of such operations,
it opens up the possibility of interleaving arithmetic operations within languagebased chain-of-thought reasoning. We explore whether this new capability comes
without an overall textual quality penalty.



**Experimental setup.** Fig. 3 illustrates one
approach for performing reasoning via
symbolic generation for a math problem in
GSM8K (Cobbe et al., 2021). Each generated computation step is coupled with an
assignment statement in Jinja, which relates the natural language explanation of
the computation with a symbolic expression. Compared to chain-of-thought reasoning (CoT; Wei et al., 2022), explicit use
of symbolic computations should lead to
more easily verifiable computational results of each step; compared to programaided language models (PAL; Gao et al.,
2022) and program of thoughts prompting (PoT; Chen et al., 2023), which recasts
problems into programs and executes it to
obtain a response, SymGen relies more on
natural language as a scaffold, embedding
symbolic computation within regular text.



Figure 3: Illustration of SymGen reasoning
on GSM8K, on an example generated by
GPT-4. Given a math question (1), the LLM
answers via direct symbolic generation, creating variables as needed (2), which can be
rendered in a user-friendly manner (3). The
syntax we use (Jinja) allows the creation of
variables based on the source text (e.g., setting total_people to 90, A ), and performing computation by referencing existing variables (e.g., calculating total_groups based on
total_people and group_size, B ). We can
moreover explain how a computation step relates to previous ones ( C ).



**Dataset.** We compare the direct SymGen ables (e.g., calculating total_groups based on
strategy against CoT and PAL on two total_people and group_size, B ). We can
datasets: GSM8K (Cobbe et al., 2021) and moreover explain how a computation step reGSM-hard (Gao et al., 2022). GSM8K are lates to previous ones ( C ).
grade school math problems like the one
illustrated in Fig. 3 that typically require
multiple steps of reasoning to solve. To create a more challenging testbed for LLMs, Gao et al. (2022) construct GSM-hard by replacing, for each problem in GSM8K, a randomly selected number with a large random number
of up to seven digits. We use a random subset of 200 problems in the test set of GSM8K



8




---PAGE BREAK---

