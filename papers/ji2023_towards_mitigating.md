## **Reasoning or Reciting? Exploring the Capabilities and Limitations of** **Language Models Through Counterfactual Tasks**

**Zhaofeng Wu** [�] **Linlu Qiu** [�] **Alexis Ross** [�] **Ekin Akyürek** [�] **Boyuan Chen** [�]

**Bailin Wang** [�] **Najoung Kim** [�] **Jacob Andreas** [�] **Yoon Kim** [�]

�MIT �Boston University
zfw@csail.mit.edu



Code Exec.


sorted(





Code Gen.



Basic Syntax



Logic



GPT-4
Performance


Default


Counterfactual


Spatial



Arithmetic
random











and substantially underperforms on counterfactual variants compared to default task instantiations.



**Abstract**


The impressive performance of recent language models across a wide range of tasks
suggests that they possess a degree of abstract reasoning skills. Are these skills general and transferable, or specialized to specific tasks seen during pretraining? To disentangle these effects, we propose an evaluation framework based on “counterfactual”
task variants that deviate from the default assumptions underlying standard tasks. Across



a suite of 11 tasks, we observe nontrivial
performance on the counterfactual variants,
but nevertheless find that performance substantially and consistently degrades compared to the default conditions. This suggests that while current LMs may possess
abstract task-solving skills to an extent, they
often also rely on narrow, non-transferable
procedures for task-solving. These results
motivate a more careful interpretation of language model performance that teases apart
these aspects of behavior.




---PAGE BREAK---

**1** **Introduction**


The striking empirical successes of language models (LMs) suggest that next-word prediction at scale
may be a viable approach for distilling the knowledge embedded in large-scale text corpora into
general-purpose interactive agents. LMs obtain impressive results on various NLP benchmarks (OpenAI, 2023; Anil et al., 2023; Anthropic, 2023; _i.a._ )
and display surprising abilities that suggest a nontrivial understanding of the world (Bubeck et al.,
2023). They have been shown to pass professional
exams (Kung et al., 2023; Nori et al., 2023; Terwiesch, 2023; _i.a._ ), exceed state-of-the-art methods
on many traditional benchmarks (Sun et al., 2023;
Sobania et al., 2023; Zhang et al., 2023a; Dhingra
et al., 2023; _i.a._ ), and surpass human performance
on tasks that require seemingly nontrivial reasoning (Chowdhery et al., 2022; Hoffmann et al., 2022;
Malinka et al., 2023; Guo et al., 2023; _i.a._ ).
Ideally, we expect a general-purpose LM to be
able to _generalize_ not only to unseen instances of
known tasks, but to new tasks. Humans, for example, can transfer their knowledge to new instances
and also flexibly adapt to novel tasks (Singley and
Anderson, 1989). To what extent does the performance of current LMs derive from their ability to
deploy task-general reasoning skills, versus their
ability to recognize and recall specific tasks seen
frequently in pre-training?
Past work has focused on instance-level generalization, but this is often complicated by data contamination issues (Dodge et al., 2021; Magar and
Schwartz, 2022; _i.a._ ). In this work, we are interested in the models’ generalizability to new task
variants, which has been less systematically studied
for LMs (though see Li et al. (2022), Mishra et al.
(2022), and Wang et al. (2022b)).
We propose to measure such task-level generalizability by taking tasks on which LMs perform well,
and altering the conditions or rules under which
these tasks are performed. The general reasoning
procedure for these tasks remains the same under
the new conditions, but the specific input-output
mapping functions are changed. We call the new
tasks _counterfactual_ _tasks_, as they deviate from
the default, generally assumed conditions for these
tasks. Figure 1 shows examples: in the top left,
default arithmetic is performed in base-10, while


We release our code, all synthetically generated data, and
[LM interactions (prompts and responses) at https://github.](https://github.com/ZhaofengWu/counterfactual-evaluation)
[com/ZhaofengWu/counterfactual-evaluation.](https://github.com/ZhaofengWu/counterfactual-evaluation)



counterfactual arithmetic is performed in base 9.
If models implement a general and transferable
task-solving procedure, we expect comparable performance on counterfactual and default tasks; if
they employ procedures tailored to default task
conditions, we expect a drop in the counterfactual
performance.


We design a suite of 11 counterfactual evaluation tasks to measure an LM’s flexibility to adapt to
new task variants across multiple categories and domains, as summarized in Figure 1. In each, the original task under the default conditions and its counterfactual variants share the same reasoning procedure but differ in their input-output mappings. We
consider traditional NLP tasks such as deductive
reasoning, non-language tasks that are nonetheless
commonly evaluated such as code generation, as
well as non-standard tasks such as drawing and spatial reasoning. The latter extralinguistic tasks test
whether LMs are able to learn conceptual structures
that mirror the structure of the non-linguistic world,
which has been suggested by recent work (Abdou
et al., 2021; Ilharco et al., 2021; Patel and Pavlick,
2022; Li et al., 2023a; Bubeck et al., 2023; Søgaard,
2023; _i.a._ ).


We evaluate the performance of GPT-4 (OpenAI, 2023), GPT-3.5, Claude (Anthropic, 2023),
and PaLM-2 (Anil et al., 2023) on tasks under both
the default and counterfactual conditions. We observe above-random counterfactual performance
for most tasks, indicating some degree of task generalizability. However, the performance on counterfactual task variants consistently and substantially
degrades relative to the performance on the default
settings. This suggests that these models’ ability
on these tasks is supported at least in part by nontransferable, default-condition-specific behaviors
rather than abstract, generalizable reasoning skills.


These results also reveal several surprising relations between model behavior on default and
counterfactual tasks (§5), including correlations
between default and counterfactual performance,
varying effectiveness of zero-shot chain-of-thought
prompting (Kojima et al., 2023), and interactions
between task- and instance-level frequency effects.
Overall, we find that small variations on the default
instantiations of tasks are challenging for models,
and thus the success of existing LMs on standard
benchmarks should not be considered as sufficient
evidence for their possession of full general capacity for the target task.




---PAGE BREAK---

**2** **Counterfactual Tasks**


We informally conceptualize each task as a function _fw_ : _X_ _→_ _Y_ that maps an **input** _x_ _∈_ _X_
under a **world model** _w_ _∈_ _W_ to an **output** _y_ _∈_ _Y_ .
World models encapsulate the conditions under
which function evaluation takes place. For example, in Python programming, _w_ might specify assumptions of Python such as indexing and operator
precedence; in arithmetic, _w_ could represent the
set of conditions required for an arithmetic operation, such as the number base. We refer to the
set of assumed default conditions, including but
not limited to the base’s being 10, as the **default**
**world**, or _w_ [default] . Intuitively, for any task, _w_ [default]

corresponds to the set of conditions underlying the
majority of task instances in text corpora. [1]

Traditional evaluations of machine learning models assess how closely a model’s learned hypothesis
_h_ estimates _fw_ by independently sampling training and test sets from the population distribution
_Dfw_, and only exposing the model to the training set for learning _h_ . However, in datasets of
scraped web text, these evaluations are subject to
potential data contamination issues (Brown et al.,
2020; Dodge et al., 2021; Magar and Schwartz,
2022; _i.a._ ). These issues may be more severe in
recent LMs: the ever-growing pretraining datasets
potentially expose the models to more evaluation instances, and the increasing sizes of recent LMs give
them more ability to memorize these instances (Carlini et al., 2020; Magar and Schwartz, 2022).
We hence consider another dimension of generalization: generalization to new task variants in
**counterfactual worlds** _w_ [cf], instead of new inputs
_x_ . This allows us to measure the extent to which a
model’s _fw_ default performance is specific to _w_ [default]

or attributable to a general implementation of the
task _f_ . [2] For arithmetic, a possible _w_ [cf] would be
one that was the same as _w_ [default] but assumed a
base other than base-10. We expect a model with
general arithmetic ability to perform similarly in


1This data-generating process can be described by the following generative model, _P_ ( _y | x, w_ ) _P_ ( _x | w_ ) _P_ ( _w_ ). From
the perspective of causal inference, our counterfactual framework can be informally seen as performing a do-operator on
this graph (Pearl, 2009).
2This setup is reminiscent of intensional models of natural
language semantics (Heim and Kratzer, 1998, §12; Von Fintel
and Heim, 2011), where _f_ is analogous to the denotation
function _·_, _x_ to its input, and _y_ to its output. By default, the
denotation is evaluated under the real world, extensionally, but � when a different possible world is specified instead, we expect
a competent system to adjust the evaluation accordingly.



other bases.
We emphasize that our goal is not to find counterfactual world models that are completely outside
the realm of human experience. Base-9 addition,
for example, is not a novel concept. Nor do we
aim to guarantee that counterfactual world models
are unobserved in a pretraining corpus. Instead,
counterfactuals are simply defined as variations on
the _default_ conditions for a task.
Concretely, we assess an LM’s task performance
with 0-shot prompting. We specify the task _f_,
the test instance _x_, and the world model _w_ in a
prompt, parse the LM’s output, and compare it to
the ground-truth label. We denote the LM’s implementation of _fw_ for a given instance _x_ to be,


_h_ ( _f, w, x_ ) = arg max _P_ LM� _y_ _[′]_ _|_ prompt _f_ ( _f, x_ ) _,_
_y_ _[′]_

prompt _w_ ( _w_ )� _,_


where the arg max is computed with an approximate decoding procedure and prompt _f_ and
prompt _w_ are prompt templates that describe tasks
and world models respectively. For each task, we
devise one or more _w_ [cf] that deviate from the default world (i.e., the default task conditions). We
evaluate both _h_ ( _f, w_ [default] _, x_ ) and _h_ ( _f, w_ [cf] _, x_ ) via
task-specific metrics. If we control _fw_ ( _x_ ) to be
similarly hard between _w_ [default] and _w_ [cf], we can
attribute the performance difference to an LM overfitting to the default instantiation of the task.


**2.1** **Counterfactual Comprehension Check**


One potential confounder is that an LM may be
failing at a particular counterfactual task by failing
to understand the prompt component that specifies
the counterfactual conditions, i.e., prompt _w_ ( _w_ [cf] ).
That is, an LM might still be reasoning in _w_ [default]

and completely _ignore_ the instructions. While this
would still be a failure of the LM, it does not necessarily represent a failure to perform the counterfactual task variant. We control for this by designing task-specific **counterfactual comprehension**
**checks** ( **CCC** s) that test an LM’s surface understanding of the specified counterfactual world.
For each (default, counterfactual) task pair, we
introduce another control task _gw_ with input _x_ _[′]_ and
output _y_ _[′]_ that is much simpler than _fw_ but still allows for the discrimination of _w_ [default] from _w_ [cf] (i.e.,
_gw_ cf( _x_ _[′]_ ) = _gw_ default( _x_ _[′]_ )). A high performance of
_P_ LM( _y_ _[′]_ _|_ prompt _g_ ( _g, x_ _[′]_ ) _,_ prompt _w_ ( _w_ [cf] )) would
indicate that prompt _w_ is effective at making the




---PAGE BREAK---

LM perform a task in _w_ [cf] . In the arithmetic example, for a base-9 counterfactual world, we use
the same prompt _w_ (base-9) to specify the counterfactual world, and check that it facilitates an
understanding of _w_ = base-9 by asking what the
next integer after _x_ _[′]_ is. If, for example, it consistently carries over digits greater than 8 and does not
carry over otherwise, this would show the effectiveness of prompt _w_ (base-9). Our CCC designs are
heuristic: as with control tasks in the probing literature (Hewitt and Liang, 2019), we rely on intuition
to craft a _gw_ that is “simpler” than _fw_ . [3]


**3** **Tasks**


In this section, we give a quick overview of the
tasks we consider. See §A for the full description
of each task and §B for all the prompts used.


**3.1** **Arithmetic**


Modern LMs have been shown to possess basic
numerical reasoning abilities (Lewkowycz et al.,
2022), with Brown et al. (2020) even reporting
near-perfect GPT-3 accuracy for two-digit additions. On the other hand, Razeghi et al. (2022) find
that LMs perform significantly better on operations
involving numbers that occur more frequently in
the pretraining data, and Li et al. (2023d) show
that symbol replacement affects the mathematical
ability of BERT (Devlin et al., 2019)-like models;
both findings point to overfitting and memorization
effects. We consider the same two-digit addition
task, the simplest arithmetic task in Brown et al.
(2020), but inspect a model’s accuracy in different
bases. We use base-8, 9, 11, and 16 as the counterfactual setup which are natural generalizations
to base-10 arithmetic. These bases were chosen to
control for task difficulty (see §7.1 for a discussion)
and also to test for how relatively uncommon (9 &
11) and common (8 & 16) bases affect performance
(see §5.1 for an analysis). To ensure the model
understands the different bases, the CCC evaluates
the successor relation under each base.


**3.2** **Programming**


Even without explicit pretraining on large amounts
of code, LMs have been found to possess decent
coding ability (Brown et al., 2020). The inclusion


3In this formulation, LM queries for CCC are separate from
the main task queries. For some tasks, it is more natural to
query about the task and CCC jointly in the same prompt, i.e.,
_P_ LM( _y, y_ _[′]_ _|_ prompt _f_ ( _f, x_ ) _,_ prompt _g_ ( _g, x_ _[′]_ ) _,_ prompt _w_ ( _w_ [cf] )).
We use this formulation instead for those tasks.



of large code corpora in LM pretraining (Gao et al.,
2021; Chowdhery et al., 2022; Touvron et al., 2023;
_i.a._ ) further improves this capability in recent LMs,
with ChatGPT sometimes outperforming state-ofthe-art approaches for bug fixing (Sobania et al.,
2023). Nevertheless, Miceli-Barone et al. (2023)
show that GPT-3 and related models are fragile under identifier swaps in programs, suggesting that
these models may only possess a shallow understanding of code. Here, we inspect an LM’s programming ability through a deeper counterfactual
perturbation: contrary to the traditional 0-based
indexing in Python, we instruct the LM to evaluate
or generate programs under a fictional language,
ThonPy, that uses 1-based indexing but is otherwise
identical to Python. 1-based indexing is a common assumption for other programming languages
such as MATLAB and R and hence provides a fair
testbed. We evaluate the LM’s performance using
the HumanEval dataset (Chen et al., 2021). The
CCC here involves the same program execution
task but on much simpler inputs, such as simple list
indexing, that do not involve deeper reasoning.


**3.3** **Basic Syntactic Reasoning**


Mahowald et al. (2023) distinguish between two
types of LM capabilities: _formal competence_ that
encompasses the knowledge of language, and _func-_
_tional competence_ which involves using language,
potentially combined with extralinguistic capacities, to interact with the world. While the other
tasks we investigate in this paper assess a model’s
functional competence, we also include an evaluation on formal competence. We revisit the
attested syntactic knowledge of LMs (Yu et al.,
2020; Linzen and Baroni, 2021; Ettinger, 2020; Pimentel and Cotterell, 2021; Belinkov, 2022; Lasri
et al., 2022; _i.a._ ) by considering a meta-linguistic
task (Beguš et al., 2023; Hu and Levy, 2023; _i.a._ ):
evaluating LMs in synthetic versions of English
with different word orders from English’s subjectverb-object (SVO) ordering. We ask the LM to
identify the main subject and the main verb of a
sentence under both the original and counterfactual
orders, where the latter is obtained from manipulating dependency trees (Ravfogel et al., 2019). The
CCC requires the model to revert simple reordered
sentences to the original SVO ordering, equivalent
to identifying these elements in a sentence.




---PAGE BREAK---

**3.4** **Natural Language Reasoning with**
**First-Order Logic**


We next consider a deductive reasoning task that
is still based on natural language. Logical reasoning is a prerequisite ability for many complex
tasks (McCarthy, 1959) and has been the focus of
much recent work (Clark et al., 2020; Tafjord et al.,
2021; Saparov and Mitchell, 2022; Saparov and
He, 2023; _i.a._ ). Nevertheless, LMs struggle with
reasoning with premises that are inconsistent with
common sense (Dasgupta et al., 2022; Yu et al.,
2023; Tang et al., 2023). Here, we undertake a similar study from the perspective of counterfactual
analysis to disentangle the effect of common sense
from a model’s actual logical reasoning capability.
Following prior work, we evaluate in an entailment format and ask LMs if a series of premises entails a conclusion. We use the FOLIO dataset (Han
et al., 2022) most of whose premises are consistent with common sense, and manually rewrite
them to violate common sense. We study if LM
performance is affected by the truthfulness of the
premises under which they operate. The CCC directly asks the model if the original or post-rewrite
premise is true, when presented both as options.


**3.5** **Spatial Reasoning**


A major debate around LMs is whether grounded
representations of meaning can be learned from
form alone (Bender and Koller, 2020; Piantadosi
and Hill, 2022; Mollo and Millière, 2023). Studies
have shown that LMs can learn meaningful world
representations through text-only training (Abdou
et al., 2021; Li et al., 2023c; Jin and Rinard, 2023).
In particular, Patel and Pavlick (2022) find that
LMs learn representations of spatial relations and
cardinal directions that can be aligned to grounded
conceptual spaces with few-shot demonstrations.
We similarly investigate an understanding of cardinal directions, but instead of evaluating whether
a model can _induce_ structured conceptual spaces,
we ask if it can _apply_ conceptual spaces to reason
about the locations of objects. Specifically, we
ask an LM for the coordinates of objects whose
positions are described using cardinal directions,
under a conventional 2D coordinate system (e.g.,
where east corresponds to (1 _,_ 0)) versus coordinate systems with swapped, rotated, and randomly
permuted axes. We expect a robust representation
to not be sensitive to such transformations. The
CCC involves asking the model to directly output



the counterfactual cardinal directions.


**3.6** **Drawing**


Despite being trained on only textual data, LMs
have been shown to be able to structure their representations of perceptual concepts such as size and
color (Abdou et al., 2021; Patel and Pavlick, 2022;
Zhang et al., 2020; Ilharco et al., 2021; _i.a._ ) in a
way that credibly mirrors the physical world. Recent LMs can even generate plausible drawings of
objects using code such as TikZ and SVG (Bubeck
et al., 2023; Zhang et al., 2023c). We evaluate the
visual understanding of LMs by asking them to
generate code for drawing various objects in the
Processing language, which Sharma et al. (2024)
found the LMs to be more adept in. Psychological
studies have shown that humans have the ability to
rotate mental representations of objects (Shepard
and Metzler, 1971; Vandenberg and Kuse, 1978).
For the counterfactual settings, we similarly ask the
LM to generate code that draws the same object,
but rotated or vertically flipped. We disallow the
use of functions such as rotate to prevent shortcut
solutions (see §7.2 for further discussion). As with
the spatial reasoning task (§3.5), an ideal model
should be robust to these settings. For the CCC,
we ask the model to draw a straight line at the top
of the canvas in addition to the object; a flipped/rotated line thus signifies an understanding of the
transformations.


**3.7** **Music**


Recent work has shown the potential of large-scale
models for music infilling (Huang et al., 2019a,b)
and generation (Agostinelli et al., 2023; Copet
et al., 2023; Ren et al., 2020). Bubeck et al. (2023)
show that even a text-only LM with no musicspecific pretraining exhibits some musical abilities,
including understanding musical structure and manipulating melodies. We investigate the extent of
LMs’ musical abilities through two tasks.
In the _chord placement_ task, we evaluate whether
LMs can provide the correct chord fret placements
for string instruments with standard or altered
string tunings. The altered tunings, known as _scor-_
_datura_, are typical in music and are used to evoke
a specific sound or effect (e.g., enabling heavier,
deeper sound in metal music). We evaluate LMs
using an existing database [4] that includes chords for
guitar and ukulele. In the counterfactual setting, we


[4https://github.com/tombatossals/chords-db](https://github.com/tombatossals/chords-db)




---PAGE BREAK---

instruct LMs to provide fret placements for a special guitar/ukulele where one or two of the strings
are altered. For guitar, we include drop-D tuning, a
popular alternative guitar tuning that allows us to
investigate whether the frequency of counterfactual
tunings affects results (see §5.1). To check whether
the model has understood the tunings, we ask for
the first three notes on each string (including open
string) as the CCC.
In the _note retreival_ task, we evaluate whether
LMs can retrieve notes from famous melodies (e.g.,
“Twinkle Twinkle Little Star”). The process of rewriting melodies in different keys, referred to as
“transposition,” is common in music (e.g., to accommodate the ranges of different singers or instruments). We evaluate LMs’ musical abilities
under transpositions by prompting them to retrieve
the _n_ -th note in a melody in either its canonical key
(default setting) or a different key (counterfactual
setting). We ask the LMs to retrieve the _n_ -th note
of the scale of the given key as the CCC.


**3.8** **Chess**

Chess playing has long been regarded as a testbed
for AI (Silver et al., 2017; Tomasev et al., 2020),
and modern LMs have exhibited abilities that imply
an understanding of chess rules (Srivastava et al.,
2023; Du et al., 2023). We test this understanding
by asking for the legality of a 4-move opening.
In the counterfactual setting, we swap the initial
positions of knights and bishops—a setup present
in a real-world chess variant “Chess 960”—and
similarly ask LMs for opening legality under this
new starting configuration. [5] We ask for the starting
positions of the knights and the bishops as the CCC.


**3.9** **SET Game**

SET is a popular card game where each card has 4
attributes with 3 different values for each attribute:

  - _color_ : (red, blue, green)

  - _shape_ : (diamond, oval, squiggle)

  - _shading_ : (solid, shaded, open)

  - _number_ : (1, 2, 3)
In each round, a player finds a SET of 3 cards in
a 12-card board whose values for each attribute
are **either** _all the same_ **or** _all unique_ . This game
has been thoroughly studied in computer science,
from the perspective of coding theory and combinatorics (Davis and Maclagan, 2003), linear algebra (Coleman and Hartshorn, 2012), and complex

5A conceptually similar analysis was performed in Li et al.
(2023c) for the game of Othello.



ity theory (Chaudhuri et al., 2003). We suspect
this popularity makes it susceptible to overfitting
by LMs and investigate this possibility. We ask the
LM to identify the card on a board that completes
a 3-card SET with two given cards. In the counterfactual setup, we invert the rule for the _number_
attribute, requiring its value to be mixed, in other
words, **neither** _all the same_ **nor** _all unique_ . For the
CCC, we ask the model for the validity of a SET
under the original rule and the counterfactual rule.


**4** **Results**


For each task, we evaluate GPT-4 (gpt-4-0314;
OpenAI, 2023), GPT-3.5 (gpt-3.5-turbo-0301),
Claude (claude-v1.3; Anthropic, 2023), and
PaLM-2 (text-bison-001; Anil et al., 2023). As
these are closed-source models, we do not have
any information regarding their size, architecture,
and pretaining details. [6] We note that the largest
PaLM-2 model is not publicly accessible, and we
can only test the second-largest version. For each
task, we experiment both with and without encouraging the model to reason step by step, by adding
the phrase “Let’s think step by step.” in our
prompts (Kojima et al., 2023; Reynolds and McDonell, 2021). Following Kojima et al. (2023), we
refer to this step-by-step setup as zero-shot chainof-thought prompting ( **0-CoT** ; Nye et al., 2021;
Wei et al., 2022). We include all prompts in §B.
Figures 2 and 3 show our results. §C contains the
numeric version. We see a consistent pattern where
LMs perform substantially worse on the counterfactual task variants, both with and without 0-shot
CoT. For most cases, LMs exhibit an above-random
counterfactual performance, suggesting some degree of the targeted ability. However, when the
CCC accuracy is high, as is usually the case for
GPT-4 and in select settings for other models too,
the gaps in default vs. counterfactual task performance demonstrate limitations in their abstract capacity to solve the target task. When the CCC accuracy is lower, the failure of counterfactual world
comprehension would be a confounder to this conclusion, but often the gaps are so large (sometimes
even dropping from near-perfect to near-zero, such
as for arithmetic) that they are nonetheless strongly


6We also explored open-source models in preliminary
experiments, but found that they possess unsatisfactory
instruction-following ability, to the point that often their output
cannot be meaningfully parsed into a prediction. We therefore
do not include these models.




---PAGE BREAK---

GPT-4 GPT-3.5 Claude PaLM-2



100


50


0
8 9 10 11 16 8 9 10 11 16


100


50


0
0 1 0 1



100


50


0
8 9 10 11 16 8 9 10 11 16


100


50


0
0 1 0 1



100


50


0
8 9 10 11 16 8 9 10 11 16


PaLM-2’s short context
length often results in
truncated output



Arithmetic

Two-digit addition


Code Exec.

Python program
evaluation


Code Gen.

Python program
generation


Basic Syntax

Main subject and verb
identification


Logic

First-order logic
deduction in
natural language


Spatial

Object coordinate
identification


Drawing

Object sketch
generation


Chords: Guitar

Fret placement
for chords


Chords: Ukulele

Fret placement
for chords



100


50


0
8 9 10 11 16 8 9 10 11 16

Base


100


50


0
0 1 0 1

Index From



0


100


50


0


100


50


0


100


50


0



100


50



100


50



100


50



100


50



100


50



100


50



100


50



100


50



0 1 0 1



0 1 0 1



0 1



0 1 0 1



0 1



0 1 0 1



0 1



0 1



0



0



0



0



100


50


0


100


50



Index From


Word Order



100


50


0


100


50


0
Y N Y N


100


50


0


100


50


0


100


50


0


100


50


0



100


50


0


100


50


0
Y N Y N


100


50


0


PaLM-2 often generates
malformated code


100


50


0


100


50


0



0
Y N Y N

Follow Common Sense?


100


50



100


50


0


100


50


0
Y N Y N


100


50


0


100


50


0


100


50


0


100


50


0



Orientation


Orientation


Tuning



Tuning

w/o 0-CoT w/ 0-CoT CCC Random


**Figure 2:** Main results. The blue and orange bars represent the default and counterfactual conditions respectively,
either with or without 0-shot chain-of-thought (0-CoT) (except code generation; see §A.2). CCC is the counterfactual
comprehension check (§2.1), but when applicable, we report it for the default setting too. Random performance is
marked whenever nontrivial. PaLM-2 here is not the largest version (§4). The CCC for code execution/generation
are identical. For spatial reasoning, we average the results from all rotation degrees. Counterfactual performance is
consistently lower than the default task performance, while CCC is usually high. §C reports numeric results.




---PAGE BREAK---

GPT-4 GPT-3.5 Claude PaLM-2



100


50


0
Y N Y N


100


50


0
Y N Y N


100


50


0
Y N Y N



100


50


0
Y N Y N


100


50


0
Y N Y N


100


50


0
Y N Y N



100


50


0
Y N Y N


100


50


0
Y N Y N


100


50


0
Y N Y N



Melody

_n_ -th note retrieval
in a melody


Chess

Opening sequence
legality identification


SET Game

Identification of
the missing card
from a SET



100


50


0
Y N Y N

C Major?


**100**


**50**


**0**
**Y** **N** **Y** **N**

Regular Board State?


100


50


0
Y N Y N

Regular Rule?



w/o 0-CoT w/ 0-CoT CCC Random


**Figure 3:** Main results (continued). The blue and orange bars represent the default and counterfactual conditions
respectively, either with or without 0-shot chain-of-thought (0-CoT). CCC is the counterfactual comprehension
check (§2.1), but when applicable, we report it for the default setting too. Random performance is marked whenever
nontrivial. PaLM-2 here is not the largest version (§4). Counterfactual performance is consistently lower than the
default task performance, while CCC is usually high. §C reports numeric results.



indicative of non-transferable, default-conditionspecific implementations of the original task. The
fact that the LMs sometimes cannot evaluate the
CCC well under the counterfactual conditions, but
can do so under the default conditions (e.g., for
arithmetic, programming, drawing, etc.) itself also
points to overfitting to the latter.


**5** **Analysis**


We now investigate how a variety of factors affect
the default and counterfactual performance trends
that we observed in §4. Unless otherwise specified,
we only consider GPT-4 with 0-shot CoT, which
has the strongest performance in our results above.


**5.1** **“Commonness” of Counterfactual**
**Conditions**


Our counterfactual worlds are not designed to be
completely alien to the LMs but only less common
than the assumed default case. In this sense, the
counterfactual-ness of these worlds is relative, and
here we take a more nuanced look at how the commonness of these counterfactual conditions affects
the default-counterfactual performance gap. For
example, in the arithmetic task, all models perform
better in bases 8 and 16, likely due to their relative abundance compared to bases 9 and 11. In
spatial reasoning, the smallest counterfactual per


formance degradation is usually from when the
north and south directions are swapped—even exceeding the default task performance for PaLM-2—
potentially because some programming libraries
use an inverted _y_ -axis, such as matplotlib (Python),
ggplot (R), and D3 (JavaScript) (see §A.5). For
chord fingering, the common alternative drop-D
tuning of guitars (DADGBE) leads to the highest
counterfactual performance for GPT-4. These correlations between the counterfactual performance
and the commonness of the counterfactual worlds
paint a more fine-grained picture than a binary default versus counterfactual distinction and point to
a memorization-like effect where the models perform better under more common conditions.


**5.2** **Proximity between Default and**
**Counterfactual Conditions**


Another axis along which the counterfactual worlds
differ is in their proximity to the default conditions. For example, for the different arithmetic
bases, bases 9 and 11 are _closer_ to base 10, but _less_
_common_ than bases 8 and 16. While the defaultcounterfactual gap is most affected by commonness
for the arithmetic task, for the guitar and ukulele
tunings (other than the drop-D tuning), the LM performance generally decreases monotonically with
increasing distance from the original tunings.
The FOLIO dataset (Han et al., 2022) enables an



---PAGE BREAK---

