## **Gemini in Reasoning: Unveiling Commonsense in** **Multimodal Large Language Models**



**Yuqing Wang**
Stanford University
ywang216@stanford.edu


**Abstract**


The burgeoning interest in Multimodal Large
Language Models (MLLMs), such as OpenAI’s
GPT-4V(ision), has significantly impacted both
academic and industrial realms. These models enhance Large Language Models (LLMs)
with advanced visual understanding capabilities, facilitating their application in a variety of
multimodal tasks. Recently, Google introduced
Gemini, a cutting-edge MLLM designed specifically for multimodal integration. Despite its
advancements, preliminary benchmarks indicate that Gemini lags behind GPT models in
commonsense reasoning tasks. However, this
assessment, based on a limited dataset (i.e., HellaSWAG), does not fully capture Gemini’s authentic commonsense reasoning potential. To
address this gap, our study undertakes a thorough evaluation of Gemini’s performance in
complex reasoning tasks that necessitate the
integration of commonsense knowledge across
modalities. We carry out a comprehensive analysis of 12 commonsense reasoning datasets,
ranging from general to domain-specific tasks.
This includes 11 datasets focused solely on language, as well as one that incorporates multimodal elements. Our experiments across four
LLMs and two MLLMs demonstrate Gemini’s
competitive commonsense reasoning capabilities. Additionally, we identify common challenges faced by current LLMs and MLLMs
in addressing commonsense problems, underscoring the need for further advancements in
enhancing the commonsense reasoning abilities
of these models. Our data and results are available at: [https://github.com/EternityYW/](https://github.com/EternityYW/Gemini-Commonsense-Evaluation/)
[Gemini-Commonsense-Evaluation/.](https://github.com/EternityYW/Gemini-Commonsense-Evaluation/)


**1** **Introduction**


Commonsense reasoning, integral to human cognition, plays a crucial role in navigating the intricacies of everyday life. Consider a scenario
where someone decides what to wear based on the
weather. This decision extends beyond the mere selection of attire; it involves understanding weather



**Yun Zhao**
Meta Platforms, Inc.
yunzhao20@meta.com


patterns, the suitability of clothing for different temperatures, and the social context of the occasion.
It’s about synthesizing diverse pieces of knowledge:
a forecast predicting rain, the practical necessity
for a raincoat, and the societal expectation of dressing appropriately for an event. This reasoning goes
beyond simply processing information; it entails
integrating varied pieces of knowledge that humans
often take for granted. A major challenge in Natural Language Processing (NLP) research is the
ambiguity and under-specification of human language. Individuals rely heavily on their commonsense knowledge and reasoning abilities to decipher
these ambiguities and infer missing information.
Commonsense reasoning has consistently posed
unique challenges in NLP research (Li et al., 2021;
Bian et al., 2023), encompassing spatial, physical,
social, temporal, and psychological aspects, along
with an understanding of social norms, beliefs, values, and the nuances of predicting and interpreting
human behavior (Liu and Singh, 2004). Models often lack this innate commonsense, hindering their
ability to contextualize data coherently, in stark
contrast to the human capacity for effortlessly understanding everyday situations (Shwartz and Choi,
2020; Bhargava and Ng, 2022).


Recent advances in Large Language Models
(LLMs) have sparked unprecedented enthusiasm
in the NLP community and beyond, significantly
enhancing a wide array of applications (Min et al.,
2021; Zhao et al., 2023; Wang et al., 2023; Kasneci et al., 2023; He et al., 2023). Building
on these achievements, Multimodal Large Language Models (MLLMs) have emerged as a pivotal focus in the next wave of AI (Wu et al.,
2023b), speculated to advance towards Artificial
General Intelligence (AGI), which aims to develop AI systems smarter than humans and beneficial for all of humanity (Rayhan et al., 2023).
The rise of MLLMs, particularly OpenAI’s GPT4V(ision) (Yang et al., 2023) and Google’s Gem



---PAGE BREAK---

ini (Team et al., 2023), marks significant progress
in this area. Among these developments, Gemini
emerges as a formidable challenger to the stateof-the-art MLLM, GPT-4V, specially engineered
for multimodal integration. Its release has ignited
constructive discussions about the current level of
AGI achievement. In widely used academic benchmarks, Gemini has attained new state-of-the-art status in the majority of tasks. However, preliminary
evaluations of Gemini, especially when compared
to models like the GPT series, have indicated potential shortcomings in its commonsense reasoning
capabilities, a fundamental aspect of human cognition. Yet, it’s important to consider that basing the
assessment of Gemini’s commonsense reasoning
abilities solely on the HellaSWAG dataset (Zellers
et al., 2019b) may not comprehensively reflect
Gemini’s full scope in this critical domain.


To address the gap in the comprehensive evaluation of Gemini’s real-world performance in commonsense reasoning tasks, our study conducts extensive experiments across 12 commonsense reasoning datasets, covering a broad spectrum of domains such as general, physical, social, and temporal reasoning. We experiment with four popular LLMs for the language dataset evaluation, including Llama2-70b (Touvron et al., 2023), Gemini Pro (Team et al., 2023), GPT-3.5 Turbo, and
GPT-4 Turbo (OpenAI, 2023). For the multimodal
dataset, we assess both Gemini Pro Vision and
GPT-4V. Our key findings are summarized as follows: (1) Overall, Gemini Pro’s performance is
comparable to that of GPT-3.5 Turbo, demonstrating marginally better average results across 11 language datasets (1.4% higher accuracy), though it
lags behind GPT-4 Turbo by an average of 8.2% in
accuracy. Moreover, Gemini Pro Vision exhibits
lower performance than GPT-4V on the multimodal
dataset, except for temporal-related questions. (2)
Approximately 65.8% of Gemini Pro’s reasoning
processes are evaluated as logically sound and contextually relevant, indicating its potential for effective application in various domains. (3) Gemini Pro
encounters significant challenges in temporal and
social commonsense reasoning, indicating key areas for further development. (4) Our manual error
analysis reveals that Gemini Pro often misunderstands provided contextual information, accounting
for 30.2% of its total errors. Furthermore, Gemini
Pro Vision struggles particularly with identifying
emotional stimuli in images, especially those in


volving human entities, which constitutes 32.6% of
its total errors.
In summary, our contributions are threefold:


(1) We provide the first thorough evaluation of
Gemini Pro’s efficacy in commonsense reasoning tasks, employing 12 diverse datasets
that span both language-based and multimodal
scenarios.


(2) Our study reveals that Gemini Pro exhibits
performance comparable to GPT-3.5 Turbo in
language-only commonsense reasoning tasks,
demonstrating logical and contextual reasoning processes. However, it lags behind GPT-4
Turbo in accuracy and encounters challenges
in temporal and social reasoning, as well as in
emotion recognition in images.


(3) Our findings lay a valuable foundation for future research in the field of commonsense reasoning within LLMs and MLLMs, highlighting the necessity to enhance specialized domains in these models and the nuanced recognition of mental states and emotions in multimodal contexts.


**2** **Commonsense Overview**


Commonsense reasoning, a fundamental aspect of
human intelligence, facilitates an intuitive understanding and interpretation of the world through
basic and often implicit knowledge and beliefs. For
instance, it involves understanding that a person
carrying an umbrella on a cloudy day likely anticipates rain, or inferring that a closed door in a
library signifies a need for quiet. In MLLMs, commonsense reasoning plays a vital role, enabling
these models to interact with and interpret human
language and visual cues in a manner that mirrors
human understanding. In our study, we explore a
variety of commonsense reasoning tasks. Definitions for each domain are provided as follows.
**General Commonsense.** This domain entails an
understanding of basic, everyday knowledge about
the world, such as recognizing that birds typically
fly and fish live in water.
**Contextual Commonsense.** This domain involves
interpreting information within specific contexts,
such as understanding that a person wearing a coat
and shivering is likely cold.
**Abductive Commonsense.** This domain is about
formulating the most plausible explanations for




---PAGE BREAK---

observations, such as inferring that wet streets are
likely due to recent rain.
**Event** **Commonsense.** This domain focuses on
understanding sequences of events and the causal
relationships between them, such as predicting that
eating spoiled food can lead to feeling sick.
**Temporal Commonsense.** This domain involves
understanding time-related concepts, such as the
fact that breakfast is typically eaten in the morning.
**Numerical Commonsense.** This domain is about
understanding numbers in everyday contexts, such
as knowing that a cube has six faces.
**Physical Commonsense.** This domain concerns
understanding the physical world, such as knowing
that a glass will break if dropped on a hard floor.
**Science Commonsense.** This domain involves the
application of scientific principles in daily life, such
as understanding that water boils at a higher temperature at sea level than in the mountains.
**Riddle** **Commonsense.** This domain challenges
creative thinking through riddles, such as deciphering a riddle where the answer is “a shadow”, requiring lateral thinking to associate intangible concepts
with physical entities.
**Social Commonsense.** This domain involves understanding social interactions, such as recognizing
that a person is likely upset if he/she is crying.
**Moral** **Commonsense.** This domain deals with
evaluating actions based on moral and ethical standards, such as understanding that stealing is generally considered wrong.
**Visual Commonsense.** This domain involves interpreting and understanding visual information in
the context of the physical and social world, such
as deducing that a person in a photo is likely running a race if they are wearing a number bib and
surrounded by other runners.


**3** **Experimental Setup**


**3.1** **Datasets**


We experiment with 12 datasets related to different types of commonsense reasoning, which include 11 language-based datasets and one multimodal dataset. The language-based datasets encompass three main categories of commonsense
reasoning problems: **General** **and** **Contextual**
**Reasoning:** (1) CommonsenseQA (Talmor et al.,
2019), focusing on general commonsense knowledge; (2) Cosmos QA (Huang et al., 2019), emphasizing contextual understanding narratives, (3)
_α_ NLI (Bhagavatula et al., 2019), introducing ab


ductive reasoning, which involves inferring the
most plausible explanation; and (4) HellaSWAG,
centering around reasoning with contextual event
sequences. **Specialized and Knowledge Reason-**
**ing:** (1) TRAM (Wang and Zhao, 2023b), testing
reasoning about time; (2) NumerSense (Lin et al.,
2020), focusing on numerical understanding; (3)
PIQA (Bisk et al., 2020), assessing physical interaction knowledge; (4) QASC (Khot et al., 2020),
dealing with science-related reasoning; and (5) RiddleSense (Lin et al., 2021), challenging creative
thinking through riddles. **Social and Ethical Rea-**
**soning:** (1) Social IQa (Sap et al., 2019), testing
the understanding of social interactions; and (2)
ETHICS (Hendrycks et al., 2020), evaluating moral
and ethical reasoning. For the multimodal dataset
(vision and language), we select VCR (Zellers et al.,
2019a), a large-scale dataset for cognition-level visual understanding. For datasets like TRAM and
ETHICS, which include multiple tasks, we extract
the commonsense reasoning part for experiments.
We employ accuracy as the performance metric for
all datasets. Table 1 provides an overview of the
datasets, as well as example questions.


**3.2** **Models**


We consider four popular LLMs for languagebased dataset evaluation, including the opensource model Llama-2-70b-chat (Touvron et al.,
2023) as well as the closed-source models Gemini
Pro (Team et al., 2023), GPT-3.5 Turbo, and GPT-4
Turbo (OpenAI, 2023). Each of these models is
accessed using its corresponding API key. Specifically, we query Gemini through Google Vertex
AI, the GPT models through the OpenAI API, and
Llama2 through DeepInfra. For the multimodal
dataset, we consider GPT-4V (gpt-4-vision-preview
in API) and Gemini Pro Vision (gemini-pro-vision
in API) in our experiments. Given the constraints
of API costs and rate limitations, we randomly select 200 examples from the validation set for each
language-based dataset following (Wang and Zhao,
2023b) and 50 examples from the validation set for
the VCR dataset following (Liu and Chen, 2023).
For all evaluations, we employ greedy decoding
(i.e., temperature = 0) during model response generation. Notably, there are instances where the
models decline to respond to certain queries, particularly those involving potentially illegal or unethical content. Sometimes, models provide answers
that are outside the scope of the options. In these




---PAGE BREAK---

cases, we categorize these unanswered questions
as incorrect.


**3.3** **Prompts**


In the evaluation of language-based datasets, we
employ two prompting settings: (1) zero-shot
standard prompting (SP) (Kojima et al., 2022),
which aims to gauge the models’ inherent commonsense capabilities in linguistic contexts, and (2)
few-shot chain-of-thought (CoT) prompting (Wei
et al., 2022), implemented to observe potential enhancements in the models’ performance. For the
multimodal dataset, we utilize zero-shot standard
prompting to assess the authentic end-to-end visual
commonsense reasoning abilities of MLLMs.


**4** **Results**


**4.1** **Overall Performance Comparison**


Table 2 demonstrates the accuracy comparison of
four LLMs under zero-shot SP and few-shot CoT
settings on 11 language-based commonsense reasoning datasets. There are several key takeaways.
First, from the model perspective, GPT-4 Turbo
outperforms the other models across the majority of datasets on average. Under the zero-shot
learning paradigm, it surpasses Gemini Pro, the
second-best performing model, by 7.3%, and shows
an even greater lead of 9.0% under the few-shot
learning paradigm. Gemini Pro exhibits marginally
higher average accuracy than GPT-3.5 Turbo, with
an increase of 1.3% under zero-shot SP and 1.5%
in the few-shot CoT scenario. It also demonstrates substantially better performance than Llama2-70b. Regarding prompting methods, the CoT approach consistently enhances performance across
all datasets, with pronounced gains observed in
datasets such as CommonsenseQA, TRAM, and
Social IQa. Lastly, from a dataset standpoint, it
is apparent that while these models exhibit commendable performance across a broad spectrum of
commonsense domains, they encounter challenges
in specific areas, particularly those involving temporal (TRAM) and social (Social IQa) dimensions
of commonsense reasoning.
For the multimodal VCR dataset, we report the
performance of GPT-4V and Gemini Pro Vision in
Table 3. The VCR consists of three subtasks: (1)
Q _→_ A, which involves generating an answer to a
question based on the visual context; (2) QA _→_ R,
which requires the model to produce a rationale for
a given answer; and (3) Q _→_ AR, which challenges



the model to both answer the question and justify
the response with appropriate rationales. In all subtasks, GPT-4V demonstrates superior performance
compared to Gemini Pro Vision, indicating a more
robust capacity for integrating visual and textual
information to provide coherent responses. In Q _→_
AR, the relatively lower performance of both models, compared to the other two subtasks, suggests
that there is considerable room for improvement
in understanding the interplay between visual cues
and commonsense reasoning.


**4.2** **Effects of Commonsense Domain**


Referring to Section 3.1, we have categorized 11
language-based datasets into three groups and presented the performance for each setting within each
group in Figure 1. Our findings indicate that GPT-4
Turbo consistently leads in performance across all
categories. The Llama-2-70b model demonstrates
marginally lower accuracy in comparison to the
other models. Gemini Pro and GPT-3.5 Turbo display comparable performances; however, Gemini
Pro slightly outperforms GPT-3.5 Turbo in two of
the three categories. Notably, its performance dip
in the Social and Ethical Reasoning group may
stem from its tendency to refuse to answer questions that could potentially involve unethical content, which we have counted as incorrect in our evaluation. Based on our experiments, among the 200
samples, Gemini Pro refuses to answer 3.0% of the
problems (6 in total) in the Social IQa dataset and
6.5% of the problems (13 in total) in the ETHICS
dataset. Overall, all models exhibit robust capabilities in handling Social and Ethical Reasoning
datasets, suggesting a relatively advanced grasp of
moral and social norms. However, there is a notable
disparity in their performance on General and Contextual Reasoning tasks, indicating a potential gap
in their understanding of broader commonsense
principles and their application in varied contexts.
The Specialized and Knowledge Reasoning category, particularly in the realms of temporal and
riddle-based challenges, highlights specific deficiencies in the models’ abilities to process complex
temporal sequences and to engage in the abstract
and creative thought required to decipher riddles.

Regarding the multimodal dataset, Figure 2 details the comparative performance between GPT4V and Gemini Pro Vision across different question types, in alignment with the guidelines of the
VCR dataset (Zellers et al., 2019a). We concen



---PAGE BREAK---

Table 1: Overview of commonsense datasets used in our experiments. “K-Way MC” signifies a multiple-choice
response format with K options. Bold text in the “Example Questions” column represents the correct answers.


**Dataset** **Domain** **Answer Type** **Example Questions**


General and Contextual Reasoning


Where is a doormat likely to be in front of?
CommonsenseQA general 5-Way MC
(A). facade; **(B). front door** ; (C). doorway; (D). entrance porch; (E). hallway.

Given the context “It wasn’t time for my book to be released... I have received
Cosmos QA contextual 4-Way MC about five rejection letters.” What may be the reason for your book getting rejected?
(A). None of the above choices; (B). I never...; (C). I felt...; **(D). It wasn’t fnished.**

Given the beginning of the story: Four Outlaws camped in Blood Gulch,
_α_ NLI abductive 2-Way MC and the end of the story: He arrested them, what is the more plausible hypothesis:
(A). They found where the sheriff was; **(B). The sheriff found where they** **were.**



HellaSWAG event 4-Way MC



Given the context “A boy in an orange shirt is playing a video game. the scene”
and the activity label “Washing face”, which of the following endings is the most
appropriate continuation of the scenario? (A). changes to safety features;
**(B). changes to the game itself** ; (C). switches to show...; (D). cuts to the boys...


Specialized and Knowledge Reasoning



Then the green ball told the orange ball that blue ball was stupid. How long was the
TRAM temporal 3-Way MC
green ball talking to the orange ball? (A). 5 weeks; (B). 24 hours; **(C). 15** **seconds** .

Complete the sentence by filling in <mask> with the most appropriate number.
NumerSense numerical Number
A classical guitar has <mask> strings. _→_ **six**

To reach the physical goal: trees, choose the more sensible solution:
PIQA physical 2-Way MC
(A). provide homes for people; **(B). provide homes for animals.**

Crabs scavanage and uses dead snail shells for what? (A). RNA; **(B). homes** ; (C). making
QASC science 8-Way MC
holes; (D). damage; (E). a hosta; (F). Protein; (G). matter; (H). building a nest.

Something very helpful if you want to go gently down a stream. (A). raft; (B). roll
RiddleSense riddle 5-Way MC
down hill; **(C). rowboat** ; (D). water; (E). roll over.


Social and Ethical Reasoning


Kai observed and improved Jan’s work. What will Kai want to do next?
Social IQa social 3-Way MC
(A). be effective; **(B). get** **praise** ; (C). get incredible.

After I got the flu virus I called my friends to tell them that I had the virus.
ETHICS moral 2-way MC
Is this acceptable? **(A). acceptable** ; (B). unacceptable.


Visual Understanding and Reasoning



VCR visual 4-Way MC



1. What is wrong with Person 2? (A). He
is not happy with what is being said to
him over the telephone; **(B). He is feeling**
**depressed** ; (C). He is high on pot; (D).
Someone has pushed him and he’s falling.
2. Given the question: What is wrong with
Person 2?, and the answer to the question:
He is feeling depressed, what is the rationale behind this answer? **(A). Person 1 is**
**talking to him probably trying to cheer**
**him up** ; (B). He looks sad and is drinking;
(C). He is walking with his head down;
(D). He is slumped down on bed and his
eyes are closed.




---PAGE BREAK---

Table 2: Performance comparison of four LLMs across 11 language-based commonsense reasoning datasets. For the
k-shot CoT setting, k is set to 5 for the majority of datasets, except HellaSWAG (k=10) and PIQA (k=1). The best
results for the k-shot setting are boldfaced, and for the 0-shot setting, underlined. GPT-4 Turbo outperforms other
models across the majority of datasets under both settings by a large margin. Gemini Pro and GPT-3.5 Turbo exhibit
comparably matched performance overall, with Gemini Pro demonstrating marginally superior commonsense
reasoning capabilities compared to GPT-3.5 Turbo on average.


**Method**

**Dataset**

Llama-2-70b Llama-2-70b Gemini Pro Gemini Pro GPT-3.5 Turbo GPT-3.5 Turbo GPT-4 Turbo GPT-4 Turbo

(0-shot, SP) (k-shot, CoT) (0-shot, SP) (k-shot, CoT) (0-shot, SP) (k-shot, CoT) (0-shot, SP) (k-shot, CoT)


CommonsenseQA 72.0 76.5 76.5 79.0 73.0 76.0 78.0 **80.0**

Cosmos QA 77.0 81.0 81.5 84.5 75.0 78.5 86.5 **88.0**

_α_ NLI 77.5 80.5 79.5 81.5 75.5 78.0 87.0 **88.0**

HellaSWAG 73.0 77.0 76.0 78.5 78.0 80.0 94.0 **95.0**

TRAM 66.0 70.0 73.5 76.0 68.5 72.0 79.5 **82.0**

NumerSense 74.0 75.5 80.0 82.0 81.5 82.5 85.0 **86.0**

PIQA 74.0 78.5 89.0 90.5 87.0 89.5 94.5 **95.5**

QASC 78.0 82.0 80.0 82.5 83.0 85.0 91.5 **92.5**

RiddleSense 62.5 66.0 75.0 82.5 71.5 75.0 94.0 **95.0**

Social IQa 71.0 77.5 73.0 78.5 73.0 78.0 82.0 **84.5**

ETHICS 88.0 89.5 87.0 87.5 94.0 95.0 97.0 **98.0**


Average 73.9 77.6 79.2 82.1 78.2 80.9 88.1 **89.5**



Table 3: Performance comparison between GPT-4V
and Gemini Pro Vision on the VCR dataset. “Q _→_
A” evaluates question-answering accuracy, “QA _→_ R”
assesses answer justification, and “Q _→_ AR” measures
the performance of both correctly answering questions
and selecting rationales. GPT-4V outperforms Gemini
Pro Vision across all subtasks.


**Method** **Q** _→_ **A** **QA** _→_ **R** **Q** _→_ **AR**


GPT-4V 80.0 72.0 56.0
Gemini Pro Vision 74.0 70.0 48.0


trate on the “Q _→_ A” subtask as it most directly
assesses the models’ visual commonsense capabilities. Considering the data sample for each type,
Gemini Pro Vision’s performance either matches or
is slightly lower than GPT-4V’s, except in temporaltype questions, where it surpasses GPT-4V. This
suggests its enhanced capability not only in recognizing but also in contextualizing time-related
elements within visual scenarios.


**4.3** **Reasoning Justification within MLLMs**


To assess the reasoning capabilities of MLLMs,
particularly their ability to provide not only correct
answers but also sound and contextually grounded
reasoning in matters of commonsense, we adopted
a systematic sampling approach. For each of the
11 language-based datasets evaluated with four
LLMs, we randomly selected 30 questions that
were correctly answered and 30 questions that were



incorrectly answered by each LLM following (Bian
et al., 2023). In cases where a dataset presented
fewer than 30 incorrect answers, we included all
available incorrect responses to ensure comprehensive analysis. After selecting these questions, we
prompted each model to explain “ _What is the ra-_
_tionale_ _behind_ _the_ _answer_ _to_ _the_ _question?_ ” The
reasoning processes provided by the models were
then manually reviewed and classified as either
True or False, based on their logical soundness and
relevance to the question. Figure 3 illustrates a comprehensive view of the average reasoning correctness across the 11 datasets, in terms of the sampled
correct and incorrect questions. In fact, not every
model had 30 incorrect questions for each dataset.
In such scenarios, we scaled the available data up
to 30 questions to ensure standardized computation. Figure 3 shows that GPT-4 Turbo’s leading
performance in both correct and incorrect answers
highlights its advanced reasoning mechanisms and
its ability to maintain coherent logic, even when the
final answers are not accurate. Additionally, Gemini Pro has emerged as a notably proficient model,
generally demonstrating commendable reasoning
abilities and offering a well-rounded approach to
commonsense reasoning. GPT-3.5, while trailing
slightly behind Gemini Pro, still demonstrates competitive reasoning abilities. Figure 4 presents two
real examples from Gemini Pro and GPT-3.5, illustrating the cases of a correct answer with a correct
rationale and an incorrect answer with an incorrect




---PAGE BREAK---

79.5


78.8


78.4

79.5

80.0


80.9


78.3


78.1

80.8



83.5


82.7

83.0


83.5



Llama-2-70b (0-shot, SP)


Llama-2-70b (few-shot, CoT)


Gemini Pro (0-shot, SP)


Gemini Pro (few-shot, CoT)


GPT-3.5 Turbo (0-shot, SP)


GPT-3.5 Turbo (few-shot, CoT)


GPT-4 Turbo (0-shot, SP)


GPT-4 Turbo (few-shot, CoT)



91.3


60 65 70 75 80 85 90 95
Average Accuracy (%)



70.9



74.9


74.4


75.4





86.5


86.4



87.8



88.9

89.5


90.2



Figure 1: Average model performance across three major commonsense reasoning categories over 11 languagebased datasets, including General and Contextual Reasoning (CommonsenseQA, Cosmos QA, _α_ NLI, HellaSWAG),
Specialized and Knowledge Reasoning (TRAM, NumerSense, PIQA, QASC, RiddleSense), and Social and Ethical
Reasoning (Social IQa, ETHICS). GPT-4 Turbo consistently exhibits superior performance in all commonsense
reasoning categories. Gemini Pro marginally surpasses GPT-3.5 Turbo in the first two categories, except for Social
and Ethical Reasoning.



rationale, respectively.
Moving to the multimodal perspective, our analysis of GPT-4V and Gemini Pro Vision on the VCR
dataset reveals notable patterns in reasoning correctness. With GPT-4V at 24% and Gemini Pro
Vision at 26%, approximately one-quarter of the
cases showed both models correctly identifying
the answers but failing to provide appropriate rationale. This discrepancy suggests that while the
models can often determine the correct outcomes,
their ability to understand or explain the underlying
reasoning behind these answers is not consistently
aligned. Furthermore, in the instances of incorrect
answers, GPT-4V and Gemini Pro Vision showed
correct rationales 16% and 22% of the time, respectively. This indicates that, despite arriving at
incorrect conclusions, the models demonstrate a capacity for effective reasoning or logical processing.
However, this does not consistently translate into
accurate outcomes, implying that while some aspects of the required knowledge are captured, other
crucial elements are likely missed.


**4.4** **Case Study:** **Gemini Pro in Commonsense**


Given our focus on evaluating the commonsense
reasoning capabilities of the Gemini Pro model,
we conduct a qualitative analysis to assess its per


formance across representative examples in four
major categories (three language-based and one
multimodal), as described in Section 3.1. To ensure an authentic end-to-end capability evaluation,
we present examples under the zero-shot learning
setting, employing standard prompting techniques.


**General (CommonsenseQA).** In the general commonsense evaluation (General and Contextual
Reasoning category) using the CommonsenseQA
dataset, consider the example question: “People are
what when you’re a stranger? (A) train (B) strange
(C) human (D) stupid (E) dangerous.” Gemini Pro
correctly chose (B) “strange,” and its reasoning process is notable. It recognized that while all options
relate to the concept of a “stranger”, only “strange”
accurately encapsulates the neutral and open-ended
nature of the question. The model effectively ruled
out other options: (A) “train”, for being too specific
and unrelated; (C) “human”, as accurate but not
capturing the question’s essence; (D) “stupid”, for
being judgmental and offensive; and (E) “dangerous”, due to its negative connotation. This selection
of “strange” demonstrates an understanding of the
unfamiliar nature associated with strangers, highlighting Gemini Pro’s capability in interpreting and
applying general commonsense knowledge appro



---PAGE BREAK---

100


90


80


70


60


50


40


30


20



























|100.0 100.0<br>GPT-4V<br>Gemini Pro Vision<br>83.3 82.482.4 83.3<br>75.0 75.0<br>66.7 66.766.7<br>50.0 50.050.0|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision||||||||||
|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision||||||||||||||||||||||
|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision||||||||||||||||||||||
|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision||||||||||||||||||||||
|83.3<br>82.4<br>75.0<br>66.7<br>100.0<br>50.0<br>83.3<br>66.7<br>82.4<br>50.0<br>66.7<br>75.0<br>50.0<br>100.0<br>GPT-4V<br>Gemini Pro Vision||||||||||||||||||||||
|orm|acti<br>nce|vity<br> com|e<br> par|xpla<br> ison|natio<br> bet|n<br>h<br> ee|ypot<br> n GP|hetica<br>  T-4|l<br>Q<br>  an|me<br>uestio<br>  d G|ntal<br>n Ty<br>   emi|pe<br>   i P|ro<br>   o V|le<br>    sion|on|sce<br>     the|ne<br>     VCR|da|tem<br>      taset|poral<br>     , cat|egor|


question type, with a focus on the “Q _→_ A” subtask. Within our sample of 50 questions, the distribution across each
type is as follows: activity (12), explanation (16), hypothetical (3), mental (4), role (5), scene (4), and temporal (6).
GPT-4V matches or surpasses Gemini Pro Vision in performance across these question types, with the exception of
the temporal category.



30


25


20


15


10


5















0

|Col1|Col2|
|---|---|
|||


|Col1|Col2|
|---|---|
|||


|Col1|Col2|
|---|---|
|||


|Col1|Col2|
|---|---|
|||

Llama2-70b Gemini Pro GPT-3.5 Turbo GPT-4 Turbo


Figure 3: Average reasoning correctness across 11 language datasets. The comparison among four LLMs is
based on a random sample of 30 correct and 30 incorrect
questions per dataset. In cases where a dataset contained
fewer than 30 incorrect questions, the data were scaled
up to maintain consistency in the sample size.


priately.
**Temporal (TRAM).** In the temporal commonsense
evaluation (Specialized and Knowledge Reasoning
category) using the TRAM dataset, consider the
example question: “He also promises to ‘come to’
him. How long does it take for him to ‘come to’
him? (A) 100 years (B) in a minute’s time (C) a



few hours.” Lacking sufficient context, especially
regarding the identities involved and the meaning
of ‘come to’, Gemini Pro was unable to provide
a definitive answer. Gemini Pro’s response highlights a significant aspect of its temporal reasoning
capabilities. It illustrates the model’s reliance on
specific contextual information to make accurate
temporal judgments. While this cautious approach
is prudent to avoid incorrect assumptions, it also
signifies a limitation in addressing ambiguous or
incomplete information – a frequent challenge in
real-world communications. This example underlines the difficulties LLMs encounter in temporal
reasoning tasks, especially when faced with scenarios that offer limited or unclear context.

**Social** **(Social** **IQa).** In assessing Gemini Pro’s
performance in social commonsense reasoning using the Social IQa dataset (Social and Ethical Reasoning category), an interesting scenario was presented: “The people bullied Sasha all her life. But
Sasha got revenge on the people. What will the people want to do next? (A) Do whatever Sasha says
(B) Get even (C) Flee from Sasha.” The correct
answer is (C), but Gemini Pro’s response is insightful. It chose (B) “Get even” as the most likely
option, reasoning that the desire for revenge is a
strong motivator and Sasha’s actions likely ignited




---PAGE BREAK---

