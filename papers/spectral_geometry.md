## **Spectral Geometry for Deep Learning: Compression** **and Hallucination Detection via Random Matrix** **Theory**

BY


Davide Ettori
B.S. in Computer Engineering, Politecnico di Milano, Milan, Italy, 2023


THESIS


Submitted as partial fulfillment of the requirements


for the degree of Master of Science in Computer Science


in the Graudate College of the


University of Illinois Chicago, 2026


Chicago, Illinois


Defense Committee:
Amit Ranjan Trivedi, Chair and Advisor
Sathya Ravi
Sourav Medya
Marco Brambilla, Politecnico di Milano


### **Accessibility Statement**

An accessible EPUB version of this document can be obtained by contacting Davide
[Ettori at detto3@uic.edu](mailto:detto3@uic.edu)


ii


### **Contents**

Accessibility Statement _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ ii


List of Figures _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ v


List of Tables _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ vi


List of Abbreviations _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ vii


Summary _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ viii


Chapter 1. Introduction _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 1
1. Goals and Research Questions _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 2
2. Thesis Structure _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 3


Chapter 2. Background _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 4
1. Background: Foundations of Deep Learning Architectures _. . . . . . . . . . . . . . . . . . . ._ 4
2. Background: Mathematical Foundations of Spectral Geometry _. . . . . . . . . . . . . . . ._ 8
3. Background: Random Matrix Theory _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 9
4. Background: Hallucinations in AI Models _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 13
5. Background: Compression in Deep Networks _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 14
6. Background: Spectral Methods in Large Models _. . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 15
7. Background: Efficiency and Reliability in Large AI Models _. . . . . . . . . . . . . . . . . . ._ 17


Chapter 3. Related Work _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 19
1. Related: Hallucination Detection _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 19
2. Related: Out-of-Distribution Detection _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 22
3. Related: Model Compression _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 24
4. Related: Random Matrix Theory in Deep Learning _. . . . . . . . . . . . . . . . . . . . . . . . ._ 28
5. Related: Multimodal Models and Hallucination _. . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 30


Chapter 4. EigenTrack: Spectral Detection Framework _. . . . . . . . . . . . . . . . . . . . . . . . . . ._ 34
1. Methodology _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 34
2. Theoretical Justification _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 40
3. Experimental Setup _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 43
4. Results _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 46
5. Ablation Studies _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 50


Chapter 5. RMT-KD: Random Matrix Distillation _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 55
1. Methodology _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 55
2. Theoretical Justification _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 61
3. Experimental Setup _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 64
4. Results _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 68


iii


CONTENTS iv


5. Ablation Studies _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 74


Chapter 6. Conclusions _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 77
1. Findings _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 77
2. Study Limitations _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 78
3. Directions for Future Research _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 80


Appendix A. Reuse of Content Published on arXiv _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 82


Bibliography _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 83


### **List of Figures**

1 Neural network (NN) _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 5

2 CNN architecture _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 6

3 Transformer architecture _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 7

4 Vision-language model (VLM) schematic _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 8

5 Wigner semicircle law _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 10

6 Marchenko–Pastur distribution _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 10

7 Tracy-Widom distribution _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 11

8 Empirical eigenvalue distribution _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 17


9 General architecture of EigenTrack _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 35

10 Layer-level spectral signature extraction _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 36

11 Temporal evolution of spectral statistics _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 39

12 AUROC for hallucination detection _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 46

13 AUROC for OOD detection _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 48

14 AUROC–latency trade-off _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 52

15 AUROC as a function of the number of generated tokens. _. . . . . . . . . . . . . . . . . . . . . ._ 53


16 Overview of the iterative RMT-KD pipeline. _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 56

17 Iterative RMT-KD training process _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 60

18 Evolution of the empirical eigenvalue distribution _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 63

19 Accuracy and parameter reduction across tasks _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 69

20 Inference speedup and power reduction _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 70

21 Memory footprint and energy efficiency _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 71

22 Accuracy–reduction tradeoff for RMT-KD. _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 75


v


### **List of Tables**

1 SOTA COMPARISON ON LLAMA _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 49

2 FULL METRICS ACROSS MODELS _. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 51


3 PERFORMANCE COMPARISON OF RMT-KD. _. . . . . . . . . . . . . . . . . . . . . . . . . . . . ._ 73


vi


### **List of Abbreviations**

**BBP** Baik–Ben Arous–P´ech´e


**CNN** Convolutional Neural Network


**FNN** Feed Forward Neural Network


**GRU** Gated Recurrent Unit


**KD** Knowledge Distillation


**KL** Kullback–Leibler


**LLM** Large Language Model


**LSTM** Long Short Term Memory


**MP** Marˇcenko–Pastur


**OOD** Out of Distribution


**RMT** Random Matrix Theory


**RNN** Recurrent Neural Network


**TW** Tracy–Widom


vii


### **Summary**

The rapid growth of deep learning and large language models has transformed artificial
intelligence across natural language processing, computer vision, and multimodal reasoning.
These advances, however, come with two persistent challenges: reliability and efficiency.
Models can produce hallucinations, misinterpret out-of-distribution inputs, and exhibit unstable behavior that undermines trust in high-stakes domains such as healthcare, law, and
finance. At the same time, their increasing scale demands enormous computational resources,
making them difficult to deploy efficiently and sustainably.
This thesis develops a unifying framework grounded in Spectral Geometry and Random
Matrix Theory (RMT) to address both reliability and efficiency in modern deep learning
systems. By studying the eigenvalue dynamics of hidden activations, we show that spectral
statistics provide a compact and interpretable lens on model behavior, capable of distinguishing structured, causal representations from noise-like variability. Two core contributions are
presented within this framework.
The first contribution, **EigenTrack**, is a real-time detector of hallucinations and outof-distribution behavior in large language and vision-language models. EigenTrack converts streaming hidden activations into spectral signatures, entropy, eigenvalue gaps, and
divergence from the Marchenko–Pastur baseline, and models their temporal evolution with
lightweight recurrent classifiers. This approach enables early detection of reliability failures
before they manifest in model outputs, while offering interpretable insights into the underlying representation dynamics. EigenTrack achieves state-of-the-art performance across
diverse architectures, demonstrating that spectral features provide stable and generalizable
signals for monitoring model trustworthiness.
The second contribution, **RMT-KD**, is a principled method for compressing deep networks through random matrix theoretic knowledge distillation. By identifying outlier eigenvalues in activation spectra as carriers of causal information, RMT-KD progressively projects
networks onto lower-dimensional subspaces while preserving accuracy through iterative selfdistillation. This process yields models that are significantly more compact and energyefficient, yet remain dense and hardware-friendly. RMT-KD attains state-of-the-art tradeoffs between compression and accuracy, outperforming traditional pruning and heuristic lowrank methods by relying on statistically grounded criteria rather than ad hoc thresholds.
Together, these contributions establish spectral geometry as a coherent foundation for
both diagnosing uncertainty and guiding compression in large-scale neural networks. By
linking eigenvalue statistics to representation quality, the thesis demonstrates how Random
Matrix Theory can provide interpretable, mathematically principled tools to make large
language models and related architectures more reliable, efficient, and trustworthy. This
spectral perspective opens new avenues for understanding deep learning systems, bridging
theoretical insights with practical methods for safe and sustainable AI deployment.


viii


CHAPTER 1

### **Introduction**


In recent years, deep learning and large language models have become central to ad

vances in artificial intelligence. Transformer-based architectures now underpin applications


ranging from natural language understanding and text generation to computer vision and


multimodal reasoning. These models have shown remarkable capabilities, often surpassing


human performance on benchmark tasks, and are increasingly being adopted in domains


such as healthcare, law, finance, and education. Their rapid adoption highlights both their


transformative potential and the growing reliance of modern society on machine intelligence.


However, the success of these systems comes with critical challenges. On the one hand,


large language models often suffer from reliability issues: they can produce hallucinations,


generate plausible but incorrect outputs, and fail dramatically when faced with out-of

distribution inputs. Such behavior undermines trust in AI, particularly in safety-critical


contexts where decisions must be both accurate and justifiable. On the other hand, the


efficiency of these models is a major obstacle. State-of-the-art architectures contain billions


of parameters, requiring enormous computational resources for training and inference. This


makes them difficult to deploy on edge devices, costly to maintain in large-scale production,


and unsustainable in terms of energy consumption.


Addressing these dual challenges of reliability and efficiency requires new methods that


are not only empirically effective but also grounded in mathematical principles. One promis

ing perspective arises from spectral geometry and Random Matrix Theory (RMT), which


provide a rigorous way to analyze the eigenvalue spectra of hidden activations in neural


networks. Eigenvalue statistics can reveal whether a model is encoding structured, causal


1


1. GOALS AND RESEARCH QUESTIONS 2


information or drifting toward noise-like behavior, offering a compact and interpretable de

scription of its internal dynamics. By leveraging these spectral insights, it becomes possible


to both diagnose failures in real time and guide principled approaches to model compression.


This thesis develops such a spectral framework for modern deep learning and large lan

guage models. It introduces two contributions built on Random Matrix Theory: EigenTrack,


a method for real-time hallucination and out-of-distribution detection, and RMT-KD, a


knowledge distillation approach for efficient network compression. Together, these methods


demonstrate that spectral analysis of neural representations can provide a unifying founda

tion for making AI systems simultaneously more trustworthy and more efficient.


**1. Goals and Research Questions**


The inspiration for this research arises from prior work conducted at the AEON Lab at


the University of Illinois Chicago, where spectral methods were first explored as a tool to


understand the behavior of large neural networks. Building on this foundation, the present


thesis investigates how spectral geometry and Random Matrix Theory can provide inter

pretable, mathematically grounded solutions to the pressing challenges of modern artificial


intelligence. The need for such approaches is clear: large language models and other deep


architectures are increasingly deployed in critical applications, yet they remain prone to


hallucinations, brittle under distributional shifts, and prohibitively costly to scale. These


limitations highlight the necessity of frameworks that not only improve performance but also


enhance trustworthiness and efficiency in a principled manner.


The central goals of this thesis are therefore twofold. First, to develop methods that


detect and anticipate failures in large language and vision-language models by analyzing the


spectral dynamics of their hidden activations. Second, to design compression strategies that


reduce the computational burden of deep networks without sacrificing accuracy, guided by


rigorous statistical laws rather than heuristics. These goals naturally lead to the following


research questions: Can spectral geometry serve as a compact and interpretable signal of


2. THESIS STRUCTURE 3


reliability in large-scale models? And can Random Matrix Theory provide a principled foun

dation for identifying and retaining the causal structure necessary for efficient compression?


Addressing these questions is the focus of this thesis, with the broader objective of advancing


the reliability and sustainability of modern deep learning.


**2. Thesis Structure**


In Chapter 2 we provide the necessary background to this research, reviewing the the

oretical foundations of spectral geometry, random matrix theory, and the fundamentals of


deep neural networks and large language models. This chapter introduces the mathematical


tools that will be applied throughout the thesis. Chapter 3 surveys related work, present

ing prior approaches to hallucination detection, out-of-distribution recognition, and model


compression. We compare black-box, grey-box, and white-box detection methods, as well


as traditional compression techniques such as pruning, low-rank approximations, and stan

dard knowledge distillation. Chapter 4 introduces the first main contribution of this thesis,


_EigenTrack_, and describes in detail its methodology, design, and experimental evaluation for


hallucination and out-of-distribution detection in large language and vision-language models.


Chapter 5 presents the second contribution, _RMT-KD_, and explains how Random Matrix


Theory can be used to guide causal knowledge distillation for efficient network compres

sion. This chapter also discusses its empirical results across natural language processing and


computer vision benchmarks. Finally, Chapter 6 concludes the thesis by summarizing the


key contributions, outlining limitations, and suggesting directions for future research at the


intersection of spectral theory and deep learning.


CHAPTER 2

### **Background**


In this chapter we review the theoretical and technological foundations of our work.


We first outline modern deep learning architectures, focusing on large language models,


convolutional networks, and vision-language systems. We then introduce the mathematical


basis of spectral geometry and random matrix theory. Next, we discuss two central challenges


in today’s models: reliability, including hallucinations and out-of-distribution errors, and


efficiency, with emphasis on compression and knowledge distillation. We also survey prior


applications of spectral methods in vision and language models, and conclude with a unified


perspective that links efficiency and reliability through spectral geometry.


**1. Background: Foundations of Deep Learning Architectures**


**From Machine Learning to Deep Learning.** Machine Learning (ML) develops al

gorithms that learn patterns from data rather than following explicit rules. Classical ML


methods such as decision trees, support vector machines, and k-nearest neighbors rely on


handcrafted features and work well on structured data. Deep Learning (DL), a subfield of


ML, uses deep neural networks to automatically learn hierarchical representations from raw


inputs. Enabled by large datasets and GPU computing, DL has driven breakthroughs in


computer vision, natural language processing, and multimodal tasks [ **1, 2** ].


**1.1. Neural Networks and Representation Learning.** Feed Forward Neural Net

works (FNNs) consist of only linear layers of interconnected neurons, defined as _h_ [(] _[l]_ [)] =


_σ_ ( _W_ [(] _[l]_ [)] _h_ [(] _[l][−]_ [1)] + _b_ [(] _[l]_ [)] ), that apply weighted sums and nonlinear activations [ **2** ]. Training with


backpropagation allows them to approximate highly complex functions. Hidden layers serve


as representations, mapping raw data to informative latent features. Modern networks are


4


1. BACKGROUND: FOUNDATIONS OF DEEP LEARNING ARCHITECTURES 5


usually overparameterized, with more parameters than training examples. Surprisingly, this


often improves generalization, stability, and representation power [ **3, 4** ].


Figure 1. Neural network (NN): features pass through hidden layers and
nonlinear activations, producing the output.


**Convolutional Neural Networks (CNNs).** CNNs transformed computer vision by


exploiting spatial locality with many convolutional filters following this equation _h_ [(] _i,j_ _[l]_ [)] [=]

��
_σ_ _m,n_ _[W]_ [ (] _m,n_ _[l]_ [)] _[x]_ [(] _i_ + _[l][−]_ _m, j_ [1)] + _n_ [+] _[ b]_ [(] _[l]_ [)][�] . Stacked convolution and pooling layers extract increasingly


abstract features, while linear layers perform classification. Landmark models such as


AlexNet, VGG, and ResNet demonstrated the scalability of CNNs [ **5, 6** ].


**Recurrent Neural Networks (RNNs).** Recurrent neural networks (RNNs) are de

signed to process sequential data by maintaining a hidden state that evolves over time. Given


an input sequence _{xt}_ _[T]_ _t_ =1 [, a vanilla RNN updates its hidden state as] _[ h][t]_ [= tanh(] _[W][xh][x][t]_ [+]


_Whhht−_ 1 + _bh_ ), and produces an output _yt_ = _Whyht_ + _by_ . While conceptually simple, standard


RNNs suffer from vanishing and exploding gradients, which limits their ability to capture


long-range dependencies.


To address this, gated architectures such as the Gated Recurrent Unit (GRU) and Long


Short-Term Memory (LSTM) were introduced. A GRU introduces two gates: the reset


gate _rt_ = _σ_ ( _Wrxt_ + _Urht−_ 1 + _br_ ) and the update gate _zt_ = _σ_ ( _Wzxt_ + _Uzht−_ 1 + _bz_ ). The

candidate hidden state is _h_ [˜] _t_ = tanh( _Whxt_ + _Uh_ ( _rt ⊙_ _ht−_ 1) + _bh_ ), and the final hidden state


1. BACKGROUND: FOUNDATIONS OF DEEP LEARNING ARCHITECTURES 6


Figure 2. CNN: convolutional filters extract features, pooling layers reduce
resolution, and linear layers classify. (Stanford CS231 Notes)


is _ht_ = (1 _−_ _zt_ ) _⊙_ _ht−_ 1 + _zt ⊙_ _h_ [˜] _t_ . These mechanisms allow the GRU to selectively retain or


overwrite information over time.


The LSTM further extends this idea by maintaining a separate memory cell _ct_ alongside


the hidden state _ht_ . It employs three gates: the input gate _it_ = _σ_ ( _Wixt_ + _Uiht−_ 1 + _bi_ ), the


forget gate _ft_ = _σ_ ( _Wf_ _xt_ + _Uf_ _ht−_ 1 + _bf_ ), and the output gate _ot_ = _σ_ ( _Woxt_ + _Uoht−_ 1 + _bo_ ). The


memory cell is updated as _ct_ = _ft ⊙_ _ct−_ 1 + _it ⊙_ tanh( _Wcxt_ + _Ucht−_ 1 + _bc_ ), and the hidden state


is given by _ht_ = _ot ⊙_ tanh( _ct_ ). By explicitly controlling how information is added, forgotten,


and exposed, LSTMs can capture dependencies over long sequences more effectively than


vanilla RNNs or GRUs.


**1.2. Transformers and Large Language Models.** Transformers, introduced by Vaswani


et al. [ **7** ], rely on self-attention to capture long-range dependencies without recurrence. The


attention mechanism computes similarity scores between all pairs of tokens in a sequence,


allowing each token to weight and aggregate information from every other token. This en

ables the model to focus on relevant context regardless of distance. Their scalability has


enabled Large Language Models (LLMs) such as BERT [ **8** ], GPT [ **9** ], and LLaMA [ **10** ].


This architecture has been shown to be effective also for multimodal data.


1. BACKGROUND: FOUNDATIONS OF DEEP LEARNING ARCHITECTURES 7


Figure 3. Transformer architecture [ **7** ]. Each encoder–decoder block is composed of multi-head self-attention layers that model contextual dependencies,
followed by position-wise feed-forward networks. Residual connections and
layer normalization ensure stable training and effective gradient flow, enabling
scalability to very large models.


**Vision-Language Models (VLMs).** VLMs integrate image encoders (CNNs or Vision


Transformers) with text encoders (Transformers) to align visual and textual modalities. This


enables tasks such as image captioning, visual question answering, and cross-modal retrieval.


CLIP [ **11** ] and LLaVA [ **12** ] are prominent examples.


**The Double-Edged Sword of Overparameterization.** Overparameterization is a


typical trait of modern neural networks. It enables models to interpolate training data while


still generalizing well, a phenomenon described by the ‘double descent” curve [ **4** ], but also


brings inefficiency, high energy cost, and deployment challenges. From the perspective of


the bias–variance trade-off [ **13** ], increasing complexity should reduce bias but raise variance,


implying an optimal model size. Yet deep networks often challenge this intuition: despite


extreme overparameterization, they can achieve both low bias and low variance with a suf

ficient amount of training data. Explaining this paradox remains an active area of research,


with spectral methods offering one promising approach.


2. BACKGROUND: MATHEMATICAL FOUNDATIONS OF SPECTRAL GEOMETRY 8


Figure 4. Vision-language model (VLM) schematic. CLIP (Contrastive Language–Image Pretraining) [ **11** ] aligns images and text by projecting them into
a shared latent space. The architecture consists of two encoders: a Vision
Transformer (or CNN) for images and a Transformer for text. During training, the model learns to maximize the cosine similarity between embeddings
of matching image-text pairs while minimizing it for mismatched pairs, using
a contrastive loss. This enables zero-shot classification by comparing the similarity of an image’s embedding to embeddings of text descriptions of potential
classes. CLIP’s ability to generalize to unseen tasks without fine-tuning has
made it a cornerstone for multimodal AI.


**2. Background: Mathematical Foundations of Spectral Geometry**


**2.1. Eigenvalues and Eigenvectors Foundations.** At the foundation of spectral ge

ometry are **eigenvalues** and **eigenvectors** . For a square matrix _A ∈_ R _[n][×][n]_, an eigenvector


_v ̸_ = 0 and corresponding eigenvalue _λ ∈_ R satisfy the relation _Av_ = _λv_ . This identifies


directions in which the linear transformation _A_ acts by pure scaling, without rotation or


distortion. The scalar _λ_ measures the amount of stretching or compression along the eigen

vector. Eigenvalues are obtained by solving the characteristic equation det( _A −_ _λI_ ) = 0,


whose roots _λ_ 1 _, . . ., λn_ represent the eigenvalues of _A_ [ **14** ]. In practice, direct computation


from the determinant is numerically unstable for large matrices, and one typically relies


on methods such as QR decomposition, singular value decomposition (SVD), or iterative


algorithms like the Lanczos procedure and power iteration [ **16** ].


**2.2. Spectral Decomposition.** For symmetric matrices, common in machine learning


(e.g., covariance matrices), the spectral theorem guarantees a decomposition _A_ = _Q_ Λ _Q_ _[T]_,


where _Q_ contains the orthogonal eigenvectors and Λ is a diagonal matrix of eigenvalues.


3. BACKGROUND: RANDOM MATRIX THEORY 9


Thus, any symmetric transformation can be expressed as scaling along orthogonal directions


[ **14** ].


**Interpretation of Eigenvalues.** Eigenvalues provide deep geometric and statistical in

sight. Geometrically, large values correspond to directions in which vectors are stretched,


while small values correspond to nearly collapsed directions. For covariance matrices, eigen

values quantify the variance of the data along the associated eigenvector directions. A few


dominant eigenvalues indicate that most variability lies in a low-dimensional subspace, while


a flatter spectrum reflects variance distributed across many directions. Hence, eigenval

ues reveal intrinsic dimensionality and help distinguish meaningful structure from noise in


high-dimensional representations [ **15** ].


**2.3. Dimensionality Reduction with Eigenvalues.** An important application of


eigenvalues is dimensionality reduction, especially PCA (Principal Component Analysis).


Consider a dataset _X ∈_ R _[n][×][d]_, where each row is a data point in _d_ dimensions. After centering


the data, the covariance matrix is computed as _C_ = [1]

_n_ _[X]_ _[T]_ _[X][ ∈]_ [R] _[d][×][d]_ [. Let] _[ λ]_ [1] _[ ≥]_ _[λ]_ [2] _[ ≥· · · ≥]_ _[λ][d]_


be the eigenvalues of _C_ with corresponding orthonormal eigenvectors _v_ 1 _, . . ., vd_ . To reduce


the dimensionality to _k < d_, we construct the projection matrix _Vk_ = [ _v_ 1 _v_ 2 _· · · vk_ ] _∈_


R _[d][×][k]_ by concatenating the top- _k_ eigenvectors. The reduced representation of the data is


then _Xred_ = _XVk ∈_ R _[n][×][k]_ . This projection maximizes the preserved variance, given by

Var( _Xred_ ) = [�] _i_ _[k]_ =1 _[λ][i]_ [, which is the largest possible among all] _[ k]_ [-dimensional linear projections.]


A more efficient approach uses SVD (Singula Values Decomposition) directly on _X_ to obtain


the principal components without explicitly forming the covariance matrix.


**3. Background: Random Matrix Theory**


**3.1. Introduction to Random Matrix Theory.** Random Matrix Theory (RMT)


studies the spectral properties of large matrices whose entries are random variables. Its cen

tral goal is to characterize the empirical spectral distribution (ESD), defined as the normal

ized histogram of eigenvalues of a random matrix as its dimensions grow. Let **M** _∈_ R _[p][×][p]_ be


a random symmetric matrix with eigenvalues _λ_ 1 _, . . ., λp_ . The empirical spectral distribution


3. BACKGROUND: RANDOM MATRIX THEORY 10


is: _µ_ **M** ( _λ_ ) = _p_ [1] - _pi_ =1 _[δ]_ [(] _[λ][ −]_ _[λ][i]_ [) where] _[ δ]_ [ is the Dirac delta. As] _[ p][ →∞]_ [,] _[ µ]_ **[M]** [ converges (in prob-]


ability) to a deterministic limiting law, such as the Wigner semicircle or Marchenko–Pastur


distribution, depending on the matrix ensemble.


Figure 5. Wigner semicircle law: eigenvalue density of large symmetric random matrices.


Figure 6. Marchenko–Pastur distribution: eigenvalue density of sample covariance matrices.


**Wigner Semicircle Law.** The Wigner Semicircle Law describes the eigenvalue density


of large symmetric matrices with i.i.d. entries of mean zero and variance _σ_ [2] [ **17** ]. Let


3. BACKGROUND: RANDOM MATRIX THEORY 11


**W** _∈_ R _[p][×][p]_ with entries _Wij_ drawn i.i.d. and **W** = **W** _[T]_ . Then, as _p →∞_, the ESD



_√_
converges to: _f_ Wigner( _λ_ ) = 2 _πσ_ 1 [2]



4 _σ_ [2] _−_ _λ_ [2] for _|λ| ≤_ 2 _σ,_ and 0 otherwise . This implies that



the eigenvalues of a purely random symmetric matrix are asymptotically confined to the


interval [ _−_ 2 _σ,_ 2 _σ_ ], establishing a universal “noise floor.”


**Marchenko–Pastur Distribution.** For covariance-type matrices, the limiting spec

trum is described by the Marchenko–Pastur (MP) law [ **18** ]. Let **X** _∈_ R _[n][×][p]_ have i.i.d. entries


with mean zero and variance _σ_ [2], and define the sample covariance matrix **C** = 1
_n_ **[X]** _[T]_ **[X]** [.]


If _n, p →∞_ with ratio _p/n →_ _c >_ 0, then the ESD of **C** converges to: _f_ MP( _λ_ ) =

2 _πcσ_ 1 [2] _λ_ ~~�~~ ( _λ_ + _−_ _λ_ )( _λ −_ _λ−_ ) _,_ _λ ∈_ [ _λ−, λ_ +] with support _λ±_ = _σ_ [2] (1 _±_ _[√]_ ~~_c_~~ ~~)~~ [2] . Thus, all eigenval

ues are confined within [ _λ−, λ_ +], forming the “bulk spectrum” of random covariance matrices.


Figure 7. Tracy-Widom distribution: The variant of the tracy-widom distribution when _β_ = 1 (matrices composed of real numbers).


**Tracy-Widom Distribution.** The Tracy–Widom distribution _Fβ_ ( _s_ ) describes the lim

iting fluctuations of the largest eigenvalue _λ_ max of large random matrices after proper cen

tering and scaling, typically as _s_ = ( _λ_ max _−_ _µn_ ) _/σn_ where _µn_ is the spectral edge and

_σn ∼_ _n_ _[−]_ [2] _[/]_ [3], which in case of the MP distribution is _λ_ max = _σ_ [2] (1 + _[√]_ ~~_c_~~ ~~)~~ [2] + _αn_ _[−]_ [2] _[/]_ [3] _X_ (where


X follows the Tracy–Widom distribution and _α_ is a costant factor). It arises universally


for Gaussian matrices with index _β_ = 1 _,_ 2 _,_ 4 corresponding to real symmetric (GOE), com

plex Hermitian (GUE), and quaternionic (GSE) cases. The distribution is defined through


the Painlev´e II equation _q_ _[′′]_ ( _x_ ) = _x q_ ( _x_ ) + 2 _q_ ( _x_ ) [3] with boundary condition _q_ ( _x_ ) _∼_ Ai( _x_ ) as


3. BACKGROUND: RANDOM MATRIX THEORY 12


_x →_ + _∞_, where Ai( _x_ ) is the Airy function. The cumulative form for _β_ = 2 is _F_ 2( _s_ ) =



exp[ _−_ - _∞_ ~~�~~
_s_ [(] _[x][ −]_ _[s]_ [)] _[ q]_ [(] _[x]_ [)][2] _[ dx]_ [], and for] _[ β]_ [ = 1 it satisfies] _[ F]_ [1][(] _[s]_ [) =]




[1] - _∞_

2 _s_ _[q]_ [(] _[x]_ [)] _[ dx]_ [].]



_F_ 2( _s_ ) exp[ _−_ [1]



This law captures the universal edge behavior of eigenvalues in random matrix theory and


appears in diverse high-dimensional systems.


**3.2. Spiked Covariance Model.** A fundamental extension of the MP model is the


spiked covariance model [ **19** ], where a low-rank signal is embedded in isotropic noise. The


population covariance is: **Σ** = _σ_ [2] **I** _p_ + [�] _i_ _[k]_ =1 _[θ][i][u][i][u]_ _i_ _[T]_ [where] _[ θ][i]_ [are spike strengths and] _[ u][i]_ [are]


orthonormal signal directions. We use Random Matrix Theory on the sample covariance


matrices obtained from the LLMs activations on the datasets to identify the spikes compo

nents and the associated signal direction, which is useful for compression by projection and


for comparing with the noise baseline.


**3.3. BBP Phase Transition.** The Baik–Ben Arous–P´ech´e (BBP) phase transition


[ **20** ] states that if the spike strength is below a critical threshold, the corresponding sample

eigenvalues remain buried inside the MP bulk. Only when _θ > σ_ [2] (1 + _[√]_ ~~_c_~~ ~~)~~ (equivalently,

when population eigenvalues exceed _λ_ + = _σ_ [2] (1+ _[√]_ ~~_c_~~ ~~)~~ [2] ), do sample eigenvalues separate from


the bulk and emerge as outliers: _λ_ outlier _> λ_ +. These detached eigenvalues carry information


about the underlying signal, while eigenvalues within the MP bulk represent noise.


The BBP transition emerges directly from the upper edge _λ_ + of the Marchenko-Pastur


bulk by considering the asymptotic location of a sample eigenvalue _λ_ [ˆ] stemming from an


isolated population spike _θ_ . This location is given by the mapping _λ_ [ˆ] ( _θ_ ) _≈_ _σ_ [2] ( _θ_ + _cθ_
_θ−_ 1 [) for]


_θ >_ 1. The phase transition occurs precisely when this isolated eigenvalue detaches from

the continuous bulk, i.e., when _λ_ [ˆ] ( _θ_ ) = _λ_ +. Substituting _λ_ + = _σ_ [2] (1 + _[√]_ ~~_c_~~ ~~)~~ [2] and solving the


equation _σ_ [2] ( _θ_ + _[cθ]_

_θ−_ 1 [) =] _[ σ]_ [2][(1+] _[√]_ ~~_[c]_~~ ~~[)]~~ [2][ for] _[ θ]_ [ yields the critical threshold] _[ θ]_ [BBP][ =] _[ σ]_ [2][(1+] _[√]_ ~~_[c]_~~ ~~[)]~~ [. Thus,]


the BBP threshold is derived by finding the minimal population signal strength required to


push a sample eigenvalue beyond the theoretical maximum _λ_ + imposed by the noise.


4. BACKGROUND: HALLUCINATIONS IN AI MODELS 13


**4. Background: Hallucinations in AI Models**


Large-scale neural networks have achieved remarkable success across natural language


processing, computer vision, and multimodal reasoning. However, their reliability remains


a central challenge. One of the most prominent issues is the phenomenon of _hallucination_,


where a model generates fluent but factually incorrect or unsupported content. Hallucina

tions can emerge even in high-performing systems due to the mismatch between training


distributions and real-world usage, or because models prioritize linguistic plausibility over


factual grounding. This raises significant concerns in domains such as healthcare, law, and


scientific discovery, where incorrect outputs may lead to serious consequences.


Closely related is the problem of _out-of-distribution_ (OOD) generalization. Models are


trained on large but finite corpora and may encounter inputs at inference that lie outside


the training distribution. In these cases, internal representations can become unstable, often


leading to either low-quality predictions or hallucinated outputs. This behavior illustrates a


fundamental limitation: deep models tend to interpolate well within their training support


but extrapolate poorly outside of it.


**4.1. Uncertanty Definition.** Understanding these challenges requires considering the


broader notion of _uncertainty_ in machine learning. Uncertainty can be decomposed into


two main types: _aleatoric_ and _epistemic_ . Aleatoric uncertainty arises from inherent noise or


ambiguity in the data itself, such as ambiguous sentences or low-quality images. Epistemic


uncertainty, on the other hand, reflects the model’s lack of knowledge and tends to dominate


when the model faces OOD inputs or insufficient training coverage. Formally, if a model


produces a predictive distribution _p_ ( _y|x, θ_ ) given parameters _θ_, aleatoric uncertainty is tied


to the conditional variability in _y_ for fixed _θ_, whereas epistemic uncertainty corresponds to


variability across different plausible parameter settings. In practice, distinguishing between


these two types is critical for reliability. For example, high aleatoric uncertainty may be


unavoidable and can be communicated through probabilistic predictions, while epistemic


uncertainty suggests the need for mechanisms such as abstention, retrieval, or further model


adaptation. However, modern neural networks often underestimate their uncertainty because


5. BACKGROUND: COMPRESSION IN DEEP NETWORKS 14


maximum-likelihood training encourages overconfident predictions. This miscalibration in

creases the likelihood that models generate hallucinations, as epistemic uncertainty is often


underestimated in their predictive distributions. Such underestimation directly undermines


reliability, especially in high-stakes applications.


**5. Background: Compression in Deep Networks**


The success of deep neural networks is partly due to their overparameterization, which


provides strong representational capacity and generalization. However, this leads to high


memory usage, inference latency, and energy costs. To mitigate these issues, several com

pression techniques have been proposed. In this section, we outline four main approaches:


knowledge distillation, pruning, quantization, and sparsity. Pruning methods aim to re

move redundant or unimportant parameters, typically weights or entire channels, from a


trained model. The underlying assumption is that many connections in overparameterized


networks contribute little to predictive performance. By eliminating these parameters, the


model becomes smaller and faster while maintaining accuracy within acceptable bounds


[ **22** ]. Quantization reduces the precision of weights and activations, replacing high-precision


floating-point representations with lower-bit formats such as 8-bit integers. This decreases


both the memory footprint and the cost of arithmetic operations, enabling deployment on


resource-constrained devices. While aggressive quantization can cause accuracy degrada

tion, careful design and post-training calibration mitigate this effect [ **23** ]. Sparsity-based


approaches enforce or exploit structured or unstructured sparsity in network parameters.


Unlike pruning, which is often applied after training, sparse methods may involve training


models under sparsity constraints or using specialized architectures. Sparse representations


reduce storage and allow faster computation on hardware that supports efficient sparse op

erations [ **24** ].


**5.1. Knowledge Distillation.** Knowledge distillation transfers knowledge from a large,


pre-trained “teacher” model to a smaller “student” model. The student is trained not only


on ground-truth labels but also to match the softened output distributions of the teacher,


6. BACKGROUND: SPECTRAL METHODS IN LARGE MODELS 15


obtained via a temperature-scaled softmax. Given teacher logits _z_ [(] _[T]_ [)] and student logits _z_ [(] _[S]_ [)],

the softened probabilities are _p_ [(] _i_ _[T]_ [)] = ~~�~~ exp( _j_ [exp(] _zi_ [(] _[z][T]_ _j_ [(][ )] _[T]_ _/τ_ [ )] _/τ_ ) ) [and] _[ p]_ _i_ [(] _[S]_ [)] = ~~�~~ exp( _j_ [exp(] _zi_ [(] _[z][S]_ _j_ [(][)] _[S]_ _/τ_ [)] _/τ_ ) ) [, where] _[ τ]_ [ is the]

temperature parameter. The training objective combines the standard cross-entropy with


the true labels _y_ and a distillation loss, typically the Kullback–Leibler divergence between


_p_ [(] _[T]_ [)] and _p_ [(] _[S]_ [)] : _L_ = _α_ CE( _y, p_ [(] _[S]_ [)] ) + (1 _−_ _α_ ) _τ_ [2] KL( _p_ [(] _[T]_ [)] _∥_ _p_ [(] _[S]_ [)] ). This allows the student to


capture not only the hard targets but also the relative similarity structure encoded in the


teacher’s predictions, leading to compact models with competitive accuracy [ **21** ].


**6. Background: Spectral Methods in Large Models**


Spectral methods study how information in high-dimensional representations is dis

tributed across directions or frequencies by inspecting the eigenstructure of matrices derived


from data or model internals (e.g., activation covariances, Jacobians, or attention maps).


In deep learning, these analyses help characterize global geometry (e.g., low-rank structure,


anisotropy, and concentration), diagnose training dynamics and generalization, and suggest


principled simplifications such as dimensionality reduction or subspace projections [ **29, 30** ].


**6.1. Spectral lenses on neural representations.** A common approach is to form


an activation matrix by stacking hidden features across samples or time and examining its


covariance spectrum. The eigenvalues of this covariance describe how variance concentrates


along principal directions; rapid decay indicates implicit low-rank structure, while heavy tails


suggest richer, multi-scale features [ **29, 30** ]. Tools like representational similarity based on


canonical correlations or centered kernel alignment compute pairwise relationships between


layer representations without requiring supervision, enabling comparisons across models,


layers, or training stages [ **27, 28** ]. These perspectives complement token-level or pixel-level


uncertainty because spectra aggregate information across many units and inputs, producing


compact, interpretable descriptors of representation quality.


**Links to random matrix theory.** When activations are approximately mean-zero and


isotropic at a given scale, the empirical covariance spectrum exhibits a bulk consistent with


random matrix predictions. The Marchenko–Pastur law describes the asymptotic eigenvalue


6. BACKGROUND: SPECTRAL METHODS IN LARGE MODELS 16


density of sample covariance matrices with aspect ratio parameter and noise variance, pro

viding a principled baseline for distinguishing noise-dominated directions from signal-bearing


outliers; deviations from this baseline, such as outlier eigenvalues or widened gaps, indicate


emergent structure [ **65** ]. For symmetric weight or Hessian-like operators with independent


fluctuations, the Wigner semicircle distribution offers a corresponding null model [ **26** ]. In


practice, deep networks rarely behave like pure noise, but these laws give calibrated reference


points: directions with eigenvalues near the bulk edge are more likely to be correlational or


redundant, while detached eigenvalues often track semantic or task-relevant factors.


**Frequency and complexity biases.** Spectral viewpoints also illuminate which func

tions neural networks learn most readily. In overparameterized regimes, many models exhibit


a “spectral bias,” fitting low-frequency or smoother components first before higher-frequency


detail as training progresses. This bias can be observed in the singular value structure of


learned features and in Fourier analyses of fitted functions, with implications for generaliza

tion and robustness [ **31** ]. Spectral diagnostics thus connect training dynamics to hypothesis


complexity and can guide early stopping or curriculum choices.


**6.2. Applications to vision and language.** In computer vision, spectra of convolu

tional features often display heavy-tailed behavior consistent with implicit regularization,


and class-conditional structure can concentrate along a small number of directions, reflect

ing semantic disentanglement in later layers [ **29, 30** ]. In language models, representation


similarity analyses quantify how linguistic abstractions emerge across depth and how atten

tion blocks reconfigure information, while covariance spectra reveal anisotropy and low-rank


tendencies in token embeddings and intermediate states [ **27, 28** ]. Across both modalities,


spectral characterization offers a unifying language to reason about layer roles, information


flow, and the trade-off between expressivity and compression.


**Implications for practice.** Because spectral summaries are global and compact, they


are useful for monitoring training health, identifying distributional shifts, and motivating ar

chitecture or optimization choices. For example, a pronounced low-rank structure motivates


7. BACKGROUND: EFFICIENCY AND RELIABILITY IN LARGE AI MODELS 17


Figure 8. Empirical eigenvalue distribution: spectrum of hidden layer activations in a deep network. The histogram shows a bulk of small eigenvalues
consistent with noise-like directions, while a few outliers (to the right of the red
line) indicate informative, structured components. Such separation between
bulk and outliers is the basis of spectral analyses in deep learning.


low-rank adapters and subspace updates, where model changes are confined to a small num

ber of directions while preserving overall behavior [ **32** ]. More broadly, comparing empirical


spectra to random matrix baselines provides quantitative signals for when representations


become structured, when they remain noise-like, and how complexity evolves with data and


compute.


**7. Background: Efficiency and Reliability in Large AI Models**


The rapid scaling of artificial intelligence (AI) systems has brought remarkable advances


in natural language processing, computer vision, and multimodal learning. At the same


time, this growth has exposed two central and often competing requirements: _efficiency_, the


ability to deploy and operate models within practical computational budgets, and _reliability_,


the assurance that predictions are robust, trustworthy, and aligned with human expectations.


While these dimensions are frequently studied in isolation, they are deeply interconnected.


Both can be analyzed through the lens of _spectral geometry_, which provides a principled


framework for understanding the dynamics of high-dimensional neural representations.


7. BACKGROUND: EFFICIENCY AND RELIABILITY IN LARGE AI MODELS 18


**Efficiency in Large-Scale AI Models.** Efficiency concerns the computational and


resource costs associated with training and deploying large models. With the rise of billion

parameter networks, challenges include inference latency, memory footprint, and energy con

sumption [ **33** ]. Scaling laws suggest that performance often grows predictably with model


size and dataset scale [ **34** ], but this trend comes at significant financial and environmental


cost. Techniques to address efficiency aim to reduce redundancy in representations, compress


parameters, or exploit low-rank structures without compromising accuracy [ **35** ]. Fundamen

tally, efficiency measures whether a model can operate in real-world conditions, from data


centers to edge devices, while preserving its utility.


**Reliability in Large-Scale AI Models.** Reliability addresses the trustworthiness of


model predictions. Large neural networks are known to hallucinate, produce spurious cor

relations, and degrade sharply on inputs outside the training distribution [ **36, 37** ]. Failures


can arise from sensitivity to noise, adversarial perturbations, or distributional shifts. Tra

ditional confidence scores, such as softmax probabilities, are poorly calibrated and fail to


capture deeper instabilities in learned representations [ **38** ]. Reliability, therefore, encom

passes robustness, calibration, interpretability, and alignment with ground-truth semantics.


It is increasingly critical as AI models are deployed in high-stakes domains such as healthcare,


finance, and law.


**7.1. Spectral Geometry as a Bridge.** Although efficiency and reliability may seem


distinct, both are tied to the geometry of neural representations. High-dimensional acti

vations often concentrate around low-rank subspaces, and their covariance spectra reveal


whether learned features are structured or dominated by noise [ **30** ]. Efficient models seek to


eliminate redundant directions, while reliable models must ensure that remaining directions


capture stable and causal signals. Spectral geometry provides the mathematical lens for uni

fying these perspectives: eigenvalue distributions and spectral entropy measure redundancy,


while spectral gaps and deviations from random matrix baselines indicate stability and ro

bustness. By analyzing activations through this shared framework, it becomes possible to


design models that are both compact and trustworthy.


CHAPTER 3

### **Related Work**


**1. Related: Hallucination Detection**


Large language models exhibit “hallucinations”, namely fluent statements that are un

faithful to sources or unverifiable by background knowledge, this jeopardizes safety in high–stakes


domains, erodes user trust through confidently wrong explanations, and can propagate mis

information at scale, empirical evidence suggests that hallucinations persist even as models


scale and fine–tune, because next–token prediction does not explicitly penalize confidently


fabricated content and because training data contain human misconceptions [ **59, 60** ]. Recent


research therefore focuses on _detection_ rather than elimination, aiming to flag risky gener

ations with minimal latency and without full model access, the literature can be organized


into black–box, grey–box, white–box, and spectral families that trade off access assumptions,


cost, and generality.


**1.1. Black-box Methods.** Black–box detectors only observe input, output, and op

tionally multiple samples, without logits or hidden states, a representative line leverages


self–consistency across stochastic re–generations, SelfCheckGPT queries the model multiple


times and measures agreement with sentence–level QA, NLI, and semantic similarity probes


to infer whether a claim is supported, reporting strong zero–resource performance across QA


and summarization [ **47** ]. Complementary work turns disagreement into a calibrated prob

ability of hallucination, a cost–effective multi–scoring framework aggregates diverse uncer

tainty signals, then performs conditional calibration to obtain risk–aware thresholds that are


competitive with heavier pipelines while reducing compute [ **61** ]. Another strand uses chains


of natural language inference, CoNLI decomposes a response into atomic claims and checks


entailment against sources, then post–edits ungrounded spans, providing a plug–and–play


19


1. RELATED: HALLUCINATION DETECTION 20


pipeline without task–specific fine–tuning [ **48** ]. Black–box methods are appealing for propri

etary models and production since they require no internals and are easy to retrofit, however


they can be expensive due to multi–sample decoding, may fail on _self–consistent_ errors where


the model repeats the same falsehood across samples, and can be sensitive to prompt format


and decoding temperature [ **59, 60** ].


**1.2. Grey-box.** Grey–box detectors exploit limited internals such as token log–probabilities


while avoiding access to hidden states, DetectGPT observes that model–generated text tends


to occupy negative curvature regions of the log–likelihood landscape, it perturbs the text


and measures average change in log–probability to separate machine from human text [ **50** ],


Fast–DetectGPT replaces masked perturbations with conditional probability curvature, sub

stantially accelerating detection while preserving accuracy [ **51** ]. Because many commer

cial APIs expose token log–probs only intermittently, recent work predicts fuller distribu

tions from partial observations to extend white–box–style criteria into proprietary settings,


Glimpse learns to reconstruct probability information and thereby enables rank, entropy,


and curvature–based detectors on API–only models [ **52** ]. Grey–box approaches are efficient


when logits are available and avoid re–generating many samples, yet they may degrade if


APIs obscure probabilities, curvature signals can correlate with distribution shift unrelated


to factuality, and they do not directly capture conflicts with external evidence [ **61** ].


**1.3. White-box Methods.** White–box detectors use internal activations, attention


maps, or intermediate logits, often enabling real–time alarms, MIND trains an unsupervised


detector on internal states collected during inference, pairing automatic data labeling with


a lightweight classifier for on–the–fly detection, and reports competitive latency–accuracy


trade–offs [ **62** ]. INSIDE argues that dense internal states retain semantic information lost


in surface tokens, introduces an _EigenScore_ derived from the eigenvalues of the covariance


of sentence embeddings to quantify response self–consistency, and proposes test–time fea

ture clipping to curb overconfident hallucinations [ **63** ]. Faithfulness scoring from attention


patterns estimates whether the tokens that should support an answer receive proportionate


attention, improving detection of unsupported rationales and explanations [ **55** ]. Finally,


1. RELATED: HALLUCINATION DETECTION 21


learning from unlabeled outputs in the wild, HaloScope estimates membership in truthful


versus untruthful subpopulations among raw LLM generations, enabling a practical truth

fulness classifier trained without manual labels and showing strong generalization across


datasets [ **64** ]. White–box methods can be accurate and low–latency because they align


with model computation, nevertheless portability to closed–source APIs is limited, internal


probes risk picking up spurious correlates, and detectors trained on one family of models


may require recalibration for another [ **59** ].


**1.4. Spectral Methods.** Spectral detectors focus on eigenvalue structure rather than


token probabilities or raw attention weights, the central idea is that under benign conditions


hidden representations behave approximately like high–dimensional noise with a bulk eigen

value distribution predicted by random matrix theory, for instance the Marchenko–Pastur


law for sample covariance, while emergent low–rank correlations under failure modes produce


outlier eigenvalues and widened spectral gaps, deviations that can be monitored during gen

eration [ **65, 66** ]. The implementation vary with the object being analyzed, attention–based


spectral methods interpret attention as a graph and compute eigenvalues of the Laplacian,


the top– _k_ Laplacian eigenvalues or their gaps feed compact probes that show state–of–the–art


accuracy among attention–centric detectors and preserve interpretability through eigengap


analysis [ **67** ]. White–box consistency metrics based on eigenvalues, for example the Eigen

Score, use the spectrum of internal sentence–embedding covariances to assess diversity versus


concentration in the generated content, with higher dispersion or bulk–like spectra indicat

ing weaker semantic grounding [ **63** ]. Spectral methods are attractive because eigenvalue


statistics compress high–dimensional geometry into a few interpretable numbers, they are


efficient to compute with truncated SVD and admit principled calibration against theoretical


baselines, at the same time they require careful control for sequence length, windowing, and


layer choice to avoid conflating harmless stylistic shifts with true hallucinations, and theoret

ical references such as Marchenko–Pastur and the Baik–Ben Arous–P´ech´e phase transition


provide guidance for separating bulk from spikes but must be adapted to non–ideal neural


settings [ **65, 66** ].


2. RELATED: OUT-OF-DISTRIBUTION DETECTION 22


**2. Related: Out-of-Distribution Detection**


Modern deep networks, especially large language and vision–language models, often op

erate under the implicit assumption that test data follow the same distribution as training


data. When this assumption is violated, model predictions can become unreliable or even


nonsensical, a condition known as the out-of-distribution (OOD) problem [ **68** ]. OOD de

tection aims to identify when an input lies outside the support of the training distribution


so that the system can abstain, trigger fallback mechanisms, or re-calibrate uncertainty


estimates. In LLMs and multimodal architectures, OOD behavior manifests as factual hal

lucinations, overconfident predictions on unseen topics, or unstable performance under do

main shifts. Since full retraining is costly, OOD detection modules are a practical route


toward safer deployment. This section reviews three major classes of methods: score-based,


representation-based, and spectral/statistical approaches, summarizing their core principles,


effectiveness, and limitations.


**2.1. Score-based Methods.** Score-based OOD detectors rely on output statistics such


as confidence, logit margins, or energy, treating extreme values as indicators of abnormality.


The most classical baseline, maximum softmax probability (MSP), measures confidence as


max _i p_ ( _yi|x_ ), where low values suggest OOD samples [ **36** ]. Although simple, MSP often


underestimates uncertainty because deep networks are overconfident even on random noise.


Temperature scaling and input perturbations were later introduced in ODIN, which perturbs


the input slightly in the direction of the gradient and rescales logits by a temperature pa

rameter to separate in- and out-distribution samples more effectively [ **69** ]. The Mahalanobis


method computes layerwise Gaussian scores by modeling class-conditional feature distribu

tions and measuring Mahalanobis distance to the closest class mean [ **70** ]. These approaches


are efficient and can be applied post-hoc without retraining. However, their calibration is


architecture- and dataset-dependent, often failing when the model’s softmax layer saturates.


Energy-based scores generalize confidence by defining an “energy” _E_ ( _x_ ) = _−T_ log [�] _i_ _[e][z][i][/T]_


from logits _zi_, where in-distribution samples concentrate near low energy [ **71** ]. Energy-based


2. RELATED: OUT-OF-DISTRIBUTION DETECTION 23


OOD detection achieves strong results in computer vision and has been extended to trans

formers and LLMs [ **72** ], offering smoother uncertainty surfaces than probability-based scores.


More recently, normalized cosine similarity and logit margin distributions have been pro

posed for sequence models [ **73** ]. Score-based methods are attractive for their simplicity and


compatibility with black-box systems but require careful threshold tuning and can conflate


epistemic and aleatoric uncertainty, limiting interpretability.


**2.2. Representation-based Methods.** Representation-based detectors leverage the


geometry of internal embeddings or subspaces, motivated by the observation that in-distribution


data form compact manifolds in hidden space while OOD inputs deviate from them. Early


work modeled feature embeddings with Gaussian mixtures or k-nearest neighbors, classify

ing high-distance samples as OOD [ **75** ]. Deep kNN (DkNN) retrieves the labels of nearest


training features at multiple layers to estimate conformity and provides layerwise reliabil

ity scores [ **74** ]. Subsequent work developed centroid-based methods such as kNN cosine


similarity, Center Loss embeddings, and Local Outlier Factor variants [ **76** ].


Advanced subspace approaches analyze the rank or span of hidden representations.


SVCCA and CKA metrics have been used to compare activation subspaces across layers and


models, showing that OOD inputs distort canonical correlations and occupy low-similarity


directions [ **27, 28** ]. Representation self-similarity metrics like Mahalanobis with shrink

age or spectral norm regularization further refine detection by discounting correlated noise.


Recent studies in transformer-based encoders use attention entropy and hidden-state vari

ance as signals of uncertainty, achieving strong transfer to text and multimodal data [ **77** ].


Representation-based approaches provide interpretability by localizing anomalies to specific


layers or embeddings, yet they require storing or approximating training features, which may


be memory-intensive for large-scale models, and they may degrade under feature collapse


caused by overfitting or adversarial fine-tuning.


**2.3. Spectral and Statistical Methods.** Spectral and statistical approaches analyze


the eigenvalue spectrum or higher-order statistics of activation covariances, connecting OOD


3. RELATED: MODEL COMPRESSION 24


detection to Random Matrix Theory (RMT). The empirical covariance Σ = [1]

_n_ _[XX]_ _[⊤]_ [of hid-]


den activations _X ∈_ R _[d][×][n]_ is expected to follow a bulk spectrum approximated by the


Marchenko–Pastur distribution under in-distribution conditions, with deviations indicating


structured or anomalous behavior. When an input is OOD, spurious correlations can induce


outlier eigenvalues or inflate the spectral tail beyond the bulk limit. This phenomenon was


observed in spectral OOD detectors such as RankFeat, which measures the effective rank of


covariance matrices as a proxy for representation collapse [ **78** ]. SpectralGap expands on this


by computing the difference between consecutive eigenvalues (the eigengap) and threshold

ing on sharp transitions that suggest emergent low-dimensional structure [ **79** ]. SNoJoE, an


efficient singular value–based detector, evaluates joint energy across layers by tracking the


spectral norm and nuclear norm evolution of activations, achieving state-of-the-art detection


in both CNNs and transformers [ **80** ].


These methods are attractive because eigenvalue spectra provide an unsupervised and


architecture-agnostic summary of distributional structure, allowing theoretical calibration


through RMT. Moreover, they align naturally with the spectral diagnostics used for relia

bility and hallucination detection, offering a principled link between OOD and uncertainty


quantification. Nonetheless, spectral detectors can be sensitive to batch size, normalization,


and numerical precision when computing eigenvalues at scale. Extensions using randomized


eigensolvers and moving-window covariance estimates partially mitigate the cost, enabling


online OOD tracking in large models [ **81, 82** ]. Overall, spectral and statistical approaches


form a rapidly growing subfield that unifies deep learning reliability analysis with classical


multivariate statistics.


**3. Related: Model Compression**


Deploying modern deep networks in latency and energy constrained settings motivates


principled compression, the goal is to reduce parameters, memory footprint, and inference


3. RELATED: MODEL COMPRESSION 25


cost while preserving accuracy and calibration. Four families dominate the literature, prun

ing, quantization, low rank factorization, and knowledge distillation. Each family spans clas

sic CNN accelerators and transformer specific variants, and each introduces distinct trade


offs between software and hardware efficiency, portability, and accuracy retention.


**3.1. Pruning.** Pruning removes parameters judged unnecessary for task performance.


Unstructured pruning zeros individual weights according to saliency, magnitude pruning


introduced in the deep compression pipeline shows that iterative prune, retrain and Huffman


coding achieves large sparsity with minimal accuracy loss in CNNs [ **83** ], while the lottery


ticket hypothesis argues that dense networks contain trainable sparse subnetworks that reach


competitive accuracy when reset and fine tuned, revealing overparameterization and the


importance of initialization [ **84** ]. Gradient and Hessian based criteria estimate sensitivity


more precisely, Taylor expansion pruning and WoodFisher approximate the loss increase from


removing a weight using first or second order information and can outperform magnitude


baselines, especially at high sparsity [ **85, 86** ]. In transformers, movement pruning encourages


weights to move toward zero during fine tuning via a dynamic L0 style regularizer, yielding


structured sparsity patterns that transfer across tasks [ **87** ].


Structured pruning removes whole channels, heads, and blocks to obtain hardware friendly


speedups. For CNNs, channel level sparsity via L1 penalty on batch norm scales, known as


network slimming, reliably shrinks models with real wall clock gains [ **88** ]. Automated search


methods, AMC and MetaPruning, couple reinforcement learning or meta networks with a


pruning controller to allocate sparsity budgets per layer under latency constraints [ **89, 90** ].


In transformers, studies show many attention heads are redundant, removing heads and even


entire feed forward sublayers can preserve accuracy with careful retraining [ **91** ]. Structured


approaches align well with dense libraries and accelerators, however they may leave accuracy


on the table relative to fine grained sparsity, and benefits depend on compiler and kernel


support for the targeted structure.


**3.2. Quantization.** Quantization maps floating point tensors to low precision integers


to reduce memory bandwidth and arithmetic cost. Post training quantization calibrates


3. RELATED: MODEL COMPRESSION 26


scales after training without access to labels, uniform 8 bit affine quantization with per


channel weights and per tensor activations is a robust baseline across CNNs and transform

ers [ **92** ]. Large language models exhibit activation outliers that break naive post training


quantization, SmoothQuant shifts range from activations into weights during calibration to


suppress outliers, enabling 8 bit activations and 8 bit or 4 bit weights at minimal perplexity


cost [ **93** ]. GPTQ proposes blockwise second order weight quantization that solves a local


least squares objective using approximate Hessian information, delivering strong 3–4 bit re

sults on LLMs with a single calibration pass [ **94** ]. AWQ further observes that a small set of


salient channels dominate error, protecting them during quantization improves 4 bit accu

racy on a wide range of decoder only models [ **95** ]. ZeroQuant and ZeroQuant v2 push post


training quantization toward 8, 6, and 4 bit on both weights and activations with tensor


wise and group wise calibration that scales to billion parameter models [ **96, 97** ].


Quantization aware training inserts fake quantization nodes during fine tuning so the


network learns to be robust to quantization noise. Learned step size quantization treats


scale as a learnable parameter and achieves state of the art low bit accuracy in CNNs


[ **98** ], while Q BERT adapts group wise quantization and Hessian guided mixed precision


to transformers, reporting strong gains at 4 bit and below [ **99** ]. Post training methods are


attractive for cost and simplicity, yet may require sizable calibration sets for LLMs, whereas


QAT yields better extremes at the expense of additional fine tuning and careful optimizer


choices. Realized speedups depend on integer kernels, activation quantization, and operator


coverage along the whole graph.


**3.3. Low rank factorization.** Low rank factorization exploits linear redundancy by


decomposing weight tensors into products of smaller tensors. For fully connected layers,


truncated SVD yields two narrow matrices that approximate the original, reducing multi

plies without sparsity [ **100** ]. For convolutions, spatial separability and tensor decompositions


apply, CP and Tucker decompositions factorize 4D kernels into sequences of smaller convo

lutions, maintaining accuracy after brief fine tuning [ **101, 102** ]. In transformers, parameter


3. RELATED: MODEL COMPRESSION 27


sharing and projection tying reduce redundancy, ALBERT shares parameters across lay

ers and factorizes the embedding matrix, substantially shrinking model size with modest


accuracy changes [ **103** ].


Adapter style low rank updates enable parameter efficient fine tuning rather than in

ference time compression, yet the same principle illuminates low intrinsic dimensionality


in language models. LoRA injects trainable low rank matrices into attention and MLP


projections while freezing the base network, matching or surpassing full fine tuning with a


small fraction of trainable parameters [ **32** ], extensions such as DoRA decouple direction and


magnitude to stabilize training and further reduce rank [ **104** ], and analyses of intrinsic di

mension argue that many NLP tasks lie in very low dimensional subspaces relative to model


size [ **105** ]. When used for compression rather than adaptation, factorization replaces large


dense projections with products of thin matrices in the deployed model, offering dense ker

nel friendliness and predictable latency, at the cost of designing ranks per layer and possible


accuracy drops if ranks are set too aggressively.


**3.4. Knowledge distillation.** Knowledge distillation transfers behavior from a high


capacity teacher to a smaller student using soft targets and auxiliary hints. The classical


formulation minimizes a convex combination of task loss and KL divergence between teacher


and student output distributions at a temperature, improving generalization of the student


[ **106** ]. FitNets introduced intermediate feature hints to guide deeper students [ **107** ], atten

tion transfer encourages students to match teacher attention maps [ **108** ], and contrastive


representation distillation aligns teacher and student embeddings by maximizing mutual


information at the instance level [ **109** ].


In NLP, distillation underpins many compact transformers. DistilBERT compresses


BERT by matching soft logits, intermediate states, and attention distributions during pre


training, halving parameters and improving speed with small accuracy loss [ **110** ]. Patient


knowledge distillation matches layer wise representations with patience, i.e., allowing the


student to align to multiple teacher layers over time, improving stability and performance


[ **111** ]. BERT of Theseus progressively replaces teacher components with student modules


4. RELATED: RANDOM MATRIX THEORY IN DEEP LEARNING 28


during fine tuning, interpreting the process as a Markov chain of network components and


delivering competitive small models [ **112** ]. MiniLM distills deep self attention by matching


value relation matrices and query key self attention distributions, yielding strong students


with tiny footprints [ **113** ]. Beyond fixed students, progressive shrinking in Once for All


training learns a super network that can instantiate sub networks of different widths and


depths, combining distillation with neural architecture adaptation to hit diverse latency


targets without retraining from scratch [ **114** ].


In vision, compact students like MobileNets and EfficientNets commonly incorporate KD


during training, and recent works explore multi teacher ensembles, self distillation without


an external teacher, and task specific distillation for detection and segmentation. Distillation


is flexible, it composes with pruning, quantization, and factorization by recovering accuracy


after structural changes, and it can target calibration and robustness objectives beyond top


1 accuracy. Its costs include teacher inference during training, potential mismatch across


domains, and sensitivity to loss weighting, temperature, and layer mapping choices.


**4. Related: Random Matrix Theory in Deep Learning**


Random Matrix Theory (RMT) provides a statistical–mechanical lens to analyze deep


neural networks, treating weight and activation matrices as high-dimensional random sys

tems. By studying the eigenvalue spectra of these matrices, RMT reveals universal behaviors


that shed light on learning dynamics, generalization, and robustness. Unlike heuristic di

agnostics, spectral geometry provides analytical tools to quantify phase transitions, implicit


regularization, and signal–noise separation in modern networks. This section summarizes key


developments applying RMT to deep learning and related areas of reliability and robustness.


**4.1. Phase Transitions.** A central insight from RMT is that eigenvalue distributions


of random covariance matrices follow the Marchenko–Pastur (MP) law, which describes the


limiting density of eigenvalues for a matrix _XX_ _[⊤]_ _/n_ when both _n_ and the feature dimension


grow large [ **65** ]. In deep networks, deviations from this theoretical bulk indicate structured


4. RELATED: RANDOM MATRIX THEORY IN DEEP LEARNING 29


correlations in learned representations. When training begins, the spectrum typically resem

bles the MP bulk; as learning progresses, a few eigenvalues detach from the bulk, forming


outliers associated with meaningful features or “spikes.” This transition is governed by the


Baik–Ben Arous–P´ech´e (BBP) threshold [ **66** ], which determines when a signal eigenvalue


separates from random noise.


Empirical studies demonstrate that layer weight spectra in trained networks display


heavy-tailed distributions following power laws rather than pure MP shapes [ **81, 115** ]. These


heavy tails reflect correlations induced by gradient descent and act as fingerprints of general

ization. The phase-transition view thus links RMT to implicit bias: well-generalizing models


operate near criticality, balancing order (signal) and chaos (noise). Conversely, overtrained


or overregularized models exhibit either degenerate (collapsed) spectra or overly flat ones,


signaling underfitting or overparameterization.


**4.2. Implicit Regularization.** RMT also explains the implicit regularization mecha

nisms that emerge in large-scale optimization. Gradient descent, batch normalization, and


dropout collectively push weight matrices toward critical spectral regimes, shaping their


eigenvalue decay [ **81** ]. This self-organized criticality aligns with the notion of _spectral bias_,


where networks learn low-frequency, smooth components first. In the spectral domain, weight


and gradient covariance spectra often show 1 _/λ_ _[α]_ power-law tails with _α_ between 2 and 5,


characterizing an intermediate regime between random noise and strong correlations.


Such spectral regularization correlates with model robustness and generalization: net

works with heavy-tailed spectra resist overfitting and exhibit better OOD performance


[ **116, 117** ]. Moreover, the Hessian spectra of well-trained models typically have a small


number of large eigenvalues corresponding to informative directions and a bulk near zero


corresponding to flat minima. This separation implies that stochastic optimization implic

itly enforces a spectral cutoff, analogous to Tikhonov regularization in inverse problems. In


this view, spectral analysis provides both a diagnostic and a theoretical justification for why


simple optimization procedures produce well-behaved models without explicit constraints.


5. RELATED: MULTIMODAL MODELS AND HALLUCINATION 30


**4.3. Subspace Analysis.** Beyond bulk statistics, RMT provides tools for subspace


analysis of learned representations. In deep networks, activations at each layer can be


viewed as samples from a high-dimensional manifold embedded in feature space. The covari

ance spectrum captures how variance is distributed across latent directions. Signal–noise


decomposition based on spectral truncation isolates meaningful subspaces from isotropic


noise, yielding insights into feature redundancy and causal directions. Studies show that


informative subspaces correspond to outlier eigenvectors, while the noise subspace follows


MP-like statistics [ **118, 81** ]. Monitoring spectral entropy or eigengaps across layers reveals


how information condenses as depth increases, forming hierarchical subspaces of decreasing


intrinsic dimension.


Recent work extends subspace spectral analysis to multimodal models. In vision–language


transformers, joint embeddings exhibit coupled spectral dynamics alignment across modal

ities manifests as correlated eigenvalue shifts between text and vision subspaces. Spectral


filtering, where activations are projected onto dominant eigenspaces, has been shown to en

hance robustness and interpretability. In particular, _EigenShield_ introduces causal subspace


filtering via RMT principles to defend against adversarial perturbations in vision–language


models, identifying and suppressing anomalous eigenmodes while preserving semantic struc

ture [ **134** ]. This line of work connects RMT-based subspace analysis to adversarial security,


suggesting that spectral geometry can serve both analytical and defensive roles.


**5. Related: Multimodal Models and Hallucination**


The integration of visual, textual, and sometimes auditory modalities has enabled a new


generation of foundation models capable of perception and reasoning across heterogeneous


data. Vision–language models (VLMs) such as CLIP, BLIP, Flamingo, LLaVA, and GPT-4V


have demonstrated impressive zero-shot capabilities on diverse multimodal tasks including


captioning, VQA, and visual reasoning [ **119, 120, 121, 122, 123** ]. However, the same


scaling that endows such flexibility also amplifies problems of reliability. Multimodal hal

lucination, the generation of textual or visual content inconsistent with the actual input


5. RELATED: MULTIMODAL MODELS AND HALLUCINATION 31


image or video, emerges as a major limitation [ **124** ]. These hallucinations can manifest as


fabricated objects in image captions, incorrect visual attributes, or semantically inconsistent


reasoning steps, threatening the deployment of VLMs in sensitive contexts such as medical


imaging, autonomous driving, or assistive technology.


**5.1. Nature of Multimodal Hallucinations.** Multimodal hallucination differs funda

mentally from text-only hallucination because it stems from misalignment between modalities


rather than internal linguistic overconfidence alone. Empirical studies classify hallucinations


into three categories: _object hallucination_, where models invent entities absent from the vi

sual scene; _attribute hallucination_, where they misdescribe color, shape, or position; and _con-_


_textual hallucination_, where textual reasoning contradicts visual evidence [ **125, 124** ]. The


underlying causes include dataset biases (co-occurrence correlations that teach the model


spurious associations), imbalance between vision and language encoders, and autoregressive


decoding that privileges linguistic priors over perceptual grounding.


Early VLMs such as OSCAR and VinVL relied heavily on object detection features,


which constrained hallucination but limited generality [ **126, 127** ]. End-to-end pretrained


architectures like CLIP and BLIP2 improved generalization but exhibited increased hallu

cination due to weaker explicit grounding. Recent works have shown that model scale and


instruction-tuning can paradoxically increase hallucination frequency even while improving


benchmark accuracy, as seen in GPT-4V and other multimodal LLMs [ **122, 123** ]. This


tension highlights a key challenge: multimodal grounding requires both accurate fusion and


balanced modality dominance.


**5.2. Detection and Mitigation Strategies.** Research into multimodal hallucination


detection adapts both linguistic and perceptual uncertainty measures. Black-box metrics


evaluate semantic alignment between generated captions and ground-truth regions using


object detectors or CLIP-based similarity scores. The VisualHallucination benchmark pro

posed quantitative metrics for hallucinated object frequency and visual faithfulness [ **124** ].


Gray-box approaches leverage attention entropy or cross-modal similarity matrices computed


from intermediate layers to detect when text tokens attend to irrelevant visual regions [ **128** ].


5. RELATED: MULTIMODAL MODELS AND HALLUCINATION 32


White-box or spectral detectors extend this by analyzing hidden covariance spectra of mul

timodal fusion layers: deviations from expected eigenvalue distributions or sharp spectral


gaps correspond to loss of alignment, providing unsupervised hallucination signals consistent


with Random Matrix Theory-based diagnostics.


Mitigation techniques span architectural, training, and decoding interventions. Architec

turally, contrastive pretraining with balanced image–text pairs (e.g., CLIP, ALIGN) reduces


dataset-induced biases [ **119, 129** ]. Training-level solutions such as grounding instruction


tuning explicitly penalize ungrounded generations by augmenting prompts with object tags


or region features [ **122, 130** ]. Reinforcement learning with human feedback (RLHF) has


also been adapted to multimodal settings, optimizing for human-judged visual faithfulness


instead of text coherence [ **123** ]. Decoding-level methods, including constrained beam search


or re-ranking by cross-modal consistency, filter out hallucinated outputs post hoc [ **125** ].


**5.3. Spectral and Causal Perspectives.** From a spectral viewpoint, multimodal fu

sion layers can be viewed as coupling two high-dimensional subspaces, visual and textual rep

resentations, whose joint covariance structure governs alignment. When properly grounded,


the joint covariance spectrum exhibits well-matched eigenmodes with smooth decay; hallu

cinations correlate with the emergence of spurious outlier eigenvalues, representing latent


directions dominated by linguistic noise or irrelevant visual activations. This aligns with re

cent analyses of eigenvalue gaps and cross-modal covariance geometry in transformer blocks


[ **81, 134** ]. Spectral regularization and subspace filtering, as explored in the context of ad

versarial robustness, can thus mitigate multimodal hallucinations by projecting activations


onto stable causal eigenspaces.


Beyond diagnosis, such RMT-informed methods offer interpretability: by tracking which


eigenvectors carry cross-modal signal versus noise, one can localize the onset of hallucination


and analyze its modality-specific source. This spectral–causal perspective suggests that mul

timodal hallucination is not an isolated failure mode but an emergent property of unbalanced


eigenspectra, where linguistic variance overwhelms visual evidence. As multimodal models


5. RELATED: MULTIMODAL MODELS AND HALLUCINATION 33


continue to scale, bridging RMT analysis, causal reasoning, and alignment training will be


crucial to achieving reliable cross-modal intelligence.


CHAPTER 4

### **EigenTrack: Spectral Detection Framework**


_This chapter is based in part on the author’s publication “EigenTrack: Spectral Activation_


_Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs”_


[ **135** ] _. The text and figures have been expanded and reformulated for clarity and completeness._


_Paper available on arXiv and submitted to ICASSP 2026 conference, under review._ _See_


_Appendix A_


**1. Methodology**


EigenTrack introduces a real-time, interpretable framework for detecting hallucinations


and out-of-distribution (OOD) behavior in large language and vision–language models. Rather


than relying on surface-level probabilities, EigenTrack monitors how the internal geometry


of activations evolves during generation. The key idea is that when a model begins to hallu

cinate or drift from the training distribution, its internal representations lose structure and


become increasingly isotropic. This degradation can be quantified through the eigenvalue


spectrum of hidden-layer covariances.


Hallucinations in large generative models often emerge gradually, not as isolated anom

alies. During early steps of generation, activations are structured and highly correlated,


reflecting a coherent internal reasoning process. As hallucination begins, these correlations


decay and the internal state becomes more random. Random Matrix Theory (RMT) provides


a natural lens for analyzing this behavior. In- distribution activations produce covariance


spectra with a few large, dominant eigenvalues. When representations lose structure, the


spectrum collapses toward the Marchenko–Pastur (MP) law, which characterizes uncorre

lated random noise. By tracking spectral statistics over time, one can therefore detect when


the model’s reasoning dynamics begin to diverge from stable behavior.


34


1. METHODOLOGY 35


Figure 9. General architecture of EigenTrack: Spectral features extracted
from hidden activations are streamed into a recurrent discrepancy detector,
which outputs early warnings of hallucination or OOD drift. In this pipeline,
the problem is framed as binary classification.


**Overview of the Framework.** EigenTrack operates as an auxiliary monitoring head


attached to any pretrained model, requiring no modification of model parameters. The


pipeline comprises three components:


_•_ **Spectral Feature Extraction:** Hidden activations from selected layers are col

lected in a sliding temporal window. Their covariance matrices are analyzed through


a truncated singular value decomposition (SVD) to obtain eigenvalues describing the


structure of the hidden-state manifold.


_•_ **Spectral Discrepancy Detection:** From each window, a compact vector of spec

tral descriptors is computed, including entropy, leading eigenvalues, spectral gaps,


and divergence from the MP distribution. These descriptors reflect how structured


or random the current representations are.


_•_ **Temporal Modeling and Early Warning:** A lightweight recurrent network ob

serves the evolution of spectral descriptors over time. By learning patterns asso

ciated with representational collapse, it outputs a probability that the model is


entering a hallucinatory or OOD regime.


1. METHODOLOGY 36


**Spectral Feature Extraction from Hidden Representations.** Let the model layers


be _L_ 1 _, L_ 2 _, . . ., Lm_ . At each generation step _t_, activations from the selected layers are con

catenated: _vt_ = [ _h_ 1 _,t∥h_ 2 _,t∥· · · ∥hm,t_ ] _,_ _hℓ,t ∈_ R _[d]_ . To capture short-term temporal evolution,

_⊤_

                                 -                                 EigenTrack maintains a sliding window of the most recent _N_ steps: _Ht_ = _vt−N_ +1 _, . . ., vt_ _∈_


1
R _[N]_ _[×][md]_ . The empirical covariance is then _Ct_ = _N_ _[H]_ _t_ _[⊤][H][t]_ [and its eigenvalues are obtained ef-]

ficiently via the truncated singular value decomposition _Ht_ = _Ut_ Σ _tVt_ _[⊤][,]_ _λt,i_ = _σNt,i_ [2] [Because]


_N ≪_ _md_, the SVD is computationally cheap, and randomized or incremental updates allow


near real-time operation during autoregressive decoding. Each _λt,i_ measures how variance


is distributed across activation directions, providing a geometric fingerprint of the model’s


internal dynamics.


Figure 10. Layer-level spectral signature: extraction and temporal tracking.
Selected layers feed activations into the spectral feature extractor, which computes eigenvalue-based descriptors and streams them to the discrepancy detector.


**Spectral Descriptors.** From the spectrum _{λt,i}_, EigenTrack derives a compact vector


_Ft_ of interpretable descriptors. It includes 22 features in total, among which we have:


_•_ **Leading Eigenvalue Sum:**



_st_ =



5


_λt,i_ (1)

_i_ =1


1. METHODOLOGY 37


measuring how much variance is concentrated in the top principal directions. Larger


values indicate low-rank, structured representations, while smaller values suggest


diffuse or noisy activations.


_•_ **Spectral Entropy:**



_λt,i_
_pt,i_ log _pt,i,_ _pt,i_ = ~~�~~
_i_ _j_ _[λ]_




 _St_ = _−_



(2)
_j_ _[λ][t,j]_



where high entropy indicates isotropic, less informative embeddings, and low entropy


reflects concentrated, coherent feature directions.


_•_ **KL Divergence from the Marchenko–Pastur Baseline:**


_DKL_          - _p_ ( _λt_ ) _∥_ _ρMP_ ( _λ_ )� (3)


where _ρMP_ ( _λ_ ) denotes the Marchenko–Pastur eigenvalue density. Small divergence


suggests noise-like, unstructured activations; larger divergence indicates meaningful


statistical structure. The same also applies when quantifying the difference in dis

tributions with the **Wasserstein distance**, which is another feature we consider in


our classifier.


_•_ **Tracy–Widom Fluctuation of the Leading Eigenvalue:** The probability that


_λt,_ 1 significantly deviates from the MP bulk edge _λ_ +:


P� _λt,_ 1 _> λ_ + + _δ_             - (4)


A substantial deviation indicates emergence of a dominant, non-random direction


(signal-bearing subspace).


_•_ **Spectral Gap Ratio:**


_λt,i_
_gt,i_ = _,_ _i ∈{_ 1 _, . . ., k}_ (5)
_λt,i_ +1


capturing how sharply leading eigenvalues detach from the rest. Larger gaps mark


transitions between structured and noise-dominated subspaces.


1. METHODOLOGY 38




_•_ **Spectral Skewness:**



3

(6)



_γt_ = [1]

_d_



_d_



_i_ =1




- ¯
_λt,i −_ _λt_

_σt_



Positive skewness implies a heavy tail of large eigenvalues (structured information),


while near-zero skewness suggests noise-like balance.


**Temporal Modeling with Recurrent Tracking.** A single spectral snapshot can re

veal instability, but hallucinations typically evolve gradually as the internal representation


quality degrades over time. To capture this temporal evolution, EigenTrack treats the de

scriptors as a time series _{F_ 1 _, F_ 2 _, . . ., FT_ _}_ and processes them using a small recurrent neural


network. The feature sequence ( _F_ 1 _, . . ., FT_ ) is modeled as a multivariate time series and pro

cessed by a lightweight recurrent model (RNN, GRU, or LSTM). At each step, _Ft_ enters the


recurrent cell, hidden states propagate contextual information, and a feed-forward head out

puts a binary logit corresponding to in-distribution versus anomalous context. Weight shar

ing across time ensures that the parameter count remains independent of sequence length,


allowing the model to learn characteristic spectral signatures associated with stable or un

stable dynamics.


This recurrent formulation provides three main advantages. First, it preserves temporal


continuity by maintaining a memory of previous spectral states, enabling the detection of


gradual drifts in representation quality. Second, it is computationally efficient, with updates


occurring in constant time per step and minimal additional overhead. Third, it preserves


causality by processing only past information, making it suitable for real-time inference and


early warning during generation. At each generation step, the hidden state evolves according


to _zt_ = RNN( _Ft, zt−_ 1), and the anomaly probability is computed as ˆ _yt_ = _σ_ ( _Wzt_ + _b_ ). If


_y_ ˆ _t > τ_, the detector raises a gating signal that can intervene by halting decoding, triggering


retrieval grounding, or lowering the generation temperature. The recurrent head is highly


compact, containing only a few thousand parameters, and can operate concurrently with


model inference without significant latency.


1. METHODOLOGY 39


Figure 11. Temporal evolution of spectral statistics: The plots illustrate
how spectral features evolve over time. Hallucinated sequences concentrate
variance in a few dominant directions, producing higher cumulative sums of
the top eigenvalues (top-left). Their spectra are flatter and more dispersed,
resulting in elevated entropy values (top-right). From a random matrix theory
perspective, if activations were purely uncorrelated noise, their eigenvalue distribution would follow the Marˇcenko–Pastur law [ **65** ]. Relative to this baseline,
hallucinated sequences remain closer to the noise-like regime, yielding lower KL
divergence values (bottom-left), while factual sequences diverge more strongly,
consistent with structured and informative dynamics. The median eigenvalue
(bottom-right) is higher and more stable for hallucinations, whereas factual
sequences show lower and more variable medians that correlate with model
confidence.


EigenTrack supports tuning of the monitored layer set _L_, window length _N_, number of


spectral features _k_, and recurrent hidden size to balance accuracy and latency. The pipeline


also extends naturally to multimodal architectures by constructing _Ht_ from cross-modal


2. THEORETICAL JUSTIFICATION 40


fusion layers or vision encoder blocks, enabling spectral monitoring across both language


and visual representations within a unified latent space.


**Efficiency and Practical Considerations.** EigenTrack is designed to be lightweight,


modular, and fast enough to operate in real time alongside large generative models. In prac

tice, only a small subset of layers is monitored, typically one every three or four transformer


blocks, since adjacent layers exhibit highly correlated spectral behavior. This sampling


strategy minimizes data transfer without sacrificing sensitivity. Covariance estimation is im

plemented through truncated or incremental SVD, which avoids full matrix reconstruction


and enables efficient eigenvalue updates at each decoding step. The recurrent head itself


contains only a few thousand parameters and performs constant-time updates, making it


several orders of magnitude smaller and faster than transformer-based temporal models. For


this reason, recurrent networks were preferred over transformers: they preserve causal pro

cessing, require negligible memory, and operate with minimal computational overhead. As a


result, EigenTrack can be integrated directly into autoregressive decoding loops, providing


continuous reliability monitoring with only a small additional latency.


**2. Theoretical Justification**


**Spectral Geometry as a Diagnostic Lens.** The theoretical foundations of Eigen

Track arise from the observation that large neural networks, despite their nonlinear nature,


exhibit emergent regularities that can be studied through the geometry of their hidden


representations. The key insight is that during normal operation, the network’s represen

tations inhabit a structured, low-dimensional manifold, while under distributional shift or


hallucination-inducing conditions, this manifold collapses toward isotropic randomness.


Modern decoder-only LLMs apply LayerNorm at every transformer block [ **9** ], centering


activations and scaling them to unit variance. Combined with the near-orthogonality of pro

jection matrices induced by weight decay and adaptive optimization methods such as Adam


[ **137** ], this makes per-token activations approximately mean-zero and isotropic. Under these


2. THEORETICAL JUSTIFICATION 41


assumptions, the activations of each layer can be modeled as random samples from an ap

proximately isotropic Gaussian ensemble, and their covariance spectra follow well-known laws


from Random Matrix Theory (RMT). Departures from this baseline indicate the emergence


of structure or instability within the model’s internal representations. When the network


encounters out-of-distribution (OOD) inputs or hallucination-inducing contexts, hidden ac

tivations exhibit correlated, low-rank perturbations that follow the spiked covariance model.


**The Random Matrix Baseline.** Random Matrix Theory provides analytical tools


to model the spectrum of large, unstructured covariance matrices. Consider a matrix _X ∈_


R _[N]_ _[×][d]_ whose entries are independent and identically distributed with zero mean and variance


_σ_ [2] . In the asymptotic limit where _N, d →∞_ and the aspect ratio _q_ = _d/N_ is fixed, the


empirical distribution of eigenvalues of the sample covariance _C_ = [1]

_N_ _[X]_ _[⊤][X]_ [ converges to the]


_Marchenko–Pastur (MP) law_ [ **65** ]. The density of eigenvalues under this null model is given


by



1
_ρ_ MP( _λ_ ) =
2 _πσ_ [2] _qλ_







( _λ_ + _−_ _λ_ )( _λ −_ _λ−_ ) _,_ _λ ∈_ [ _λ−, λ_ +] (7)



with the support edges defined as _λ±_ = _σ_ [2] (1 _±_ _[√]_ ~~_q_~~ ~~)~~ [2] This distribution characterizes the “noise


floor” of high-dimensional representations. Any activation covariance spectrum that aligns


closely with _ρ_ MP can be interpreted as statistically equivalent to a random ensemble with no


significant correlations. Conversely, systematic deviations, such as the emergence of outlier


eigenvalues or heavy tails, signal the existence of structured correlations and dependencies


in the underlying representations. A similar concept, related to the top eigenvalue deviation,


is modeled by the Tracy-Widom distribution.


An analogous principle holds for uncentered activations or correlation matrices whose


entries are symmetrized rather than formed from _X_ _[⊤]_ _X_ . In this setting, the eigenvalue density


converges to the _Wigner semicircle law_ [ **131** ], which describes the distribution of eigenvalues


of symmetric random matrices. The Marchenko–Pastur, Tracy-Widom and Wigner laws form


the theoretical backbone of EigenTrack’s null model: they represent the spectral geometry


of complete randomness against which meaningful structure can be contrasted.


2. THEORETICAL JUSTIFICATION 42


**Signal Emergence on the Spiked Covariance Model.** The empirical covariance of a


trained model’s activations rarely conforms to the pure MP law. Instead, it follows a _spiked_


_covariance model_ [ **132** ], in which a low-rank signal subspace is embedded within a high

dimensional noisy background: _Ct_ = _σ_ [2] _I_ + [�] _i_ _[k]_ =1 _[θ][i][u][i][u]_ _i_ _[⊤]_ [Here,] _[ σ]_ [2] _[I]_ [ represents isotropic noise,]

and _{ui}_ _[k]_ _i_ =1 [are orthogonal directions capturing coherent, semantically meaningful structure.]


The scalars _θi_ denote the signal strengths along those directions. In the spectral domain,


the presence of such low-rank perturbations produces outlier eigenvalues that detach from


the MP bulk. This phenomenon is governed by the _Baik–Ben Arous–P´ech´e (BBP) phase_


_transition_ [ **20** ], which defines a critical signal-to-noise threshold:


_λs > σ_ [2] (1 + _[√]_ ~~_q_~~ ~~)~~ = _⇒_ signal eigenvalue separates from the bulk. (8)


When _λs_ falls below this threshold, the signal direction becomes indistinguishable from noise,


and its corresponding eigenvalue reabsorbs into the random bulk. This threshold delineates


the boundary between ordered and disordered regimes in the representation space. Within


deep neural networks, such transitions correspond to phases where latent representations


lose alignment with meaningful semantic directions—an effect that empirically correlates


with hallucinations or failures of grounding in LLMs and VLMs.


**From Static Spectra to Temporal Dynamics.** Traditional RMT analyses treat _Ct_ as


a static object, characterizing one snapshot of representational geometry. However, in autore

gressive models, activations evolve dynamically with each generated token. Let _{ρt_ ( _λ_ ) _}_ _[T]_ _t_ =1


denote the sequence of spectral densities observed across a generation trace. EigenTrack


extends the RMT framework by modeling this sequence as a stochastic process in the space


of spectral measures. This approach captures not only instantaneous anomalies but also the


temporal precursors of instability.


Formally, define a set of spectral statistics _ϕt_ = _f_ ( _ρt_ ) summarizing the current spectral


state (e.g., curvature, skewness, or divergence from MP baseline). The trajectory _{ϕt}_


can then be regarded as a dynamical system: _ϕt_ +1 = _F_ ( _ϕt_ ) + _ϵt_ where _F_ denotes the


underlying transition operator induced by the model’s internal dynamics and _ϵt_ is a stochastic


3. EXPERIMENTAL SETUP 43


perturbation. When the model processes coherent, in-distribution input, _F_ is approximately


stationary: the geometry of the activation manifold evolves smoothly. Under hallucination


or distributional shift, however, the transition dynamics become unstable, producing abrupt


spectral fluctuations and non-stationarity in _ϕt_ . These deviations serve as early-warning


signals of representational collapse.


**Implications for Model Reliability.** In summary, the theoretical justification of Eigen

Track rests on three pillars:


(1) **Random Matrix Theory:** establishes the null hypothesis of isotropic randomness,


represented by the Marchenko–Pastur and Tracy–Widom distribution.


(2) **Spiked Covariance Theory:** explains how deviations from the MP law arise


from coherent, task-aligned structure, and how their disappearance signals repre

sentational collapse.


(3) **Temporal Spectral Dynamics:** connects the evolution of eigenvalue distributions


to the stability of the model’s internal manifold during generation.


Together, these principles provide a rigorous mathematical framework linking spectral


geometry to reliability in deep generative models. EigenTrack operationalizes this connection


by tracking the time evolution of the spectrum, transforming abstract random-matrix theory


into a practical diagnostic of hallucination risk.


**3. Experimental Setup**


**Overview of Evaluation Protocol.** The experimental evaluation of EigenTrack is de

signed to assess its ability to detect hallucinations and out-of-distribution (OOD) behavior


across a wide range of open-source large language models (LLMs) and vision-language models


(VLMs). All experiments are conducted on publicly available architectures from the Hug

gingFace Hub, spanning models from 1B to 8B parameters. The evaluated families include


LLaMa, Qwen, Mistral, and LLaVa, each tested in both base and instruction-tuned variants.


For every model, generation is limited to sequences of up to 128 tokens. During infer

ence, the full stream of hidden activations across layers is captured in real time. To facilitate


3. EXPERIMENTAL SETUP 44


efficient computation, caching mechanisms are activated and configured to retain all inter

mediate hidden states. The generation temperature is fixed at 0.2 to ensure deterministic


and stable token sampling, reducing stochastic variability across runs.


**Hallucination Detection Setup.** To evaluate hallucination detection, we adopt a con

trolled and reproducible QA-based pipeline built upon the HaluEval dataset, which derives


from HotpotQA. Each passage is paired with two types of questions: its true, factually


grounded question, and a randomly selected unrelated question generated by LLaMa-8B.


The corresponding answers to these paired questions yield factual (non-hallucinated) and


hallucinated model outputs respectively.


This setup involves three distinct interacting components, instantiated as separate mod

els:


_•_ **Main Model:** The model under analysis (typically the smallest one in the family)


whose hidden activations are monitored by EigenTrack.


_•_ **Question Generator:** A larger model (LLaMa-8B) used to produce semantically


unrelated or misleading questions that trigger hallucinations.


_•_ **Answer Judge:** A separate LLM employed to automatically verify factual consis

tency between the question, passage, and generated response.


Through this tri-model interaction, the system constructs a large, automatically labeled


dataset of hallucinated versus truthful generations, without the need for manual annotation.


For multimodal evaluation, the same procedure is extended to VLMs such as LLaVa, where


text passages are combined with images from the Flickr8k dataset. Again, LLM-as-a-judge


is employed to verify if the answer was factual or hallucinated.


**Out-of-Distribution Detection Setup.** EigenTrack is also evaluated on OOD detec

tion tasks to test its sensitivity to semantic domain shift. For this purpose, the WebQues

tions dataset is used as the in-distribution (ID) source, while the Eurlex dataset, composed


of European legal-domain queries, is treated as the OOD counterpart. Since Eurlex data lies


outside the pretraining distribution of the tested models, it naturally induces OOD behavior.


3. EXPERIMENTAL SETUP 45


As in the hallucination experiments, each model receives one question at a time and


generates a textual response of up to 128 tokens, with activations streamed during decoding.


For VLMs, we give images from Flickr8k as context and a prompt to describe them for


ID samples, for OOD ones, we ask to respond to EurLex questions. This consistent setup


allows EigenTrack to be tested across both unimodal and multimodal configurations without


architecture-specific modifications.


**Classifier Training.** At each generation step, EigenTrack converts the hidden activa

tions into a compact spectral representation. This produces a time series of spectral de

scriptors, already defined in the previous section, which capture the temporal evolution of


the representation geometry. Each sequence of spectral vectors serves as input to a light

weight recurrent classifier designed to distinguish between stable (in-distribution, factual)


and unstable (OOD, hallucinated) trajectories. Three recurrent architectures are considered:


a simple RNN, a GRU, and an LSTM. Each classifier consists of a linear input projection, a


single recurrent layer, and a binary output head producing a probability score for anomalous


behavior. The recurrent hidden state dimension is set to 16. All classifiers are trained using


the Adam optimizer with learning rate 10 _[−]_ [3] and weight decay 10 _[−]_ [4] . Training is performed


with a binary cross-entropy loss over labeled sequences. A final linear projection layer is


applied after the recurrent unit to map the hidden representation to the two output logits.


**Evaluation Metrics.** Detection performance is measured using the area under the re

ceiver operating characteristic curve (AUROC). This metric quantifies the separability be

tween the positive (hallucinated or OOD) and negative (factual or ID) classes. An AUROC


of 0.5 corresponds to random guessing, while a score of 1.0 indicates perfect discrimination.


All results are reported as mean AUROC values averaged over multiple runs and datasets


for both LLM and VLM settings.


This evaluation protocol ensures that the results reflect not only instantaneous perfor

mance but also the robustness of the proposed spectral tracking method across different


architectures, modalities, and failure types.


4. RESULTS 46


**4. Results**


This section reports the empirical performance of EigenTrack on hallucination detection


and out-of-distribution (OOD) shift identification across language-only and vision–language


backbones. We summarize results with two comparative bar plots and two tables. For each


artifact, we provide an interpretive commentary that links the observed trends back to the


spectral-temporal design of the method.


Figure 12. AUROC for hallucination detection: across LLaMa (1B/3B/7B),
Qwen (1.8B/7B), Mistral-7B, and LLaVa-7B using RNN, GRU, and LSTM
temporal heads.


12 reveals a consistent and interpretable trend across all evaluated model families. Eigen

Track achieves strong performance, maintaining AUROC values between 0.82 and 0.94 across


the full spectrum of model sizes. Among the recurrent architectures, GRUs deliver the high

est overall performance, followed closely by LSTMs, while vanilla RNNs trail slightly behind.


This ranking underscores the advantage of gated recurrent mechanisms—particularly their


4. RESULTS 47


ability to retain long-term dependencies and selectively filter spectral fluctuations over time.


The improved sensitivity of GRUs suggests that hallucination events are preceded by subtle,


low-frequency variations in spectral statistics that require controlled temporal integration


rather than simple memory recurrence.


A clear correlation emerges between model size and detection performance. Within the


LLaMa family, for instance, AUROC improves steadily from approximately 0.84 in LLaMa

1B to nearly 0.89 in LLaMa-7B. This positive scaling trend indicates that larger models,


by virtue of their higher representational capacity, generate more structured and separable


spectral signatures. These richer eigenvalue dynamics allow EigenTrack’s recurrent back

end to identify deviations from the stable spectral regimes that characterize factual, coherent


generations. The strongest results are observed on 7B-scale architectures, particularly Qwen

7B and LLaVa-7B, where GRUs reach AUROC values exceeding 0.93, marking a high degree


of discriminative reliability. 12 demonstrates that EigenTrack generalizes effectively across


diverse foundation models and that its spectral-temporal framework benefits both from larger


model capacities and from the inductive bias of gated recurrence. The consistency of these


results supports the hypothesis that hallucinations manifest as measurable perturbations


in the eigenvalue geometry of activation covariances, patterns that GRU-based tracking


captures with remarkable precision.


The AUROC results for OOD detection, shown in 13, exhibit a pattern closely mirror

ing that of hallucination detection while achieving overall higher performance levels. Across


all tested models, EigenTrack attains AUROC scores ranging from approximately 0.85 to


0.96, reflecting robust and consistent OOD sensitivity. Once again, GRU-based classifiers


consistently outperform LSTM and vanilla RNN counterparts, underscoring the importance


of gated temporal memory in capturing the gradual spectral shifts that accompany domain


transitions. The performance gap between GRUs and simpler recurrent cells is less pro

nounced in smaller models such as LLaMa-1B and Qwen-1.8B, suggesting that even basic


4. RESULTS 48


Figure 13. AUROC for OOD detection: on the same set of models using
RNN, GRU, and LSTM temporal heads. Same idea to classify OOD instead
of hallucination.


recurrence can effectively capture spectral drift when the representational complexity is lim

ited, while gated mechanisms become more advantageous as model scale and spectral richness


increase.


A clear upward trend is observed with increasing model scale. Larger architectures such


as LLaMa-7B and LLaVa-7B exhibit AUROC values surpassing 0.92, implying that more


complex networks produce activation spectra with richer statistical structure and sharper


deviations under distributional shift. These results demonstrate that EigenTrack’s spectral


features generalize across scales, capturing the transition from in-distribution to out-of

distribution regimes through eigenvalue dispersion and Marchenko–Pastur divergence.


Interestingly, OOD detection tends to outperform hallucination detection overall, as do

main shifts often trigger broader and more coherent changes in the global spectral landscape


4. RESULTS 49


than the subtler local instabilities preceding hallucinations. This observation aligns with the


theoretical expectation that shifts in data manifold geometry induce distinct, high-amplitude


spectral perturbations that are easier to isolate. By integrating these spectral descriptors


with lightweight GRU-based temporal modeling, EigenTrack achieves strong OOD general

ization with minimal architectural overhead, validating its efficiency and interpretability as


a spectral–temporal reliability framework for large-scale models.


Table 1. SOTA COMPARISON ON LLAMA


**Hallucination Detection** **OOD Detection**


**Method** **1B** **3B** **7B** **Method** **1B** **3B** **7B**


**EigenTrack** **84.2** **86.1** **89.4** **EigenTrack** **85.5** **89.2** **92.4**


LapEigvals 78.5 81.9 87.1 Cosine Distance 81.9 87.7 92.0


INSIDE 75.3 83.1 81.0 Energy Score 83.2 85.2 89.0


SelfCheckGPT 73.9 80.4 80.9 Max Softmax Prob 70.1 71.0 72.0


HaloScope 82.0 82.7 86.1 ODIN 80.1 84.2 92.1


1 presents the comparison between EigenTrack and representative state-of-the-art detec

tors on the LLaMa model family. It shows the AUROC comparison on LLaMa models


(1B/3B/7B) between EigenTrack and representative baselines for hallucination and OOD


detection. Each half of the table shows one task (Hallucination Detection and OOD Detec

tion).


For hallucination detection, EigenTrack achieves AUROC values of 84.2, 86.1, and 89.4


on the 1B, 3B, and 7B models, respectively, surpassing spectral baselines such as LapEigvals


(81.9 on LLaMa-3B, 87.1 on LLaMa-7B) and self-consistency approaches like HaloScope


and INSIDE, which remain below 83 on smaller models. The performance gap widens with


model scale, emphasizing the capacity of EigenTrack’s temporal spectral modeling to capture


evolving internal dynamics that become more pronounced in larger networks.


5. ABLATION STUDIES 50


In the OOD detection setting, EigenTrack again dominates across scales, reaching 92.4


AUROC on LLaMa-7B compared to 92.0 for Cosine Distance and 89.0 for Energy Score, while


simpler baselines such as Max Softmax Probability fall sharply below 72. These differences


underscore that static confidence-based metrics saturate quickly, whereas EigenTrack’s use of


evolving spectral statistics maintains discriminative power across both subtle and pronounced


domain shifts.


These capture global representation dynamics that surface-level confidence methods (Max


Softmax, ODIN) and snapshot spectral analyses (LapEigvals) miss. Baseline OOD methods


are score-based without OOD supervision; their AUROC values are obtained by sweeping


thresholds over in-distribution scores, ensuring threshold-independent and fair comparison.


2 shows the comprehensive EigenTrack results for hallucination and out-of-distribution


(OOD) detection across multiple language and vision–language backbones. The table reports


both the AUROC and the F1 score for each model and temporal head. AUROC quantifies


the overall discriminative power of EigenTrack in distinguishing normal from anomalous


states, providing a threshold-independent indicator of reliability detection quality. The F1


score complements this by measuring the balance between precision and recall at the optimal


operating point, reflecting how effectively the detector can identify failure instances without


excessive false alarms. Together, these metrics capture both the robustness and practical


usability of the spectral–temporal detection framework.


**5. Ablation Studies**


**Accuracy–Latency Trade-off.** This section investigates how the spectral dynamics


captured by EigenTrack depend on temporal and architectural parameters. In particular, we


study how the sliding-window size and the number of generated tokens influence the overall


detection accuracy, measured by the Area Under the Receiver Operating Characteristic


(AUROC). These ablation studies clarify the trade-off between response time and reliability,


providing practical guidance for deploying EigenTrack in different operating regimes.


5. ABLATION STUDIES 51


Table 2. FULL METRICS ACROSS MODELS


**AUROC Full** **F1 Full**


**Model** RNN GRU LSTM RNN GRU LSTM


LLaMa 1B 0.799 0.842 0.831 0.750 0.790 0.783


LLaMa 3B 0.832 0.861 0.844 0.779 0.808 0.794


LLaMa 7B 0.853 0.894 0.872 0.805 0.851 0.825


Qwen 1.8B 0.724 0.824 0.821 0.672 0.798 0.774


Qwen 7B 0.842 0.931 0.922 0.794 0.881 0.870


Mistral 7B 0.864 0.888 0.871 0.812 0.839 0.819


LLaVa 7B 0.902 0.941 0.934 0.853 0.892 0.887


LLaMa 1B 0.825 0.855 0.852 0.776 0.814 0.802


LLaMa 3B 0.858 0.892 0.871 0.810 0.841 0.821


LLaMa 7B 0.879 0.924 0.897 0.829 0.874 0.847


Qwen 1.8B 0.762 0.872 0.846 0.713 0.821 0.796


Qwen 7B 0.867 0.948 0.936 0.817 0.898 0.885


Mistral 7B 0.883 0.906 0.892 0.832 0.855 0.842


LLaVa 7B 0.923 0.958 0.946 0.873 0.906 0.897


**Sliding-Window Length and Latency.** Figure 14 shows the evolution of AUROC


as a function of inference latency across several large language and vision-language models,


including LLaMa-1B, LLaMa-3B, LLaMa-7B, Qwen-1.8B, Qwen-7B, Mistral-7B, and LLaVa

7B. Each curve corresponds to a distinct model family evaluated using different sliding

window lengths for the GRU-based classifier that tracks spectral features over time.


Shorter windows enable EigenTrack to capture rapid fluctuations in the covariance spec

trum of activations, thereby improving sensitivity to early anomalies. This finer temporal


resolution leads to higher AUROC at the cost of increased computational load, since more


windows must be processed per sequence. As the window length grows, fewer updates are


5. ABLATION STUDIES 52


required, reducing latency but also coarsening the temporal signal. The curves in Figure 14


reveal that accuracy saturates between approximately 25 and 50 tokens, marking an optimal


trade-off region where inference time remains below 10 milliseconds while AUROC exceeds


0.88 for most models. This balance highlights that EigenTrack can be flexibly tuned depend

ing on whether the application prioritizes real-time responsiveness or maximum detection


precision.


Figure 14. AUROC–latency trade-off: across large language and visionlanguage models. Shorter windows yield finer temporal resolution but higher
latency, while longer windows improve efficiency at minor cost in accuracy.
The shaded area indicates the optimal trade-off region. Experiments were conducted on a reduced sample (20%) of the Hallucination dataset, using GRU
as the classification head.


**Evolution with Number of Generated Tokens.** Figure 15 illustrates AUROC as


a function of the number of generated tokens. The results show that all evaluated models


start near chance-level performance when only a few tokens have been produced, as the


internal representations are still weakly informative. As generation progresses, the AUROC


5. ABLATION STUDIES 53


rises sharply within the first few tokens, reaching a stable plateau once sufficient contextual


information has accumulated in the hidden activations. Across model families, the strongest


improvement occurs between 8 and 16 tokens, after which the gains gradually diminish.


This pattern reveals that hallucination and out-of-distribution cues emerge early in the


generation process. The stabilization of AUROC beyond approximately 32 tokens implies


that additional computation provides limited improvement, suggesting that EigenTrack can


detect reliability issues well before they manifest in the output text. In practice, this enables


efficient monitoring strategies: short responses can be fully analyzed, whereas for long-form


generation only the initial segments need continuous spectral tracking to achieve comparable


accuracy.


Figure 15. AUROC as a function of the number of generated tokens: accuracy improves sharply during early generation and saturates beyond 32 tokens, indicating that spectral signatures of hallucination and distributional
shift arise early in the decoding process. Experiments were conducted on the
Hallucination dataset, using GRU as the classification head.


5. ABLATION STUDIES 54


**Discussion.** These ablation studies show that EigenTrack delivers stable, interpretable


performance across diverse architectures and temporal setups. Its reliability stems from


the spectral evolution of activations rather than parameter tuning. The early AUROC


stabilization indicates that hallucination-related spectral shifts appear almost immediately,


enabling efficient real-time use. This balance of latency and precision makes EigenTrack a


flexible framework, well-suited for both research diagnostics and large-scale safety monitoring


in production.


CHAPTER 5

### **RMT-KD: Random Matrix Distillation**


_This chapter is based in part on the author’s publication “RMT-KD: Random Matrix Theo-_


_retic Causal Knowledge Distillation”_ [ **136** ] _. The text and figures have been expanded and re-_


_formulated for clarity and completeness. Paper available on arXiv and submitted to ICASSP_


_2026 conference, under review. See Appendix A_


**1. Methodology**


**Problem Setting.** The goal of Random Matrix Theoretic Knowledge Distillation (RMT

KD) is to compress a trained network while preserving accuracy, latency, and energy effi

ciency. RMT-KD departs from sparsity- or heuristic-rank methods by identifying and keeping


only the causal directions of hidden activations, as revealed by their spectral geometry. At a


high level, training proceeds in stages: once validation accuracy crosses a stability threshold,


the current layer is analyzed spectrally on a small calibration subset; outlier eigen-directions


beyond the Marchenko–Pastur (MP) noise bulk are deemed informative and retained; a linear


projection block implements the reduction; the reduced model self-distills from the previous


checkpoint to recover accuracy; the loop repeats layer by layer until a target reduction or a


quality bound is reached.


**Data Collection for Spectral Analysis.** Let a calibration set be formed by sampling


a small fraction of the training data that is representative of the task distribution (ten


percent by default). Hidden activations from the target layer are collected by a forward pass


without gradient updates, forming a column-stacked activation matrix **X** _∈_ R _[d][×][n]_ . Here,


dimension _d_ is the layer width and _n_ is the number of collected samples or tokens. The


empirical covariance is then computed **Σ** = [1]

_n_ **[XX]** _[⊤]_ [. For transformer blocks,] **[ X]** [ is taken after]


55


1. METHODOLOGY 56


Figure 16. Overview of the iterative RMT-KD pipeline: training runs normally until the validation metric exceeds a user-specified threshold; at that moment the selected layer is analyzed with Random Matrix Theory on a held-out
calibration subset. The MP bulk edge is estimated to separate noise-like from
structure-bearing components. A projection block is inserted that maps activations to the causal subspace spanned by outlier eigenvectors, and downstream
widths are resized accordingly. The new (narrower, dense) model is then finetuned with a self-distillation objective that aligns its outputs to those of the
pre-reduction teacher checkpoint. The same train–analyze–reduce–distill cycle
repeats across layers until benefits saturate or targets on parameters, latency,
or power are met.


the feed-forward or projection sub-block depending on the reduction site; for convolutional


networks it is taken channel-wise by flattening spatial dimensions into the sample axis.


**Estimating the Noise Bulk via the MP Law.** Under near-isotropic, mean-zero


activations typical of normalized deep layers, the eigenvalue distribution of the empirical


covariance exhibits a noise bulk captured by the MP density. The aspect ratio is defined as


_q_ = _[d]_

_n_ [. The MP support is parameterized by a noise variance] _[ σ]_ [2][ with edges]


_λ−_ = _σ_ [2] (1 _−_ _[√]_ ~~_q_~~ ~~)~~ [2] and _λ_ + = _σ_ [2] (1 + _[√]_ ~~_q_~~ ~~)~~ [2] (9)


RMT-KD initializes _σ_ [2] from a robust statistic of the spectrum (the median eigenvalue by


default), then refines _σ_ [2] by minimizing the squared error between the empirical histogram


1. METHODOLOGY 57


and the MP density on the interval [ _λ−, λ_ +]. The aggressiveness of reduction can be tuned


by shifting the initialization quantile upward, which increases _λ_ + and thereby classifies more


components as noise.


_σ_ 0 [2] [= Quantile]         - _{λi}_ _[d]_ _i_ =1 _[, τ]_         - with _τ ∈_ [0 _._ 4 _,_ 0 _._ 6] (10)


_σ⋆_ [2] [= arg min] _σ_ [2] ��� _ρ_ ( _λ_ ) _−_ _ρ_ MP( _λ_ ; _σ_ 2 _, q_ ) ��22 (11)


**Estimating** _σ_ [2] **from Eigenvalues of Sample Covariance Matrix.** In random matrix


theory, when the data matrix _X ∈_ R _[N]_ _[×][p]_ has i.i.d. entries with mean 0 and variance _σ_ [2], the


sample covariance matrix _S_ = _X_ _[T]_ _X/N_ has eigenvalues _λi_ whose empirical mean satisfies


1 - _p_ 1 - _N_ - _p_
_p_ _i_ =1 _[λ][i]_ [ =] _Np_ _i_ =1 _j_ =1 _[X]_ _ij_ [2] [. By the Law of Large Numbers, this converges to][ E][[] _[X]_ _ij_ [2] [] =]


_σ_ [2] (since those are assumed to have 0 mean) as _N, p →∞_ . Therefore, using the mean


eigenvalue to estimate _σ_ [2] is justified under these ideal conditions. The median eigenvalue


also approximates _σ_ [2], being more robust to outliers in the spikes section. However, this


is still an approximation because: (1) finite _N, p_ cause sampling error, (2) neural network


activations are not perfectly i.i.d., and (3) the presence of spike eigenvalues from correlated


signals biases the mean upward.


**Selecting Causal Directions.** With the refined bulk edge _λ_ + estimated, outliers are


identified by thresholding the spectrum.


_I_ out = _{ i ∈{_ 1 _, . . ., d}_ : _λi > λ_ + _}_ (12)


Let **U** out _∈_ R _[d][×][k]_ denote the eigenvectors corresponding to these _k_ = _|I_ out _|_ outlier eigenvalues.


The causal projection is then formed as an orthonormal basis from **U** out (by construction it


is orthonormal) **P** = **U** _[⊤]_ out [. The reduced activation is the image of the original activation]


through **P**, such that **h** _[′]_ = **Ph** . In code, the projection is implemented as a fixed linear layer


initialized with **P** and optionally fine-tuned jointly with the rest of the network. Downstream


modules are resized from width _d_ to width _k_ . For transformers, reductions are applied


1. METHODOLOGY 58


to token-embedding, intermediate feed-forward, or projection dimensions; for convolutional


networks, activations are reshaped so that spatial locations act as samples and channels


as features. The resulting activation matrix is projected through the learned matrix **P** in


feature space and then reshaped back to the original image tensor dimensions, effectively


reducing the channel dimension without using explicit convolutional filters.


**Self-Distillation for Stability Across Reductions.** To prevent catastrophic forget

ting, each reduction step starts from a teacher checkpoint (pre-reduction) and trains the


student (post-reduction) to match both labels and teacher logits. The total loss is a convex


combination of task cross-entropy and a Kullback–Leibler alignment term.


_L_ = _α L_ CE + (1 _−_ _α_ ) KL� **p** teacher _∥_ **p** student        - (13)


A temperature can be used inside the softmax of both distributions for smoother targets; the


teacher is an exponential-moving-average or the last full-precision checkpoint. The coefficient


_α_ is chosen to maintain validation accuracy while allowing the student to adapt to its lower

dimensional subspace.


**Progressive Layerwise Schedule.** RMT-KD proceeds over a chosen layer order (shallow

to-deep by default). After each reduction and short fine-tuning, validation is re-checked. The


process halts when any of the following happens: target compression or latency is achieved,


the retained-outlier ratio falls below a minimum threshold, or validation drops beyond an


allowed tolerance.


_k_
Stop if or ∆ValAcc _< −ϵ_ or Params _≤_ Target (14)
_d_ _[< ρ]_ [min]


The outlier ratio criterion guards against over-aggressive cuts in deep layers whose spectra


carry denser structure.


1. METHODOLOGY 59


**Computational Complexity.** The spectral step requires covariance formation and an


eigen-decomposition. With _d_ as layer width and _n_ as calibration size, the costs are


_O_ ( _nd_ [2] ) for **XX** _[⊤]_ and _O_ ( _d_ [3] ) for eigen-decomposition. (15)


Because the calibration set is small and only a few layers are reduced per stage, this over

head is negligible relative to training time. Crucially, the reduced networks remain _dense_ :


projection uses compact matrix–vector multiplies that map efficiently to standard GPU ker

nels; no sparse backends or custom kernels are required. On-device memory decreases with


width, lowering activation and weight footprints; data movement and compute scale down


accordingly, yielding both latency improvements and power savings.


**Implementation Notes.** To make the procedure robust in practice, we adopt the fol

lowing defaults, which were effective across transformers and ResNets:


_τ ∈_ [0 _._ 4 _,_ 0 _._ 5] for initializing _σ_ [2] _,_ _α ∈_ [0 _._ 3 _,_ 0 _._ 7] _,_ temperature _∈_ [0 _._ 5 _,_ 1 _._ 5] (16)


Calibration fraction _≈_ 0 _._ 1 _,_ MP fit via _ℓ_ 2 histogram matching on [ _λ−, λ_ +] (17)


For BERT-like models, reductions at the feed-forward inner width and projection dimensions


bring the largest gains; for ResNet-50, activations are flattened so that spatial positions serve


as samples and channels as features, followed by RMT-based projection in feature space and


reshaping back to the original tensor. After each projection insertion, a brief warm-up phase


can be applied to stabilizes training before resuming full fine-tuning.


1. METHODOLOGY 60


**Example Pseudocode of RMT-KD Process.**


**Input:** trained checkpoint _M,_ target layer index _ℓ,_ calibration set _D_ cal


**1.** Collect activations **X** on _D_ cal


**2. Σ** _←_ [1]

_n_ **[XX]** _[⊤]_

**3.** Initialize _σ_ 0 [2] [from spectrum quantile] _[ τ]_


**4.** _σ⋆_ [2] _[←]_ [arg min] _[σ]_ [2] _[ ∥][ρ]_ [�] _[ −]_ _[ρ]_ [MP][(] _[σ]_ [2] _[, q]_ [)] _[∥]_ [2] 2

**5.** _λ_ + _←_ _σ⋆_ [2][(1 +] _[ √]_ ~~_[q]_~~ ~~[)]~~ [2]


**6.** Select outliers _{λi > λ_ + _}_ and eigenvectors **U** out


**7. P** _←_ **U** _[⊤]_ out _[,]_ **h** _[′]_ _←_ **Ph**


**8.** Insert projection block; resize downstream widths to _k_


**9.** Fine-tune with _L_ = _αL_ CE + (1 _−_ _α_ ) KL( **p** teacher _∥_ **p** student)


**10.** Restart for next layer or stop


Figure 17. Iterative RMT-KD training process: the model is trained until
the validation accuracy surpasses a threshold, after which Random Matrix
Theory is applied layer by layer to compute eigenvalue spectra, insert causal
projections, and rebuild the network with reduced dimensions. The cycle
continues until all target layers are processed and the final compressed model
is validated and saved.


**Why This Design Is Sensible.** The MP bulk fitting provides a statistically grounded


noise floor, eliminating ad hoc cutoffs that plague PCA-style truncation and heuristic rank


selection. Selecting directions by outlier tests aligns the model with a spiked covariance pic

ture in which only a small set of eigen-directions carry task structure. Performing projection


2. THEORETICAL JUSTIFICATION 61


as a dense linear map preserves hardware efficiency. Finally, self-distillation smooths the op

timization landscape across discrete architecture changes, ensuring that compressed models


inherit the function learned by their predecessors rather than relearning from scratch.


**2. Theoretical Justification**


**Random Matrix Theory as a Statistical Lens.** Random Matrix Theory (RMT) pro

vides a rigorous mathematical framework to describe the spectral behavior of large matrices


whose entries are random or exhibit weak correlations. When applied to deep learning,


RMT enables the separation of meaningful, task-relevant structures in neural activations


from random fluctuations that arise during training. In high-dimensional regimes, RMT


offers asymptotic laws that govern the eigenvalue spectra of covariance matrices.


Consider a hidden activation matrix _X ∈_ R _[d][×][n]_ formed by the activations of _d_ neurons


across _n_ samples. The empirical covariance matrix is defined as Σ = [1]

_n_ _[XX]_ _[⊤]_ [whose eigenval-]

ues _{λi}_ _[d]_ _i_ =1 [encode the variance distribution of the learned representations. RMT predicts]


that for large _d, n_, the eigenvalue density of such random covariance matrices converges to the


Marchenko–Pastur (MP) distribution [ **65** ], given by _ρ_ MP( _λ_ ) = 2 _πλqσ_ 1 [2] ~~�~~ ( _λ_ + _−_ _λ_ )( _λ −_ _λ−_ ) _, λ ∈_


[ _λ−, λ_ +] where _q_ = _d/n_, _σ_ [2] is the variance of the entries of _X_, and the bulk edges are defined

as _λ±_ = _σ_ [2] (1 _±_ _[√]_ ~~_q_~~ ~~)~~ [2] . Eigenvalues within the interval [ _λ−, λ_ +] form the _bulk_, correspond

ing to random noise. In contrast, eigenvalues _λi > λ_ + represent statistically significant


deviations, so-called outliers, which capture structured or causal information within the ac

tivations. This separation between noise and structure provides a statistically grounded rule


for identifying informative directions in representation space.


**BBP Phase Transition.** The theoretical foundation for distinguishing signal from noise


lies in the spiked covariance model [ **132** ]. In this model, the covariance matrix is assumed


to consist of a low-rank structured component plus isotropic noise: Σ = Σ _s_ + _σ_ [2] _Id_ where


Σ _s_ encodes the task-relevant directions (the “spikes”). When the signal strength exceeds a


critical threshold, that is known as the Baik–Ben Arous–P´ech´e (BBP) transition [ **20** ], the


2. THEORETICAL JUSTIFICATION 62


corresponding eigenvalues detach from the MP bulk. The BBP threshold is defined as:


_λ_ BBP = _σ_ [2] (1 + _[√]_ _c_ ) _,_ where _c_ = _[d]_ (18)

_n_ _[.]_


Eigenvalues _λi > λ_ BBP correspond to genuine signals whose eigenvectors align with semanti

cally meaningful or causal dimensions of the data distribution. This mechanism explains how


deep representations evolve from random initializations to structured manifolds as learning


progresses: early in training, the spectrum follows the MP law closely, whereas later stages


exhibit prominent outliers signifying emergent structure.


**Application to Neural Representations.** Deep neural networks, particularly large


language models (LLMs) and vision models (CNNs, VLMs), produce hidden representations


that are extremely high-dimensional but redundantly parameterized. Empirically, their co

variance spectra display a characteristic profile: a large noisy bulk and a small number of


outliers that dominate the representational variance [ **30** ]. RMT thus provides a principled


criterion for model compression and subspace selection, in contrast to heuristic thresholds


used in principal component analysis (PCA) or low-rank approximations.


For a given layer, once the activation covariance Σ is computed from a calibration subset,


the noise variance _σ_ [2] is initialized as the median eigenvalue and refined by minimizing the _ℓ_ 2


distance between the empirical histogram and the MP distribution. The resulting _λ_ + acts as a


dynamic cutoff separating signal from noise. All eigenvectors associated with _λi > λ_ + define a


projection matrix _P ∈_ R _[k][×][d]_, where _k < d_, mapping the activations onto a lower-dimensional


causal subspace: _Y_ = _PX_ . This transformation eliminates noisy or redundant components


while retaining the statistically validated causal structure. The process is repeated layer by


layer, yielding compact, dense networks whose internal representations are aligned with the


intrinsic structure of the data.


**Empirical Evidence and Spectral Interpretation.** Figure 18 visualizes the empir

ical eigenvalue spectra of hidden layer activations in BERT-base at increasing depths. The


2. THEORETICAL JUSTIFICATION 63


red dashed line marks the upper edge _λ_ + predicted by the Marchenko–Pastur law. In the em

bedding block (top left), most eigenvalues cluster near zero, with few extending beyond _λ_ +,


indicative of unstructured, noise-dominated representations. As training proceeds through


successive Transformer blocks, more eigenvalues detach from the bulk, forming distinct out

liers that capture meaningful linguistic or contextual information.


Figure 18. Evolution of the empirical eigenvalue distribution: activation
covariances across BERT-base layers, trained on SST dataset (from GLUE
data). The red dashed line denotes the upper bulk edge _λ_ + predicted by the
Marchenko–Pastur law. The emergence of outlier eigenvalues across layers
indicates the progressive formation of structured, causal representations.


This spectral evolution demonstrates that deep layers encode increasingly specialized,


semantically rich features, while random-like fluctuations dominate earlier layers. Hence,


RMT-guided filtering provides a natural boundary for compression: early layers can be


3. EXPERIMENTAL SETUP 64


aggressively reduced without losing meaningful structure, while deeper layers require con

servative compression to preserve critical information.


**Spectral Geometry for Reliability.** Beyond its role in compression, the spectral anal

ysis of hidden activations connects to broader principles of spectral geometry and information


stability. Large outlier eigenvalues correspond to directions of high curvature or concentrated


information flow, whereas the bulk reflects isotropic noise and instability.


This geometric viewpoint suggests that a model’s reliability, its robustness against hallu

cination or out-of-distribution drift, can also be inferred from spectral signatures. A stable,


well-calibrated model maintains a consistent separation between bulk and outlier regions,


indicating a balanced representation of structure and noise. Conversely, the collapse or


inflation of eigenvalue spectra may signal overfitting, undertraining, or instability under


perturbations.


Thus, the same spectral framework that governs compression in RMT-KD also underpins


diagnostic tools like EigenTrack, establishing a unifying theoretical basis across efficiency and


reliability dimensions. In summary, RMT provides:


_•_ A statistically grounded threshold ( _λ_ + or _λ_ BBP) for separating informative structure


from random noise.


_•_ A causal interpretation of eigenvalue outliers as stable, meaningful directions in


representation space.


_•_ A framework for constructing low-dimensional, dense, and interpretable subspaces


without ad hoc heuristics.


By embedding this spectral analysis into an iterative self-distillation loop, RMT-KD trans

forms deep networks into efficient, causally consistent systems, reducing redundancy while


preserving semantic power.


**3. Experimental Setup**


**Overview.** To evaluate the effectiveness of the proposed RMT-KD framework, we con

duct a series of controlled experiments across both language and vision domains. The goal of


3. EXPERIMENTAL SETUP 65


this setup is to assess whether Random Matrix Theory (RMT)-guided layer compression can


significantly reduce model size and computational cost while preserving accuracy and repre

sentational quality. All experiments are implemented in `PyTorch` using CUDA acceleration,


ensuring full reproducibility and compatibility with common training pipelines.


**Model Architectures.** We evaluate RMT-KD on two representative classes of deep


networks: Transformers for language modeling and Convolutional Neural Networks (CNNs)


for image recognition. This dual evaluation demonstrates the generality of the proposed


framework across architectures that differ in structural topology, activation statistics, and


spectral properties.


**BERT-base.** We adopt a 12-layer Transformer encoder identical to the BERT-base


configuration [ **8** ], comprising 12 self-attention heads per layer, hidden size _d_ = 768, and feed

forward dimension _dff_ = 3072. The model totals approximately 139 million parameters. The


training uses WordPiece tokenization with a vocabulary of 30k tokens. The base model serves


as the initial teacher in the RMT-KD iterative distillation loop, from which progressively


reduced students are generated by projecting activation subspaces based on spectral analysis.


**BERT-tiny.** As a smaller baseline, we also include the 6-layer TinyBERT variant with


44 million parameters. This model has a reduced hidden dimension of _d_ = 384 and 6


attention heads per layer. Evaluating RMT-KD on BERT-tiny provides insight into the


limits of compression in already compact architectures, where redundancy is minimal.


**ResNet-50.** For computer vision experiments, we use the standard 50-layer residual


network [ **138** ] consisting of bottleneck convolutional blocks and batch normalization layers.


The architecture contains 23.5 million parameters and achieves high baseline accuracy on


CIFAR-10. Applying RMT-KD to this convolutional backbone illustrates the adaptability


of the spectral compression principle beyond attention-based architectures.


All models are initialized from scratch using Xavier initialization and trained end-to-end


before any compression step. This ensures that the activation covariances analyzed during


RMT calibration reflect converged, semantically meaningful features.


We evaluate performance on three datasets representing complementary domains:


3. EXPERIMENTAL SETUP 66


**GLUE Benchmark.** The General Language Understanding Evaluation (GLUE) suite


includes multiple text classification and sentence-pair tasks such as SST-2 (sentiment anal

ysis), QNLI (question–answer entailment), and QQP (duplicate question detection). Each


subtask uses standard train/validation/test splits. Text sequences are tokenized to a maxi

mum length of 128 tokens and padded to form uniform mini-batches. Training and evaluation


follow the official GLUE evaluation protocol, with metrics such as accuracy and F1 score.


**AG News.** This dataset contains 120k news articles categorized into four classes (World,


Sports, Business, Sci/Tech). It is used to test the scalability of RMT-KD to large text


corpora. Documents are preprocessed using lowercasing and punctuation normalization,


and truncated to 256 tokens.


**CIFAR-10.** For vision tasks, we use the CIFAR-10 dataset containing 50k training and


10k test images across ten object categories. Each image has a resolution of 32 _×_ 32 pixels.


Standard augmentations including random horizontal flip, random crop with padding of 4


pixels, and per-channel normalization are applied. These augmentations ensure that the


covariance statistics estimated by RMT reflect robust and diverse activations.


**Training Procedure.** All experiments are conducted on a NVIDIA RTX 6000 GPU


with 48 GB of memory. GPU power consumption and energy efficiency are measured using


NVIDIA’s System Management Interface ( `nvidia-smi` ) at 1-second resolution to estimate


average draw during inference. Training follows a two-phase procedure: _(i)_ baseline model


training and _(ii)_ RMT-guided compression with self-distillation.


**Phase I – Baseline Training.** Each baseline model is trained using the AdamW


optimizer with initial learning rate _η_ 0 = 1 _×_ 10 _[−]_ [4] for Transformers and _η_ 0 = 1 _×_ 10 _[−]_ [3] for


CNNs. The learning rate decays linearly during the epochs by a factor of 0 _._ 1. We use weight


decay _λ_ = 0 _._ 0005 and dropout _p_ = 0 _._ 1. Batch sizes are set to 32 for language tasks and 128


for vision tasks, while the numeber of epochs range from 5 to 10. Training continues until


validation accuracy saturates or exceeds a pre-defined threshold _τval_ (typically 90% of the


expected baseline score).


3. EXPERIMENTAL SETUP 67


**Phase II – RMT-KD Compression.** Once convergence is reached, a calibration


subset _D_ cal is sampled, comprising 10% of the training data. Hidden activations _X ∈_ R _[d][×][n]_


from each target layer are extracted to compute empirical covariance matrices Σ = [1]

_n_ _[XX]_ _[⊤]_ [.]


The eigenvalue spectrum _{λi}_ of Σ is estimated using efficient eigendecomposition routines


in `NumPy` . The empirical distribution is compared to the Marchenko–Pastur (MP) law, and


the noise variance _σ_ [2] is initialized as the median of the eigenvalues (or any other chosen


quantile as we’ll see in the ablation studies) and then refined by minimizing the _ℓ_ 2 distance


between the histogram of _{λi}_ and the expected MP density. Eigenvalues _λi > λ_ +, where _λ_ +


is the upper edge of the MP support, are selected as _causal directions_ . Their corresponding


eigenvectors form a projection matrix _P ∈_ R _[k][×][d]_, which maps activations to the reduced


subspace R _[k]_ .


This projection is implemented as a fixed linear layer, inserted between the original layers


of the model. After projection, the model undergoes fine-tuning with a self-distillation loss


that enforces alignment between the original logits _p_ old and the new logits _p_ new:


_L_ = _α L_ task + (1 _−_ _α_ ) _D_ KL( _p_ old _∥p_ new) (19)


where _α_ = 0 _._ 7 balances the supervised task loss and the distillation regularizer. The proce

dure is repeated iteratively across multiple layers, halting when the target compression ratio


or accuracy threshold is reached.


**Evaluation Metrics.** We assess each model on three dimensions:


_•_ **Predictive performance:** classification accuracy and F1 score on the test sets.


_•_ **Computational efficiency:** wall-clock inference time and throughput (samples/sec).


_•_ **Resource usage:** model parameter count, disk size, and GPU power consumption.


Inference latency is measured with batch size 1 to emulate real-time applications, aver

aged over 1000 forward passes. Energy measurements integrate instantaneous power over


the full inference window.


4. RESULTS 68


All training scripts are implemented in modular PyTorch classes to ensure reusability.


Random seeds are fixed for all experiments to ensure determinism. Model checkpoints,


spectra, and projection matrices are saved using `dill` for transparent tracking of compression


stages.


**4. Results**


**Overview.** The effectiveness of the proposed RMT-KD framework is assessed across


diverse model families and datasets to verify its ability to jointly enhance efficiency and


preserve performance. The experiments focus on three representative settings: BERT-base


and BERT-tiny trained on the GLUE benchmark tasks (SST, QQP, and QNLI), and ResNet

50 on CIFAR-10. The analysis includes a comprehensive evaluation of accuracy, compression


ratio, inference speed, energy and power consumption, as well as memory footprint.


All models are fine-tuned under identical optimization and data conditions. The RMT

KD variants differ only in their use of spectral projections derived from the eigenvalue struc

ture of hidden activations. This ensures that any observed variation in behavior can be


attributed to the Random Matrix–based distillation itself, rather than to external hyperpa

rameter tuning or architectural modifications. The following subsections present quantitative


results and interpretive commentary for each major aspect of evaluation.


**Accuracy and Parameter Reduction.** Figure 19 summarizes the fundamental per

formance–compression trade-off achieved by RMT-KD. Across all datasets, the Random


Matrix–guided projection significantly reduces the number of parameters while maintaining,


and in several cases slightly improving, classification accuracy. For BERT-base, parameter


counts fall by approximately 81%, yet the accuracy on SST and QQP increases modestly.


This improvement arises from the elimination of noisy and redundant directions in the rep

resentation space, allowing the model to operate on a more structured, low-dimensional


manifold that retains only causally relevant components.


For the smaller BERT-tiny, compression remains substantial (around 58%) but accuracy


drops minimally, staying within one percentage point of the baseline. This indicates that


4. RESULTS 69


Figure 19. Accuracy and parameter reduction across tasks: accuracy comparison between baseline and RMT-KD models (bars, left axis) together with
parameter count reduction (lines, right axis). RMT-KD attains up to 80.9%
parameter reduction with negligible or positive accuracy variations.


even compact models harbor spectral redundancy, and that RMT-KD successfully extracts


and preserves the dominant signal modes. The behavior of ResNet-50 on CIFAR-10 further


confirms the generality of this principle: despite being convolutional and non-transformer

based, its spectral structure follows the same pattern, leading to almost half the parameters


being pruned with negligible performance loss.


These results demonstrate that the eigenvalue-based selection criterion acts as a robust


proxy for representational importance. The alignment between theoretical prediction and


empirical observation reinforces the claim that the Marchenko–Pastur bulk effectively char

acterizes noise-like fluctuations, while the eigenvalue outliers capture task-relevant structure.


**Inference Speed and Power Consumption.** The improvements in computational


efficiency are illustrated in Figure 20. The RMT-KD models consistently achieve faster


4. RESULTS 70


Figure 20. Inference speedup and power reduction: relative inference
speedup (bars, left axis) and power consumption (lines, right axis) of RMTKD models compared with baseline implementations. Values above the bars
indicate multiplicative speed gains with respect to the uncompressed model.


inference across all configurations, confirming that the spectral projections yield measurable


hardware-level benefits. On BERT-base, inference throughput increases nearly threefold on


SST and QNLI, reaching speedups of 2.88× and 2.82× respectively. This gain originates from


the substantial dimensionality reduction of intermediate representations, which reduces the


cost of matrix multiplications in both the attention and feedforward blocks.


Power measurements show a correlated reduction, with average consumption during in

ference decreasing by nearly half. The overall power profile becomes smoother, with fewer


peaks corresponding to heavy tensor operations. This behavior reflects a denser yet lighter


model whose computation remains contiguous and well-aligned with GPU acceleration pat

terns. The smaller BERT-tiny variants, while starting from a more efficient baseline, still


4. RESULTS 71


benefit from a consistent 30–35% improvement in runtime and reduced wattage, demon

strating that the method scales proportionally across model sizes. Even the convolutional


ResNet-50 sees a slight gain, evidencing that RMT-KD can complement conventional CNN


compression methods without architectural redesign.


From a systems standpoint, these results indicate that spectral compression transforms


the efficiency landscape of transformer models. By maintaining dense matrices while shrink

ing their dimensions according to principled statistical rules, RMT-KD avoids the memory


access inefficiencies typical of sparse pruning and quantization. The resulting models exhibit


both algorithmic and energy-level optimization, suitable for large-scale deployment.


Figure 21. Memory footprint and energy efficiency: comparison of memory
footprint (bars, left axis) and total energy consumption per inference (lines,
right axis) for baseline and RMT-KD models. Energy values represent accumulated consumption over entire forward passes.


**Memory and Energy Efficiency.** Figure 21 highlights the impact of RMT-KD on


memory usage and energy expenditure. The compression achieved in the parameter domain


4. RESULTS 72


directly translates into reduced storage and runtime memory. For the BERT-base family,


the total memory footprint drops by roughly 81%, shrinking from over 500 MB to near


100 MB, while maintaining functional equivalence in downstream predictions. The energy


cost per inference, measured over full sequences, decreases proportionally, indicating that


the reduction in model size and power draw compounds multiplicatively to yield significant


energy savings.


The smaller BERT-tiny models exhibit around 55% memory reduction and roughly 50%


lower energy requirements, a nontrivial improvement given their already compact nature. In


the convolutional setting, ResNet-50 benefits less dramatically due to its structured and spa

tially constrained design, yet still achieves approximately 10% energy reduction. Importantly,


these gains are obtained without introducing sparsity or hardware-specific optimization; the


compressed models remain fully dense, enabling efficient parallel execution on GPUs and


TPUs.


The combined evidence across the three plots underscores the dual virtue of the RMT

KD approach: a substantial acceleration of inference and reduction of computational cost,


accompanied by preserved or improved predictive reliability. The compression operates not


as a heuristic simplification but as a mathematically grounded projection derived from the


spectral geometry of the network itself. By identifying and preserving the eigenmodes that


carry causal structure while discarding noise-like components, the method simultaneously


enhances both efficiency and interpretability.


The results presented above confirm the central hypothesis of this thesis: Random Ma

trix Theory provides a rigorous and practical foundation for understanding and optimizing


deep neural networks. The consistent correlation between eigenvalue outlier preservation


and empirical performance demonstrates that the spectral geometry of neural activations


is a reliable indicator of representational content. Through the RMT-KD process, models


are distilled not by arbitrary pruning or quantization rules, but through a data-driven, the

oretically justified projection that aligns with statistical mechanics principles. The gains


4. RESULTS 73


reported in Figures 19–21 collectively show that spectral methods can reconcile efficiency


and reliability, yielding models that are simultaneously faster and smaller.


Table 3. PERFORMANCE COMPARISON OF RMT-KD


**BERT-base** **BERT-tiny** **ResNet-50**


**Method** **Red.** **Acc.** **Red.** **Acc.** **Red.** **Acc.**


**RMT-KD** **80.9%** **+1.8%** **58.8%** **+1.4%** **47.7%** **+0.7%**


DistilBERT 42.7% +0.2% 54.8% +0.4%  

Theseus 48.3% +0.6% 53.0% +0.1%  

PKD 40.5% -1.0% 50.1% -0.8%  

AT  -  - 42.2% +0.4%


FitNet  -  - 40.6% +0.2%


CRD  -  - 45.4% +0.6%


3 shows the BERT results evaluated on the GLUE benchmark, and ResNet-50 results


reported on CIFAR-10. Theseus = BERT-of-Theseus, PKD = Patient Knowledge Distil

lation, AT = Attention Transfer, FitNet = FitNets (Hints for Thin Deep Nets), and CRD


= Contrastive Representation Distillation. Missing entries (—) indicate that NLP and CV


baselines are evaluated separately. It summarizes the comparative performance of RMT-KD


against several state-of-the-art distillation and compression baselines across both natural


language processing (NLP) and computer vision (CV) domains. For NLP tasks on the


GLUE benchmark, RMT-KD achieves remarkable compression on BERT-base, reducing pa

rameters by over 80% while simultaneously improving accuracy by +1.8% compared to the


original model. Even the lighter BERT-tiny variant benefits substantially, with nearly 60%


reduction and a modest accuracy improvement of +1.4%. In the vision domain, ResNet

50 achieves a 47.7% reduction with a consistent accuracy gain of +0.7%, indicating that


RMT-KD generalizes effectively across modalities. Compared to popular baselines such as


DistilBERT and Theseus, RMT-KD consistently provides larger compression ratios while


5. ABLATION STUDIES 74


also improving or maintaining accuracy. For instance, while DistilBERT compresses BERT

base by roughly 43%, its performance improvement is limited to +0.2%, suggesting that


conventional distillation retains redundant subspaces. By contrast, RMT-KD identifies and


preserves only the causal spectral components, those corresponding to outlier eigenvalues be

yond the Marchenko–Pastur threshold, ensuring that only statistically meaningful directions


are maintained. This theoretically grounded approach enables aggressive compression with

out the instability or heuristic tuning typical of pruning-based and low-rank factorization


methods.


On computer vision benchmarks, RMT-KD also surpasses advanced distillation strategies


such as Contrastive Representation Distillation (CRD) and Attention Transfer (AT), achiev

ing both higher compression and slightly better accuracy. The consistent positive deltas


across all evaluated settings emphasize the robustness of the proposed random-matrix-guided


projection and its ability to produce dense, hardware-efficient student models. Overall, 3


highlights how RMT-KD bridges the gap between mathematical rigor and practical compres

sion, establishing a new balance between efficiency and performance in deep neural networks.


**5. Ablation Studies**


**Quantile Sensitivity Trade-off.** A crucial hyperparameter in RMT-KD is the ini

tialization quantile used to estimate the noise variance _σ_ [2] in the Marchenko–Pastur (MP)


distribution. This quantile determines the upper bulk edge _λ_ + and, consequently, the cut

off threshold separating random from causal eigenvalues. Adjusting this quantile directly


controls the aggressiveness of compression: higher quantiles enlarge _λ_ +, classifying more


eigenvalues as noise and discarding more dimensions, while lower quantiles retain a broader


subspace and yield more conservative reductions.


To systematically study the influence of this parameter, we conducted an ablation ex

periment across a wide range of quantiles, from 0% to 100%, on both natural language and


vision benchmarks. For each quantile, the complete RMT-KD pipeline was executed, involv

ing iterative self-distillation, RMT-guided projection, and fine-tuning. 22 reports model


5. ABLATION STUDIES 75


accuracy and parameter reduction percentages for BERT-base on the GLUE datasets (SST,


QQP, QNLI) and ResNet-50 on CIFAR-10.


Figure 22. Accuracy–reduction tradeoff for RMT-KD: Accuracy–reduction
tradeoff as a function of the initial quantile used to initialize _σ_ [2] . The shaded
region (40–50% quantile) marks the regime providing optimal balance between
accuracy retention and compression across tasks. Experiments are conducted
with BERT-base model on all datasets.


**Observations Across Quantiles.** At low quantiles (below 20%), the estimated _σ_ [2]


is small, leading to narrow MP bulk edges and the preservation of nearly all eigenvalues.


In this regime, compression is minimal, parameter reduction remains below 30%, but task


accuracy remains essentially identical to the baseline, confirming that RMT-KD behaves


conservatively when the noise threshold is underestimated.


As the quantile increases toward the median (40–50%), a marked transition occurs. The


empirical bulk edges _λ±_ now align closely with the statistical envelope predicted by the spiked


covariance model, effectively filtering out noisy directions while retaining meaningful, struc

tured components of the feature space. Across all datasets, this median regime yields the best


compromise between compactness and fidelity: parameter reduction reaches 60–80% while


5. ABLATION STUDIES 76


accuracy degradation remains within 1–2 percentage points. Interestingly, in some cases


(notably SST and QNLI), accuracy slightly improves, suggesting that RMT-guided filtering


removes redundant or detrimental representations, acting as a form of implicit regularization.


Beyond the 60% quantile, compression becomes increasingly aggressive. The bulk edge


_λ_ + grows large enough that even moderately informative eigenvalues are pruned, leading to


excessive dimensionality reduction and a steep accuracy decline. On CIFAR-10, for instance,


the accuracy curve exhibits a sharp drop beyond 70%, while parameter reduction saturates


near its maximum ( _≈_ 90%). This asymmetry underscores a critical insight: overestimating


_σ_ [2] effectively shifts the MP distribution toward high variance, erasing weak but meaningful


correlations in the activations.


The ablation also highlights how model architecture mediates spectral redundancy. Transformer

based models such as BERT display larger tolerance to compression, maintaining near

baseline performance up to 80% parameter reduction. This behavior aligns with the known


overparameterization of Transformer layers, where multiple heads and intermediate projec

tions encode overlapping features. Conversely, convolutional networks like ResNet-50 exhibit


more limited redundancy; performance degradation begins earlier, reflecting tighter coupling


between representational width and task accuracy. These results confirm that the optimal


quantile region is model- and domain-invariant, consistently falling between 40% and 50%


across both NLP and vision tasks. This provides a robust rule-of-thumb for practitioners:


initializing _σ_ [2] at the median eigenvalue delivers statistically valid MP fitting while balancing


efficiency and generalization.


CHAPTER 6

### **Conclusions**


**1. Findings**


This thesis set out to investigate how Spectral Geometry and Random Matrix Theory


can provide a principled foundation for improving both the reliability and efficiency of large


language models and related deep architectures. Two main contributions were developed:


EigenTrack, a real-time detector of hallucination and out-of-distribution behavior, and RMT

KD, a compression framework based on random matrix theoretic knowledge distillation.


Together, these studies establish a coherent view of how eigenvalue dynamics in neural


activations can serve as compact, interpretable signatures of model behavior.


The first major finding concerns reliability. By tracking the temporal evolution of spec

tral statistics, such as entropy, eigenvalue gaps, and deviations from the Marchenko–Pastur


law, EigenTrack demonstrated that hidden activations carry early-warning signals of model


failure. Unlike surface-level uncertainty measures, which are tied to output probabilities,


spectral features capture global properties of representation geometry across multiple layers.


This makes them well suited for hallucination and out-of-distribution detection. The integra

tion of lightweight recurrent classifiers further showed that temporal modeling is essential:


reliability failures emerge not from isolated states but from gradual drifts in representation


dynamics. The empirical evaluations confirmed that EigenTrack surpasses existing black

box, grey-box, and white-box methods, while offering interpretable insights into why and


when failures occur.


The second major finding relates to efficiency. With RMT-KD, the thesis introduced a


compression method that leverages the separation between noise-like bulk eigenvalues and in

formative outliers in the activation spectrum. By iteratively projecting networks onto causal


subspaces defined by outlier eigenvectors and reinforcing stability through self-distillation,


77


2. STUDY LIMITATIONS 78


RMT-KD achieved substantial reductions in model size, inference latency, and energy con

sumption while maintaining state-of-the-art accuracy. Importantly, the models remain dense


and hardware-friendly, distinguishing this approach from sparsity-driven methods that often


sacrifice deployability. This demonstrates that random matrix principles can move beyond


theory to provide practical, scalable tools for model compression.


Taken together, these findings reveal that spectral analysis offers a unifying language for


addressing two of the most pressing challenges in modern AI. The same eigenvalue-based


perspective that uncovers early signs of hallucination can also guide principled compression


strategies. Beyond technical performance, this work shows that spectral geometry makes


deep learning systems more interpretable by linking failure modes and efficiency gains to


mathematically well-characterized properties of high-dimensional data. The contributions of


this thesis thus lie not only in new methods, but also in establishing a spectral framework


that bridges reliability and efficiency in large-scale artificial intelligence.


**2. Study Limitations**


While the contributions of this thesis demonstrate the potential of spectral geometry and


Random Matrix Theory for improving the reliability and efficiency of large language models,


several limitations must be acknowledged. These limitations concern both the scope of the


experiments conducted and the generalizability of the proposed methods.


First, the evaluation of EigenTrack was performed primarily on open-source large lan

guage models and vision-language models of moderate scale, typically up to several billion


parameters. While results show strong performance across these settings, it remains to be


verified how well the method extends to frontier-scale models with hundreds of billions of


parameters, where training dynamics and activation statistics may differ substantially. Sim

ilarly, the hallucination and out-of-distribution datasets employed, though diverse, cannot


capture the full breadth of real-world scenarios in which reliability is critical. Broader testing


on more diverse domains, particularly safety-critical contexts such as medicine or law, will


be necessary before the method can be fully deployed in practice.


2. STUDY LIMITATIONS 79


Second, RMT-KD, though effective in compressing BERT and ResNet architectures, was


not extensively validated on multimodal or generative models. The iterative self-distillation


procedure relies on stable calibration subsets, and its effectiveness may be sensitive to distri

butional shifts between training and deployment environments. Moreover, while the method


reduces model size without inducing sparsity, the cubic complexity of eigenvalue decompo

sition still poses challenges for extremely large layers, even if approximations or randomized


solvers can mitigate this cost.


Third, both EigenTrack and RMT-KD depend on assumptions inherent to Random Ma

trix Theory, such as the applicability of the Marchenko–Pastur law and the spiked covariance


model to high-dimensional neural activations. While these assumptions were empirically sup

ported in the experiments, their validity may vary across architectures, training regimes, and


optimization strategies. Further theoretical analysis is needed to rigorously establish when


spectral signatures reliably distinguish between structure and noise in deep learning.


Finally, the thesis focused on methodological contributions rather than integration into


end-to-end systems. Although EigenTrack and RMT-KD each address an important chal

lenge, their combined deployment in real-world pipelines, where models must be simultane

ously reliable, efficient, and adaptive, was not explored. This integration remains a crucial


step toward translating spectral methods into practice.


3. DIRECTIONS FOR FUTURE RESEARCH 80


**3. Directions for Future Research**


The results of this thesis suggest several promising directions for future research at the


intersection of spectral geometry, random matrix theory, and deep learning. These directions


span theoretical investigations, methodological innovations, and practical applications.


From a theoretical standpoint, there is a need to deepen the mathematical understand

ing of spectral phenomena in large neural networks. While Random Matrix Theory provides


valuable tools such as the Marchenko–Pastur law and spiked covariance models, the behavior


of real-world deep architectures often deviates from these idealized assumptions. Future work


could focus on extending RMT to account for non-i.i.d. structures, correlated activations,


and non-linear dynamics characteristic of transformers and multimodal networks. Estab

lishing rigorous conditions under which spectral outliers consistently align with meaningful


representations would strengthen the interpretability and reliability of spectral methods.


On the methodological side, future research could explore hybrid approaches that combine


spectral signatures with complementary sources of information. For reliability, integrating


EigenTrack with output-based uncertainty estimation, attention analysis, or causal probing


could provide a multi-layered defense against hallucination and distributional shift. For effi

ciency, RMT-KD could be combined with quantization or low-rank factorization to further


reduce resource demands while preserving the advantages of dense, hardware-friendly repre

sentations. Another promising direction is the adaptation of spectral compression techniques


to generative models and vision-language architectures, where efficiency and controllability


are both critical.


At the application level, deploying spectral methods in real-world systems offers sig

nificant opportunities. EigenTrack could be integrated into large language model serving


pipelines as an early-warning mechanism, enabling safer deployment in domains such as


healthcare, law, or education. Similarly, RMT-KD could be leveraged to bring powerful mod

els onto mobile and edge devices, facilitating sustainable AI adoption in resource-constrained


3. DIRECTIONS FOR FUTURE RESEARCH 81


environments. Extending these methods to frontier-scale models will also be crucial, requir

ing advances in scalable spectral analysis such as randomized eigensolvers or distributed


implementations.


In conclusion, this thesis provides an initial step toward a spectral approach to reliable


and efficient deep learning. Expanding these methods across scales, domains, and archi

tectures represents a rich research direction that can significantly advance the safety and


accessibility of artificial intelligence in the years to come.


APPENDIX A

### **Reuse of Content Published on arXiv**


Portions of this thesis include material previously published as preprints on arXiv. The


following works are included:


_•_ **EigenTrack: Spectral Activation Feature Tracking for Hallucination and**


**Out-of-Distribution Detection in LLMs and VLMs**, arXiv:2509.15735.


_•_ **RMT-KD: Random Matrix Theoretic Causal Knowledge Distillation**,


arXiv:2509.15724.


According to arXiv’s reuse policy:


_“If you are the copyright holder of the work, you do not need arXiv’s permission to reuse the_


_full text.”_ (Source: `https://info.arxiv.org/help/license/reuse.html` )


I am the copyright holder of these works and therefore retain the right to reuse their


content in this thesis.


82


### **Bibliography**


[1] L. Yann, B. Yoshua, H. Geoffrey, “Deep learning,” _Nature_, vol. 521, no. 7553, pp. 436–444, 2015.

[2] G. Ian, B. Yoshua, C. Aaron, _Deep_ _Learning_ . MIT Press, 2016. Available at:
[https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

[3] Z. Chiyuan, B. Samy, H. Moritz, R. Benjamin, V. Oriol, “Understanding deep learning requires rethinking
generalization,” in _International Conference on Learning Representations_, 2017.

[4] B. Mikhail, H. Daniel, M. Siyuan, M. Soumik, “Reconciling modern machine learning practice and the
classical bias-variance trade-off,” _Proceedings of the National Academy of Sciences_, vol. 116, no. 32,
pp. 15849–15854, 2019.

[5] K. Alex, S. Ilya, H. G. E, “ImageNet classification with deep convolutional neural networks,” in _Advances_
_in Neural Information Processing Systems_, 2012.

[6] H. Kaiming, Z. Xiangyu, R. Shaoqing, S. Jian, “Deep residual learning for image recognition,” in _Pro-_
_ceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 2016.

[7] V. Ashish, S. Noam, P. Niki, U. Jakob, J. Llion, G. A. N, K. �Lukasz, P. Illia, “Attention is all you need,”
in _Advances in Neural Information Processing Systems_, 2017.

[8] D. Jacob, C. Ming-Wei, L. Kenton, T. Kristina, “BERT: Pre-training of deep bidirectional transformers
for language understanding,” in _Proceedings of NAACL-HLT_, 2019.

[9] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language models are unsupervised
[multitask learners,” OpenAI, 2019. Available at: https://openai.com/research/language-unsupervised.](https://openai.com/research/language-unsupervised)

[10] T. Hugo, L. Thibaut, I. Gautier, M. Xavier, L. Marie-Anne, L. Timoth´ee, R. Baptiste, G. Naman,
H. Eric, A. Faisal, others, “LLaMA: Open and efficient foundation language models,” _arXiv preprint_
_arXiv:2302.13971_, vol., no., pp., 2023.

[11] R. Alec, K. J. Wook, H. Chris, R. Aditya, G. Gabriel, A. Sandhini, S. Girish, A. Amanda, M. Pamila,
C. Jack, K. Gretchen, S. Ilya, “Learning transferable visual models from natural language supervision,”
in _Proceedings of ICML_, 2021.

[12] L. Haotian, L. Chunyuan, W. Qingyang, L. Y. Jae, “Visual Instruction Tuning,” _arXiv preprint_
_arXiv:2304.08485_, vol., no., pp., 2023.

[13] G. Stuart, B. Elie, D. Ren´e, “Neural networks and the bias/variance dilemma,” _Neural Computation_,
vol. 4, no. 1, pp. 1–58, 1992.

[14] H. R. A, J. C. R, _Matrix Analysis_ . Cambridge University Press, 2012.

[15] T. L. N, B. David, _Numerical Linear Algebra_ . SIAM, 1997.

[16] G. G. H, V. L. C. F, “Matrix computations and eigenvalue problems,” _Journal of Computational and_
_Applied Mathematics_, vol. 123, no. 1-2, pp. 35–65, 2000.

[17] C. Tony, “The Wigner semicircle law and eigenvalue distribution of random matrices,” _SIAM Review_,
vol. 34, no. 2, pp. 260–266, 1992.

[18] Y. Pavlo, “A short introduction to random matrix theory,” _arXiv preprint arXiv:1608.04850_, vol., no.,
pp., 2016.

[19] P. Debashis, “Asymptotics of sample eigenstructure for a large dimensional spiked covariance model,”
_Statistica Sinica_, vol., no., pp. 1617–1642, 2007.

[20] B. Jinho, B. A. Gerard, P. Sandrine, “Phase transition of the largest eigenvalue for nonnull complex
sample covariance matrices,” _Annals of Probability_, vol. 33, no. 5, pp. 1643–1697, 2005.

[21] H. Geoffrey, V. Oriol, D. Jeff, “Distilling the knowledge in a neural network,” _arXiv preprint_
_arXiv:1503.02531_, vol., no., pp., 2015.

[22] H. Song, M. Huizi, D. W. J, “Deep compression: Compressing deep neural networks with pruning,
trained quantization and Huffman coding,” in _International Conference on Learning Representations_
_(ICLR)_, 2016.


83


BIBLIOGRAPHY 84


[23] G. Amir, K. Sehoon, D. Zhen, Y. Zhewei, M. M. W, K. Kurt, “A survey of quantization methods for
efficient neural network inference,” _arXiv preprint arXiv:2103.13630_, vol., no., pp., 2021.

[24] E. Utku, G. Trevor, M. Jacob, C. P. Samuel, E. Erich, “Rigging the lottery: Making all tickets winners,”
in _International Conference on Machine Learning (ICML)_, 2020.

[25] M. V. A., P. L. A., “Distribution of eigenvalues for some sets of random matrices,” _Mathematics of the_
_USSR-Sbornik_, vol. 1, no. 4, pp. 457–483, 1967.

[26] W. E. P., “On the distribution of the roots of certain symmetric matrices,” _Annals of Mathematics_,
vol. 67, no. 2, pp. 325–327, 1958.

[27] R. Maithra, G. Justin, Y. Jason, S. Jascha, “SVCCA: Singular vector canonical correlation analysis
for deep learning dynamics and interpretability,” in _Advances in Neural Information Processing Systems_
_(NeurIPS)_, 2017.

[28] K. Simon, N. Mohammad, L. Honglak, H. Geoffrey, “Similarity of neural network representations revisited,” in _International Conference on Machine Learning (ICML)_, 2019.

[29] P. Vardan, H. Xiao, D. D. L., “The power-law spectrum of deep network gradients,” _Nature Communi-_
_cations_, vol. 11, no. 1, pp. 1–12, 2020.

[30] M. C. H., M. M. W., “Implicit self-regularization in deep neural networks: Evidence from random
matrix theory and implications for learning,” _Journal of Machine Learning Research_, vol. 22, no. 165,
pp. 1–73, 2021.

[31] R. Nasim, B. Aristide, A. Devansh, D. Felix, L. Min, H. F. A., B. Yoshua, C. Aaron, “On the spectral
bias of neural networks,” in _International Conference on Machine Learning (ICML)_, 2019.

[32] H. E. J., S. Yelong, W. S. Biderman, L. Hongyu, C. Zeyuan, W. Philip, C. Weizhu, “LoRA: Low-rank
adaptation of large language models,” in _International Conference on Learning Representations (ICLR)_,
2022.

[33] S. Emma, G. Ananya, M. Andrew, “Energy and policy considerations for deep learning in NLP,” _arXiv_
_preprint arXiv:1906.02243_, vol., no., pp., 2019.

[34] K. Jared, M. Sam, H. Tom, others, “Scaling laws for neural language models,” _arXiv preprint_
_arXiv:2001.08361_, vol., no., pp., 2020.

[35] H. Song, M. Huizi, D. W. J, “Deep compression: Compressing deep neural networks with pruning,
trained quantization and Huffman coding,” in _International Conference on Learning Representations_
_(ICLR)_, 2016.

[36] H. Dan, G. Kevin, “A baseline for detecting misclassified and out-of-distribution examples in neural
networks,” in _International Conference on Learning Representations (ICLR)_, 2017.

[37] O. Yaniv, F. Emily, R. Jie, others, “Can you trust your model’s uncertainty? Evaluating predictive
uncertainty under dataset shift,” in _Advances in Neural Information Processing Systems (NeurIPS)_,
2019.

[38] G. Chuan, P. Geoff, S. Yu, W. K. Q, “On calibration of modern neural networks,” in _International_
_Conference on Machine Learning (ICML)_, 2017.

[39] J. Ziwei, L. Nayeon, F. Rita, Y. Tiezheng, S. Dan, X. Yan, I. Etsuko, B. Yejin, C. Delong, D. Wenliang,
C. H. Shu, M. Andrea, F. Pascale, “Survey of Hallucination in Natural Language Generation,” _arXiv_
_preprint arXiv:2202.03629_, vol., no., pp., 2022.

[40] H. Lei, Y. Weijiang, M. Weitao, Z. Weihong, F. Zhangyin, W. Haotian, C. Qianglong, P. Weihua,
F. Xiaocheng, Q. Bing, L. Ting, “A Survey on Hallucination in Large Language Models: Principles,
Taxonomy, Challenges, and Open Questions,” _arXiv preprint arXiv:2311.05232_, vol., no., pp., 2023.

[41] J. Ziwei, B. Yejin, others, “A Comprehensive Survey of Hallucination in Large Language Models,” _ACM_
_Computing Surveys_, vol., no., pp., 2025.

[42] L. Stephanie, H. Jacob, E. Owain, “TruthfulQA: Measuring How Models Mimic Human Falsehoods,”
in _ACL_, 2022.

[43] L. Chenyang, others, “Loki’s Dance of Illusions: A Comprehensive Survey on Hallucinations in LLMs,”
_arXiv preprint arXiv:2507.02870_, vol., no., pp., 2025.

[44] D. Alexander, H. Katherine, M. Dan, A. Ben, A. Babak, B. Alex, C. Christina, D. Jonathan, E. Jacob,
H. M. D., others, “Underspecification Presents Challenges for Credibility in Modern Machine Learning,”
_arXiv preprint arXiv:2011.03395_, vol., no., pp., 2020.

[45] X. Zeyu, others, “Hallucination is Inevitable: An Innate Limitation of Large Language Models,” _arXiv_
_preprint arXiv:2401.11817_, vol., no., pp., 2024.


BIBLIOGRAPHY 85


[46] K. A. Tauman, O. R. Team, “Why Language Models Hallucinate,” _OpenAI Technical Report_, vol., no.,
pp., 2025.

[47] M. Potsawee, L. Adian, G. M. J. F., “SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
for Generative Large Language Models,” in _EMNLP_, 2023.

[48] L. Deren, L. Yaxi, H. Mengya, W. Mingyu, Y. Vincent, C. Emily, K. Eslam, “Chain of Natural
Language Inference for Reducing Large Language Model Ungrounded Hallucinations,” _arXiv preprint_
_arXiv:2310.03951_, vol., no., pp., 2023.

[49] K. Lorenz, G. Yarin, F. Sebastian, “Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation,” _arXiv preprint arXiv:2302.09664_, vol., no., pp., 2023.

[50] M. Eric, L. Yoonho, K. Alexander, M. C. D., F. Chelsea, “DetectGPT: Zero-Shot Machine-Generated
Text Detection Using Probability Curvature,” in _ICML_, 2023.

[51] B. Guangsheng, Z. Yanbin, T. Zhiyang, Y. Linyi, Z. Yue, “Fast-DetectGPT: Efficient ZeroShot Detection of Machine-Generated Text via Conditional Probability Curvature,” _arXiv preprint_
_arXiv:2310.05130_, vol., no., pp., 2023.

[52] B. Guangsheng, Z. Yanbin, H. Juncai, Z. Yue, “Glimpse: Enabling White-Box Methods to Use Proprietary Models for Zero-Shot LLM-Generated Text Detection,” in _ICLR_, 2025.

[53] S. Weihang, W. Changyue, A. Qingyao, H. Yiran, W. Zhijing, Z. Yujia, L. Yiqun, “Unsupervised RealTime Hallucination Detection Based on the Internal States of Large Language Models,” in _Findings of_
_ACL_, 2024.

[54] S. Zhongxiang, Z. Xiaoxue, Z. Kai, S. Yang, X. Jun, Z. Xiao, Y. Weijie, L. Han, “ReDeEP: Detecting
Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability,” in _ACL_, 2025.

[55] S. Anand, S. Sheng, D. Tri, others, “AttentionScore: Faithfulness Detection from Attention Patterns in
LLMs,” _arXiv preprint arXiv:2311.09516_, vol., no., pp., 2024.

[56] S. Yixin, H. Yue, Y. Chenglu, S. Yicheng, W. Dong, S. Lei, C. Lin, “RankFeat: Rank-1 Feature Removal
for Out-of-Distribution Detection,” in _NeurIPS_, 2022.

[57] M. Yihan, L. Zhiyu, L. Yiyou, Z. James, “Spectral Normalized Joint Energy for Multi-Label Out-ofDistribution Detection,” _arXiv preprint arXiv:2405.04759_, vol., no., pp., 2024.

[58] B. Jakub, J. Denis, S. Albert, G. Bogdan, K. Tomasz, “Hallucination Detection in LLMs Using Spectral
Features of Attention Maps,” in _ICML_, 2025.

[59] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. Bang, D. Chen, W. Dai, H. S. Chan,
A. Madotto, P. Fung, “Survey of Hallucination in Natural Language Generation,” _ACM Computing_
_Surveys_, vol. 55, no. 12, pp. 248:1–248:38, 2023.

[60] S. Lin, J. Hilton, O. Evans, “TruthfulQA: Measuring How Models Mimic Human Falsehoods,” in _Pro-_
_ceedings of ACL_, 2022.

[61] S. Valentin, J. Fu, G. Detommaso, S. Xu, G. Zappella, B. Wang, “Cost-Effective Hallucination Detection
for LLMs,” _arXiv preprint arXiv:2407.21424_, vol., no., pp., 2024.

[62] W. Su, C. Wang, Q. Ai, Y. Hu, Z. Wu, Y. Zhou, Y. Liu, “Unsupervised Real-Time Hallucination
Detection based on the Internal States of Large Language Models,” in _Findings of ACL_, 2024.

[63] C. Chen, K. Liu, Z. Chen, Y. Gu, Y. Wu, M. Tao, Z. Fu, J. Ye, “INSIDE: LLMs’ Internal States Retain
the Power of Hallucination Detection,” in _Proceedings of ICLR_, 2024.

[64] X. Du, C. Xiao, Y. Li, “HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection,” _arXiv preprint arXiv:2409.17504_, vol., no., pp., 2024.

[65] V. A. Marˇcenko, L. A. Pastur, “Distribution of eigenvalues for some sets of random matrices,” _Mathe-_
_matics of the USSR-Sbornik_, vol. 1, no. 4, pp. 457–483, 1967.

[66] J. Baik, G. B. Arous, S. P´ech´e, “Phase transition of the largest eigenvalue for nonnull complex sample
covariance matrices,” _The Annals of Probability_, vol. 33, no. 5, pp. 1643–1697, 2005.

[67] J. Binkowski, D. Janiak, A. Sawczyn, B. Gabrys, T. Kajdanowicz, “Hallucination Detection in LLMs
Using Spectral Features of Attention Maps,” _arXiv preprint arXiv:2502.17598_, vol., no., pp., 2025.

[68] J. Yang, K. Han, Y. Wang, “Generalized Out-of-Distribution Detection: A Survey,” _IEEE Transactions_
_on Pattern Analysis and Machine Intelligence_, vol. 45, no. 2, pp. 1359–1381, 2021.

[69] S. Liang, Y. Li, R. Srikant, “Enhancing The Reliability of Out-of-distribution Image Detection in Neural
Networks,” in _International Conference on Learning Representations (ICLR)_, 2018.

[70] K. Lee, K. Lee, H. Lee, J. Shin, “A Simple Unified Framework for Detecting Out-of-Distribution Samples
and Adversarial Attacks,” in _Advances in Neural Information Processing Systems (NeurIPS)_, 2018.


BIBLIOGRAPHY 86


[71] W. Liu, X. Wang, J. Owens, Y. Li, “Energy-based Out-of-distribution Detection,” in _Advances in Neural_
_Information Processing Systems (NeurIPS)_, 2020.

[72] H. Jiang, L. Song, J. Zhang, D. Yu, “Energy-Based Out-of-Distribution Detection for Large Language
Models,” _arXiv preprint arXiv:2309.02586_, vol., no., pp., 2023.

[73] L. Xiao, Y. Wang, X. Yang, Y. Li, “Sequence-level Out-of-Distribution Detection via Normalized Logit
Similarity,” _arXiv preprint arXiv:2405.07688_, vol., no., pp., 2024.

[74] N. Papernot, P. McDaniel, “Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust
Deep Learning,” in _NeurIPS_, 2018.

[75] H. Sun, Z. Wang, Y. Li, “ReAct: Out-of-Distribution Detection With Rectified Activations,” _Advances_
_in Neural Information Processing Systems (NeurIPS)_, vol., no., pp., 2022.

[76] J. Tack, S. Mo, J. Jeong, J. Shin, “CSI: Novelty Detection via Contrastive Learning on Pretrained
Features,” in _NeurIPS_, 2020.

[77] H. Yilmaz, S. Hooker, “OOD Detection for Text via Attention Entropy and Representation Variance,”
_arXiv preprint arXiv:2208.10545_, vol., no., pp., 2022.

[78] L. Sun, J. Zhang, B. Han, T. Liu, “RankFeat: Rank-Preserving Feature Regularization for Out-ofDistribution Generalization and Detection,” _IEEE Transactions on Neural Networks and Learning Sys-_
_tems_, vol., no., pp., 2022.

[79] S. Park, H. Lee, S. Ahn, “SpectralGap: Eigenvalue Gap-Based Out-of-Distribution Detection,” _arXiv_
_preprint arXiv:2403.07881_, vol., no., pp., 2024.

[80] M. Ahmed, K. Wu, X. Wang, Y. Li, “SNoJoE: Singular Value Based Joint Energy for Out-of-Distribution
Detection,” _arXiv preprint arXiv:2404.05209_, vol., no., pp., 2024.

[81] C. H. Martin, M. W. Mahoney, “Implicit Self-Regularization in Deep Neural Networks: Evidence from
Random Matrix Theory,” _Journal of Machine Learning Research_, vol. 22, no., pp. 1–73, 2021.

[82] P. Donos, M. G. Bianchi, P. Favaro, “Spectral Shifts in Neural Network Representations Reveal OOD
and Adversarial Examples,” _arXiv preprint arXiv:2012.00883_, vol., no., pp., 2020.

[83] S. Han, H. Mao, W. J. Dally, “Deep Compression: Compressing Deep Neural Networks with Pruning,
Trained Quantization and Huffman Coding,” in _International Conference on Learning Representations_
_(ICLR)_, 2016.

[84] J. Frankle, M. Carbin, “The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks,”
in _International Conference on Learning Representations (ICLR)_, 2019.

[85] P. Molchanov, S. Tyree, T. Karras, T. Aila, J. Kautz, “Pruning Convolutional Neural Networks for
Resource Efficient Inference,” in _International Conference on Learning Representations (ICLR)_, 2017.

[86] S. P. Singh, D. Alistarh, “WoodFisher: Efficient Second-Order Approximation for Neural Network
Compression,” in _Advances in Neural Information Processing Systems (NeurIPS)_, 2020.

[87] V. Sanh, T. Wolf, A. M. Rush, “Movement Pruning: Adaptive Sparsity by Fine-Tuning,” in _Advances_
_in Neural Information Processing Systems (NeurIPS)_, 2020.

[88] Z. Liu, J. Li, Z. Shen, G. Huang, S. Yan, C. Zhang, “Network Slimming: Learning Slimmable Networks
via Channel Pruning,” in _International Conference on Computer Vision (ICCV)_, 2017.

[89] Y. He, J. Lin, Z. Liu, H. Wang, L. Li, S. Han, “AMC: AutoML for Model Compression and Acceleration
on Mobile Devices,” in _European Conference on Computer Vision (ECCV)_, 2018.

[90] Z. Liu, J. Wang, others, “MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning,” in _International Conference on Computer Vision (ICCV)_, 2019.

[91] P. Michel, O. Levy, G. Neubig, “Are Sixteen Heads Really Better than One?,” in _Advances in Neural_
_Information Processing Systems (NeurIPS)_, 2019.

[92] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, D. Kalenichenko, “Quantization
and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,” in _Proceedings of the_
_IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2018.

[93] Z. Xiao, J. Lin, S. Han, “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
Language Models,” in _International Conference on Machine Learning (ICML)_, 2023.

[94] E. Frantar, D. Alistarh, “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers,” in _International Conference on Learning Representations (ICLR) Workshops_, 2022.

[95] J. Lin, Z. Liu, H. Wang, L. Zhu, S. Han, “AWQ: Activation-aware Weight Quantization for LLMs,”
_arXiv preprint arXiv:2306.00978_, vol., no., pp., 2023.


BIBLIOGRAPHY 87


[96] Z. Yao, Y. Li, H. Lin, S. Shen, others, “ZeroQuant: Efficient and Affordable Post-Training Quantization
for Large-Scale Transformers,” in _Advances in Neural Information Processing Systems (NeurIPS)_, 2022.

[97] Z. Yao, H. Lin, others, “ZeroQuant-V2: Exploring Post-Training Quantization for Diffusion Models,”
_arXiv preprint arXiv:2210.17323_, vol., no., pp., 2022.

[98] S. K. Esser, J. L. McKinstry, D. Bablani, R. Appuswamy, D. S. Modha, “Learned Step Size Quantization,” in _International Conference on Learning Representations (ICLR)_, 2019.

[99] S. Shen, Z. Gan, Y. Shen, others, “Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT,”
in _Proceedings of the AAAI Conference on Artificial Intelligence_, 2020.

[100] E. L. Denton, W. Zaremba, J. Bruna, Y. LeCun, R. Fergus, “Exploiting Linear Structure Within
Convolutional Networks for Efficient Evaluation,” in _Advances in Neural Information Processing Systems_
_(NeurIPS)_, 2014.

[101] V. Lebedev, Y. Ganin, M. Rakhuba, I. Oseledets, V. Lempitsky, “Speeding-up Convolutional Neural
Networks Using Fine-tuned CP-Decomposition,” in _International Conference on Learning Representa-_
_tions (ICLR) Workshops_, 2015.

[102] Y. Kim, E. Park, others, “Compression of Deep Convolutional Neural Networks for Fast and Low
Power Mobile Applications,” in _International Conference on Learning Representations (ICLR)_, 2016.

[103] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, R. Soricut, “ALBERT: A Lite BERT for
Self-supervised Learning of Language Representations,” in _International Conference on Learning Repre-_
_sentations (ICLR)_, 2020.

[104] H. Liu, S. Diao, Q. Dong, others, “DoRA: Weight-Decomposed Low-Rank Adaptation,” _arXiv preprint_
_arXiv:2402.09353_, vol., no., pp., 2024.

[105] A. Aghajanyan, S. Gupta, L. Zettlemoyer, “Intrinsic Dimensionality Explains the Effectiveness of
Language Model Fine-Tuning,” in _Association for Computational Linguistics (ACL)_, 2021.

[106] G. Hinton, O. Vinyals, J. Dean, “Distilling the Knowledge in a Neural Network,” in _NeurIPS Deep_
_Learning and Representation Learning Workshop_, 2015.

[107] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, Y. Bengio, “FitNets: Hints for Thin Deep
Nets,” in _International Conference on Learning Representations (ICLR)_, 2015.

[108] S. Zagoruyko, N. Komodakis, “Paying More Attention to Attention: Improving the Performance of
Convolutional Neural Networks via Attention Transfer,” in _International Conference on Learning Rep-_
_resentations (ICLR)_, 2017.

[109] Y. Tian, D. Krishnan, P. Isola, “Contrastive Representation Distillation,” in _International Conference_
_on Learning Representations (ICLR)_, 2020.

[110] V. Sanh, L. Debut, J. Chaumond, T. Wolf, “DistilBERT, a Distilled Version of BERT: Smaller, Faster,
Cheaper and Lighter,” in _NeurIPS Workshop on Energy Efficient Machine Learning_, 2019.

[111] S. Sun, Y. Cheng, Z. Gan, J. Liu, “Patient Knowledge Distillation for BERT Model Compression,” in
_Proceedings of EMNLP-IJCNLP_, 2019.

[112] C. Xu, W. Zhou, T. Ge, F. Wei, M. Zhou, “BERT of Theseus: Compressing BERT by Progressive
Module Replacing,” in _Proceedings of EMNLP_, 2020.

[113] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, M. Zhou, “MiniLM: Deep Self-Attention Distillation for
Task-Agnostic Compression of Pre-Trained Transformers,” in _Advances in Neural Information Processing_
_Systems (NeurIPS)_, 2020.

[114] H. Cai, C. Gan, T. Wang, Z. Zhang, S. Han, “Once-for-All: Train One Network and Specialize it for
Efficient Deployment,” in _International Conference on Learning Representations (ICLR)_, 2020.

[115] C. H. Martin, M. W. Mahoney, “Heavy-Tailed Universality Predicts Trends in Test Accuracy for Deep
Neural Networks,” _Journal of Machine Learning Research_, vol. 21, no. 163, pp. 1–71, 2020.

[116] J. Pennington, S. S. Schoenholz, S. Ganguli, “Resurrecting the Sigmoid in Deep Learning Through
Dynamical Isometry: Theory and Practice,” in _Advances in Neural Information Processing Systems_
_(NeurIPS)_, 2018.

[117] C. H. Martin, M. W. Mahoney, “Spectral Universality and Tikhonov Regularization in Deep Learning,”
_arXiv preprint arXiv:2301.06318_, vol., no., pp., 2023.

[118] L. Sagun, U. Evci, V. U. Guney, Y. Dauphin, L. Bottou, “Empirical Analysis of the Hessian of OverParametrized Neural Networks,” _arXiv preprint arXiv:1706.04454_, vol., no., pp., 2017.


BIBLIOGRAPHY 88


[119] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin,
J. Clark, G. Krueger, I. Sutskever, “Learning Transferable Visual Models From Natural Language Supervision,” in _Proceedings of the International Conference on Machine Learning (ICML)_, 2021.

[120] J. Li, D. Li, C. Xiong, S. C. Hoi, “BLIP: Bootstrapped Language-Image Pre-training for Unified
Vision-Language Understanding and Generation,” in _Proceedings of ICML_, 2022.

[121] J. Alayrac, J. D. Fauw, a. others, “Flamingo: A Visual Language Model for Few-Shot Learning,” in
_Advances in Neural Information Processing Systems (NeurIPS)_, 2022.

[122] H. Liu, C. Li, Q. Wu, Y. J. Lee, “Visual Instruction Tuning,” _arXiv preprint arXiv:2304.08485_, vol.,
no., pp., 2023.

[123] OpenAI, “GPT-4 Technical Report,” _arXiv preprint arXiv:2303.08774_, vol., no., pp., 2023.

[124] D. Li, H. Liu, C. Li, J. Yang, Y. J. Lee, “Multimodal Hallucination in Large Vision-Language Models:
A Survey,” _arXiv preprint arXiv:2310.16291_, vol., no., pp., 2023.

[125] A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell, K. Saenko, “Object Hallucination in Image
Captioning,” in _Proceedings of the Conference on Empirical Methods in Natural Language Processing_
_(EMNLP)_, 2018.

[126] X. Li, X. Yin, C. Li, X. Hu, P. Zhang, L. Zhang, L. Wang, H. Hu, L. Dong, F. Wei, Y. Choi, J. Gao,
“OSCAR: Object-Semantics Aligned Pre-training for Vision-Language Tasks,” in _Proceedings of ECCV_,
2020.

[127] P. Zhang, X. Li, X. Hu, others, “VinVL: Making Visual Representations Matter in Vision-Language
Models,” in _Proceedings of CVPR_, 2021.

[128] Y. Yin, J. Li, C. Wang, Y. Li, “MHalDet: Multimodal Hallucination Detection via Attention Entropy
and Cross-Modal Alignment,” _arXiv preprint arXiv:2312.10457_, vol., no., pp., 2023.

[129] C. Jia, Y. Yang, Y. Xia, others, “Scaling Up Visual and Vision-Language Representation Learning
With Noisy Text Supervision,” in _Proceedings of ICML_, 2021.

[130] D. Li, X. Li, J. Li, Y. J. Lee, “InstructBLIP: Towards General-purpose Vision-Language Models with
Instruction Tuning,” _arXiv preprint arXiv:2305.06500_, vol., no., pp., 2023.

[131] W. E. P, “On the distribution of the roots of certain symmetric matrices,” _Annals of Mathematics_,
vol. 67, no. 2, pp. 325–327, 1958.

[132] J. I. M, “On the distribution of the largest eigenvalue in principal components analysis,” _Annals of_
_Statistics_, vol. 29, no. 2, pp. 295–327, 2001.

[133] A. Shun-ichi, _Information Geometry and Its Applications_ . Springer, 2016.

[134] N. Darabi, D. Naik, S. Tayebati, D. Jayasuriya, R. Krishnan, and A. R. Trivedi, “EigenShield: Causal
Subspace Filtering via Random Matrix Theory for Adversarially Robust Vision-Language Models,” arXiv
[preprint arXiv:2502.14976, 2025. Available at: https://arxiv.org/abs/2502.14976.](https://arxiv.org/abs/2502.14976)

[135] D. Ettori, N. Darabi, S. Tayebati, R. Krishnan, M. Subedar, O. Tickoo, A. R. Trivedi, “EigenTrack:
Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and
VLMs,” _arXiv preprint arXiv:2509.15735_, vol., no., pp., 2025.

[136] D. Ettori, N. Darabi, S. Senthilkumar, A. R. Trivedi, “RMT-KD: Random Matrix Theoretic Causal
Knowledge Distillation,” _arXiv preprint arXiv:2509.15724_, vol., no., pp., 2025.

[137] K. Seijin, A. Yassir, v. O. Johannes, “Weight decay induces low-rank attention layers,” _arXiv preprint_
_arXiv:2410.23819_, vol., no., pp., 2024.

[138] H. Kaiming, Z. Xiangyu, R. Shaoqing, S. Jian, “Deep residual learning for image recognition,” in
_Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 2016.


