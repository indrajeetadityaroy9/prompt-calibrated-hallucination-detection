_July 2024_

# **PaliGemma: A versatile 3B VLM for transfer**


**Lucas Beyer** [*,][†] **, Andreas Steiner** [*] **, André Susano Pinto** [*] **, Alexander Kolesnikov** [*] **, Xiao Wang** [*] **, Daniel Salz,**
**Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner,**
**Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar,**
**Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer,**
**Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi**
**Xiong, Radu Soricut, Jeremiah Harmsen and Xiaohua Zhai** [*,][†]

*Core team, †Project lead


**PaliGemma is an open Vision-Language Model (VLM) that is based on the SigLIP-So400m vision encoder**
**and** **the** **Gemma-2B** **language** **model.** **It** **is** **trained** **to** **be** **a** **versatile** **and** **broadly** **knowledgeable** **base**
**model that is effective to transfer.** **It achieves strong performance on a wide variety of open-world tasks.**
**We evaluate PaliGemma on almost 40 diverse tasks including standard VLM benchmarks, but also more**
**specialized tasks such as remote-sensing and segmentation.**


### **1. Introduction**

PaliGemma is an open model, continuing the line
of PaLI vision-language models in a combination
with the Gemma family of language models.


PaLI is a series of state-of-the-art visionlanguage models, starting with the first PaLI [23]
showing promising scaling results up to 17 B,
using classification pretrained ViT [131] and
mT5 [126] language model. PaLI-X [24] and
PaLM-E [36] then pushed this further, combining ViT-22 B [29] and a 32 B UL2 [104] language
model or the 540 B PaLM [28] language model,
respectively, and getting further increased performance on vision-language tasks, albeit saturating
performance on standard image classification and
retrieval tasks. Finally, PaLI-3 [25] demonstrates
that through better pretraining with SigLIP [133]
and more careful multimodal data curation, a 2 B
vision and 3 B language model ( _i.e_ . a 5 B visionlanguage model) matches the 10x larger PaLI-X
and 100x larger PaLM-E across most benchmarks.


PaliGemma continues this trend, combining the
400 M SigLIP and the 2 B Gemma models [82]
into a sub-3 B VLM that still maintains performance comparable to PaLI-X, PaLM-E, and PaLI-3.


Gemma [82] is a family of auto-regressive
decoder-only open large language models built
from the same research and technology used to
create the Gemini [7] models. The models come
in different sizes (2 B, 7 B), both pretrained and


_Corresponding author(s):_ _lbeyer,xzhai@google.com_
© 2024 Google DeepMind. All rights reserved



instruction fine-tuned. PaliGemma uses the 2 B
pretrained version.


The main goal of our work is to provide a
versatile base VLM. Hence, we show that it
reaches state-of-the-art results not only on standard COCO captions, VQAv2, InfographicVQA
and others, but also on more exotic RemoteSensing VQA, TallyVQA, several video captioning and QA tasks, as well as referring expression
_segmentation_ (see full task list in Appendix B).

### **2. Related work**


Over the course of the past few years, visionlanguage models have gained considerable importance in computer vision. The first generation,
spearheaded by CLIP [94] and ALIGN [49] by
scaling up ConVIRT [135] and VirTex [32], is
an extension of large-scale classification pretraining [55, 131], to leverage all data from the web
without the need for onerous human labeling, replacing a fixed and large set of classes by a caption
embedding instead. The caption embeddings are
mostly obtained using language encoders (similar to BERT [33]) and allow to open up the
vocabulary of classification and retrieval tasks.
The second generation, akin to T5 [95] in language, is a unification of captioning and questionanswering tasks via generative encoder-decoder
modeling [27, 111, 120, 138], often backed
by the progress in generative language models.




---PAGE BREAK---

PaliGemma: A versatile 3B VLM for transfer









Figure 1 | PaliGemma’s architecture: a SigLIP image encoder feeds into a Gemma decoder LM.



These were then further scaled up by, among others, Flamingo [6], BLIP-2 [62] and, PaLI [23]. Finally, most recent works [7, 70, 87, 113] perform
an additional “instruction tuning” step that is intended to make the raw model more user-friendly.
In addition to building systems, several recent
more systematic studies [59, 81, 107] aim to find
out what really matters in VLMs. PaliGemma is an
open base VLM without instruction tuning, and
this report answers a few more questions regarding what matters. More discussion in Appendix A.

### **3. Model**


In this section we present details about
PaliGemma’s architecture and training. Several
of our decisions are further ablated in Section 5.


At a high level, PaliGemma is a VLM, taking as
input one or more images, and a textual description of the task (the prompt or question, which
we often refer to as the `prefix` ). PaliGemma
then autoregressively generates a prediction in
the form of a text string (the answer, which we
often refer to as the `suffix` ).


This simple image+text in, text out API is
flexible enough to cover many standard tasks,
such as image classification, captioning, visual
question-answering and dialogue. Additionally,
as shown in the literature, by converting more



complex structured outputs into “text”, this API
can also cover more tasks such as: detection [22],
instance segmentation [25, 115], panoptic segmentation, depth prediction, colorization, and
many more [56, 73, 139]. This conversion can be
hand-engineered and task-specific, such as done
in pix2seq [22] for detection, or learned as is
the case for segmentation [56] and dense output
tasks in general.


During PaliGemma’s pretraining, we limit ourselves to “text” covering natural language, object
detection, and instance segmentation, but this
API remains versatile and the pretrained models
can be finetuned for other output types.


**3.1.** **Architecture**


PaliGemma consists of three components:


 - An image encoder, for which we use a
publicly available SigLIP [133] checkpoint,
specifically the “shape optimized” [5] ViTSo400m image encoder. This model was
contrastively pretrained at large scale via the
sigmoid loss, and has shown state-of-the-art
performance, especially for its small size.

 - A decoder-only language model, for which
we use the publicly available Gemma-2B
v1.0 [82] raw pretrained checkpoint, which
strikes a great balance between performance


2




---PAGE BREAK---

PaliGemma: A versatile 3B VLM for transfer



Figure 2 | PaliGemma’s Prefix-LM masking: block
attention throughout image and prefix, autoregressive attention on the suffix. Each square indicates whether the row can attend to the column.


and size. As we will show, this language
model is good enough to match or surpass
the performance of VLMs using much larger
language models, including previous PaLIs.

 - A linear layer projecting SigLIP’s output tokens into the same dimensions as Gemma2B’s vocab tokens, so they can be concatenated. In early experiments, we found that
more complicated alternatives ( _e.g_ . MLPs)
do not provide a clear advantage, and hence
decided to use the simplest option (Sec 5.5).


The image is passed through the image encoder, which turns it into a sequence of _𝑁_ img tokens. The text is converted into _𝑁_ txt tokens using Gemma’s SentencePiece [58] tokenizer, and
embedded with Gemma’s vocabulary embedding
layer. The image tokens are projected with the
(zero initialized) linear projection. Then the sequence of input tokens to the decoder is created
as follows (and also as visible in Figure 2):

```
 tokens = [image tokens...,
      BOS, prefix tokens..., SEP,
      suffix tokens..., EOS, PAD...]

```

We always resize the image to a fixed square
size (224, 448, or 896 pixels). This leads to a
fixed number of image tokens per model variant (respectively 256, 1024, or 4096 tokens),



which we place in the front, making image tokens
straightforward to interpret without the need for
special location markers. The BOS token then
marks the start of text tokens. We use `\n` as SEP
token, it does not appear in any of our prefixes.
We also tokenize SEP separately to avoid it being
merged (by the tokenizer) with either the end of
the prefix or the beginning of the suffix. In order to maximize model capacity for such a small
model, we have full (unmasked) attention on the
whole input, _i.e_ . the image and prefix tokens.
In this way, image tokens can also "lookahead"
at the task at hand (prefix) in order to update
their representation. The suffix is our output and
necessarily covered by an auto-regressive mask,
including the PAD tokens. When we mention sequence length ( _𝑁_ txt), we typically mean prefix
and suffix combined, ignoring image tokens.


**3.2.** **Pretraining**


The training of PaliGemma follows the same steps
as previous PaLI models, with only small modifications. Training consists of several stages, which
we detail in this section:


 - **Stage0:** Unimodal pretraining - we use existing off-the-shelf components.

 - **Stage1:** Multimodal pretraining - long pretraining on a carefully chosen mixture of
multimodal tasks. Notably, nothing is frozen.

 - **Stage2:** Resolution increase - short continued pretraining at higher resolution.

 - **Stage3:** Transfer - turn the base model into
a task-specific specialist.


_**3.2.1.**_ _**Stage0:**_ _**Unimodal pretraining**_


First, the unimodal components of the model are
pretrained individually, in order to benefit from
their well-studied and scaled training recipes. For
PaliGemma specifically, we do not perform any
custom unimodal pretraining, instead relying on
existing publicly available checkpoints.


Following PaLI-3’s strong experimental results,
we use a SigLIP image encoder. While PaLI-3
(and others [6, 26]) use a large image model such
as ViT-G, we use the much smaller but similarly
strong “shape optimized” ViT-So400m model.


3




---PAGE BREAK---

PaliGemma: A versatile 3B VLM for transfer



PaLI traditionally uses an encoder-decoder language model; however all recently publicly released language models are decoder-only Transformers. We opt for the Gemma-2B model, which
strikes a good balance between size and performance. Larger language models, such as the popular 7 B or 70 B sizes, are often significantly better
at tasks like mathematical reasoning. However,
PaLI-3 has shown that across a wide range of
vision-language tasks, a well-trained small 5 B
model (2 B vision + 3 B language) can attain the
same performance as the much larger 55 B PaLI-X
(22 B vision + 32 B language) and 562 B PaLM-E
(22 B vision + 540 B language), including tasks
such as ScienceQA. With PaliGemma we continue
this push for smaller models and show that we
can keep the same performance with less than
3 B total parameters.


_**3.2.2.**_ _**Stage1:**_ _**Multimodal pretraining**_


In this stage, we combine the unimodal models
as explained in Section 3.1 and train the whole
model on a broad mixture of large-scale visionlanguage tasks. Contrary to most recent VLMs,
our core goal is to train a base model that finetunes well to a wide range of tasks, not merely to
align the modalities. Intuitively, we want a mix
of tasks which force the model to acquire a broad
range of “skills”, regardless of the task’s user (or
benchmark) friendliness out of the box. More on
this in Section 3.2.5.


It is common practice, also followed by previous PaLI versions, to keep the image encoder
frozen during the first multimodal pretraining
stage. This is partially due to findings as in
LiT [132] reporting multimodal tuning of pretrained image encoders degrading their representations. However, more recent work such as
CapPa [110] and LocCa [115] have shown that
captioning and other harder-to-learn tasks can
provide valuable signal to image encoders, allowing them to learn spatial and relational understanding capabilities which contrastive models
like CLIP or SigLIP typically lack. Hence, again
in the spirit of learning more skills during pretraining, we depart from common practice and
do not freeze the image encoder. However, the
challenges outlined in LiT remain. In order to



avoid destructive supervision signal from the initially unaligned language model, we use a slow
linear warm-up for the image encoder’s learningrate (Figure 3), which ensures that the image
encoder’s quality is not deteriorated from the initially misaligned gradients coming through the
LLM.


We train Stage1 at resolution 224px (hence,
_𝑁_ img = 256 image tokens) and sequence length
_𝑁_ txt = 128 for a total of 1 billion examples. While
we provide an ablation in Section 5.1 showing
that a 10x to 30x shorter Stage1 still provides
good results on popular benchmarks, we wish
to imbue as much visual knowledge to the base
model as possible, and cover a broad set of concepts, cultures, and languages [17, 37, 68, 85,
92, 93, 136].


_**3.2.3.**_ _**Stage2:**_ _**Resolution increase**_


The model resulting from Stage1 is already a
useful base model for many tasks (see example images in Appendix B). However, it only
understands images at 224 × 224 pixel resolution, which is too small for several tasks. For
instance, detection and segmentation of smaller
objects, and tasks related to reading smaller texts
such as charts, infographics, or documents, all
strongly benefit from higher resolution (see Table 1). Hence, we train two further model checkpoints for increased resolution, first to 448 × 448
and then to 896 × 896 pixel resolution.


Since stage1 took care of providing the model
with a broad set of knowledge and skill, stage2
can focus on extending the model’s ability to
parse higher-resolution images. We thus run
Stage2 with fewer total examples, while increasing the cost and information density of each example. For resolution 448, we train for an additional
50 M examples, and for resolution 896, we add
another 10 M examples.


For simplicity, Stage2 consists of the exact same
mixture of tasks and datasets as Stage1, but with
significantly increased sampling of tasks that require high resolution. Additionally, these upweighted tasks all can be modified to provide
much longer suffix sequence lengths. For instance,
for OCR tasks, we can simply request the model


4




---PAGE BREAK---

PaliGemma: A versatile 3B VLM for transfer



to read _all_ text on the image in left-to-right, topto-bottom order. For detection and segmentation
tasks, we can request the model to detect or segment _all_ objects for which annotation is provided.
Hence, we also increase the text sequence length
to _𝑁_ txt = 512 tokens.


While PaLI has always had this resolution increasing stage, and for image classification the
importance of resolution is long known [55, 109],
several recent works [81, 114, 121] have raised
the importance of resolution in VLMs too. We add
to this body of knowledge by providing several
ablation studies regarding Stage2 in Section 5.7.


_**3.2.4.**_ _**Stage3:**_ _**Transfer**_


The result of Stages 1 and 2 is a family of three
PaliGemma checkpoints, at 224px, 448px, and
896px resolution, which are pre-equipped with
broad visual knowledge. However, these checkpoints are not “user (or benchmark) friendly” as
their pretraining has focused solely on density of
learning signal, as opposed to usable interface.


These base models need to be transferred to
serve their intended final purpose. That could
take the form of fine-tuning on a specific, specialized task, such as COCO Captions, Remote
Sensing VQA, Video Captioning, or InfographicQA. Adapt to new inputs such as multiple images (NLVR2) or bounding boxes draw in the
image (WidgetCap). Or it could take the form of
instruction [70] or even chat [46] tuning.


To show the effectiveness of the base models, we transfer them to a wide range of individual academic benchmarks, using a simple unified transfer recipe with few hyper-parameters.
And to showcase the versatility beyond academic
tasks, we also provide a “mix” transfer checkpoint,
which transfers to a subset of these tasks at the
same time, along with detailed captioning and
long question-answering data. While this is not
instruction tuning, it is a step in that direction.


We also transfer PaliGemma to tasks which take
multiple images as input. NLVR2 is one such task,
which asks one question about two images, and
requires looking at both to give the correct answer. Other such tasks are standard short-video



understanding tasks subsampled to 16 frames. In
all these cases, we follow PaLI-3 and encode each
image separately, then concatenate the image
tokens without any special separator or embedding tokens. Thus, 16 frames at 224px resolution
result in _𝑁_ img = 4096 image tokens, the same
amount as a single image at 896px resolution.


For all transfers, we perform fine-tuning of all
the model parameters. The hyper-parameters we
modify per-task are the following, in decreasing
order of importance:


 - Resolution ( _i.e_ . checkpoint): **224**, 448, 896.

 - Epochs: **1, 3, 10**, 30, 100.

 - Learning-rate: 3e-5, **1e-5**, 3e-6.

 - Label-smoothing: **0.0**, 0.1, 0.3.

 - Dropout in the LLM: **0.0**, 0.1, 0.3.

 - Weight decay: **0.0** or 0.1 × learning-rate.

 - Freeze ViT: **false**, true.

 - Beam-search may benefit captioning.


The above are typical values we suggest exploring,
with the recommended initial attempt value in
bold. We provide the best setting for each individual task in Appendix J. We study the sensitivity
to transfer hyper-parameters in Section 6.2, and
the “transferability” in general in Section 6, showing that good results can be achieved with the
aforementioned initial attempt values.


_**3.2.5.**_ _**Pretraining task mixture**_


Just like for previous PaLI models, the pretraining (Stage1 and Stage2) is designed to result
in a model that transfers well, not necessarily a
model that is usable out of the box (“0 shot”).
The intuition here is that we want a mix of tasks
which force the model to acquire a broad range
of “skills”. We prefix each task with its unique
prefix to avoid conflicting learning signals across
skills [14]. At transfer time (Stage3), the model
then merely needs to recognize which skill is useful for the task, and rewire itself to use that while
following the output syntax and vocabulary of
the task. In our experience, these can all be done
relatively quickly and based on few examples
(Section 6.3). We do not use any of our transfer datasets during pretraining, and furthermore
remove all near-duplicates of their images from


5




---PAGE BREAK---

PaliGemma: A versatile 3B VLM for transfer



0 200M 400M 600M 800M 1000M
Examples seen



the pretraining datasets [55].


Largely following previous PaLI works, these
are the pretraining tasks:

```
 caption {lang}
```

We include the simple captioning objective on
various datasets, including WebLI in over 100 languages, and CC3M-35L. Previous PaLIs use an
encoder-decoder language model with the SplitCap objective, however for PaliGemma with the
decoder-only language model, plain captioning
is a more informative and simpler objective.

```
 ocr
```

Concatenation (in raster order) of all text on the
image transcribed by a public OCR system. Potentially skipping random snippets of OCR in order
to fit sequence length without biasing recognition
towards the beginning of raster order.

```
 answer en {question}
```

Generated VQA on CC3M-35L following [19] with
questions in 35 languages but English answers.
Additionally, English-only object-centric questions
on OpenImages following [91]:
listing: `What` `objects` `are` `in` `the` `image?`,
presence: `Is` `{thing}` `in` `the` `image?`,
multi-object presence: `Which` `of` `{thing},`
`{thing}...` `are` `in` `the` `image?`,
and newly, counting: `How` `many` `{thing}?` .

```
 question {lang} {English answer}
```

Generated VQG on CC3M-35L following [19] generating questions in 35 languages, for a given
English answer.

```
 detect {thing} ; {thing} ; ...
```

Multi-object detection similar to Pix2Seq [22] on
generated open-world data via pseudo-labeling
as described in OWL-ViTv2 [83].

```
 segment {thing} ; {thing} ; ...
```

Multi-object instance segmentation as in PaLI3 [25] on generated open-world data similar to
OWL-ViTv2 [83] and SAM [54].

```
 caption <ymin><xmin><ymax><xmax>
```

Grounded captioning of what is in the box, following LocCa [115]. The box is indicated by the
same location tokens as used in detection and segmentation: normalized image coordinates binned
to 1024 tokens.



Figure 3 | Learning-rate schedule across stages.


Notably distinct from the widely used LLaVa’s
GPT-4 generated instruction following data, none
of PaliGemma’s pretraining tasks is the output of
a larger commercial VLM.


Finally, we believe that it is important to detect
and remove all images in our pretraining datasets
which are near-duplicates of images in the transfer tasks we evaluate in this report [55], as well
as a few more popular computer vision benchmarks. Doing so, we more accurately capture
PaliGemma’s capability to transfer to new tasks.


_**3.2.6.**_ _**Other pretraining details**_


Throughout pretraining, we use an "infinite"
learning-rate schedule following [131], which
provides a straightforward way of chaining several stages without decaying the learning-rate
between them. Figure 3 shows the full schedule:
pretraining is one continuous rsqrt curve for all
stages. The transfer can then act as a cooldown,
fully annealing the learning rate. We recommend
transferring with a simple setup that tunes the
full model using a cosine learning-rate schedule
with a short linear warm-up and decaying to zero.
This is not well represented by Figure 3 due to its
comparatively short duration.


The model was entirely trained in the opensource `big_vision` codebase [12] on Cloud
TPUv5e [38]. However, some of the pretraining
datasets remain private. During training, we partition data, as well as model parameters and optimizer state (Zero-DP style [96]) across all available devices using JAX [16] with GSPMD [125].
This fully-sharded data-parallel (FSDP [137])
sharding strategy is achieved by constructing
global arrays and annotating the sharding accordingly, with the XLA compiler [97] taking care of
the concrete implementation of the computation


6



1.0


0.5


0.0








---PAGE BREAK---

PaliGemma: A versatile 3B VLM for transfer



and communication between devices. We measured a model FLOPS utilization (MFU) of 55%,
resulting in 5189 tokens/second/device. Model
parameters and optimizer state are kept in float32
to guarantee stable training, but we verified that
inference works just as well with bfloat16 model
parameters.


One training run of the final PaliGemma model
using TPUv5e-256 takes slightly less than 3 days
for Stage1 and 15h for each Stage2. Stage1 sees
slightly less than 350 B tokens, and both Stage2
combined about 90 B tokens. Transfers take between 20min and 10h on TPUv3-32, depending
on the task.


In order to avoid model brittleness to different
image processing details in different frameworks,
we randomize the image preprocessing details
such as resize method, JPEG encoding, and apply
very slight `inception_crop` .

### **4. Results**


In order to verify the transferability of PaliGemma
to a wide variety of tasks, we transfer the pretrained models on more than 30 academic benchmarks via fine-tuning. Importantly, none of these
tasks or datasets are part of the pretraining data
mixture, and their images are explicitly removed
from the web-scale pretraining data. Results are
presented in Table 1.


To select the hyper-parameters for transfer, we
first sweep the parameters mentioned in Section 3.2.4, starting from the recommended value.
We do not necessarily perform the full crossproduct, and we sometimes extend or supersample the range, if it seems promising. Importantly,
we make any such decisions and hyper-parameter
choices based on the transfer task’s validation
split, and if none is provided, we hold out a small
“minival” set from the training data. Once we
found good hyper-parameter values for a task, we
re-train using the full training and validation data,
and report final test numbers. Details on tasks,
metrics, data splits are in Appendix B and final
hyper-parameters in Appendix J. In Section 6.2
we show that a single recommended value for
each hyper-parameter without any exploration



Table 1 | Results (1 random run of 5) obtained
with PaliGemma. Tasks marked with ⌞ indicate
zero-shot evaluation of the transferred model
above. Where numbers depend on server submissions we report standard deviation from validation splits. Highlighted rows indicate resolution sensitive tasks. Per-task details and hyperparameters are in Appendix B and J.


Task 224px 448px 896px


**Image captioning**
COCOcap 141.9 ±0.3 144.6 ±0.5  ⌞NoCaps 121.7 ±0.3 123.6 ±0.7  COCO-35L (en) 139.2 ±0.4 141.2 ±0.6  COCO-35L (avg34) 113.7 ±0.3 115.8 ±0.1  ⌞XM3600 (en) 78.0 ±0.8 80.0 ±0.3  ⌞XM3600 (avg35) 41.9 ±0.0 42.4 ±0.1  Screen2Words 117.6 ±0.7 119.6 ±0.7  

**Visual question answering**
VQAv2 83.2 ±0.4 85.6 ±0.2  ⌞MMVP 47.3 45.3  ⌞POPE 86.0 87.0  ⌞Objaverse Multiview 62.7 62.8  OKVQA 63.5 ±0.3 63.2 ±0.2  AOKVQA-MC 76.4 ±0.4 76.9 ±0.3  AOKVQA-DA 61.9 ±0.6 63.2 ±0.5  GQA 65.6 ±0.4 67.0 ±0.3  ⌞xGQA (avg7) 57.3 ±0.2 57.9 ±0.5  NLVR2 90.0 ±0.2 88.9 ±0.3  ⌞MARVL (avg5) 80.6 ±0.3 76.8 ±0.3  AI2D 72.1 ±0.7 73.3 ±0.7  ScienceQA 95.4 ±0.3 95.9 ±0.2  RSVQA-lr 92.6 ±0.3 93.1 ±0.7  RSVQA-hr (test) 92.6 ±0.1 92.8 ±0.1  RSVQA-hr (test2) 90.6 ±0.1 90.5 ±0.2  
VizWizVQA 73.7 ±0.2 75.5 ±0.5  TallyQA (simple) 81.7 ±0.2 84.9 ±0.1  TallyQA (complex) 69.6 ±0.1 72.3 ±0.2  CountBenchQA 81.9 ±1.6 83.1 ±2.1  OCR-VQA 72.3 ±0.4 74.6 ±0.4 74.9


**Image segmentation**
RefCOCO (testA) 75.7 ±0.1 77.9 ±0.1 78.7
RefCOCO (testB) 70.7 ±0.2 72.4 ±0.2 73.9
RefCOCO+ (testA) 71.9 ±0.1 74.2 ±0.2 76.1
RefCOCO+ (testB) 64.5 ±0.6 64.5 ±0.2 66.9
RefCOCOg (test) 68.2 ±0.1 71.0 ±0.1 72.7


**Video input**
ActivityNet-QA 50.8 ±0.4  -  ActivityNet-CAP 34.6 ±0.6  -  MSRVTT-QA 50.1 ±0.1  -  MSRVTT-CAP 70.5 ±0.9  -  MSVD-QA 60.2 ±0.3  -  VATEX 79.7 ±0.4  -  

7




---PAGE BREAK---

PaliGemma: A versatile 3B VLM for transfer



0


-20%


-40%


-60%


-80%

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
|ne||1M<br>3M<br>10M<br>30M<br>100M 300M<br>1|1M<br>3M<br>10M<br>30M<br>100M 300M<br>1|1M<br>3M<br>10M<br>30M<br>100M 300M<br>1|1M<br>3M<br>10M<br>30M<br>100M 300M<br>1|1M<br>3M<br>10M<br>30M<br>100M 300M<br>1|



Figure 4 | Relative regret of transfers when varying the amount of pretraining during Stage 1. Per
task plot in appendix K.1.


works almost as well on most tasks.


For all but video tasks, we report results on at
least two resolutions to provide an impression
of which tasks benefit from increased resolution.
We provide many resolution-related ablations in
Section 5.7.


Notably, we have not found any significant benefit from data augmentation. We simply resize the
input images to a square fixed resolution, even
for tasks such as RefCOCO segmentation (more
on that in Section 5.7 and Appendix C).

### **5. Ablations**


We conduct diverse ablations to gain deeper understanding of what matters for training and
transferring VLMs. Unless noted otherwise, all
ablations are run with the same setup as the main
models, except for making the Stage1 pretraining 10x shorter ( _i.e_ . 100 M examples seen), and
transfer results are reported on validation sets
instead of withheld test-sets. For each experiment, we present only the salient result summary
in the main text, but we provide a full per-task
breakdown of results in the Appendix.


**5.1.** **Multimodal pretraining duration**


With its 1 B examples seen, our multimodal pretraining (Stage1) is on the longer side, similar
to BLIP-2 [62], InternVL [26], QwenVL [10],
Idefics2 [59] (all around 1 B), but unlike
ShareGPT4-v [21], Mini-Gemini [65], LLaVa [70]
and its derivatives (around 1 M).



8





79

77

75

73

71





79

77

75

73

71












|Loss on prefix too|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|Loss on prefix too<br>|||||||
|Loss on prefix too<br>|||||||
|:<br> <br>u<br> p<br> a<br>o<br>g<br>o<br>a<br>g<br>h<br><br> u<br>r<br>h<br>t<br>i<br>e<br>d<br>s<br> a<br>l<br>**. **<br> a<br>n<br>k<br>i<br>r<br>t<br> <br> <br>o<br>p<br> h<br>e<br>c<br>e|<br>Suf<br>re<br> pl<br> nd<br> th<br>er<br>n. <br>tio<br> St<br> a<br>en<br>se <br> th<br>e r<br>s,<br>ng.<br>rs<br> f<br>. T<br>  go<br>e n<br> **C**<br>bl<br>in<br> br<br>rst<br>ess<br>eg<br> on<br>` pr`<br>n i<br>ate<br> e i<br>ns <br>all<br>n b|+P<br> 5 | <br> y t<br>  rig<br> e b<br> pre<br> W<br>ns,<br>ag<br>  co<br>dix <br> th<br> ree<br> es<br> and<br> So<br> onl<br>or <br>he<br>  od<br> ot<br>**aus**<br>ate <br>g le<br> eak<br>, w<br>ive<br>y w<br>  the<br>` ef`<br> s th<br> in<br>  ma<br> w<br>y c<br> ars|ref<br> Lea<br>  he<br>  ht:<br>  est<br> tra<br>e r<br> al<br>e1,<br>  mpl<br> K.<br>e b<br>  le<br> ult<br>  sk<br> me<br> y d<br>a b<br> 10<br>   tra<br>  sig<br>**al**<br> se<br> ar<br> do<br>e i<br> ma<br> hi<br>   “i<br>` ix` <br>  at<br>  th<br>  ge<br>hic<br> on<br>  al|+I<br> rni<br>   aut<br>  w<br>   of<br> ini<br>un <br> l th<br> an<br>  ete<br>1. <br>est<br>  arn<br>  sho<br>  ipp<br>  ta<br>  et<br>roa<br> 0 M<br>   de-<br>  nif<br>** ma**<br>ver<br> nin<br> wn<br>nve<br> ski<br> ch<br>   npu<br> tok<br>   it a<br>  e “<br>   tok<br>h r<br> frm<br>  so i|mg<br> ng<br>   o-r<br>het<br>    ou<br> ng<br> pr<br>  e<br>d <br>   b<br>For<br> tr<br>  ing<br>  ws<br>  in<br>  sk<br>  eri<br>d <br>   p<br>   of<br>  ca<br>** sk**<br>al <br> g<br>  in<br>sti<br> ng<br>  allo<br>   t”<br> en<br>    llo<br>   thi<br>   en<br>epr<br> ed<br>   nc|<br>   e<br>h<br>    r<br> <br>e<br> <br>s<br>   r<br> <br>a<br> <br> <br>  g<br>   a<br>  o<br>a<br>   r<br> <br>  n<br>** i**<br>o<br>  o<br> <br>g<br> . <br> <br>    p<br> s<br>    w<br>   n<br>   s<br><br> <br>   l|


|nsfer tasks|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|Task-<br>prompt?<br>|Task-<br>prompt?<br>|Task-<br>prompt?<br>|Task-<br>prompt?<br>|Task-<br>prompt?<br>|
|Task-<br>prompt?<br>|||||
|u<br>   e<br> t<br>     o<br>  e<br>i<br>    d<br>w<br>   -<br> <br>fe<br>  e<br>   t<br>   a<br>    a<br>  e<br> <br>   a<br>     a<br>   h<br>**  a**<br>u<br>  c<br>   p<br>e <br>li<br>   f<br> <br>  e<br>     m<br>   g<br>    w<br>n<br>   F<br>|Yes<br>  p f<br>   ssiv<br> o i<br>     wl<br>   n<br>nin<br>    o<br> th<br>   do<br> ca<br>r r<br>  s f<br>    sh<br>   ge<br>    fec<br>   a l<br>div<br>   ini<br>     bla<br>   ur<br>**  nd**<br>r k<br>  tiv<br>   end<br> de<br>Ge<br>   ull<br>     of t<br>   al<br>     ore<br>   ”<br>     c<br>t t<br>   igu<br>    the|<br>   o<br> <br>  n<br>     e<br>   o<br><br> <br><br> <br>s<br><br>   o<br>    o<br>   1<br> <br>    i<br>e<br>   n<br> <br>   t<br> <br><br>  e<br> <br>s<br> <br> <br>      h<br>   s<br> <br>    p<br>     a<br><br> <br>|No<br>   r S<br>   e m<br>  clu<br>     dg<br>   t b<br>gs <br>    n<br>e i<br>   n<br>e o<br>esu<br>   r e<br>    rte<br>    e<br>    ted<br>    ttl<br>rs<br>   g<br>     tio<br>   in<br>**   le**<br>ey <br>   in<br>   ix <br>ig<br>mm<br>    (bi<br>      e<br>   o<br>      to<br>    ro<br>     n<br>e <br>   re <br>     au|t<br> <br> <br>     e<br>    e<br> o<br>     t<br><br>    a<br>f<br>l<br>    a<br>    r<br>    n<br> <br>    e<br>e <br>    d<br>     n<br>   g<br>**   a**<br> c<br> <br> K<br>n<br> <br>    -<br>       d<br>    F<br>      k<br>    c<br>      a<br>q<br> 5<br>     t|




---PAGE BREAK---

