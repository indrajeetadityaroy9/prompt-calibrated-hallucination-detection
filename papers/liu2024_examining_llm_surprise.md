## **Trackable Agent-based Evolution Models at Wafer Scale**

Matthew Andres Moreno 1,2,3,*, Connor Yang 4, Emily Dolson 5,6, and Luis Zaman 1,2

1Department of Ecology and Evolutionary Biology, University of Michigan, Ann Arbor, United States
.
2Center for the Study of Complex Systems, University of Michigan, Ann Arbor, United States
3Michigan Institute for Data Science, University of Michigan, Ann Arbor, United States
4Undergraduate Research Opportunities Program, University of Michigan, Ann Arbor, United States
5Department of Computer Science and Engineering, Michigan State University, East Lansing, United States
6Program in Ecology, Evolution, and Behavior, Michigan State University, East Lansing, United States
*corresponding author: _morenoma@umich.edu_
**Abstract**



Continuing improvements in computing hardware are poised to
transform capabilities for _in silico_ modeling of cross-scale phenomena underlying major open questions in evolutionary biology and
artificial life, such as transitions in individuality, eco-evolutionary
dynamics, and rare evolutionary events. Emerging ML/AI-oriented
hardware accelerators, like the 850,000 processor Cerebras Wafer
Scale Engine (WSE), hold particular promise. However, many
practical challenges remain in conducting informative evolution
experiments that efficiently utilize these platforms’ large processor
counts. Here, we focus on the problem of extracting phylogenetic
information from agent-based evolution on the WSE platform.
This goal drove significant refinements to decentralized _in silico_
phylogenetic tracking, reported here. These improvements yield
order-of-magnitude performance improvements. We also present
an asynchronous island-based genetic algorithm (GA) framework
for WSE hardware. Emulated and on-hardware GA benchmarks
with a simple tracking-enabled agent model clock upwards of
1 million generations a minute for population sizes reaching 16
million agents. This pace enables quadrillions of agent replication
events a day. We validate phylogenetic reconstructions from these
trials and demonstrate their suitability for inference of underlying
evolutionary conditions. In particular, we demonstrate extraction,
from wafer-scale simulation, of clear phylometric signals that
differentiate runs with adaptive dynamics enabled versus disabled.
Together, these benchmark and validation trials reflect strong
potential for highly scalable agent-based evolution simulation that
is both efficient and observable. Developed capabilities will bring
entirely new classes of previously intractable research questions
within reach, benefiting further explorations within the evolutionary
biology and artificial life communities across a variety of emerging
high-performance computing platforms.


**Introduction**


A quintessential characteristic of computational artificial life
experiments is the near total malleability of the simulacrum
(Pattee, 1989). Indeed, exploration of arbitrary possibilities ‘as
they could be’ is the core of artificial life’s role as a tool for
inquiry (Langton, 1997). Such near-limitless freedom to realize
arbitrary system configurations, however, can obscure an intrinsic
limitation of most computational artificial life work: scale.
Take, for instance, the Avida platform, which instantiates populations of self-replicating computer programs for evolution experiments. When running on a single CPU, this system can support
about 20,000 generations per day, or about two hundred million
individual replication cycles daily (Ofria et al., 2009). By way
of comparison, _E. coli_ populations within individual flasks of the



Lenski Long-Term Evolution Experiment undergo only six doublings per day, meaning their generations take orders of magnitude
longer than Avidians (Good et al., 2017). (In continuous culture,
though, the rate can be c. 72 generations per day.) Indeed, such
capability for fast generational turnover has been a key motivation
for using artificial life systems to study evolution. However, the
effective population size of flasks in the Long-Term Evolution Experiment is orders of magnitude larger than Avida’s population size:
30 million vs. 10,000. Consequently, these systems actually exhibit a similar number of replication events per day. This pattern of
dramatically faster generation times than those observed in nature
and dramatically smaller populations largely generalizes across
artificial life systems. Of course, any such comparisons should also
note profound discrepancies between the genetic, phenotypic, and
environmental richness of biological organisms and ALife models.
Crucially, however, the scale of population size can greatly
impact subjects of artificial life research, like transitions in individuality, ecological dynamics, and rare evolutionary innovations
(Dolson & Ofria, 2021; Taylor, 2019; Taylor et al., 2016). Crossscale dynamics are also crucial to many key real-world challenges.
For example, in evolutionary epidemiology, interactions between
within-host infection dynamics and population-level epidemiological patterns determine the evolutionary trajectory of the population
(Schreiber et al., 2021). However, because capabilities of current
silicon-based processors are not expected to improve markedly in
the foreseeable future (Sutter et al., 2005), the scale-up necessary
to progress on these key frontiers will demand many-processor
computation. Application of parallel and distributed computation,
however, imposes compromises to the convenience, flexibility, observability, interpretability, total reliability, and perfect replicability
enjoyed under classical centralized, serial models of computation.
Encouragingly, these challenges are already implicit to much of
biology; the productivity of research involving natural organisms
evidences that they are surmountable and even hints at strategies
that can be used to solve them. Here, we explore alignment of digital evolution to HPC accelerator hardware at the extreme cutting
edge of massively distributed computation, and use techniques inspired by those applied to natural organisms to mitigate limitations
of distributed computation with respect to tracking phylogenies.



**Progress Toward Scale-up in Artificial Life**


Achieving highly scalable artificial life and digital evolution
systems involves two distinct engineering considerations. First,




---PAGE BREAK---

as with any high-performance scientific computing, careful
design is required to appropriately divvy computation and
communication workloads across available hardware. Second,
given the exceptionally discretionary nature of artificial life
modeling, we can intentionally tailor simulation semantics to suit
underlying hardware capabilities. Ackley’s ongoing work with
the T2 Tile Project and ULAM exemplifies a strong synthesis
of this engineering duality (Ackley & Ackley, 2016). At the level
of simulation semantics, Ackley formulates update procedures
in terms of local interactions between discrete, spatially situated
particles. This design provides for efficient one-to-one mapping
between simulation space and hardware components, minimizing
requirements for intra-hardware connectivity and preventing
global impacts from on-the-fly augmentations or reductions
of available hardware. The ULAM framework then ties into
implementation-level infrastructure necessary to accomplish performant, best-effort lock/release of spatial event windows spanning
bordering hardware units (Ackley et al., 2013). Ackley’s work is
distinguished, in fact, in dipping to a yet-lower level of abstraction
and tackling design of bespoke, modular distributed processing
hardware (Ackley, 2011, 2023; Ackley & Williams, 2011).
Several additional digital evolution projects have made notable
headway in synthesizing artificial life models with sophisticated,
scalable technical backing, achieving rich interactions among
numerous parallelized simulation components. Harding demonstrated large-scale cellular automata-based artificial development
systems, achieved through GPU-parallelized instantiations of
a genetic program (Harding & Banzhaf, 2007). Early work by
Ray with Network Tierra used an island model to distribute
digital organisms across federated servers, with migration handled
according to the real-time latencies and topology of the underlying
network (Ray, 1995). More recently, Heinemann’s continuation
of the ALIEN project has leveraged GPU acceleration to achieve
spectacularly elaborate simulations with rich interactions between
numerous spatially-situated soft body agents (Heinemann, 2008).
Likewise, the Distributed Hierarchical Transitions in Individuality
(DISHTINY) project has incorporated event-driven agent-agent
interaction schemes amenable to best-effort, asynchronous interlocution (Moreno & Ofria, 2022; Moreno et al., 2021b). GPU-first
agent-based modeling (ABM) packages like Flame GPU also
tackle this problem of hardware-simulacrum matching, albeit
framed at a higher level of abstraction (Richmond et al., 2010).
Beyond ALife, broader realms of application-oriented evolutionary
computation have folded in with many-processor computation,
most commonly through island-model and director-worker
evaluation paradigms (Abdelhafez et al., 2019; Cantu-Paz, 2001).´


**Untapped Emerging Hardware**

In retrospect, connectionist artificial intelligence turns out to
have been profoundly scale-dependent. The utility and ubiquity
of ANNs have exploded in tandem with torrential growth
in training set sizes, parameter counts, and training FLOPs
(Marcus, 2018). Recruitment of multi-GPU training for image
classification, requisite particular accommodating adjustments to
the underlying deep learning architecture, is commonly identified
as the watershed moment to this transformation (Krizhevsky
et al., 2012). Commercial investment in AI capabilities then set



in motion a virtuous cycle of further-enabling hardware advances
(Jouppi et al., 2017). Indeed, the scaling relationship between
deep learning and training resources has itself become a major
area of active study, with expectation for this virtuous cycle to
continue through the foreseeable future (Kaplan et al., 2020).
A major upshot of the deep learning race is the emergence
of spectacularly capable next-generation compute accelerators
(Emani et al., 2021; Jia et al., 2019; Medina & Dagan, 2020;
Zhang et al., 2016). Although tailored expressly to deep learning
workloads, these hardware platforms represent an exceptional
opportunity to leapfrog progress on grand challenges in artificial
life. The emerging class of fabric-based accelerators, led by
the 850,000 core Cerebras CS-2 Wafer-Scale Engine (WSE)
(Lauterbach, 2021; Lie, 2022), holds particular promise as a
vehicle for artificial life models. This architecture interfaces
multitudinous processing elements (PEs) in a physical lattice, with
PEs executing independently with private on-chip memory and
interacting locally through a network-like interface.
In this work, we explore how such hardware might be recruited
for large-scale digital evolution, demonstrating a genetic algorithm
implementation tailored to the dataflow-oriented computing model
of the CS-2 platform. Indeed, rapid advances in the capability
of accelerator devices, driven in particular by market demand
for deep learning operations, are anticipated to drive advances
in agent-based model capabilities (Perumalla et al., 2022). The
upcoming CS-3 chip, for instance, supports clustering potentially
thousands of constituent accelerators (Moore, 2024).


**Maintaining Observability**


Orthogonalities between the fundamental structure and objectives
of AI and artificial life methods will complicate any effort to requisition AI hardware for artificial life purposes. In common use, deep
learning operates as a black box medium (Loyola-Gonzalez, 2019)
(but not always (Mahendran & Vedaldi, 2015)). This paradigm
de-emphasizes accessibility of inner state. In contrast, artificial life
more often functions as a tool for inquiry. This goal emphasizes
capability to observe and interpret underlying simulation state
(Horgan, 1995; Moreno et al., 2023). (A similar argument holds
for ALife work driven by artistic objectives, as well.)
Unfortunately, scale complicates simulation observability. It
is not uncommon for the volume and velocity of data streams
from contemporary simulation to outstrip hardware bandwidth
and storage capacity (Klasky et al., 2021). Extensive engineering
effort will be required to ensure large-scale simulation retains
utility in pursuing rigorous hypothesis-driven objectives.
Here, we confront just a single aspect of simulation observability within distributed evolutionary simulation: phylogenetic history
(i.e., evolutionary ancestry relationships). Phylogenetic history
plays a critical role in many evolution studies, across study domains
and _in vivo_ and _in silico_ model systems alike (Faith, 1992; French
et al., 2023; Kim et al., 2006; Lenski et al., 2003; Lewinsohn et al.,
2023; Moreno et al., 2021a; Stamatakis, 2005). Phylogenetic analysis can trace the history of notable evolutionary events (e.g., extinctions, evolutionary innovations), but also characterize more general
questions about the underlying mode and tempo of evolution (Hernandez et al., 2022; Lewinsohn et al., 2023; Moreno et al., 2023;
Shahbandegan et al., 2022). Particularly notable, recent work has




---PAGE BREAK---

Figure 1: **Strategy for trackable distributed evolution simula-**
**tion.** Hstrat markers ( ) attached to replicating agents ( ) in a many-CPU
runtime (left panel) enable _post hoc_ estimation of relatedness between
lineages, enabling approximate phylogenetic reconstruction (right panel).


used comparison of observed phylogenies against those produced
under differing simulation conditions to test hypotheses describing
underlying dynamics within real-life evo-epidemiological systems
(Giardina et al., 2017; Voznica et al., 2022). Additionally, _in silico_,
phylogenetic information can even serve as a mechanism to guide
evolution in application-oriented domains (Burke et al., 2003;
Lalejini et al., 2024a,b; Murphy & Ryan, 2008).


**Decentralized Phylogenetic Tracking**


Most existing artificial life work uses centralized tracking to
maintain an exact, complete record of phylogenetic history
comprising all parent-offspring relationships that have existed over
the course of a simulation (Bohm et al., 2017; De Rainville et al.,
2012; Dolson et al., 2024; Garwood et al., 2019; Godin-Dubois
et al., 2019; Ray, 1992). Typically, records of extinct lineages are
pruned to prevent memory bloat (Moreno et al., 2024c). Although
direct tracking is well suited to serial simulation or centralized
controller-worker schemes, runtime communication overheads and
sensitivity to data loss impede scaling to highly distributed systems

- particularly those with lean memory capacity like the Cerebras
WSE (Moreno et al., 2024c). To overcome this limitation, we
have developed reconstruction-based approaches to _in_ _silico_
phylogenetic tracking (Moreno et al., 2022a). These approaches
require no centralized data collection during simulation runtime;
instead, they use _post hoc_ comparisons among end-state agent
genomes to deduce approximate phylogenetic history - akin
to how DNA-based analyses describe natural history. Figure 1
summarizes this reconstruction-based strategy.


Although analogous work with natural biosequences is notoriously challenging and data-intensive (Lemmon & Lemmon, 2013;
Neyman, 1971), the recently-developed hereditary stratigraphy
annotation architecture is explicitly designed for fast, accurate,
and data-lean reconstruction. Designed to attach on underlying
replicators as a neutral annotation (akin to noncoding DNA), it is
a general-purpose technique potentially applicable across diverse
study domains (Cohen, 1987; Friggeri et al., 2014; Liben-Nowell
& Kleinberg, 2008). In a stroke of convergent thinking, Ackley
(2023) reports use of “bar code” annotations on his self-replicators
to achieve a measure of coarse-grained lineage tracing.



**Contributions**


In this paper, we report new software and algorithms that harness
the Cerebras Wafer-Scale Engine to enable radically scaled-up
agent-based evolution while retaining key aspects of observability
necessary to support hypothesis-driven computational experiments.
Implementation comprises two primary aspects:
1. an asynchronous island-based genetic algorithm (GA) suited
to the memory-scarce, highly-distributed, data-oriented WSE
architecture, and
2. a fundamental reformulation of hereditary stratigraphy’s core
storage and update procedures to achieve fast, simple, resourcelean annotations compatible with unconventional, resourceconstrained accelerator and embedded hardware like the WSE.
Both are implemented in Cerebras Software Language (CSL)
and validated using Cerebras’ SDK hardware emulator. We use
benchmark experiments to evaluate the runtime performance
characteristics of the new hereditary stratigraphy algorithms in
isolation and in the integrated context providing tracking-enabled
support for the island-model GA. In conjunction, we report
emulated and on-device trials that validate phylogenetic
reconstructions and demonstrate their suitability for inference of
underlying evolutionary conditions.
Results from both experiments are promising. We find that new
surface-based algorithms greatly improve runtime performance.
Scaled-down emulator benchmarks and early on-hardware trials
indicate potential for simple agent models — with phylogenetic
tracking enabled — to achieve on the order of quadrillions of agent
replication events a day at full wafer scale, with support for population sizes potentially reaching hundreds of millions. Further, using
proposed techniques, phylogenetic analyses of simulations spanning hundreds of thousands of PEs succeed in detecting differences
in adaptive dynamics between alternate simulation conditions.


**Methods**


This section recounts the core mechanics of hereditary stratigraphy
methods for phylogenetic tracking and describes new lightweight
“surface” data structures used in this work. Then, we describe
the asynchronous compute-communicate implementation strategy
behind the testbed island-based genetic algorithm used for WSE
validation and benchmarking experiments.


**Distributed Phylogenetic Tracking**


Natural history of biological life operates under no extrinsic provision for interpretable record-keeping, yet efforts to study it have
proved immensely fruitful. This fact bodes well that scaled-up
simulation studies can succeed as a platform for rich hypothesisdriven experiments despite potential sacrifices to aspects of
observability currently enjoyed with centralized, storage-rich
processing. Indeed, observational and analytical strategies already
developed to confront limitations in biological data can solve, or at
least guide, work with massively distributed evolution simulations.
In biology, mutational drift encodes ancestry information in
DNA genomes. Our proposed methods are analogous, with
ancestry information captured within genomes themselves rather
than external tracking. Phylogenetic history can then be estimated
after the fact, as Figure 1 depicts. This strategy reduces runtime




---PAGE BREAK---

3.3e+03
2.4e+03
1.5e+03
6.0e+02
0.0e+00



0 32 64 96 128 160 192 224
Buffer Position



**steady retention policy** **tilted retention policy**



0 32 64 96 128 160 192 224 256
Buffer Position



10 [3]
10 [2]
10 [1]
10 [0]
0



2000



0



0 375 750 1125 1500 1875 2250 2625 3000
Ingestion Time Point



0 375 750 1125 1500 1875 2250 2625 3000 0 375 750 1125 1500 1875 2250 2625 3000
Ingestion Time Point Ingestion Time Point

Figure 2: **Surface-based hereditary stratigraphy implementations.** Visualizations of steady (left) and tilted (right) surface site selection policies.
Top-row heatmaps show evolution of time-since-last-deposition for each site on a 256-bit field over the course of 4,096 time steps. The bottom row
shows retention spans for 3,000 ingested time points. Vertical lines span durations between ingestion and elimination for differentia appended at successive
time points. Time points previously eliminated are marked in red. Time elapses from bottom to top in both visualizations.



communication and is resilient to germane modes of data loss (e.g.,
dropped messages, hardware crash-out). Note that, in addition to
sampling end-state genomes, extracting “fossil” genomes over the
course of simulation can allow lineages that end up going extinct
to be included in phylogenetic reconstruction.

Recent work introducing _hereditary_ _stratigraphy_ (hstrat)
methodology has explored how best to organize genetic material to
maximize reconstruction quality and minimize memory footprint
(Moreno et al., 2022a,b). Hstrat material can be bundled with
agent genomes in a manner akin to non-coding DNA, entirely
neutral with respect to agent traits and fitness.

The hereditary stratigraphy algorithm associates each generation
along individual lineages with an identifying “fingerprint” value,
referred to as a differentia. On birth, offspring receive a new
differentia value and append it to an inherited chronological
record of past values, each corresponding to a generation along
that lineage. Under this scheme, mismatching differentiae can
be used to delimit the extent of common ancestry. This semantic
streamlines _post_ _hoc_ phylogenetic reconstruction to a simple
trie-building procedure (Moreno et al., 2024c).

To save space, differentia may be pruned away. However, care
must be taken to ensure retention of checkpoint generations that
maximize coverage across evolutionary history. Reducing the
number of bits per differentia can also provide many-fold memory
space savings. These savings can be re-invested to increase the
quantity of differentia retained, improving the density of record
coverage across elapsed generations. The cost of this shift is
an increased probability of spurious differentia value collisions,
which can make two lineages appear more closely related than
they actually are. We anticipate that most use cases will call for
differentia sized on the order of a single bit or a byte. Indeed,
single-bit differentiae have been shown to yield good quality
phylogenies using only 96 bits per genome (Moreno et al., 2024a).

Small differentia size intensifies need for a lean data structure
to back differentia record management. Shrinking differentia to
a single bit would be absurd if each is accompanied by a 32-bit
generational timestamp. To delete timestamps, though, we need
means to recalculate them on demand. As such, all described
algorithms can deduce timestamps of retained differentia solely
from their storage index and the count of elapsed generations.

Lastly, design of hstrat annotations must also consider how



available storage space should be allocated across the span
of history. In one possible strategy, which we call _**“steady”**_
_**policy**_, retained time points would be distributed evenly across
history. By contrast, under _**“tilted” policy**_ more recent time points
are preferred (Han et al., 2005; Zhao & Zhang, 2006). Note
that prior hereditary stratigraphy work refers to them instead
as “depth-proportional” and “recency-proportional resolution”
(Moreno et al., 2022a). Comparisons of reconstruction quality
have shown that tilted policy gives higher quality reconstructions
from the same amount of reconstruction space in most — but not
all — circumstances (Moreno et al., 2024a). This pattern follows
an intuition that high absolute precision is more useful to resolving
recent events than more ancient ones. In practice, it may be
desirable to use a hybrid approach that allocates half of available
annotation space to each strategy (Moreno et al., 2024a). The
bottom panels of Figure 2 contrast steady versus tilted behaviors.


**Surface-based Hereditary Stratigraphy Algorithms**

At the outset of this project, several problematic aspects of porting
existing hereditary stratigraphy algorithms to the WSE became
apparent. Issues stemmed, in part, from a fundamental feature
of these algorithms: organization of retained strata in contiguous,
sorted order. This design imposes various drawbacks:

- _wasted space_ : an annotation size cap can be guaranteed, but
a percentage of available space typically goes unused;

- _high-level_ _feature_ _dependencies:_ in places, existing column
code uses complex data structures with dynamically allocated



Figure 3: **Column vs. surface-based hereditary stratigraphy.**
Contrast of existing sorted-order “column”-based stratum retention
framework with proposed explicitly addressed “surface”-based approach.












































































---PAGE BREAK---

memory to perform operations like set subtraction; and

in sorted order. Figure 3 contrasts the two approaches.

and the bottom panels showing the resulting distributions of
retained time points across history. We leave formal descriptions
of underlying indexing algorithms to future work. However, reference Python implementations can be found in provided software
materials. For this work, we also translated the tilted algorithm
to the general-purpose Zig programming language and then to
the Zig-like Cerebras Software Language for use on the WSE.
These algorithms are notable in providing a novel and highly
efficient solution to the more general problem of curating dynamic
temporal cross-samples from a data stream, and may lend
themselves to a broad set of applications outside the scope of
phylogeny tracking (Moreno et al., 2024b).


**WSE Architecture and Programming Model**

The Cerebras Wafer-Scale Engine comprises a networked grid of
independently executing compute cores (Processing Elements or
PEs). Each PE contains dedicated message-handling infrastructure,
which routes tagged 32-bit packets (“wavelets”) to neighboring
PEs and/or to be processed locally, according to a programmed
rule set. Each PE contains 48kb of private on-chip memory
wideth single clock cycle latency; communication to neighboring
PEs, too, incurs low latency (Buitrago & Nystrom, 2021). PEs
support standard arithmetic and flow-control operations, as well
as vectorized 32- and 16-bit integer and floating-point operations.
The WSE device is programmed by providing “kernel” code,
written in the Cerebras Software Language (CSL), to be executed
on each PE. This language’s programming model purposefully
reflects characteristics of the underlying WSE architecture. CSL
uses an event-driven framework, with tasks defined to respond
to wavelet activation signals exchanged between — and within

- PEs. Scheduling of active tasks occurs via hardware-level
microthreading, which allows for some level of concurrency.
Special faculties are provided for asynchronous send-receive
operations that exchange buffered data between PEs.
For further detail, we refer readers to extensive developer
documentation made available by Cerebras through their SDK
program. Access can be requested, currently free of charge, via
their website. For this initial work, we evaluated some CSL code



on a virtualized 3 _×_ 3 PE array emulated with conventional CPU
hardware, while other experiments used hundreds of thousands
of PEs on a physical CS-2 device.


**Asynchronous Island-model Genetic Algorithm**

We apply an island-model genetic algorithm, common in
applications of parallel and distributed computing to evolutionary
computation, to instantiate a wafer-scale evolutionary process
spanning PEs (Bennett III et al., 1999). Under this model, PEs
host independent populations that interact through migration (i.e.,
genome exchange) between neighbors.
Our implementation unfolds according to a generational
update cycle. Migration, depicted as blue-red arrows, is handled
first. Each PE maintains independent immigration buffers and
emigration buffers dedicated to each cardinal neighbor, depicted by
Figure 4 in solid blue and red, respectively. On simulation startup,
asynchronous receive operations are opened to accept genomes
from each neighboring PE into its corresponding immigration
buffer. At startup, additionally, each emigration buffer is populated
with genomes copied from the population and an asynchronous
send request is opened for each. Asynchronous operations are
registered to on-completion callbacks that set a per-buffer “send-”
or “receive-complete” flag variable. In this work, we size send
buffers to hold one genome and receive buffers to hold four
genomes. The main population buffer held 32 genomes.
Subsequently, the main update loop tests all completion flags.
For each immigration flag that is set, buffered genomes are
copied into the main population buffer, replacing randomly
chosen population members. Then, the flag is reset and a new
receive request is initiated. Likewise, for each emigration flag
set, corresponding send buffers are re-populated with randomly
sampled genomes from the main population buffer. Corresponding
flags are then reset and new send requests are initiated. Figure 4b
summarizes the interplay between send/receive requests, callbacks,
flags, and buffers each generation (“main cycle”).
The remainder of the main update loop handles evolutionary
operations within the scope of the executing PE. Each genome








































































---PAGE BREAK---

steady

tilted
trivial

steady

tilted

trivial









Figure 5: **Hereditary stratigraphy algorithm benchmarks.** Comparison of per-generation operation time for column- and surface-based steady
and tilted retention policies, lower is better. Top and bottom panels show Python and Zig implementations, respectively. Trivial is a simple hardcoded
retention decision, provided as a baseline control. Background hatching indicates significant outcomes (Mann-Whitney U test; _n_ =20).



within the population is evaluated to produce a floating point
fitness value. For this initial work, we use a trivial fitness function
that explicitly models additive accumulation of beneficial/deleterious mutations as a floating point value within each genome. After
evaluation, tournament selection is applied. Each slot in the next
generation is populated with the highest-fitness genome among
_n_ =5 randomly sampled individuals, ties broken randomly.
Finally, a mutational operator is applied across all genomes in
the next population. Experiments used a simple Gaussian mutation
on each genome’s stored fitness value, with sign restrictions used
to manipulate the character of selection in some treatments. At
this point, hereditary stratigraphy annotations — discussed next

- are updated to reflect an elapsed generation.
The next generation cycle is then launched by a self-activating
wavelet, repeating until a generation count halting condition is met.


**Genome Model**

We used a 96-bit genome for clade reconstruction trials, shown in
Supplementary Figure 8 (Moreno et al., 2024d). At the outset of
simulation, the first 16 bits of founding genomes were randomized
and, subsequently, were inherited without mutation, thus identifying descendants of the same founding ancestor. The next 80 bits
were used for hereditary stratigraph annotation, 16 bits for a generation counter and the remaining 64 as a tilted-retention surface with
single-bit differentiae. Other experiments use a 128-bit genome
layout, with the first 32 bits used for a floating point “fitness”
value and the generation counter upgraded from 16 to 32 bits.


**Software and Data Availability**

Software, configuration files, and executable notebooks for this work are available via Zenodo at
[doi.org/10.5281/zenodo.10974998.](https://doi.org/10.5281/zenodo.10974998) Data and
supplemental materials are available via the Open Science
[Framework at osf.io/bfm2z/ (Foster & Deardorff, 2017).](https://osf.io/bfm2z/)
Hereditary stratigraphy utilities are published in the hstrat
Python package (Moreno et al., 2022b). This project used data
formats from the ALife Standards project (Lalejini et al., 2019)
and benefited from open-source scientific software (Cock et al.,
2009; Dolson et al., 2024; Harris et al., 2020; Huerta-Cepas et al.,
2016; Hunter, 2007; Moreno, 2024a,b; Moreno & Rodriguez
Papa, 2024; Moreno & Yang, 2024; pandas developers, 2020;
Virtanen et al., 2020; Waskom, 2021; Wes McKinney, 2010).
WSE experiments reported in this work used the CSL compiler
and CS-2 hardware emulator bundled with the Cerebras SDK
v1.0.0 (Selig, 2022), available via request from Cerebras. SDK
utilities can be run from any Linux desktop environment,
regardless of access to Cerebras hardware. We accessed CS-2
hardware through PSC Neocortex (Buitrago & Nystrom, 2021).



**Results and Discussion**


Here, we report a series of benchmarks that evaluate the viability
of our proposed approaches to harness the CS-2 accelerator for
digital evolution simulations and extract phylogenetic records
from said simulations. First, we compare the performance
characteristics of new surface-oriented hereditary stratigraphy
methods against preceding column-based implementation to
determine the extent to which they succeed in streamlining
runtime operations. Then, we benchmark phylogeny-tracked
WSE genetic algorithm implementations. To assess the runtime
overhead of surface-based tracking, we compare against
benchmarking results with phylogenetic tracking disabled.
Finally, we validate the credibility of the presented end-to-end
annotation-to-reconstruction pipeline by reviewing clade structure
and phylometric properties from emulated and on-hardware runs.


**Surface Algorithm Benchmark on Conventional CPU**


We performed microbenchmark experiments to test computational
efficiency of new surface-based algorithms. Trials measured
the real-time speed of sequential generation updates on one
annotation with capacity for 64 differentiae. Benchmarks used
both Python, for comparability with existing column algorithm
implementations, and Zig, to assess performance under compiler
optimization. Figure 5 overviews results.
Python implementations of the surface tilted and steady
algorithms both took around 4.2 microseconds per operation
(standard error [SE] 50ns and 66ns; _n_ = 20). For context,
this speed was about 4 _×_ the measured time for a surface
placement using a trivial calculation (SE 0.05; _n_ = 20). As
expected, column implementations of steady and tilted fared
much worse, taking about 7 _×_ and 34 _×_ the execution time per
operation compared to the surface operations. In both cases,
surface implementations significantly outperformed their column
counterparts (Mann-Whitney U test, _α_ =0 _._ 05).
Zig microbenchmarks clock tilted surface annotation updates
at 230 ns per operation (SE 0.9ns; _n_ = 20). For context,
this time is a little more than twice that required for a main
memory access in contemporary computing hardware (Velten
et al., 2022). Our results measure the operation at 49 _×_ the
measured time for a trivial placement calculation (SE 0.03;
_n_ =20). Zig steady implementation clocks 511 ns per operation
(SE 4; _n_ = 20), 110 _×_ trivial (SE 0.8ns; _n_ = 20). Note that
speedup of Zig implementations relative to Python reflects
an intrinsic performance penalty due to interpreter overhead
of Python evaluation. However, comparisons among Python
implementations and among Zig implementations are nonetheless
informative because all are on equal footing in this regard.




---PAGE BREAK---

0-0xb89b
8-0x5daa



Low-hanging speedups and optimizations exist to further
improve the per-operation surface update performance achieved
in practice. Half of update operations on surfaces with single-bit
differentiae can be skipped entirely, owing to 50% probability that
randomization fails to change a stored differentia value. Further,
simulations with synchronous or near-synchronous generations
can cache calculated surface-placement indices, meaning they
would only need to be computed once for an entire subpopulation.
Another possibility is to coarsen temporal resolution, only
updating annotations at intervals every _n_ th generation.


**Wafer-Scale Engine Island-model GA Benchmark**


Next, we used an emulator to characterize the expected
performance of our island model genetic algorithm on WSE
hardware and estimate the magnitude of simulation that might
be achieved with on-device execution. We used the emulator’s
per-PE clock cycle counters to measure the amount of real time
elapsed over the course of a 40-generation-cycle simulation. We
tested using a 3 _×_ 3 PE collective with the tagged 3-word genome
and a per-PE population size of 32, applying a tilted hereditary
stratigraphy every generation. PEs completed a mean of 24,138
generation cycles per second (SEM 99; _n_ =9). As an indicator
of inter-PE exchange throughput, each PE immigrated a mean of
118 genomes (SEM 11; _n_ =9) over 40 elapsed generation cycles.

What scale of simulation does this performance imply at full
scale on CS-2 hardware? Across eight on-device, tracking-enabled
trials of 1 million generations (described in the following section),
we measured a mean simulation rate of 17,688 generations per
second for 562,500 PEs (750 _×_ 750 rectangle) with run times
slightly below one minute. Trials with 1,600 PEs (40 _×_ 40)
performed similarly, completing 17,734 generations per second.

Multiplied out to a full day, 17,000 generations per second
turnover would elapse around 1.5 billion generations. With 32
individuals hosted per each of 850,000 PEs, the net population
size would sit around 27 million at full CS-2 wafer scale. (Note,
though, that available on-chip memory could support thousands
of our very simple agents per PE, raising the potential for a
net population size on the order of a billion agents.) A naive
extrapolation estimates on the order of a quadrillion agent
replications could be achieved hourly at full wafer scale for such a
very simple model. We look forward to conducting more thorough
benchmarking and scaling experiments in future work.

How fast is simulation without hstrat instrumentation? We
repeated our hardware-emulated benchmark with 32-bit genomes
stripped of all instrumentation. Under these conditions, PEs
completed on average 47,175 generations per second (SEM 220;
_n_ =9) and immigrated 118 genomes (SEM 12; _n_ =9).

These timings measure phylogeny tracking as approximately
equivalent to that of the other aspects of simulation, combined.
Given the highly minimalistic nature of the agent model and selection process, this result is highly promising. In actual use, most
experiments will likely involve a much more sophisticated agent
model, so relative overhead of tracking will be diminished. Additionally, these results do not include caching and coarsening strategies discussed above, which would speed tracking up considerably.



Figure 6: **Clade** **Reconstruction** **Trial.** Example phylogenies
reconstructed from runs of increasing duration on a virtual grid of nine
hardware-simulated PEs. Founding genomes were tagged with random
16-byte identifier values, which were held constant throughout simulation
(Supplementary Figure 8). Color-coding indicates each sampled taxon’s
founding ancestor according to this identifier value. Simulation was
performed with neutral selection.


**Clade Reconstruction Trial**


To assess overall correctness of our Cerebras Software Language
(CSL) surface algorithm implementation, we reviewed the
clade structure of sample phylogenetic reconstructions from
emulated WSE hardware. Full reconstruction quality testing of the
underlying new surface algorithms themselves is provided in other
recent work (Moreno et al., 2024a). (There, we found quality to be
comparable to existing column algorithms.) For these experiments,
we tagged genomes with a 16-bit randomly generated identifier at
simulation startup, as shown in Supplementary Figure 8 (Moreno
et al., 2024d). We instantiated populations over a 3 _×_ 3 PE
collective for durations of 25, 50, 100, and 250 generation cycles
with neutral selection. Phylogenetic reconstruction used one
end-state genome sampled per PE.
Figure 6 shows reconstructed phylogenies from each duration.
As expected, taxa belonging to the same founding clade (shown
by color) are reconstructed as more closely related to each
other than to other taxa. Additionally, and also as expected,
the number of distinct remaining founding clades diminishes
monotonically with increasing simulation duration. These
consistencies corroborate the general integrity of our CSL surface
implementation. Note, however, the incidence of moderate
overestimations for relatedness between independently-tagged
clades throughout. As mentioned earlier, and discussed in greater
depth elsewhere (Moreno et al., 2024a), this is an expected artifact
of hereditary stratigraphy with single-bit differentiae. Applications
requiring greater reconstruction precision can opt for larger
differentia sizes and/or higher differentia counts.


**On-hardware Trial**


Finally, we set out to assess the performance of our pipeline at
full wafer scale. For these experiments, we used the four-word,
tracking-enabled genome layout, with the full first 32 bits
containing a floating point fitness value. We defined two



5-0x228a



1-0x8aa2



(a)


(b)


(c)


(d)



4-0xbe17



6-0xca2
2-0x7896
7-0x6e3b



3-0x748f
0-0x748f



3-0x5c57



0 5 10 15 20 25 30



6-0x82c8
4-0x82c8
0-0xaf0b

8-0x55cf
7-0x55cf
1-0x97c0
5-0xf38d
3-0x172c



2-0x4489



0 10 20 30 40 50 60



0-0x51fc



7-0x3738



4-0x7d9
3-0x7d9
5-0x7d9



8-0x722b
6-0x722b

[2-0][x98c9]
1-0x98c9

100

4-0x82dd
7-0x82dd
6-0x82dd



0 20 40 60 80 100 120



1-0xebe0
8-0xebe0



5-0xebe0
2-0xebe0



0 50 100 150 200 250

Generations




---PAGE BREAK---

1e7



sum branch
lengths



1.5

1.0

0.5



regime

adaptive
purifying



mean evolutionary
distinctiveness


1 25 56
num PEs (10k)



1 25 56
num PEs (10k)



7500

5000

2500



(a) adaptive vs. purifying phylometric structure


(b) adaptive regime (c) purifying regime

Figure 7: **On-hardware Trial.** Results from phylogenetic reconstruction of 1 million generation on-hardware simulations. Panel 7a compares
phylometric readings from purifying-only and adaptation-enabled configurations, 4 replicates each. Panels 7b and 7c juxtapose example 750 _×_ 750
PE simulation phylogenies under each configuration regime. Phylometrics
were calculated from reconstructions with 10k sampled end-state agents.
For legibility, phylogeny visualizations were further subsampled to 1k
end-state agents. Top phylogenies are linear-scaled. Bottom phylogenies
are log-scaled with ultrametric correction to better show topology.


treatments: _**purifying-only**_ _**conditions**_, where 33% of agent
replication events decreased fitness by a normally-distributed
amount, and _**adaption-enabled conditions**_, which added beneficial
mutations that increased fitness by a normally-distributed amount,
occurring with 0.3% probability. These beneficial mutations
introduced the possibility for selective sweeps. As before, we
used tournament size 5 for both treatments. We performed four
on-hardware replicates of each treatment instantiated on 10k
(100 _×_ 100), 250k (500 _×_ 500) and 562.5k (750 _×_ 750) PE arrays.
We halted each PE after it elapsed 1 million generations.
Upon completion, we sampled one genome from each PE. Then,
we performed an agglomerative trie-based reconstruction from subsamples of 10k end-state genomes (Moreno et al., 2024c). Figure
7 compares phylogenies generated under the purifying-only and
adaption-enabled treatments. As expected (Moreno et al., 2023),
purifying-only treatment phylogenies consistently exhibited greater
sum branch length and mean evolutionary distinctiveness, with the
effect on branch length particularly strong. These structural effects
are apparent in example phylogeny trees from 562.5k PE trials
(Figures 7b and 7c). Successful differentiation between treatments
is a highly promising outcome. This result not only supports the
correctness of our methods and implementation, but also confirms
the capability of reconstruction-based analyses to meaningfully
describe dynamics of very large-scale evolution simulations.


**Conclusion**


Computing hardware with transformative capabilities is presently
coming to market. This fact presents an immediate opportunity
to bring orders-of-magnitude greater simulation scale to bear
on grand challenges in artificial life. It is not unreasonable to
anticipate the possibility that with such resources some aspects of
these open questions will be revealed to harbor more-is-different
dispositions, in which larger scales reveal qualitatively different



dynamics (Anderson, 1972). Riding the coattails of AI-workloaddriven hardware development, itself largely driven by profound
more-is-different payoffs in deep learning, provides perhaps the
most immediate means toward this possibility.
Such an endeavor is a community-level challenge that will
require significant resources and collaborative effort. Work
presented here is an early step in methods and infrastructure
development necessary to scale up what is possible in digital
evolution research. We have demonstrated new algorithms
and software for phylogeny-enabled agent-based evolution
on a next-generation HPC accelerator hardware platform.
Microbenchmarking results show that proposed instrumentation
algorithms achieve several-fold improvement in computational
efficiency. Related work shows new algorithms to improve
reconstruction quality in some cases, too (Moreno et al., 2024a).
Benchmarks confirm that, including tracking operations, simple
models at wafer scale can achieve quintillions of replications
per day. In future work, it will be necessary to move beyond
proof-of-concept and explore the limits of these capabilities in
the context of more demanding, interaction-intensive models.
Special characteristics set agent-based digital evolution apart
from many other HPC application domains and position it as a
potentially valuable testbed for innovation and leadership. Among
these factors are challenging workload heterogeneity (varying
within a population and over evolutionary time), resiliency
of global state to local perturbations, and perhaps substantial
freedom to recompose underlying simulation semantics to
accommodate hardware capabilities. Indeed, artificial life and
digital evolution have played host to notable boundary-pushing
approaches to computing at the frontiers of computing modalities
such as best-effort computing, reservoir computing, global-scale
time-available computing, and modular tile-based computing in
addition to more traditional cluster and GPU-oriented approaches
(Ackley, 2020, 2023; Heinemann, 2008; Miikkulainen et al., 2024;
Moreno et al., 2021b; Ray, 1995). Work done to scale up digital
evolution simulation should be done with an eye for contributions
back to broader HPC constituencies.
In this vein, presented “surface” indexing algorithms stand to
benefit larger classes of stream curation problems, situations in
which a rolling feed of sequenced observations must be dynamically downsampled to ensure retention of elements representative
across observed history (Moreno et al., 2024b). In particular, to further benefit observable agent-based modeling, we are interested in
exploring applications that sample time-series activity at simulation
sites or distill coarsened agent histories (e.g., position over time).
Our goal in this work is to build new capabilities that empower
research agendas across the digital evolution and artificial life
community. To this end, we have prioritized making our CSL and
Python software easily reusable by other researchers. In particular,
CSL code implementing the presented island-model GA is modularized and extensible for drop-in customization to instantiate any
fixed-length genome content and fitness criteria. We look forward
to collaboration in broader tandem efforts to harness the Cerebras
platform, and other emerging hardware, in follow-on work.


**Acknowledgement**


Computational resources were provided by the MSU Institute for Cyber



---PAGE BREAK---

