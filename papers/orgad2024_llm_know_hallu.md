# Full Characterization of the Depth Overhead for Quantum Circuit Compilation with Arbitrary Qubit Connectivity Constraint

Pei Yuan and Shengyu Zhang


Tencent Quantum Laboratory, Tencent, Shenzhen, Guangdong 518057, China


In some physical implementations of quantum computers, 2-qubit operations can
be applied only on certain pairs of qubits. Compilation of a quantum circuit into
one compliant to such qubit connectivity constraint results in an increase of circuit
depth. Various compilation algorithms were studied, yet what this depth overhead is
remains elusive. In this paper, we fully characterize the depth overhead by the _routing_
_number_ of the underlying constraint graph, a graph-theoretic measure which has been
studied for 3 decades. We also give reduction algorithms between different graphs,
which allow compilation for one graph to be transferred to one for another. These
results, when combined with existing routing algorithms, give asymptotically optimal
compilation for all commonly seen connectivity graphs in quantum computing.

### 1 Introduction


Quantum computation has demonstrated a substantial advantage over its classical counterpart in
solving significant problems such as integer factorization [1], search [2] and a wide variety of
others, by structurally designed algorithms [3] as well as variational algorithms [4]. Quantum
hardware technologies have seen considerable advancements in recent years [5–14], enabling the
execution of quantum algorithms. A crucial aspect of this execution involves the compilation of
quantum algorithms into quantum circuits, typically composed of 1- and 2-qubit gates.
Despite the theoretically proven advantages, the practical implementation of quantum algorithms and quantum circuits faces numerous challenges, one of which is the qubit connectivity
constraint. On certain hardware platforms such as superconducting [5, 7, 8], quantum dots [9–13]
and cold atoms [15–17], 2-qubit gates can only act on certain pairs of qubits, while operations on
two distant qubits are usually achieved by a sequence of SWAP operations [18–25]. This qubit
connectivity constraint can be naturally modeled by a connected _constraint_ _graph_ _G_ = ( _V, E_ ),
where the vertex set _V_ represents the qubits, and the edge set _E_ specifies the pairs of qubits on
which 2-qubit gates can act. A circuit respecting the _G_ constraint is termed _G-compliant_ in this
paper.
Various constraint graphs exist on real hardware, including path [5, 26], bilinear chains [5, 6],
2D-grids [7, 8], brick walls [5], cycle-grids [27] and trees [5]. Future connectivity patterns may
even broaden this variety. The wide diversity of the constraint graphs calls for systematic studies
on the compilation overhead due to the connectivity constraint. In this paper, we focus on circuit
depth, which typically corresponds to the circuit’s execution time. In the NISQ era [28], the


Pei Yuan: peiyuan@tencent.com


Shengyu Zhang: shengyzhang@tencent.com




---PAGE BREAK---

execution time is particularly relevant as most variational quantum algorithms require executing
the same circuit thousands of times and use the sample average to estimate the expectation of an
observable on the circuit’s final state.
Several questions arise about the depth overhead. On a given constraint graph _G_, what is the
smallest depth overhead, as a measure of the graph, that a compiler can possibly achieve? How do
we actually compile a circuit achieving this minimal depth increase? When designing a quantum
chip, how should we lay out the qubits to ensure their connectivity facilitates a small circuit depth
overhead? In this paper, we address these questions by fully characterizing the depth overhead for
_any_ given graph, with an explicit compilation algorithm given and the optimality shown.
A widely adopted generic method compiling a quantum circuit under qubit connectivity constraint is to insert SWAP gates into the original circuit to bring (the states in) two distant qubits
together, apply the two-qubit operations, and then move them back [18–25]. This method is also
the focus of this paper.
Before diving into details, let us specify the measure for depth overhead. Take a common
universal set of 1-qubit and 2-qubit gates [1] . Given an _n_ -qubit circuit _C_ with depth _d_ ( _C_ ) assuming
all-to-all qubit connectivity, we need to compile it to a circuit _C_ _[′]_ with depth _d_ ( _C_ _[′]_ ) that respects the
constraint graph _G_ . We use the ratio of _d_ ( _C_ _[′]_ ) _/d_ ( _C_ ) as the overhead measure for circuit instance
_C_, aiming to compile any _C_ with a small overhead. Formally, we study the following measure
doh (standing for _depth overhead_ )



doh( _G_ ) := max min
_C_ _C_ _[′]_ _∼C_ :
_G_ -compliant



_d_ ( _C_ _[′]_ )

(1)
_d_ ( _C_ ) _[.]_



Here the minimum is over all _G_ -compliant circuits _C_ _[′]_ equivalent to _C_ and obtained from _C_ by
inserting SWAP gates, and the maximum is over all _n_ -qubit circuits _C_ . Namely, we hope to find
a good compiling algorithm _C_ _→_ _C_ _[′]_ such that the depth increase ratio _d_ ( _C_ _[′]_ ) _/d_ ( _C_ ) is controlled
for any possible input circuit _C_ [2] .
While there are a few compilation algorithms working for a few specific graph constraints

[18–25], the studies fall short in two aspects. Firstly, the proposed routing algorithms for the
specific graphs were not always optimal. For instance, Ref. [29] proposed a QAOA circuit that is
hardware-compliant under a grid constraint. The method was to first find a long path in the grid
and then to use the known SWAP routing algorithm on the path. For a grid of size _[√]_ _n_ _×_ _[√]_ _n_, this
results in a _O_ ( _n_ ) depth overhead. Later we will show a superior routing algorithm with _O_ ( _[√]_ _n_ )
depth overhead, and show that it is optimal. Secondly, the compilation algorithms so far are _ad_
_hoc_ for different specific graphs, and no systematic studies on general graphs have been conducted.
This paper is the first to provide a unified, provably optimal algorithm for _all_ graphs.


Main results. In this work, we present a unified algorithm for qubit routing for any given constraint graph _G_, with the depth overhead fully characterized by the routing number rt( _G_ ), a wellstudied graph theoretic measure with a long history. We demonstrate that for all connected graphs
_G_,


doh( _G_ ) = Θ(rt( _G_ )) _._ (2)


1The 1-qubit gates are not restricted by the qubit connectivity. Common choices for the 2-qubit gates
include CNOT, CZ, iSWAP, etc. Note that a SWAP gate can be easily realized by three CNOT gates.


2Here we use ratio _d_ ( _C_ _′_ ) _/d_ ( _C_ ) rather than difference _d_ ( _C_ _′_ ) _−_ _d_ ( _C_ ) because _d_ ( _C_ _′_ ) increases linearly with
_d_ ( _C_ ) for a generic circuit _C_, while the difference _d_ ( _C_ _[′]_ ) _−_ _d_ ( _C_ ) can be arbitrarily large. (Though it is also
possible to consider multiple layers of _C_ together in compression, this semantic compression does not have
much advantage for a _generic_ depth- _d_ circuit _C_, as each layer can be arbitrary and different layers do
not admit a good compression. Even for the rare cases admitting significant semantic compression, the
computational complexity of finding such a good compression is high.)




---PAGE BREAK---

Specifically, on any graph _G_, we provide an algorithm to compile an arbitrarily given circuit _C_
(with no connectivity constraint) into another circuit _C_ _[′]_ with the depth increase ratio bounded by
_O_ (rt( _G_ )) from above. Furthermore, we show that this is the best possible outcome—one cannot
compile a generic _C_ with a depth increase factor asymptotically better than _O_ (rt( _G_ )).
As a graph theoretic measure that has been studied for about three decades, the routing number
rt( _G_ ) has been pinned down for many specific graphs _G_ such as paths, grids, trees, complete
bipartite graphs, hypercubes, etc. [30–35]. By combining our algorithm with these known routing
methods, we immediately obtain compilation algorithms for quantum circuits under these graph
constraints.
Additionally, we introduce a reduction algorithm between different graphs, enabling us to
derive efficient routing algorithms for new constraint graphs from existing algorithms on some
basic graphs. To demonstrate this, we construct compilers for IBM’s brick-wall graphs [5] and
Rigetti’s cycle-grid graphs [27] by reducing SWAP networks on those graphs to the ones on the
2D-grid.


Previous work. The result in Eq. (2) bears resemblance to the canonical work on lower bounding quantum circuit size complexity by the geodesic distance on the unitaries manifold [36, 37].
However, our work diverges in two significant ways: (1) we provide matching upper and lower
bounds, and (2) our approach is more operational as it presents an explicit algorithm to compile a
given arbitrary circuit _C_ to a _G_ -compliant circuit _C_ _[′]_ with depth _d_ ( _C_ _[′]_ ) = _O_ ( _d_ ( _C_ ) _·_ rt( _G_ )).
Ref. [38] explores the qubit connectivity constraint for general and special classes of unitaries,
such as those for quantum state preparation (QSP), by exploiting techniques in earlier work [39]
and carefully arranging qubits such that most two-qubit gates act on nearby qubits. The paper
shows that, somewhat surprisingly, the qubit connectivity constraint does not significantly increase
circuit complexity for these classes of unitaries either in the worst or a generic case. For example, it
gives a parametrized QSP circuit of depth _O_ (2 _[n]_ _/n_ ) to generate a given arbitrary _n_ -qubit quantum
state, while even circuits _without_ the qubit connectivity constraint require the depth of the same
order of depth [40]. Though this might suggest that the qubit connectivity constraint does not
increase circuit complexity, it is important to note that the worst-case or generic unitary matrices
have exponentially high complexity. Thus Ref. [38] merely demonstrates that the connectivity
constraint does not exacerbate these already complex cases. However, our primary concern in
practice lies with efficient (e.g. polynomial depth) circuits—After all, these are the ones to be
used in the future. Results in Ref. [38] do not provide insight into how the connectivity constraint
affects these _e_ ffi _cient_ circuits, and particularly do not rule out the possibility that shallow circuits
significantly suffer from the constraint. This work shows that, fortunately, this is not the case: If a
circuit _C_ with all-to-all connections has depth _d_, then it can always be compiled to a _G_ -compliant
circuit _C_ _[′]_ with depth at most _O_ ( _d·_ rt( _G_ )) = _O_ ( _dn_ ). In particular, if _d_ ( _C_ ) is a polynomial function
of _n_, so is the depth of _C_ _[′]_ .
Ref. [14] considered routing for partial permutations, in which one only needs to map _k_ _<_ _n_
vertices _xi_ to _k_ other vertices _yi_, and the rest _n −_ _k_ vertices can be mapped arbitrarily. When
the circuit depth is concerned with, the paper gave a reduction (“Greedy Depth Mapper”) from a
partial routing protocol to circuit compilation. Their algorithm is essentially the same as ours in
Lemma 2. However, this compilation can be very loose as explained after Lemma 2. They also
gave a number of partial routing protocols, which may complement our result: They gave efficient
routing methods, which may be utilized by our reduction to obtain efficient compilation methods
(though one should also be careful about the difference between partial and full permutation).


Organization. The rest of this paper is organized as follows. In Section 2, we introduce notations and review some previous results. In Section 3, we show the full characterization of the




---PAGE BREAK---

depth overhead. Then we demonstrate a reduction of routing numbers between different graphs
and construct routing algorithms for practice qubit connectivity constraints in Section 4. Finally,
we discuss our results in Section 5.

### 2 Preliminaries


Notation. Let [ _n_ ] := _{_ 1 _,_ 2 _, . . ., n}_ . Let _|S|_ denote the size of set _S_ . In this paper we study
general undirected graphs _G_ = ( _V, E_ ), where _V_ is the set of vertices and _E_ is the set of edges. We
denote the number of vertices by _n_, and sometimes identify the vertex set _V_ with the set [ _n_ ]. For a
vertex _u ∈_ _V_ and a subset _S_ _⊆_ _V_, the neighbor of _u_ inside _S_ is _NS_ ( _u_ ) = _{v_ _∈_ _S_ : ( _u, v_ ) _∈_ _E}_ .
We drop the subscript _S_ and just write _N_ ( _u_ ) if _S_ = _V_ . For a subset _S_ _⊆_ _V_, the _induced subgraph_
of _G_ on _S_ is
_G|S_ = ( _S, E_ _[′]_ ) with _E_ _[′]_ = _{_ ( _u, v_ ) _∈_ _E_ : _u ∈_ _S, v_ _∈_ _S}._ (3)


The _diameter_ of a graph _G_ is the largest distance of two vertices, denoted by


_diam_ ( _G_ ) := max (4)
_u,v∈V_ _[d]_ [(] _[u, v]_ [)] _[,]_


where _d_ ( _u, v_ ) is the distance of vertices _u_ and _v_ on graph _G_ (i.e. the number of edges on the
shortest path between _u_ and _v_ ). A _matching_ in an undirected graph _G_ = ( _V, E_ ) is a vertex-disjoint
subset of edges _M_ _⊆_ _E_ .


Quantum circuit compilation and depth overhead. The qubit connectivity constraint can be
modeled by a connected graph _G_ = ( _V, E_ ). A two-qubit gate can be applied to a pair of qubits
_i, j_ _∈_ _V_ if and only if ( _i, j_ ) _∈_ _E_ . We refer to _G_ as the _constraint graph_ and a circuit satisfying
the above constraint as a _G-compliant circuit_ . When _G_ is the complete graph _Kn_, we say that the
circuit has all-to-all connectivity.
The quantum circuit compilation problem studied in this paper is as follows: Given an _n_  qubit input quantum circuit _C_ consisting of 1- and 2-qubit gates with all-to-all connectivity, and a
constraint graph _G_ with _n_ vertices (identified with the _n_ qubits), construct a _G_ -compliant circuit
_C_ _[′]_ equivalent to _C_ by adding SWAP gates. Here two circuits are equivalent if they implement the
same unitary operation. To measure the quality of the hardware-compliant circuit, we introduce
the following concept of _depth overhead_ :


doh( _G, C_ ) := min _C_ _[′]_ _[d]_ [(] _[C][′]_ [)] _[/d]_ [(] _[C]_ [)] _[,]_ (5)


where _d_ ( _C_ ) and _d_ ( _C_ _[′]_ ) denote the depth of circuits _C_ and _C_ _[′]_, respectively, and the minimum is
over all _C_ _[′]_ that satisfy the above compilation requirement. The _depth_ _overhead_ of a constraint
graph _G_ is then defined as
doh( _G_ ) := max doh( _C, G_ ) _._ (6)
_C_


where the maximum is taken over all _n_ -qubit circuits _C_ with all-to-all qubit connectivity.


Permutations. The set of all permutations on set [ _n_ ] is denoted by _Sn_ . A permutation _π_ _∈_
_Sn_ is a _transposition_ if _π_ = ( _a_ 1 _, a_ 2)( _a_ 3 _, a_ 4) _· · ·_ ( _a_ 2 _k−_ 1 _, a_ 2 _k_ ) with distinct _a_ 1 _, a_ 2 _, . . ., a_ 2 _k_, where
( _a_ 2 _i−_ 1 _, a_ 2 _i_ ) means to exchange _a_ 2 _i−_ 1 and _a_ 2 _i_ . A permutation _π_ is a transposition if and only if it
satisfies _π_ [2] = _id_, the identity permutation. It is well known that any permutation can be written
as the composition of two transpositions.


Lemma 1 ( [41]). _Any_ _π_ _∈_ _Sn_ _can_ _be_ _decomposed_ _as_ _π_ = _σ_ 1 _◦_ _σ_ 2 _,_ _where_ _σ_ 1 _, σ_ 2 _∈_ _Sn_ _are_
_transpositions._




---PAGE BREAK---

Routing number rt( _G_ ). The routing number is defined by the following game [30]. Given a
connected graph _G_ = ( _V, E_ ) with _n_ vertices, we place one pebble at each vertex _v_ _∈_ _V_, and move
the pebbles in rounds. In each round _i_, we are allowed to select a matching _Mi_ _⊆_ _E_ and swap the
two pebbles at _u_ and _v_ for all edges ( _u, v_ ) _∈_ _Mi_ . For any permutation _π_ _∈_ _Sn_, rt( _G, π_ ) is the
_minimum_ number of rounds in which one can move all pebbles from their initial positions _v_ to the
destinations _π_ ( _v_ ). For any graph _G_, the routing number is defined as


rt( _G_ ) = max (7)
_π∈Sn_ [rt][(] _[G, π]_ [)] _[.]_

### 3 Full characterization of depth overhead


In this section, we present a protocol for constructing a _G_ -compliant circuit with the depth overhead of at most _O_ (rt( _G_ )) for any given circuit _C_ in Section 3.1. Then we demonstrate that for any
_G_, there exists a circuit _C_ with depth overhead at least Ω(rt( _G_ )) in Section 3.2, thereby offering a
complete characterization of the depth overhead of _G_ .
Computing the depth overhead for a given _C_ and _G_ turns out to be NP-hard; see Appendix A
for the proof. However, the asymptotic behavior of doh( _G_ ) can be fully characterized by a graph
measure called the _routing number_, denoted by rt( _G_ ), of a graph _G_, as defined in Eq. (7).


3.1 Hardware-compliant circuit construction and depth overhead upper bound


In this section, we first present a compiling algorithm by the maximum matching. Second, we give
a graph partition algorithm. Third, based on the graph partition algorithm, we present the general
compiling algorithm and the depth overhead upper bound.
Before diving into detailed constructions, let us first compare the measures doh( _G_ ) and rt( _G_ ),
and explain why the upper bound doh( _G_ ) = _O_ (rt( _G_ )) does not immediately follow from their
definitions, despite the apparent similarities. For easier comparison, let us formulate doh( _G_ ) in a
language of games similar to that for the routing number rt( _G_ ) (Eq. (7)): we compile circuit _C_
layer by layer, and for each layer, suppose the two-qubit gates are on pairs ( _u_ 1 _, v_ 1), _. . ._, ( _uk, vk_ ),
then we need to use SWAP gates to move _ui_ and _vi_ next to each other, apply the gate, and move
them back. This formulation highlights immediate similarities between doh( _G_ ) and rt( _G_ ): (1)
Both can be viewed as games in rounds, with each round consisting of SWAP operations. (2) Both
measures represent the minimum number of rounds.
However, also note that there are some _key di_ ff _erences_ between the two measures: (1) In circuit
compilation, _ui_ and _vi_ need to be moved to _next_ to each other, while in graph routing, _i_ needs to
be moved to _π_ ( _i_ ). Note that given a routing algorithm to move each _i_ to _π_ ( _i_ ) for any permutation
_π_, it is still hard to move _ui_ and _vi_ next to each other because we cannot simply find a neighbor _vi_ _[′]_
of _vi_ and let _π_ ( _ui_ ) = _vi_ _[′]_ [; for example, if all] _[ v][i]_ [’s have degree][ 1][ and are all connected to a common]
“port” node _v_ to reach the rest of the graph, then all _vi_ _[′]_ [equal to] _[ v]_ [, making the map] _[ π]_ [(] _[u][i]_ [) =] _[ v]_ [ not]
a permutation. We will need to handle this type of bottleneck issue in designing the protocol for
doh( _G_ ). (2) In circuit compilation, it suffices that _ui_ and _vi_ are next to each other at _some_ time
step _ti_ _≤_ doh( _G_ ) (different pairs ( _ui, vi_ ) may have different _ti_ ), while in graph routing, all _i_ need
to be moved to _π_ ( _i_ ) exactly at time rt( _G, π_ ). This gives us some freedom to design the doh( _G_ )
protocol.
One basic property that will be used later is a linear upper bound of rt( _G_ ) [30]:


rt( _G_ ) _≤_ 3 _n._ (8)


A routing protocol induces a SWAP circuit in a natural way: For any permutation _π_ on vertices
in graph _G_ and any routing protocol in the definition of rt( _G, π_ ), if two pebbles at two vertices _i_




---PAGE BREAK---

and _j_ are swapped, then we apply a swap operation on qubits _i_ and _j_ in the SWAP circuit. Then
the following unitary transformation _Uπ_


_|x_ 1 _· · · xn⟩_ _−−→|_ _[U][π]_ _xπ_ (1) _· · · xπ_ ( _n_ ) _⟩,_ _∀xi_ _∈{_ 0 _,_ 1 _}, i ∈_ [ _n_ ] _,_ (9)


can be realized by a rt( _G, π_ )-depth circuit consisting of swap operations.
Before we give the general compiling algorithm, we first give a lemma which can compile
circuits for graphs with a large matching.


Lemma 2. _For any connected graph G with the maximum matching size ν, we can construct_
_G-compliant_ _circuits_ _with_ _the_ _depth_ _overhead_ _at_ _most_ _O_ (rt( _G_ ) _· n/ν_ ) _._ _That_ _is,_ doh( _G_ ) =
_O_ (rt( _G_ ) _· n/ν_ ) _._


_Proof._ Fix a given _n_ -qubit circuit _C_ . For each layer _Ci_ with at least one two-qubit gate,
suppose that the 2-qubit gates are _Ci_ 1 _, . . ., Cik_ on pairs _{_ ( _u_ 1 _, v_ 1) _, . . .,_ ( _uk, vk_ ) _}_ of qubits,
respectively, where 1 _≤_ _k_ _≤_ _n/_ 2 and the 2 _k_ vertices _u_ 1 _, v_ 1 _,_ _. . ._ _, uk, vk_ are all distinct. Let
_{_ ( _x_ 1 _, y_ 1) _,_ _. . ._ _,_ ( _xν, yν_ ) _}_ be a maximum matching of _G_ . We can compile this layer of _Ci_ to
a _G_ -compliant circuit in depth _O_ (rt( _G_ )) as follows:


1. Apply all single-qubit gates in _Ci_ .


2. Pick any permutation _π_ that permutes _uj_ to _xj_ and _vj_ to _yj_, for all _j_ _∈_ [ _ν_ ]. Run the
circuit _Uπ_ (in Eq. (9)) of depth at most rt( _G_ ).


3. Apply the 2-qubit gates _Cij_ on ( _xj, yj_ ), for all _j_ _∈_ [ _ν_ ];


4. Run _Uπ_ _[†]_ [,] [the] [reverse] [process] [of] [Step] [2][.]


If _k_ _≤_ _ν_, then this implements _Ci_ already in depth 2 _·_ rt( _G_ ) + 2. If _k_ _>_ _ν_, then this
implements the first _ν_ two-qubit gates among _k_ ones. Repeat the last three steps in the
above procedure until all 2-qubit gates are handled, which needs _⌈k/ν⌉_ iterations. Each
iteration needs at most 2 _·_ rt( _G_ )+1 depth, thus the overall depth overhead is 1+(2rt( _G_ )+
1) _· ⌈k/ν⌉_ = _O_ (rt( _G_ ) _· n/ν_ ). 

Note that Lemma 2 can already give compiling algorithms with depth overhead at most _O_ (rt( _G_ ))
for some specific graphs _G_, including 1D-chain, 2D-grid, IBM’s brick wall or Rigetti’s bilinear cycle, binary tree, etc, all of which have a matching of size Θ( _n_ ). But for graphs with a small matching size (an extreme example is the star graph which has the maximum matching size _ν_ = 1), the
bound of _O_ (rt( _G_ ) _· n/ν_ ) is very loose.
Lemma 2 has a clear intuition that the existence of a large matching facilitates moving the
pebble around. Actually, it is even tempting to conjecture that this dependence of _O_ ( _n/ν_ ) is
inevitable since a bottleneck in a graph does make simultaneous pebble moving inefficient due to
traffic congestion. However, this bottleneck also affects rt( _G_ ) protocols and should be inherently
characterized in the rt( _G_ ) measure. What we need to do is to technically relate the difficulty in
the two measures and construct a reduction from one to the other. Next, we give details on how to
remove the _O_ ( _n/ν_ ) factor in Lemma 2.
To improve it to the optimal bound of _O_ (rt( _G_ )), one idea is to partition the constraint graph
into vertex-disjoint connected subgraphs, each having _O_ (1) diameter and containing at most
_O_ (rt( _G_ )) vertices. We aim to move each pair of qubits ( _ui, vi_ ) on which a two-qubit gate acts
to one of these subgraphs (different pairs may go to different subgraphs). If this can be achieved,
then we can implement two-qubit gates within one subgraph efficiently. Indeed, since each subgraph has diameter _O_ (1), it takes _O_ (1) SWAP gates to implement one gate and since the subgraph
has size rt( _G_ ), all the gates inside this subgraph can be done by _O_ (rt( _G_ )) SWAP gates, which




---PAGE BREAK---

takes at most _O_ (rt( _G_ )) rounds. Also note that the routings in different subgraphs can be carried
out in parallel. Thus the overall overhead is _O_ (rt( _G_ )).
The challenge is that it is not always possible to achieve such a good partition of the constraint
graph. We present a good graph partition algorithm in Lemma 4, for which we will first show the
following bottleneck lemma.


Lemma 3 (bottleneck). _For_ _a_ _connected_ _graph_ _G_ = ( _V, E_ ) _,_ _suppose_ _that_ _there_ _exist_ _vertex_
_sets_ _V_ 1 _, V_ 2 _⊆_ _V_ _such_ _that_


_1._ _V_ 1 _∩_ _V_ 2 = _∅;_


_2._ _for_ _any_ _u ∈_ _V_ 1 _,_ _N_ ( _u_ ) _⊆_ _V_ 2 _,_ _where_ _N_ ( _u_ ) := _{w_ _∈_ _V_ : ( _w, u_ ) _∈_ _E}._


_Then_ _the_ _routing_ _number_ _of_ _G_ _satisfies_







rt( _G_ ) = Ω




- _|V_ 1 _|_
_|V_ 2 _|_



_._ (10)



_Proof._ Let us label the pebbles in _V_ 1 as 1 _,_ 2 _, . . ., |V_ 1 _|_ . Let _s_ := _⌊|V_ 1 _|/_ 2 _⌋_ . Define a permutation _π_ := (1 _,_ 1 + _s_ )(2 _,_ 2 + _s_ ) _· · ·_ ( _s,_ 2 _s_ ). Note that if we move the pebble at vertex _i_ to
the vertex _i_ + _s_, the pebble must go through _V_ 2 since _N_ ( _i_ ) _⊆_ _V_ 2 by assumption. Since at
most _|V_ 2 _|_ pebbles in _V_ 1 can go through _V_ 2 in one round, moving _s_ pebbles needs at least
_s/|V_ 2 _|_ rounds. Therefore,


rt( _G_ ) _≥_ rt( _G, π_ ) _≥_ _s/|V_ 2 _|_ = Ω( _|V_ 1 _|/|V_ 2 _|_ ) _._


                              

Next, we present the graph partition algorithm in Lemma 4, which outputs two families of
vertex sets _W_ and _W_ _[′]_ .


Lemma 4. _There_ _exists_ _an_ _algorithm_ _which,_ _on_ _any_ _n-vertex_ _connected_ _graph_ _G_ = ( _V, E_ ) _,_
_outputs_ _two_ _families_ _of_ _vertex_ _sets_


_W_ = _{W_ 1 _,_ _. . .,_ _Ws_ : _Wi_ _⊆_ _V,_ _∀i ∈_ [ _s_ ] _},_ (11)

_W_ _[′]_ = _{W_ 1 _[′][,]_ _[. . .,]_ _[W]_ _t_ _[ ′]_ [:] _[ W][ ′]_ _j_ _[⊆]_ _[V,]_ _[∀][j]_ _[∈]_ [[] _[t]_ []] _[}][,]_ (12)


_for_ _some_ _s, t ∈_ [ _n_ ] _,_ _satisfying_ _the_ _following_ _properties._


_1._ _(disjointness)_ _For_ _any_ _distinct_ _i, i_ _[′]_ _∈_ [ _s_ ] _and_ _distinct_ _j, j_ _[′]_ _∈_ [ _t_ ] _,_ _we_ _have_ _Wi ∩_ _Wi′_ = _∅_
_and_ _Wj_ _[′]_ _[∩]_ _[W][ ′]_ _j_ _[′]_ [=] _[ ∅][;]_

_2._ _(coverage)_ _|_ [�] _i∈_ [ _s_ ] _[W][i][|]_ [ +] _[ |]_ [ �] _j∈_ [ _t_ ] _[W][ ′]_ _j_ _[| ≥]_ _[n/]_ [2] _[;]_


_3._ _(size_ _bound)_ _For_ _any_ _i_ _∈_ [ _s_ ] _and_ _j_ _∈_ [ _t_ ] _,_ _we_ _have_ 2 _≤|Wi|_ = _O_ (rt( _G_ )) _and_ 2 _≤_
_|Wj_ _[′][|]_ [ =] _[ O]_ [(][rt][(] _[G]_ [))] _[;]_


_4._ _(diameter_ _bound)_ _For_ _all_ _i_ _∈_ [ _s_ ] _and_ _j_ _∈_ [ _t_ ] _,_ _the_ _induced_ _subgraphs_ _Gi_ = _G|Wi_ _and_
_G_ _[′]_ _j_ [=] _[ G][|]_ _Wj_ _[′]_ _[all]_ _[have]_ _[diameter]_ _[at]_ _[most]_ [2] _[.]_


_The_ _algorithm_ _runs_ _in_ _time_ _O_ ( _n_ [3] ) _._




---PAGE BREAK---

_Proof._ The family _W_ is constructed as follows. First find a maximal matching


_M_ = _{_ ( _w_ 1 _, w_ 2) _,_ ( _w_ 3 _, w_ 4) _,_ _. . .,_ ( _w_ 2 _s−_ 1 _, w_ 2 _s_ ) _}_ (13)


of _G_ . Put
_W_ = _{Wi_ : _i ∈_ [ _s_ ] _},_ where _Wi_ = _{w_ 2 _i−_ 1 _, w_ 2 _i} ⊆_ _V,_ _∀i ∈_ [ _s_ ] _._ (14)


Then _W_ satisfies the properties 1, 4 and 3. Indeed, each _G|Wi_ is essentially an edge and
thus the two nodes are connected, different _G|Wi_ ’s are disjoint as _M_ is a matching, and
_|Wi|_ = 2 = _O_ (rt( _G_ )).
The family _W_ _[′]_ is constructed as follows. Define vertex set _T_ = _V_ _−∪i∈_ [ _s_ ] _Wi_, the
vertices not in the maximal matching _M_ . Note that _T_ is an independent set, i.e. any two
vertices _a, b_ _∈_ _T_ are not connected; otherwise, we could have added the edge ( _a, b_ ) in _M_
to form a larger matching, contradicting _M_ being maximal. Define set


            -             -             _S_ := _w_ _∈_ _Wi_ : _NT_ ( _w_ ) � _∅_ (15)

_i∈_ [ _s_ ]


to contain those vertices with a connection to _T_ . The family _W_ _[′]_ is constructed by Algorithm 1.


Algorithm 1: Construction of _W_ _[′]_


1 Input: Connected graph _G_ = ( _V, E_ ), vertex sets _T, S_ _⊆_ _V_ .

2 Output: A family _W_ _[′]_ of sets.

1: Initialize _Nw_ [0] [:=] _[ ∅]_ [,] _[∀][w]_ _[∈]_ _[S]_ [.]

2: Initialize _A_ 0 _,p_ := _∅_, _∀p ∈_ [ _|T_ _|_ ].

3: _T_ 1 := _T_, _S_ 1 := _S_, _k_ 1 := _|S_ 1 _|_ .

4: for _p_ = 1 to _|T_ _|_ do

5: for _i_ = 1 to _|S|_ do

6: _Ai,p_ := _∅_ .

7: if there are at least 1 and at most _⌈|Tp|/kp⌉_ neighbors of _wi_ in _Tp −_ [�] _[i]_ _r_ _[−]_ =1 [1] _[A][r,p]_
then



8: Let _Ai,p_ contain all these neighbors.



9: else if there are more than _⌈|Tp|/kp⌉_ neighbors of _wi_ in _Tp −_ [�] _[i]_ _r_ _[−]_ =1 [1] _[A][r,p]_ [then]



10: Let _Ai,p_ contain arbitrary _⌈|Tp|/kp⌉_ many of these neighbors.



11: end if



12: Let _Nw_ _[p]_ _i_ [:=] _[ N]_ _w_ _[p][−]_ _i_ [1] _∪_ _Ai,p_ .



13: end for



14: if _|_ [�] _i_ _[|][S]_ =1 _[|]_ _[N]_ _w_ _[p]_ _i_ _[|≥|][T]_ _[|][/]_ [2] [then]



15: return _W_ _[′]_ := _{Nw_ _[p]_ _i_ _[∪{][w][i][}]_ [ :] _[ |][N]_ _w_ _[p]_ _i_ _[| ≥]_ [1] _[,]_ _[i][ ∈]_ [[] _[|][S][|]_ []] _[}]_ [and] [end] [the] [whole] [program.]



16: end if



17: Set _Tp_ +1 := _Tp\_ [�] _r_ _[|][S]_ =1 _[|]_ _[A][r,p]_ [.]



18: Set _Sp_ +1 := _{wi_ : _|Ai,p|_ = _⌈|Tp|/kp⌉, i ∈_ [ _|S|_ ] _}_ .



19: Set _kp_ +1 := _|Sp_ +1 _|_ .

20: end for


In the algorithm, _Tp_, _Ai,p_, _Sp_, _kp_ and _Nw_ _[p]_ _i_ [are] [defined] [as] [follows.] [The] [set] _[T][p]_ [contains]
those vertices in _T_ not selected in the first _p−_ 1 iterations. In the _p_ -th iteration, the set _Ai,p_
denotes the neighbor set within _Tp −_ [�] _[i]_ _r_ _[−]_ =1 [1] _[A][r,p]_ [of] [vertex] _[w][i]_ _[∈]_ _[S]_ [with] [cardinality] [bounded]
by _⌈|Tp|/kp⌉_ . The set _Sp_ contains all vertices _wi_ _∈_ _S_ that have exactly _⌈|Tp−_ 1 _|/kp−_ 1 _⌉_
neighbors being chosen in the ( _p −_ 1)-th iteration. The number _kp_ denotes the size of




---PAGE BREAK---

