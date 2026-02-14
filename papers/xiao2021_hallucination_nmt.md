**ANOSOV** **ENDOMORPHISMS** **ON** **THE** **2-TORUS:**

**REGULARITY** **OF** **FOLIATIONS** **AND** **RIGIDITY**


MARISA CANTARINO AND REGIS [´] VARAO [˜]


Abstract. We provide sufficient conditions for smooth conjugacy between two
Anosov endomorphisms on the 2-torus. From that, we also explore how the regularity of the stable and unstable foliations implies smooth conjugacy inside a class
of endomorphisms including, for instance, the ones with constant Jacobian. As
a consequence, we have in this class a characterization of smooth conjugacy between special Anosov endomorphisms (defined as those having only one unstable
direction for each point) and their linearizations.


1. Introduction


In this work we study rigidity results for Anosov endomorphisms: hyperbolic maps
that are not necessarily invertible. The term _rigidity_ is associated with the idea that
the value of an invariant or a specific property of the system determines its dynamics.
In our case, we want to determine the smooth conjugacy class of the system, and the
invariant will be given by its Lyapunov exponents.
Let us recall briefly the context for the invertible case. Given a closed manifold
_M_, for a _C_ [1] Anosov diffeomorphism _f_ : _M_ _→_ _M_, any nearby _C_ [1] diffeomorphism _g_ is
topologically conjugate to _f_, that is, there exists _h_ : _M_ _→_ _M_ such that _g ◦_ _h_ = _h ◦_ _f_ .
This conjugacy _h_ is H¨older continuous, but if _h_ is _C_ [1] and _x_ a periodic point such
that _f_ _[p]_ ( _x_ ) = _x_, then

_D_ ( _h ◦_ _f_ _[p]_ ) _x_ = _D_ ( _g_ _[p]_ _◦_ _h_ ) _x_

= _⇒_ _Dfx_ _[p]_ [= (] _[Dh][x]_ [)] _[−]_ [1] _[ ◦]_ _[Dg]_ _h_ _[p]_ ( _x_ ) _[◦]_ _[Dh][x][.]_

Therefore, if the conjugacy between _f_ and _g_ is smooth, then the matrices _Dfx_ _[p]_ _[Dg]_ _h_ _[p]_ ( _x_ )
are conjugate. The conjugacy of these matrices is a necessary condition for the smooth
conjugacy, and it is natural to ask whether it is a sufficient condition. It turns out that
this condition on the periodic points is the main property of the sufficient condition,
as can be seen from the results obtained in the invertible setting [8, 13, 14, 15].
By considering maps that do not have an inverse, we have a few interesting behaviors. In contrast to Anosov diffeomorphisms, Anosov endomorphisms are not
structurally stable in general. Feliks Przytycki [29] used the fact that we can perturb


2020 _Mathematics_ _Subject_ _Classification._ Primary: 37C15; Secondary: 37D20.
_Key_ _words_ _and_ _phrases._ Smooth dynamics, Hyperbolic dynamics, Non-invertible dynamics.
M. C. was partially financed by the Coordena¸c˜ao de Aperfei¸coamento de Pessoal de N´ıvel Superior

- Brasil (CAPES) - grant 88882.333632/2019-01 and the Funda¸c˜ao Carlos Chagas Filho de Amparo
`a Pesquisa of the State of Rio de Janeiro (FAPERJ) E-26/202.014/2022.
R. V. was partially financed by CNPq and Fapesp grants 18/13481-0 and 17/06463-3.
This is an author-created, un-copyedited version of an article accepted for publication in
Nonlinearity. The publisher is not responsible for any errors or omissions in this version of
the manuscript or any version derived from it. The Version of Record is available online at
`[https://doi.org/10.1088/1361-6544/acf267](https://doi.org/10.1088/1361-6544/acf267)` .


1




---PAGE BREAK---

2 MARISA CANTARINO AND REGIS [´] VARAO [˜]


a linear Anosov endomorphism to create many different unstable directions to show
that a perturbation of an Anosov endomorphism may not be conjugated to it, since
a conjugacy should preserve stable and unstable manifolds. Additionally, he proved
that Anosov endomorphisms are structurally stable if and only if they are invertible
or expanding maps.
We say that an Anosov endomorphism is _special_ if each point has only one unstable
direction. Although an Anosov endomorphism may not be structurally stable, it is
conjugated to its linearization if and only if it is special [3, 32, 28].
For hyperbolic maps on surfaces, the conjugacy between _Dfx_ _[p]_ [and] _[Dg]_ _h_ _[p]_ ( _x_ ) [for] [cor-]
responding periodic points _x_ and _h_ ( _x_ ) is equivalent to _f_ and _g_ having the same Lyapunov exponents _λ_ _[u/s]_ _f_ ( _x_ ) = _λ_ _[u/s]_ _g_ ( _h_ ( _x_ )) on these points. We prove that this condition
is indeed sufficient to guarantee smooth conjugacy.


**Theorem** **A.** _Let_ _f, g_ : T [2] _→_ T [2] _be_ _C_ _[k]_ _,_ _k_ _≥_ 2 _,_ _Anosov_ _endomorphisms_ _topologically_
_conjugated_ _by_ _h_ : T [2] _→_ T [2] _homotopic_ _to_ _Id._ _If_ _the_ _corresponding_ _periodic_ _points_ _of_ _f_
_and_ _g_ _have_ _the_ _same_ _Lyapunov_ _exponents,_ _then_ _the_ _conjugacy_ _h_ _is_ _C_ _[k]_ _._ _In_ _particular,_
_if_ _f_ _and_ _g_ _are_ _C_ _[∞]_ _,_ _then_ _h_ _is_ _also_ _C_ _[∞]_ _._


This kind of result was addressed for the invertible case on T [2] by Rafael de la Llave,
Jos´e Manuel Marco and Roberto Moriy´on in a serie of works [23, 7, 24, 9, 8] and for
T [3] by Andrey Gogolev and Misha Guysinsky [13, 12], and it has counterexamples in
higher dimensions [8], in which more hypotheses are required [14], [15].
For the non-invertible setting, the above theorem extends some of the results obtained independently by Fernando Micena [25, Theorem A, Theorem B]. On [25,
Theorem A], _f_ and _g_ need not to be special, as in our case, but he requires the topological conjugacy to preserve SRB measures instead of the condition on Lyapunov
exponents of periodic points. The core of the proof is [25, Lemma 4.2], where he
obtains a candidate to smooth conjugacy as the limit of solutions of differential equations over local unstable manifolds. On [25, Theorem B], he requires, in addition to
the condition on Lyapunov exponents of periodic points, that _f_ is strongly special,
meaning that each point has only one unstable direction and that the stable manifold
is dense. Then he uses a conformal metric on unstable manifolds to, again, solve
some differential equations and find a candidate for _C_ [1] conjugacy. Since it is _C_ [1], it
preserves SRB measures, and then he applies [25, Theorem A]. Additionaly, after the
submission of our work, in a recent preprint, Ruihao Gu and Yi Shi prove a similar
result in [16, Theorem 1.3].
Similarly to the approaches for the invertible setting, we apply a regularity lemma
by Jean-Lin Journ´e [20] to “spread” the regularity of _h_ along transversal foliations to
the same regularity in a whole neighborhood of a point. However, we have to do so
locally, since _h_ is not invertible. Additionally, there are several subtleties along the
steps of the proof, such as fixing inverses locally in a well-defined way.
Recently, Jinpeng An, Shaobo Gan, Ruihao Gu and Yi Shi [1] proved that an noninvertible _C_ [1+] _[α]_ Anosov endomorphism on T [2] is special if and only if every periodic
point for _f_ has the same stable Lyapunov exponent. In particular, _λ_ _[s]_ _f_ [(] _[x]_ [)] _[≡]_ _[λ][s]_ _A_ [for]
every _x_ _∈_ T [2] periodic for _f_, where _A_ is the linearization of _f_ . Then, if _f_ is special,
we have the following corollary of Theorem A.


**Corollary** **1.** _Let_ _f_ : T [2] _→_ T [2] _be_ _a_ _C_ _[k]_ _,_ _k_ _≥_ 2 _,_ _special_ _Anosov_ _endomorphism_ _and_ _A_
_its_ _linearization._ _If_ _λ_ _[u]_ _f_ [(] _[x]_ [)] _[≡]_ _[λ][u]_ _A_ _[for]_ _[every]_ _[x]_ _[∈]_ [T][2] _[periodic]_ _[for]_ _[f]_ _[,]_ _[then]_ _[the]_ _[conjugacy]_
_h_ _between_ _f_ _and_ _A_ _is_ _C_ _[k]_ _._ _In_ _particular,_ _if_ _f_ _is_ _C_ _[∞]_ _,_ _then_ _h_ _is_ _also_ _C_ _[∞]_ _._




---PAGE BREAK---

RIGIDITY FOR ANOSOV ENDOMORPHISMS ON T2 3


It remains to provide examples of non-special Anosov endomorphisms on T [2] that
are conjugated but with the conjugacy not being _C_ [1], giving then conditions to apply
Theorem A in its full generality. If _f_ and _g_ are conjugated, by lifting the conjugacy
to the inverse limit space, _f_ and _g_ are inverse-limit conjugated. Then, a necessary
condition is that _f_ and _g_ have the same linearization, since by Nobuo Aoki and
Koichi Hiraide [3, Theorem 6.8.1] _f_ and its linearization are inverse-limit conjugated.
Besides, the conjugacy must necessarily be a homeomorphism between _Wf_ _[u]_ [(˜] _[x]_ [)] [and]
_Wg_ _[u]_ [(˜] _[h]_ [(˜] _[x]_ [))] [for] [each] [of] [the] [unstable] [directions.] [Fernando] [Micena] [and] [Ali] [Tahzibi] [[][27][]]
proved that, if _f_ is not special, there is a residual subset _R_ _∈_ T [2] such that every
_x_ _∈R_ has infinitely many unstable directions. This suggests the complexity of this
problem.
**Question:** Under which conditions do we have a topological conjugacy between
two non-special Anosov endomorphisms on T _[n]_ with the same linearization?
An answer to this question for T [2] is given in [16, Theorem 1.1], in which they prove
that, if _f_ and _g_ are non-invertible and homotopic, then _f_ is topologically conjugate
to _g_ if, and only if, the corresponding periodic points of _f_ and _g_ have the same stable
Lyapunov exponents. Then Theorem _A_ can be reformulated as follows.


**Corollary** **2.** _Let_ _f, g_ : T [2] _→_ T [2] _be_ _C_ _[k]_ _,_ _k_ _≥_ 2 _,_ _homotopic_ _Anosov_ _endomorphisms._
_If_ _the_ _corresponding_ _periodic_ _points_ _of_ _f_ _and_ _g_ _have_ _the_ _same_ _Lyapunov_ _exponents,_
_then_ _they_ _are_ _conjugated_ _and_ _the_ _conjugacy_ _h_ _is_ _C_ _[k]_ _._ _In_ _particular,_ _if_ _f_ _and_ _g_ _are_
_C_ _[∞]_ _,_ _then_ _h_ _is_ _also_ _C_ _[∞]_ _._


To provide a context in which Theorem A can be applied, our approach is towards
the regularity of the stable and unstable directions. This is inspired by a conjecture
for Anosov diffeomorphisms on compact manifolds, which states that _C_ [2] regularity
of the stable and unstable foliations of an Anosov diffeomorphism would imply the
same regularity for the conjugacy with the Anosov automorphism homotopic to it

[10, 35]. A similar result for _C_ [1+] _[β]_ stable and unstable foliations cannot hold, since in
surfaces the stable and unstable manifolds are _C_ [1+] _[β]_ but the conjugacy is generally
not better than H¨older continuous.
Another inspiration is given by the fact that the absolute continuity of stable and
unstable foliations is a central property used to prove the ergodicity of conservative
Anosov diffeomorphisms [2]. That is, a regularity condition on the foliation implies
a very specific behavior of the map.
We then consider the following context: what properties may we require on the
stable or unstable foliations of a special Anosov endomorphism to get regularity for
the conjugacy map?
Instead of absolute continuity, we work with a uniform version of absolute continuity - called the _UBD_ _(uniform_ _bounded_ _density)_ property, as defined by F. Micena
and A. Tahzibi [26], see Definition 2.9. In [34] it is shown that a smooth conservative
partially hyperbolic diffeomorphism on T [3] is smoothly conjugate to its linearization
if and only if the center foliation has the UBD property. This can be seen as a sharp
result, since it is given as an example on [33] a conservative partially hyperbolic diffeomorphism on T [3] such that the center foliation is _C_ [1] but the conjugacy map is
not _C_ [1] . Hence, a uniformity condition in the densities is the natural candidate for
rigidity results. Since the center foliation does not exist in our context, we work with
a regularity condition for the unstable foliation. In a work in preparation, Marisa
Cantarino, Sime˜ao Targino da Silva and R´egis Var˜ao [6] prove that the UBD property
is equivalent do the holonomies having uniformly bounded Jacobians.




---PAGE BREAK---

4 MARISA CANTARINO AND REGIS [´] VARAO [˜]


For the sake of simplicity, we state the results for endomorphisms with constant
Jacobian as a particular case, and we treat the more general context on Section 4,
for a class of maps that satisfy a condition analogue to conservativeness.


**Theorem B.** _Let f_ : T [2] _→_ T [2] _be a C_ [2] _Anosov endomorphism with constant Jacobian_
_and_ _A_ _its_ _linearization._ _If_ _the_ _unstable_ _foliation_ _of_ _f_ _is_ _absolutely_ _continuous_ _with_
_uniformly_ _bounded_ _densities,_ _then_ _λ_ _[σ]_ _f_ _[≡]_ _[λ]_ _A_ _[σ]_ _[for]_ _[σ]_ _[∈{][u, s][}][.]_


In the above result _f_ is not required to be special. Note that, in general, _f_ does not
have a global unstable foliation, since each point may have more than one unstable
leaf, so when we say “the unstable foliation of _f_ has the UBD property”, we are
actually looking at a foliation on the universal cover of T [2] (see Definition 2.9).
We introduce a more general condition than constant Jacobian for _f_, that we call
_quasi_ _preservation_ _of_ _densities_ (see Definition 4.4). It means that the conditional
measures are “controlled” under iterations of _f_, which is automatic if the Jacobian
is constant (see Lemma 4.1). In Subsection 4.3, we see that Theorem B is a special
case of the following.


**Theorem** **C.** _Let_ _f_ : T [2] _→_ T [2] _be_ _a_ _C_ _[∞]_ _Anosov_ _endomorphism_ _with_ _quasi_ _preserva-_
_tion_ _of_ _densities_ _along_ _its_ _invariant_ _foliations,_ _and_ _A_ _its_ _linearization._ _If_ _the_ _stable_
_and_ _unstable_ _foliations_ _of_ _f_ _are_ _absolutely_ _continuous_ _with_ _uniformly_ _bounded_ _densi-_
_ties,_ _then_ _λ_ _[σ]_ _f_ _[≡]_ _[λ]_ _A_ _[σ]_ _[for]_ _[σ]_ _[∈{][u, s][}][.]_


By joining Theorems A and B for _f_ special and _g_ = _A_, we have that our regularity
condition on the unstable foliation implies equality of Lyapunov exponents, which
implies that _h_ is as regular as _f_ . Conversely, if _h_ is _C_ _[∞]_, the unstable leaves of _F_
are taken to unstable lines of _A_ by a lift _H_ of _h_, which implies the UBD property of
the unstable foliation of _F_ . The same holds for the stable foliation, and Corollary 3
follows.


**Corollary** **3.** _Let_ _f_ : T [2] _→_ T [2] _be_ _a_ _C_ _[∞]_ _special_ _Anosov_ _endomorphism_ _with_ _constant_
_Jacobian_ _and_ _let_ _A_ _be_ _its_ _linearization._ _The_ _unstable_ _foliation_ _of_ _f_ _is_ _absolutely_
_continuous_ _with_ _uniformly_ _bounded_ _densities_ _if_ _and_ _only_ _if_ _f_ _is_ _C_ _[∞]_ _conjugate_ _to_ _A._


Corollary 3 is the natural formulation of a result similar to Theorem 1.1 from [34]
for Anosov endomorphisms on T [2] . Indeed, working with preservation of volume for
maps that are not invertible is more subtle: a conservative endomorphism may not
have constant Jacobian. Instead, we ask for both foliations to have the UBD property
and to have their induced volume densities _quasi_ _preserved_ _under_ _f_ (see Definition
4.4), with the case with _f_ having constant Jacobian as a particular case. They are
necessary conditions in Corollary 3: if an Anosov endomorphism _f_ is _C_ [1] conjugated
to its linearization, then the unstable and stable foliations have the UBD property
and present quasi preservation of densities with respect to _f_, as we see in Section 4.


1.1. **Structure** **of** **the** **paper** **and** **comments** **on** **the** **results.** In Section 2 we
introduce formally some of the aforementioned notions, as well as some properties
necessary for the proofs. Regarding Theorem A, that we prove in Section 3, we want
to apply Journ´e’s regularity lemma [20] to take the regularity of the conjugacy on
transverse foliations to the whole manifold. To avoid complications from the fact that
_f_ and _g_ are not invertible, we will apply this lemma locally on the universal cover,
and this requires some adaptations. We proceed (as done in [13] for the weak unstable direction) by constructing, for the unstable direction, a function _ρ_ and using it to




---PAGE BREAK---

RIGIDITY FOR ANOSOV ENDOMORPHISMS ON T2 5


obtain the regularity of _h_ along unstable leaves. For stable leaves, an analogous argument holds. Even knowing that there are SRB measures for Anosov endomorphisms

[31], their definition involves the inverse limit space, and to avoid it our definition of
_ρ_ is adapted to a local foliated box projected from the universal cover. The function
_ρ_ is, indeed, proportional to the density of the conditional of the SRB measure on
the unstable leaves where it is well defined.
The proof of Theorem A relies on the low dimension of T [2] to construct metrics on
each unstable leaf that “behave regularly” under _f_ . We use these metrics to prove
that the conjugacy is uniformly Lipschitz restricted to the leaf in Lemma 3.3. After
that, we prove that the conjugacy restricted to each leaf is in fact _C_ [1] . Additionally,
we promote this _C_ [1] regularity to the regularity of _f_ and _g_ with the argument that
the densities of the conditionals of the SRB measure on each unstable leaf are _C_ _[k][−]_ [1],
so we are using the fact that the unstable foliation is one dimensional to claim that
our densities _ρ_ are _C_ _[k][−]_ [1] . Since the argument for the stable foliation is analogous, it
cannot be easily transposed to higher dimensions. Besides, a similar theorem for T _[n]_,
_n ≥_ 4, would need additional hypothesis, as it includes the invertible case [8].
For the proof of Theorem B in Section 4, we need the stable and unstable distributions of a lift of an Anosov endomorphism to be _C_ [1] . For this, we see in Proposition
3 that a codimension one unstable distribution for a lift of an Anosov endomorphism
to the universal cover is _C_ [1], again making use of the fact that we are working on T [2] .
The strategy to prove Theorem B is to construct, along unstable leaf segments of the
lift _F_, invariant measures uniformly equivalent to the length that grow at the same
rate that the unstable leaves of _A_, then the unstable Lyapunov exponents of both _f_
and _A_ are the same. These measures are defined for leaves of a full volume set, and
we use the regularity of stable holonomies - provided by the fact that the stable
distribution is _C_ [1] - to construct similar measures at each point and conclude that
_λ_ _[u]_ _f_ _[≡]_ _[λ]_ _A_ _[u]_ [.]


2. Preliminary concepts


Along this section, we work with concepts and results needed for the proofs on
sections 3 and 4.


2.1. **Inverse** **limit** **space.** For certain dynamical aspects, such as unstable directions, we need to analyze the past orbit of a point. Since every point has more than
one preimage, there are several choices of past, and we can make each one of these
choices a point on a new space, defined as follows.


**Definition** **2.1.** Let ( _X, d_ ) be a compact metric space and _f_ : _X_ _→_ _X_ continuous.
The _inverse_ _limit_ _space_ (or _natural_ _extension_ ) to the triple _X_, _d_ and _f_ is

_•_ _X_ [˜] = _{x_ ˜ = ( _xk_ ) _∈_ _X_ [Z] : _xk_ +1 = _f_ ( _xk_ ), _∀k_ _∈_ Z _}_,

_•_ ( _f_ [˜] (˜ _x_ )) _k_ = _xk_ +1 _∀k_ _∈_ Z e _∀x_ ˜ _∈_ _X_ [˜],




_•_ _d_ [˜] (˜ _x,_ ˜ _y_ ) = [�]

_k_



_d_ ( _xk, yk_ )

.
2 _[|][k][|]_



With this definition, ( _X,_ [˜] _d_ [˜] ) is a compact metric space and the shift _f_ [˜] is continuous
and invertible. If _π_ : _M_ [˜] _→_ _M_ is the projection on the 0th coordinate, _π_ (˜ _x_ ) = _x_ 0,
then _π_ is continuous.
There have been several advances on the study of non-invertible systems by making
use of the inverse limit space. To name a few, it is the natural environment to search
for structural stability [5, 4], and it provides tools to explore the measure-theoretical




---PAGE BREAK---

6 MARISA CANTARINO AND REGIS [´] VARAO [˜]


properties of these systems [30]. Although the inverse limit space is not a manifold in
general, in the Axiom A (which includes the case of Anosov Endomorphisms) case it
can be stratified by finitely many laminations whose leaves are unstable sets [5, §4.2].
In the torus case, the inverse limit space has a very specific algebraic and topological structure, more precisely, the inverse limit space has the structure of a compact
connected abelian group with finite topological dimension, which is called a _solenoidal_
_group_ [3, §7.2]. It can also be seen as a fiber bundle ( _X, X, π, C_ [˜] ), where the fiber _C_
is a Cantor set [3, Theorem 6.5.1].


2.2. **Anosov** **endomorphisms:** **Some** **properties.** Let _M_ be a closed _C_ _[∞]_ Riemannian manifold.


**Definition** **2.2** ([29]) **.** Let _f_ : _M_ _→_ _M_ be a local _C_ [1] diffeomorphism. _f_ is an _Anosov_
_endomorphism_ (or _uniformly_ _hyperbolic_ _endomorphism_ ) if, for all _x_ ˜ _∈_ _M_ [˜], there is, for
all _i ∈_ Z, a splitting _TxiM_ = _E_ _[u]_ ( _xi_ ) _⊕_ _E_ _[s]_ ( _xi_ ) such that

_•_ _Df_ ( _xi_ ) _E_ _[u]_ ( _xi_ ) = _E_ _[u]_ ( _xi_ +1);

_•_ _Df_ ( _xi_ ) _E_ _[s]_ ( _xi_ ) = _E_ _[s]_ ( _xi_ +1);

_•_ there are constants _c_ _>_ 0 and _λ_ _>_ 1 such that, for a Riemannian metric on
_M_,
_||Df_ _[n]_ ( _xi_ ) _v|| ≥_ _c_ _[−]_ [1] _λ_ _[n]_ _||v||_, _∀v_ _∈_ _E_ _[u]_ ( _xi_ ), _∀i ∈_ Z,
_||Df_ _[n]_ ( _xi_ ) _v|| ≤_ _cλ_ _[−][n]_ _||v||_, _∀v_ _∈_ _E_ _[s]_ ( _xi_ ), _∀i ∈_ Z.
_E_ _[s/u]_ ( _xi_ ) is called the _stable/unstable_ _direction_ for _xi_ . When it is needed to make
explicit the choice of past orbit, we denote by _E_ _[u]_ (˜ _x_ ) the unstable direction at the
point _π_ (˜ _x_ ) = _x_ 0 with respect to the orbit _x_ ˜.


This definition includes Anosov diffeomorphisms (if _f_ is invertible) and expanding
maps (if _E_ _[u]_ ( _x_ ) = _TxM_ for each _x_ ). Throughout this work, however, we consider the
case in which _E_ _[s]_ is not trivial, excluding the expanding case, unless it is mentioned
otherwise.
A point can have more than one unstable direction under an Anosov endomorphism, even though the stable direction is always unique. Indeed, we find in [29] an
example in which a point has uncountable many unstable directions.
Note that there is not a global splitting of the tangent space; the splitting is
along a given orbit _x_ ˜ = ( _xi_ ) on _M_ ˜ . But _x_ ˜ induces a hyperbolic sequence, and
one can apply the Hadamard–Perron theorem in the same way it is done for Anosov
diffeomorphisms to prove that _f_ has _local stable_ and _local unstable manifolds_, denoted
by _Wf,R_ _[s]_ [(˜] _[x]_ [) and] _[ W][ s]_ _f,R_ [(˜] _[x]_ [),] [tangent to the stable and unstable directions [][29][,] [Theorem]
2.1]. Additionally, for _R >_ 0 sufficiently small, these manifolds are characterized by

_Wf,R_ _[s]_ [(˜] _[x]_ [) =] _[ {][y]_ _[∈]_ _[M]_ [:] _[ ∀][k]_ _[≥]_ [0,] _[d]_ [(] _[f][ k]_ [(] _[y]_ [)] _[, f][ k]_ [(] _[x]_ [0][))] _[ < R][}]_


and

_Wf,R_ _[u]_ [(˜] _[x]_ [) =] _[ {][y]_ _[∈]_ _[M]_ [:] _[ ∃][y]_ [˜] _[∈]_ _M_ [˜] such that _π_ (˜ _y_ ) = _y_ e _∀k_ _≥_ 0, _d_ ( _y−k, x−k_ ) _< R}._


The _global_ _stable/unstable_ _manifolds_ are

_Wf_ _[s]_ [(˜] _[x]_ [) =] _[ {][y]_ _[∈]_ _[M]_ [:] _[ d]_ [(] _[f][ k]_ [(] _[y]_ [)] _[, f][ k]_ [(] _[x]_ [0][))] _−−−−→k→∞_ 0 _}_


and

_Wf_ _[u]_ [(˜] _[x]_ [) =] _[ {][y]_ _[∈]_ _[M]_ [:] _[ ∃][y]_ [˜] _[∈]_ _M_ [˜] such that _π_ (˜ _y_ ) = _y_ e _d_ ( _y−k, x−k_ ) _−−−−→k→∞_ 0 _}._


Moreover, these manifolds are as regular as _f_ . The stable manifolds do not depend
on the choice of past orbit for _x_ 0, but the unstable ones do.




---PAGE BREAK---

RIGIDITY FOR ANOSOV ENDOMORPHISMS ON T2 7


In the case that the unstable directions do not depend on _x_ ˜, that is, _E_ _[u]_ (˜ _x_ ) = _E_ _[u]_ (˜ _y_ )
for any _x,_ ˜ ˜ _y_ _∈_ _M_ [˜] with _x_ 0 = _y_ 0, then we say that _f_ is a _special_ _Anosov_ _endomorphism_ .
Hyperbolic toral endomorphisms are examples of special Anosov endomorphisms, as
the unstable direction of each point is given by its unstable eigenspace.
The fact that Anosov endomorphisms that are not invertible or expanding maps
are not structurally stable was proven by R. Ma˜n´e and C. Pugh [22] and F. Przytycki

[29] in the 1970’s, when they introduced the concept of Anosov endomorphisms as
we know today. Ma˜n´e and Pugh also proved the following proposition, which is very
useful to generalize properties of Anosov diffeomorphisms to endomorphisms.


**Proposition** **1** ([22]) **.** _Let_ _M_ _be_ _the_ _universal_ _cover_ _of_ _M_ _and_ _F_ : _M_ _→_ _M_ _a_ _lift_
_for_ _f_ _._ _Then_ _f_ _is_ _an_ _Anosov_ _endomorphism_ _if_ _and_ _only_ _if_ _F_ : _M_ _→_ _M_ _is_ _an_ _Anosov_
_diffeomorphism._ _Additionally,_ _the_ _stable_ _bundle_ _of_ _F_ _projects_ _onto_ _that_ _of_ _f_ _._


Most results for Anosov diffeomorphisms require _M_ to be compact. Even though
universal covers are not generally compact, since _F_ is a lift for a map on a compact
space, _F_ carries some uniformity, which allows us to prove some results that were
originally stated for compact spaces for the lifts proven in Proposition 1 to be Anosov
diffeomorphisms.


**Proposition** **2** ([27]) **.** _If_ _f_ : _M_ _→_ _M_ _is_ _a_ _C_ [1+] _[α]_ _Anosov_ _endomorphism,_ _α_ _>_ 0 _,_
_and_ _F_ : _M_ _→_ _M_ _is_ _a_ _lift_ _for_ _f_ _to_ _the_ _universal_ _cover,_ _then_ _there_ _are_ _WF_ _[u]_ _[and]_ _[W][ s]_ _F_
_absolutely_ _continuous_ _foliations_ _tangent_ _to_ _EF_ _[u]_ _[and]_ _[E]_ _F_ _[s]_ _[.]_


The above proposition is stated in [27] as Lemma 4.1, and the absolute continuity
is defined on Subsection 2.5. The proof is the same as for the compact case, as
_F_ projects on the torus, then its derivatives are periodic with respect to compact
fundamental domains. With the same argument, we can prove the following just as
it is in [21, §19.1].


**Proposition** **3.** _Let_ _f_ : _M_ _→_ _M_ _be_ _an_ _Anosov_ _endomorphism_ _and_ _F_ : _M_ _→_ _M_ _a_
_lift_ _for_ _f_ _to_ _the_ _universal_ _cover._ _If_ _the_ _unstable_ _distribution_ _of_ _F_ _has_ _codimension_
_one,_ _then_ _it_ _is_ _C_ [1] _._


In particular, if dim _M_ = 2, then we can apply the arguments in Proposition 3 for
_F_ and _F_ _[−]_ [1], since _F_ is invertible, and both the stable and unstable distributions are
_C_ [1] . This also implies that the stable and unstable holonomies (see Definition 2.3)
are _C_ [1], . In general, these distributions and the associated holonomies are H¨older
continuous (see [21, §19.1]).


**Definition** **2.3.** Given a foliation _F_, we define the _holonomy_ _h_ Σ1 _,_ Σ2 : Σ1 _→_ Σ2
between two local discs Σ1 and Σ2 transverse to _F_ by _q_ _�→F_ ( _q_ ) _∩_ Σ2, where _F_ ( _q_ ) is
the leaf of _F_ containing q.


That is, a holonomy moves the point _q_ through its leaf on _F_ . For Anosov endomorphisms, we have transverse foliations on the universal cover, so the _stable_ _holonomy_
can have local unstable leaves as the discs. When there is no risk of ambiguity, we
denote it simply by _h_ _[s]_ . The same goes for the unstable holonomy.
Another important feature of Anosov endomorphisms on tori is transitivity. The
following theorem is a consequence of results in [3] for _topological Anosov maps_, which
are continuous surjections with some kind of expansiveness and shadowing property.
Anosov endomorphisms are particular cases of topological Anosov maps that are
differentiable.




---PAGE BREAK---

8 MARISA CANTARINO AND REGIS [´] VARAO [˜]


**Proposition** **4** ([3]) **.** _Every_ _Anosov_ _endomorphism_ _on_ T _[n]_ _is_ _transitive._


_Proof._ By [3, Theorem 8.3.5], every topological Anosov map _f_ on T _[n]_ has its nonwandering set as the whole manifold, that is, Ω( _f_ ) = T _[n]_, which implies transitivity. 

2.3. **Conjugacy.** We say that _A_ : T _[n]_ _→_ T _[n]_ is the _linearization_ of an Anosov endomorphism _f_ : T _[n]_ _→_ T _[n]_ if _A_ is the unique linear toral endomorphism homotopic to _f_ .
Much of the behavior of _f_ can be inferred by the one of _A_ . In fact, if _f_ is invertible
or expansive, _f_ and _A_ are topologically conjugate. In the more general non-invertible
setting, this conjugacy does not exist if _f_ is not special, since a conjugacy should
preserve stable and unstable manifolds.
The version of Theorem A given by F. Micena [25, Theorem B] requires the Anosov
endomorphism to be _strongly_ _special_ - that is, each point only has one unstable
direction and _Wf_ _[s]_ [(] _[x]_ [)] [is] [dense] [for] [each] _[x]_ _[∈]_ _[M]_ [—] [in] [order] [to] [guarantee] [the] [existence]
of conjugacy with its linearization and to prove that it is smooth. This relies on
Proposition 5 stated below and given by Aoki and Hiraide in [3] for topological
Anosov maps.


**Proposition** **5** ([3]) **.** _If_ _f_ : T _[n]_ _→_ T _[n]_ _is_ _a_ _strongly_ _special_ _Anosov_ _endomorphism,_
_then_ _its_ _linearization_ _A_ _is_ _hyperbolic_ _and_ _f_ _is_ _topologically_ _conjugate_ _to_ _A._


For Anosov diffeomorphisms, the density of the stable or unstable leaves of each
point is equivalent to transitivity. Whether every Anosov diffeomorphism is transitive
is still an open question.
However, for general Anosov endomorphisms even in the transitive case, the stable
manifolds may be not dense. For example, consider the linear Anosov endomorphism
on T [3] induced by the matrix











_A_ =



2 1 0
1 1 0
0 0 2



 _._



It is easy to check that dim _E_ _[u]_ = 2, dim _E_ _[s]_ = 1, _A_ : T [3] _→_ T [3] is transitive and
_WA_ _[u]_ [(] _[x]_ [)] [is] [dense] [in] [T][3] [for] [each] _[x]_ [,] [but,] [if] _[x]_ [ = (] _[x]_ [1] _[, x]_ [2] _[, x]_ [3][)] _[ ∈]_ [T][3][,] _[W][ s]_ _A_ [(] _[x]_ [)] [is] [restricted] [to]
T [2] _× {x_ 3 _}_, then it is not dense.
In the same year, Naoya Sumi proved that the hypothesis of density on the stable
set is not required [32]. More recently, Moosavi and Tajbakhsh [28], very similarly to
Aoki and Hiraide and to Sumi, extended this result to topological Anosov maps on
nil-manifolds. As a consequence, we have the following.


**Proposition** **6** ([32, 28]) **.** _An_ _Anosov_ _endomorphism_ _f_ : T _[n]_ _→_ T _[n]_ _is_ _special_ _if_ _and_
_only_ _if_ _it_ _is_ _conjugate_ _to_ _its_ _linearization_ _by_ _a_ _map_ _h_ : T _[n]_ _→_ T _[n]_ _homotopic_ _to_ _Id._


Even a small perturbation of a special Anosov endomorphism may be not special,
and a perturbation can, in fact, have uncountable many unstable directions. This is
an obstruction to topological conjugacy. On the universal cover, however, we do have
a conjugacy.
If _f_ : T _[n]_ _→_ T _[n]_ is an Anosov endomorphism, by [3, Proposition 8.2.1] there is a
unique continuous surjection _H_ : R _[n]_ _→_ R _[n]_ on the universal cover with


_•_ _A ◦_ _H_ = _H_ _◦_ _F_ ;

_•_ _H_ is uniformly close to _Id_ ;

_•_ _H_ is uniformly continuous.




---PAGE BREAK---

