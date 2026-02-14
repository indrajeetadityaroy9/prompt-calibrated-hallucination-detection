**Gauge Theories of Josephson Junction Arrays:** **Why Disorder Is Irrelevant for the Electric**
**Response of Disordered Superconducting Films**


C. A. Trugenberger [1]

1 _SwissScientific_ _Technologies_ _SA,_ _rue_ _du_ _Rhone_ _59,_ _CH-1204_ _Geneva,_ _Switzerland_



**JOSEPHSON JUNCTION ARRAYS AS A MODEL FOR**
**PLANAR SUPERCONDUCTORS**


Superconductivity (for a review, see [1]) is predicated on
the formation of a ground state in which magnetic interactions
are much stronger than electric ones. When the thickness of a
superconducting film is decreased, however, electric interactions become stronger and stronger, until the superconducting
ground state is destroyed in favour of different phases [2–4].
These phase transitions, which can also be driven by an applied magnetic field at fixed (small) thickness, are accompanied by the dissociation of the material into an emergent granular structure of islands of charge condensate [5]. In this configuration, superconductivity is due to Josephson tunnelling
between islands when global phase coherence is established.
The other phases form when global phase coherence is lost.
These granular superconductors near their quantum phase
transition can be modelled by Josephson junction arrays
(JJAs) [6], fabricated regular quadratic lattices of superconducting islands deposited on a substrate and coupled by
Josephson junctions (for a review, see [7]). The phase transitions are typically driven by tuning the Josephson coupling
_EJ_ with respect to a mostly fixed charging energy _EC_ of the
islands. In some of the most recent implementations, e.g.,
the Al islands are deposited on a semiconductor InAs substrate and the Josephson coupling is driven by a voltage gate

[8]. Typical island dimensions are _O_ (1µ _m_ ), while their typical
distances are _O_ (100 _nm_ ).
It is often believed that the quantum transition destroying
superconductivity is due to the disorder embodied by the irregular granularity [9]. Here, we show that it is not so; this
quantum transition is caused by electric interactions becoming ever stronger, and this increase in the electric coupling has
nothing to do with disorder, as can be witnessed in the fact
that it also occurs in perfectly regular JJAs.
In discussions of disorder, one starts from a disorder-free
Hamiltonian, which defines the spectrum of excitations. Then,
one lets some parameters of this Hamiltonian become random
variables, be it the potential energy of electrons, as in the original formulation [10], the Ising couplings, when discussing
spin glasses (for a review, see [11]) or other couplings in general. For planar superconductors, the appropriate disorder-free
Hamiltonian to start with is that of Josephson junction arrays,
which, in their classical limit, are embodiments of the XY
model, which is the paradigm of 2D superconductivity (for
a review, see [12]). The disorder to be added consists then in



allowing random sizes (and shapes) of the local condensate islands and random distances between them (the graph connectivity is typically held fixed in these discussions to maintain
the existence of the superconducting phase). This disorder
models the typical structure of irregular islands of condensate
characterizing superconducting films [5], and is reflected in
the couplings _EJ_ and _EC_ becoming random variables centred
around their typical JJA values.


Starting from the correct disorder-free Hamiltonian ensures
that one deals with the correct spectrum of excitations. It is
these excitations that are affected by the disorder embodied
in the random couplings. For example, in the 1D disordered
Schwinger model, one should start from the bosonized version of the Hamiltonian, since linear confinement implies that
the spectrum consists only of neutral mesons [13]. In the
present case, the correct disorder-free system precisely consists of JJAs. As we now show, the phases obtained in this system when superconductivity is destroyed are either topological, and hence disorder becomes “transparent”, or do not have
charged excitations in the spectrum, and disorder can thus influence possibly thermal properties but not electric transport.
This has the consequence that they are genuine new phases of
matter.


In a nutshell, what happens is as follows. On a 2D JJA,
quantum phase slips are equivalent to vortex tunnelling on the
dual array. When vortices can tunnel, charges and vortices can
both be out of condensate simultaneously. But as soon as this
happens, charges and vortices are frozen into a topological
ground state by their mutual statistics interactions: this is the
origin of the Bose metal. Moreover, when vortex tunnelling
events proliferate, electric fields are squeezed into electric flux
tubes between charges and holes. This causes a linearly confining potential and the infinite resistance of the superinsulating phase.


However, there is one important difference between superconducting films and JJAs. Films are made of one single material, it is only the condensate that dissociates into superconducting islands; JJAs typically have physical islands of a different material than the exposed substrate between them. As
we will show, this implies that to see all phases in JJAs, one
most probably needs two driving parameters, with one setting
the charge tunnelling strength _EJ_ between the islands and one
governing the vortex tunnelling strength _EC_ on the “dual array” between them.




---PAGE BREAK---

2



**JOSEPHSON JUNCTION ARRAYS: THE STANDARD**
**TREATMENT**


Josephson junction arrays (for a review, see [7]) are
quadratic arrays of spacing ℓ of superconducting islands with
nearest neighbours Josephson couplings of strength _EJ_ . Each
island has a capacitance _C_ 0 to the ground and a mutual capacitance _C_ to its neighbours. The Hamiltonian for such a system
is



integral representation [6]




- +π

Dφ exp(− _S_ ),
−π



_Z_ = 

{ _q_ **x** }

  - β
_S_ =



1
**x** _i q_ **x** φ˙ **x** + 4 _E_ C [′] _[q]_ **[x]** _C_ 0/ _C_ −∇ [2] _[q]_ **[x]**




 _dt_
0




 + _EJ_ (1 − cos (∆ _i_ φ **x** )), (6)

**x**, _i_



where β = 1/ _T_ is the inverse temperature. In (6), (Euclidean)
time has to be considered also as discrete, as generally appropriate when degrees of freedom can change only in integer
steps. We introduce thus a discrete time step ℓ0, which is the
typical time scale associated with tunnelling events. We thus
substitute the time integrals and space sums over a lattice with
nodes **x** by a sum over space-time lattice nodes _x_, with _x_ [0] = _t_
denoting the discrete time direction. Also, in what follows we
shall consider the purely quantum theory at zero temperature
by letting β →∞; incorporating a finite temperature is easily
performed by restricting the time sums to a finite domain. Denoting the (forward) finite-time differences by ∆0, we obtain




- �2 - - �� [�]
_V_ **y** - _V_ **x** + _EJ_ 1 − cos φ **y** - φ **x**,



_H_ = 

**x**



_C_ 0



0 
2 _[V]_ **x** [2][+]



< **xy** 



- _C_

2



(1)
where boldface characters denote the sites of the array, < **xy** indicates nearest neighbours, _V_ **x** is the electric potential of the
island at **x**, and φ **x** the phase of its order parameter. Introducing the forward and backwards lattice derivatives (which are
exchanged under summation by parts):


∆ _i f_ ( **x** ) = _f_ ( **x** + [ˆ] _i_ ) − _f_ ( **x** ),
∆ˆ _i f_ ( **x** ) = _f_ ( **x** ) − _f_ ( **x**           - ˆ _i_ ), (2)


where [ˆ] _i_ denotes a unit vector in direction _i_, and the corresponding lattice Laplacian ∇ [2] = ∆ [ˆ] _i_ ∆ _i_, we can rewrite the
Hamiltonian as




 - 1
_S_ = _x_ _i_ _j_ 0∆0φ + 4ℓ0 _E_ C [′] _[j]_ [0] _C_ 0/ _C_ −∇ [2] _[j]_ [0]



_Z_ = 

{ _j_ 0}




- +π

Dφ exp(− _S_ ),
−π




 + ℓ0 _E_ J (1 − cos (∆ _i_ φ)), (7)

_x_, _i_



_H_ = 

**x**



1 - _C_ 0 − _C_ ∇ [2][�] _V_ **x** + _EJ_ (1 − cos (∆ _i_ φ **x** )), (3)
2 _[V]_ **[x]**

**x**, _i_



where we have chosen natural units so that _c_ = 1, ℏ = 1,
ε0 = 1 and we have set the lattice spacing ℓ = 1 for ease of
presentation.
The phases φ **x** are quantum-mechanically conjugated to the
charges (Cooper pairs) 2 _eq_ **x**, _q_ **x** ∈ _Z_ on the islands, where _e_
is the electron charge. The Hamiltonian (3) can be expressed
in terms of charges and phases by noting that the electric potentials _V_ **x** are determined by the charges 2 _eq_ **x** via a discrete
version of Poisson’s equation:


           - _C_ 0 − _C_ ∇ [2][�] _V_ **x** = 2 _eq_ **x** . (4)


Using this in (3) we obtain



where we now denote the integer charge degrees of freedom
by _j_ 0 for reasons to become clear in a moment. Going over
to the Villain representation, the partition function can be formulated as




- +π

Dφ exp(− _S_ ),
−π



_Z_ = 

{ _ai_ },{ _j_ 0}




D _ji_



1       **x** 4 _E_ C [′] _[q]_ **[x]** _C_ 0/ _C_ −∇ [2] _[q]_ **[x]** [ +] **x**, _i_



_H_ = 


_E_ J (1 − cos (∆ _i_ φ **x** )),
**x**, _i_



(5)
wheredimensional Yukawa potential of mass _E_ C [′] [≡] _[e]_ [2][/][2] _[C]_ [.] [The integer charges] _[ q]_ [√] _C_ **[x]** [interact via a two-] 0/ _C_ . We have denoted the coupling of this potential as _E_ C [′] [to signify that, con-]
trary to what is normally assumed, it is not the full charging
energy _E_ C of the islands, as we will now show.

The partition function of the JJA admits a phase-space path



 - 1 1
_S_ = _x_, _i_ _i_ _j_ 0∆0φ + _iji_ (∆ _i_ φ + 2π _ai_ ) + 4ℓ0 _E_ C [′] _[j]_ [0] _C_ 0/ _C_ −∇ [2] _[j]_ [0][ +] 2ℓ0 _E_ J _[j]_ (8) _i_ [2] [,]


where _ai_ ∈ Z are integers and _ji_ represents the total charge
current in direction [ˆ] _i_ .


**ADDING QUANTUM PHASE SLIPS**


In 1D Josephson junction chains, quantum phase slips [14]
are crucial in the regime of low temperatures and, accordingly,
they are routinely taken into account (for a review, see [15]).
For some unknown reason, however, the corresponding tunnelling events are mostly neglected for 2D JJAs, which leads
to the wrong results. Let us first note that _ai_ in (8) constitutes a
lattice gauge field. If we make a transformation _ai_ → _ai_ +∆ _i_ λ,
with an integer λ, we can absorb λ into φ with a shift by a multiple of 2π of its integration domain. We can then shift the definition of the integers _ai_ to re-establish the original integration




---PAGE BREAK---

domain and the original action, showing that the gauge transformation does indeed leave the partition function invariant.
As a consequence, only the transverse, pseudo-scalar components of _ai_ constitute gauge-invariant quantities; these are
the vortices in the model. Note two very important, and also
often overlooked, facts: these are core-less vortices, characterized only by their gauge structure, i.e., the circulation of
the phases around an array plaquette. As such they can tunnel
without dissipation from one site of the dual array to a neighbouring one. Moreover, contrary to Cooper pairs, they are not
Noether charges but purely topological ones. Since they have
no core, they can not only tunnel from one site of the dual
array to another, but they can also appear/disappear on one
site in tunnelling events that change the topological quantum
number, called instantons (for a review see [16]).
In 1D Josephson junction chains, quantum phase slips are
local quantum tunnelling events in which the phase of the condensate at one particular island undergoes a 2π flip over the
typical time scale ℓ0. In 2D JJAs, instead, quantum phase
slips are half-lines of simultaneous such flips of alternating
chirality, ending in one particular island, as shown in Figure 1. These configurations are nothing other than half-lines
in which the gauge field _ai_ alternatingly increases or decreases
by one unit. Since there are nowhere vortices but on the two
plaquettes based on the endpoint, the only gauge-invariant
quantity in this configuration is the quantum phase slip at the
endpoint, corresponding to the displacement of a vortex from
one plaquette based there to the adjacent one, as shown in Figure 1. Thus, again, we have gauge-invariant instantons. These
must be taken into account at low temperatures.


FIG. 1. A half-line of simultaneous and alternating 2π phase flips.
The only gauge-invariant degree of freedom is the quantum phase
slip at the endpoint, corresponding to the displacement of a vortex
from one plaquette based on this endpoint to the adjacent one.


Quantum phase slips in JJAs amount thus to vortex tunnelling on the dual array (centres of the plaquettes), the dual
phenomenon to Cooper pair tunnelling on the array. This tunnelling _cannot be neglected_ when studying the quantum phase



3


structure of the arrays. We must thus add to the action a vortex kinetic term dual to the corresponding tunnelling current
for Cooper pairs [17, 18]. Of course, this has an important
consequence: a vortex moving in one direction creates a 2D
logarithmic Coulomb potential between the islands in the perpendicular direction.
Before showing how this can be incorporated into the action
we have to pause a moment to introduce the gauge-invariant
lattice version of the curl operator ϵµαν∂α, where the Greek letters denote the three possible directions on the Euclidean 3D
lattice, with sites denoted by { _x_ } [ **?** ]. We first introduce forward and backward finite differences also in the (Euclidean)
time direction,

∆0 _f_ ( _x_ ) = _[f]_ [(] _[x]_ [ +][ ℓ][0][ ˆ][µ][)][ −] _[f]_ [(] _[x]_ [)],

ℓ0

∆ˆ 0 _f_ ( _x_ ) = _[f]_ [(] _[x]_ [)][ −] _[f]_ [(] _[x]_ [ −] [ℓ][0][ ˆ][µ][)] . (9)

ℓ0


Then, we introduce forward and backwards shift operators


_S_ µ _f_ ( _x_ ) = _f_ ( _x_ + _d_ µˆ),
_S_ ˆ µ _f_ ( _x_ ) = _f_ ( _x_        - _d_ µˆ), (10)


where µˆ denotes a unit vector in direction µ and _d_ = 1 in
the spatial directions, _d_ = ℓ0 in the Euclidean time direction.
Summation by parts on the lattice interchanges both the two
finite differences (with a minus sign) and the two shift operators. Gauge transformations are defined by using the forward
finite differences. In terms of these operators, one can then
define two lattice curl operators


_K_ µν = _S_ µϵµαν∆α, _K_ ˆ µν = ϵµαν ˆ∆α ˆ _S_ ν, (11)


where no summation is implied over the equal indices µ and
ν. Summation by parts on the lattice interchanges also these
two operators (without any minus sign). Gauge invariance is
then guaranteed by the relations


_K_ µν∆ν = ∆ [ˆ] µ _K_ µν = 0, _K_ ˆ µν∆ν = ∆ˆ µ _K_ ˆ µν = 0 . (12)


Note that the product of the two curl operators gives the lattice
Maxwell operator


_K_ µα _K_ [ˆ] αν = _K_ [ˆ] µα _K_ αν = −δµν∆+ ∆µ∆ [ˆ] ν, (13)


where ∆= ∆ [ˆ] µ∆µ is the 3D Laplace operator.
Using this notation, the vortex number density is ϕ0 =
_K_ 0 _iai_ . Since the vortex three-current is conserved, we can
write it completely in terms of a gauge field _a_ µ after introducing a Lagrange multiplier _a_ 0: ϕµ = _K_ µν _a_ ν. The partition
function, including quantum phase slips, is then



_i_ _j_ 0 (∆0φ + 2π _a_ 0) + _iji_ (∆ _i_ φ + 2π _ai_ ) + 1 _ji_ 2 + π [2]
_x_ 2ℓ0 _E_ J 4ℓ0




- +π

Dφ exp(− _S_ ),
−π



_Z_ = 

{ _a_ µ},{ _j_ 0}




D _a_ 0D _ji_



_S_ = 


~~(~~ ϕ14) [2] _i_ [.]
4ℓ0 _E_ C




---PAGE BREAK---

The Gauss law associated with the Lagrange multiplier _a_ 0
leads, in the Coulomb gauge, to the 2D Coulomb interaction
term


   - 1
_S_ Coulomb =     - · · + 4ℓ0 _E_ C _j_ 0 −∇ [2] _[j]_ [0][ +][ . . .] [,] (15)

_x_


for charges. This shows that, on sufficiently large samples, the
charging energy of the islands is dominated by the 2D logarithmic Coulomb interaction associated with vortex tunnelling
(quantum phase slips) and not by the screened interaction due
to the finite capacitances. This can lead only to finite-size corrections to the dominant _EC_ in (14).
At this point, we note that the charge current _j_ µ is conserved
and, hence, it can be represented as the dual field strength associated with a second emergent gauge field _b_ µ as _j_ 0 = _K_ 0 _ibi_,
_ji_ = _Ki_ 0 _b_ 0 + _Ki jb j_, where _b_ 0 is a real variable, while _bi_ are
integers. We then use Poisson’s formula,



4


quadratic in the emergent electric fields, which are orthogonal to the total charge and vortex currents. These are the
Josephson tunnelling currents plus local fluctuation terms deriving from time-dependent vortex currents for charges, and
vice versa for vortices. The action is thus a non-relativistic
version of the Maxwell–Chern–Simons gauge action corresponding to infinite magnetic permeability, in which emergent
magnetic fields are suppressed. Gauge fields are topologically
massive [19], the topological gap coinciding with the Josephson plasma frequency of the array, ωP = ~~[√]~~ 8 _EJ_ _EC_ . The dimensionless parameter g = ~~�~~ π [2] _EJ_ /2 _EC_, measuring the relative strength of magnetic and electric interactions, drives the
quantum phase structure. Note that the quantity _EC_ here is
different from the previously introduced _EC_ [′] [:] [the former is the]
long-range logarithmic potential induced by vortex tunnelling,
the latter is the sub-dominant short-range component arising
from the island capacitances.
The quantum phase structure is determined by the condensation, or lack thereof, of integer electric ( _Qi_ ) or magnetic
( _M_ µ) strings on a 3D Euclidean lattice [17]. Both their energy and their entropy are proportional to their length and the
condensations are governed by energy/entropy balance conditions. The resulting quantum phase structure is shown in
Figure 2.







_f_   - _n_ µ� =   _n_ µ _k_ µ




_dn_ µ _f_  - _n_ µ� e _[i]_ [2][π] _[n]_ [µ] _[k]_ [µ], (16)



_k_ µ



turning a sum over integers { _n_ µ} into an integral over real variables, to make all components of the gauge fields _a_ µ and _b_ µ
real, at the price of introducing integer link variables _Qi_ and
_Mi_,



_Z_ = 

{ _Qi_ }


_S_ = 





{ _Mi_ }




- - +π
D _a_ µD _b_ µ Dφ exp(− _S_ ),

−π





1 2 π [2]
_i_ 2π _a_ µ _K_ µν _b_ ν + _ji_ +
_x_ 2ℓ0 _E_ J 4ℓ0



Finally, we note that the quantities _K_ [ˆ] µν∆νφ are the circulations of the array phases around the plaquettes orthogonal
to the direction µ in 3D Euclidean space-time, and are thus
quantized as 2π integers. We can thus absorb the quantities

- _K_ ˆ _i_ 0∆0φ + _K_ ˆ _ij_ ∆ _j_ φ� in a redefinition of the integers _Mi_, and define _K_ [ˆ] 0 _i_ ∆ _i_ φ = 2π _M_ 0. The original integral over the phases φ
can then be traded for a sum over the vortex numbers _M_ 0,



FIG. 2. The quantum phase structure of JJAs.



_Z_ = 

{ _Qi_ }


_S_ = 





{ _M_ µ}




D _a_ µD _b_ µ exp(− _S_ ),



ϕ [2] _i_ [+] _[ i]_ [2][π] _[a][i][Q][i]_ [ +] _[ i]_ [2][π] _[b]_ [µ] _[M]_ [µ][ .]
4ℓ0 _E_ C /ℓ



_i_ 2π _a_ µ _K_ µν _b_ ν + 1 _ji_ 2 + π [2]
_x_ 2ℓ0 _E_ J 4ℓ0



(18)of



This is the gauge theory of JJAs [17, 18]. The quantities
_Qi_ and _Mi_ represent the Josephson currents of Cooper pairs
and the dual vortex tunnelling currents, respectively, while _M_ 0
is the vortex number. Charges and vortices interact via two
emergent gauge fields _a_ µ and _b_ µ. The first, infrared-dominant
term in the gauge action is the lattice version of the topological Chern–Simons term [19]. The remaining two terms are



In addition to the dimensionless conductance parameter g,
there is another relevant parameter η [17], which is a function
of the ratio ωPℓ0 of the two characteristic frequencies in the
problem, [µ][ .] the plasma frequency and the tunnelling frequency
1/ℓ0. When η < 1, there is a direct “first-order” (coexistence

(18)of two phases) quantum transition between superconducting

and superinsulating phases as g is decreased below the resistance quantum at _RQ_ = 6.45 _k_ Ω at g = 1, i.e., electric interactions become stronger. When η - 1, an intermediate Bose
metal state [17] (for a review, see [20, 21]) appears in the interval 1/η < g < η; as we now review [22], this phase is actually
a bosonic topological insulator [23, 24]. Exactly this quantum
phase structure has been recently derived experimentally in an
irregular In/InO composite array, although the driving mag



---PAGE BREAK---

netic field was too low to detect superinsulation [25].
When magnetic interactions dominate ( _EJ_ ≫ _EC_ ), electric
strings _Qi_ condense. This means that Josephson currents of
charges percolate through the sample, establishing global superconductivity. This superconductivity, with dissipationless
vortices whose confinement/deconfinement drives the thermal
phase transition to a resistive state is a type-III superconductivity, not described by the usual Ginzburg–Landau theory

[26], and is not confined to 2D but exists also in 3D [27].
It has also been proposed to described the physics of high- _Tc_
cuprates [28]. We shall not discuss this phase further here, but
we shall rather focus on what happens when superconductivity is destroyed by strong electric interactions.
For η > 1, there is an intermediate domain in which neither
electric nor magnetic strings condense: no superconducting
currents, and vortices are gapped excitations. In this intermediate domain, the infrared-dominant action for the the JJA
reduces to the topological Chern-Simons term


_S_ TI =        - _i_ 2π _a_ µ _K_ µν _b_ ν . (19)


_x_


This is the action of a bosonic topological insulator [23,
24]. In this phase, both charges and vortices are frozen in
the bulk. The ground state wave function of this state has
been derived in [29]; it consists of an integer-filling composite
quantum incompressible fluid of charges and vortices at g = 1
with excess charges and vortices for g - 1 forming a Wigner
crystal, with charges being in excess of vortices for g > 1 and
vice versa.
While the bulk is completely frozen at zero temperature,
there remain edge currents, where edges may be internal to the
sample, forming a percolation structure [30]. On these edges,
we also have the usual 1D quantum phase slips [14, 15], corresponding to vortices moving across the edges. These cause
the observed metallic saturation of the resistance, which is the
origin of the name “Bose metal” for this phase, first predicted
in [17]. Of course, since there is an overabundance of charges
for g - 1, the resistance is lower here than in the region
g < 1 where we have an overabundance of vortices. Correspondingly, when the temperature is raised, the resistance
increases in the regime g - 1, which is the origin of the alternative name “failed superconductor”, while it decreases in
the opposite regime g - 1, giving rise to the alternative name
“failed insulator”. Failed superconductors and failed insulators, however, are two faces of the same medal, the intermediate bosonic topological insulator [22].
The bosonic topological insulator is the physical embodiment of a field-theoretic anomaly involving Chern–Simons
gauge fields [31]. In topologically massive gauge theories,
the limit _m_ →∞ does not commute with quantization because of the phase space reduction this limit entails [31]. In
physical applications, the topological gauge theory in (19)
must always be considered as the _m_ →∞ limit of the full
theory (18) with dynamical terms; otherwise, wave functionals would not be normalizable. This implies, in particular,
that phase and charge are _not_ a canonically conjugate pair, as



5


would follow from the pure Chern–Simons term (19). Therefore, charges and vortices can be both out of condensate, even
when their gap is finite and they can be excited in the bulk.
Of course, at _T_ = 0, they are immediately frozen into a topological ground state by the mutual statistics interactions, giving rise to the topological insulator/Bose metal phase via edge
transport [22].
When electric interactions dominate ( _EC_ ≫ _EJ_ ), there is
a condensation of vortices while Josephson currents are suppressed. To establish the nature of this phase, let us couple the
total electric current _j_ µ to the real electromagnetic gauge field
_A_ µ,



is the string tension, _G_ = _O_ (1), and we have reinstated physical units. This is the phenomenon of confinement, known
from strong interactions (for a review, see [34]), with electric




  _S_ → _S_ + _i_



_A_ µ _j_ µ = _S_ + _i_  _x_ _x_



_A_ µ _K_ µν _b_ ν, (20)

_x_



and integrate over the emergent gauge fields _a_ µ and _b_ µ to obtain the electromagnetic effective action. In the limit ℓ0ω _P_ ≫
1, this is given by




   -   - g
_S_ eff _A_ µ, _Mi_ =
4πℓ0ω _P_




- ( _Fi_ - 2π _Mi_ ) [2], (21)


_x_



where _Fi_ are the spatial components of the dual electromagnetic vector strength _F_ µ = _K_ [ˆ] µν _A_ ν, and where the integers _Mi_
have to be summed over in the partition function. This is a
deep non-relativistic version of Polyakov’s compact QED action [32, 33], in which only electric fields survive. Its form
shows that the action is periodic under shifts _Fi_ → _Fi_ + 2π _Ni_,
with integer _Ni_, and that the gauge fields are thus indeed compact, i.e., angular variables defined on the interval [−π, +π].
The integers _Mi_ can be decomposed into transverse and longitudinal components, of which neither alone has to be an integer, only the sum is so constrained. The transverse components can be absorbed into a redefinition of _Fi_ ; the longitudinal components can be represented as

_Mi_ [L] [=] ∇ [∆][2] _[i]_ _[m]_ [,] (22)


where _m_ ∈ Z are magnetic monopole instantons [32, 33].
These represent tunnelling events in which vortices appear/disappear on a single plaquette, thereby interpolating between different topological sectors. The proliferation of such
instantons at small g has a momentous consequence.
One is used to the fact that electromagnetic fields mediate Coulomb forces between static charges, a 1/| **x** | potential
in 3D, or a log| **x** | potential in 2D. The monopole plasma in
the compact version of QED, however, drastically changes
this and generates a linearly confining potential σ| **x** | between
charges of opposite sign, where



σ = [ℏ][ω] _[P]_

ℓ




~~�~~
16 e [−] 2ℓ0πgω _P_ _[G]_, (23)
πgℓ0ω _P_




---PAGE BREAK---

fields playing the role of chromo-electric fields and Cooper
pairs playing the role of quarks. An electric flux tube (string)
dual to Abrikosov vortices holds together charges in neutral
pion excitations. There is no charged excitation in the spectrum for arrays larger than the pion size, and this is the origin
of the infinite electric resistance in this phase, dual to the infinite conductivity in the superconducting phase. This state of
matter is known as a superinsulator, and was first predicted in

[17] (for a review, see [35]). Superinsulators [17, 36–39], with
their divergent electrical resistance, have been experimentally
detected in InO [40], TiN [36], NbTiN [41], and NbSi [42]. A
recent measurement of the dynamic response of superinsulators confirmed that the potential holding together ± charges is
indeed linear [43].


**WHY DISORDER AND DISSIPATION ARE NOT RELEVANT**
**FOR THE QUANTUM PHASES OF GRANULAR FILMS**


We are now in the position to explain why disorder is irrelevant for the quantum phases of granular superconductors.
As mentioned in the introduction, JJAs are the closest ordered
Hamiltonian system on which to add positional and size disorder of the granules. This disorder then affects the behaviour of
the excitations in the spectrum of JJAs. In the topological insulator phase, disorder can indeed help to pin the excess bulk
charges and vortices around the integer-filling ground state
when g deviates from its central value g = 1. But even this
is not necessary; these excitations form a Wigner crystal [29],
exactly as can happen in the fractional quantum Hall effect

[44]. And, indeed, the topological insulator state is clearly detected in perfectly ordered JJAs [8]. Finally, the edge currents
are symmetry-protected and thus transparent to Anderson localization.
In the superinsulating phase, there are no U(1) charged excitations in the spectrum; they are confined (for large enough
samples). As such, disorder may influence the neutral excitations responsible for the thermal properties, but not the electric transport properties. The infinite resistance (even at finite
temperatures) is due exclusively to strong electric interactions
first preventing Bose condensation and then becoming linearly
confining by an instanton plasma. This is analogous to the situation in the disordered Schwinger model (1D QED), where
confinement is kinematic. First, one has to identify the correct
spectrum of neutral mesons and only then can one even speak
of disorder [13]. Of course, disorder can influence itself the
strength of the Coulomb interaction, but this leads only to a
renormalization of _EC_, i.e., of g [45].
Finally, dissipation by single-electron tunnelling is often
mentioned as a relevant phenomenon for granular superconductors. However, in the bosonic topological insulator phase,
the only possible dissipation is due to quantum phase slips,
since edge states are symmetry protected. In the superinsulating phase, there are no single electrons in the spectrum; they
are confined too. The only region where single-electron dissipation may become relevant is at higher temperatures, near



6


the thermal transition, where the string tension becomes very
small, in perfect analogy to the dual superconductors.


**WHY THE FULL PHASE STRUCTURE IS NOT YET SEEN**
**IN JJAS**


The g ≥ 1 ( _R_ ≤ _RQ_ ) segment of the bosonic topological insulator phase (failed superconductor) has been recently experimentally detected in Al/InAs JJAs, confirming that disorder is
irrelevant for this phase [8]. As mentioned in the introduction,
this has been achieved by a voltage gate, which depletes the
available charges for tunnelling, which amounts essentially
to varying _EJ_ at fixed _EC_ . Apparently, it is not possible to
reach low enough values of g by this procedure alone, without destroying superconductivity of the Al islands themselves.
In our opinion, one should consider a more regular arrangement of superconducting islands and exposed semiconductor
substrate, forming something like a checkerboard with, say,
the white squares being the superconducting islands and the
black squares the “dual” islands of exposed substrate, and introduce a second, separate procedure to govern the dual, vortex tunnelling coupling _EC_ . We predict that once this can be
achieved, the described phase structure will be exposed also
for completely ordered JJAs.


[1] Tinkham, M. Introduction to Superconductivity; Dover Publications: New York, NY, USA, 1996.

[2] Efetov, K.B. Phase transitions in granulated superconductors.

JETP **1980**, 51, 1015–1022.

[3] Haviland, D.; Liu, Y.; Goldman, A. Onset of superconductivity
in the two-dimensional limit. Phys. Rev. Lett. **1989**, 62, 2180–
2183.

[4] Fisher, M.P.A. Quantum phase transitions in disordered twodimensional superconductors. Phys. Rev. Lett. **1990**, 65, 923–
926.

[5] Sac´ep´e, B.; Dubouchet, T.; Chapelier, C.; Sanquer, M.; Ovadia, M.; Shahar, D.; Ioffe, L. Localization of preformed Cooper
pairs in disordered superconductors. Nat. Phys. **2011**, 7, 239–
244.

[6] Fazio, R.; Sch¨on, G. Charge and vortex dynamics in arrays of
tunnel junctions. Phys. Rev. **1991**, B43, 5307–5320.

[7] Fazio, R.; van der Zant, H. Quantum phase transitions and vortex dynamics in supercondcucing networks. Phys. Rep. **2001**,
355, 235–334.

[8] Bøttcher, C.; Nichele, F.; Kjaergaard, M.; Suominen, H.J.;
Shabani, J.; Palmstrøm, C.J.; Marcus, C.M. Superconducting, insulating and anomalous metallic regimes in a gated twodimensional semiconductor–superconductor array. Nat. Phys.
**2018**, 14, 1138–1145.

[9] Sac´ep´e, B.; Feigel’man, M.; Klapwijk, T.M. Quantum breakdown of superconductivity in low-dimensional materials. Nat.
Phys. **2020**, 16, 734–746.

[10] Anderson, P. Absence of diffusion in certain random lattices.

Phys. Rev. **1958**, 109, 1492–1505.

[11] Mezard, M.; Parisi, G.; Virasoro M.-A. Spin Glass Theory and

Beyond; World Scientific: singapore, 1993.




---PAGE BREAK---

[12] Minnhagen, P. The two-dimensional Coulomb gas, vortex unbinding and superfluid-superconducting films. Phys. Rep **1987**,
59, 1001–1066.

[13] Nandkishore, R.M.; Sondhi, S.L. Many body localization with
long range interactions. Phys. Rev. **2017**, X7, 041021.

[14] Golubev, D.S.; Zaikin, A.D. Quantum tunnelling of the order
parameter in superconducting nanowires. Phys. Rev. **2001**, B64,
014504.

[15] Arutyunov, K.Y.; Golubev, D.S.; Zaikin, A.D. Superconductivity in one dimension. Phys. Rep. **2008**, 464, 1–70.

[16] Coleman, S. _Aspects of Symmetry_ ; Cambridge University Press:
Cambridge, UK, 1985.

[17] Diamantini, M.C.; Sodano, P.; Trugenberger, C.A. Gauge theories of Josephson junction arrays. Nucl. Phys. **1996**, B474, 641–
677.

[18] Trugenberger, C.; Diamantini, M.C.; Poccia, N.; Nogueira,
F.S.; Vinokur, V.M. Magnetic monopoles and superinsulation
in Josephson junction arrays. Quant. Rep. **2020**, 2, 388–399.

[19] Deser, S.; Jackiw, R.; Templeton, S. Three-dimensional massive gauge theories. Phys. Rev. Lett. **1982**, 48, 975.

[20] Phillips, P.; Dalidovich, D. The elusive Bose metal. Science
**2003**, 302 243–247.

[21] Kapitulnik, A.; Kivelson, S.A.; Spivak, B. Anomalous metals:
Failed superconductors. Rev. Mod. Phys. **2109**, 91, 011002.

[22] Diamantini, M.C.; Mironov, A.Y.; Postolova, S.M.; Liu, X.;
Hao, Z.; Silevitch, D.M.; Vinokur, V.M. Bosonic topological intermediate state in the superconductor-insulator transition.
Phys. Lett. **2020**, A384, 126570.

[23] Lu, Y.-M.; Viswhwanath, A. Theory and classification of interacting integer topological phases in two dimensions: A ChernSimons approach. Phys. Rev. **2012**, B86 125119.

[24] Wang, C.; Senthil, T. Boson topological insulators: A window
into highly entangled quantum phases. Phys. Rev. **2013** **B87**,
235122.

[25] Zhang, X.; Palevski, A.; Kapitulnik, A. Anomalous metals:
From “failed superconductor” to “failed insulator”. Proc. Natl.
Acad. Sci. USA **2022**, 119 e2202496119.

[26] Diamantini, M.C.; Trugenberger, C.A.; Vinokur, V.M. How
planar superconductors cure their infrared divergences. JHEP
**2022**, 10 100.

[27] Diamantini, M.C.; Trugenberger, C.A.; Vinokur, V.M. Type-III
Superconductivity. Adv. Sci. **2023**, 1 2206523.

[28] Diamantini, M.C.; Trugenberger, C.A.; Vinokur, V.M. Topological nature of high-temperature superconductivity. Adv.



7


Quantum Technol. **2021**, 4, 2000135.

[29] Diamantini, M.C.; Trugenberger, C.A.; Vinokur, V.M.
Superconductor-to-insulator transition in absence of disorder.
Phys. Rev. **2021**, B103, 174516.

[30] Chalker, J.T.; Coddington, P.D. Percolation, quantum tunnelling and the integer Hall effect. J. Phys. **1988**, C21, 2665.

[31] Dunne, G.; Jackiw, R.; Trugenberger, C.A. Topological (ChernSimons) quantum mechanics. Phys. Rev. **1990**, D41, 661–666
(1990).

[32] Polyakov, A.M. Compact gauge fields and the infrared catastrophe. Phys. Lett. **1975**, 59, 82–84.

[33] Polyakov, A.M. _Fields_ _and_ _Strings_ ; Harwood Academic Publisher: Chur, Switzerland, 1987.

[34] Greensite, J. _An_ _Introduction_ _to_ _the_ _Confinement_ _Problem_ ;
Springer: Berlin/Heidelberg, Germany, 2011.

[35] Trugenberger, C.A. _Superinsulators,_ _Bose_ _Metals,_ _high-Tc_ _Su-_
_perconductors:_ _The_ _Quantum_ _Physics_ _of_ _Emergent_ _Magnetic_
_Monopoles_ ; World Scientific: Singapore, 2022.

[36] Vinokur, V.M. Superinsulator and quantum synchronization.

Nature **2008**, 452, 613–615.

[37] Baturina, T.I.; Vinokur, V.M. Superinsulator-superconductor
duality in two dimensions. Ann. Phys. **2013**, 331, 236–257.

[38] Diamantini, M.C.; Trugenberger, C.A.; Vinokur, V.M. Confinement and asymptotic freedom with Cooper pairs. Comm. Phys.
**2018**, 1, 77.

[39] Diamantini, M.C.; Trugenberger, C.A.; Vinokur, V.M. Quantum magnetic monopole condensate. Comm. Phys. **2021**, 4, 25.

[40] Sambandamurthy, G.; Engel, L.W.; Johansson, A.; Peled,
E.; Shahar, D. Experimental evidence for a collective insulating state in two-dimensional superconductors. Phys. Rev. Lett.
**2005**, 94 017003.

[41] Mironov, A.Y. Charge Bereszinskii-Kosterlitz-Thouless transition in superconducting NbTiN films. Sci. Rep. **2018**, C6 1181–
1203.

[42] Humbert, V.; Ortu˜no, M.; Somoza, A.M.; Berg´e, L.; Dumoulin,
L.; Marrache-Kikuchi, C.A. Overactivated transport in the localized phase of the superconductor-insulator transition. Nat.
Commun. **2021**, 12, 6733.

[43] Mironov, A.; Diamantini, M.C.; Trugenberger, C.A.; Vinokur,
V.M. Relaxation electrodynamics of superinsulators. Sci. Rep.
**2022**, 12, 19918.

[44] Kim K.-S.; Kivelson, S.A. The quantum Hall effect in absence
of disorder. NPJ Quantum Matter **2021**, 6, 22.

[45] Finkel’shtein, A.M. Superconducting transition temperature in
amorphous films. JETP Lett. **1987**, 45, 46–49.




---PAGE BREAK---

