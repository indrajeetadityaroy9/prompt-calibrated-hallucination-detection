## Subspace method based on neural networks for solving the partial differential equation

Zhaodong Xu [a,c], Zhiqiang Sheng [a,b,] _[∗]_


_aNational_ _Key_ _Laboratory_ _of_ _Computational_ _Physics,_ _Institute_ _of_ _Applied_ _Physics_ _and_
_Computational_ _Mathematics,_ _Beijing,_ _100088,_ _China_
_bHEDPS,_ _Center_ _for_ _Applied_ _Physics_ _and_ _Technology,_ _and_ _College_ _of_ _Engineering,_
_Peking_ _University,_ _Beijing,_ _100871,_ _China_
_cGraduate_ _School_ _of_ _China_ _Academy_ _of_ _Engineering_ _Physics,_ _Beijing_ _100088,_ _China_


**Abstract**


We present a subspace method based on neural networks (SNN) for solving
the partial differential equation with high accuracy. The basic idea of our
method is to use some functions based on neural networks as base functions
to span a subspace, then find an approximate solution in this subspace. We
design two special algorithms in the strong form of partial differential equation. One algorithm enforces the equation and initial boundary conditions
to hold on some collocation points, and another algorithm enforces _L_ [2] -norm
of the residual of the equation and initial boundary conditions to be 0. Our
method can achieve high accuracy with low cost of training. Moreover, our
method is free of parameters that need to be artificially adjusted. Numerical
examples show that the cost of training these base functions of subspace is
low, and only one hundred to two thousand epochs are needed for most tests.
The error of our method can even fall below the level of 10 _[−]_ [10] for some tests.
The performance of our method significantly surpasses the performance of
PINN and DGM in terms of the accuracy and computational cost.


_Keywords:_ Subspace, neural networks, base function, training epochs, least
squares.


_∗_ Corresponding author.
_Email_ _addresses:_ `xuzhaodong_math@163.com` (Zhaodong Xu),
`sheng_zhiqiang@iapcm.ac.cn` (Zhiqiang Sheng )


_Preprint_ _submitted_ _to_ _Elsevier_




---PAGE BREAK---

**1.** **Introduction**


Due to the rapid development of machine learning methods, the method
based on neural networks attracts more and more attention. Since neural
networks can be used to approximate any function, they can be used to
approximate the solution of partial differential equation(PDE). Researchers
have proposed many numerical methods based on neural networks, which
can be a promising approach for solving the partial differential equation.
Many machine learning methods for solving the partial differential equation are based on deep neural networks, such as physical information neural
networks (PINN)[22], deep Galerkin method (DGM)[24], deep Ritz method
(DRM)[7], and weak adversarial networks (WAN)[35]. The main difference
between these methods is in the construction of the loss function. For example, the loss functions of PINN and DGM are based on the _L_ [2] -norm of the
residuals of partial differential equation in strong form. The loss function of
the deep Ritz method is based on the energy functional corresponding to the
weak form of partial differential equation. The weak adversarial networks
method constructs a loss function by minimizing an operator norm induced
from the weak form of partial differential equation.
A method using deep learning approach for interface problem is proposed
in [30]. They reformulate the equation in variational form, use deep neural networks to represent the solution of the equation, and use a shallow
neural networks to approximate inhomogeneous boundary conditions. Another approach to solve this problem is proposed in [8], utilizing different
neural networks in different sub-domains since the solution may change dramatically across the interface. A multi-scale fusion networks is constructed
in [34]. This multi-scale fusion networks can better capture discontinuity,
thereby improving accuracy. A deep learning method based on PINN for
multi-medium diffusion problem is proposed in [33]. They add the interface
continuity condition as a loss term to the loss function and propose a domain separation strategy. A cusp-capturing PINN to solve interface problem
is proposed in [26], which introduces a cusp-enforced level set function to the
networks.
The PINN method is used to solve one dimensional and two dimensional
Euler equations that model high-speed aerodynamic flows in [17]. A conservative PINN on discrete sub-domains for nonlinear conservation laws is
proposed in [11]. Thermodynamically consistent PINN for hyperbolic systems is presented in [21]. PINN is used to solve the inverse problems in


2




---PAGE BREAK---

supersonic flows in [12]. A PINN method with equation weight is introduced
in [14], which introduces a weight such that the neural networks concentrate
on training the smoother parts of the solutions.
Due to the curse of dimensionality, deep neural networks methods have
been widely applied for solving high-dimensional partial differential equations. A type of tensor neural networks is introduced in [27, 28]. They
develop an efficient numerical integration method for the functions of the
tensor neural network, and prove the computational complexity to be the
polynomial scale of the dimension. A machine learning method solving highdimensional partial differential equation by using tensor neural networks and
a posteriori error estimator is proposed in [29]. They use a posteriori error
estimator as the loss function to update these parameters of tensor neural
networks.
In recent years, many methods based on shallow neural networks have
also received attention, such as methods based on extreme learning machine
(ELM)[10] and random feature methods. ELM-based methods for solving the
partial differential equation have been developed [3, 4, 5, 6, 13, 19, 23, 25]. A
numerical method for solving linear and nonlinear partial differential equations based on neural networks by combining the ideas of extreme learning
machines, domain decomposition, and local neural networks is proposed in

[3]. The weight/bias coefficients of all hidden layers in the local neural networks are all preset random values in the interval [ _−Rm, Rm_ ], where _Rm_ is a
hyperparameter, and only the weight coefficients of the output layer need to
be solved by the least squares method. A modified batch intrinsic plasticity
method for pre-training the random coefficients is proposed in [4] in order to
reduce the impact of the hyperparameter on accuracy. A method based on
the differential evolution algorithm to calculate the optimal or near-optimal
value of the hyperparameter is given in [6]. An approach for solving the
partial differential equation based on randomized neural networks and the
Petrov-Galerkin method is proposed in [23]. They allow for a flexible choice
of test functions, such as finite element basis functions. A local randomized neural networks method with discontinuous Galerkin methods for partial differential equation is developed in [25], which uses randomized neural
networks to approximate the solutions on sub-domains, and uses the discontinuous Galerkin method to glue them together. A local randomized neural
networks method for interface problems is developed in [13]. A discontinuous
capturing shallow neural networks method for the elliptical interface problem
is developed in [9].


3




---PAGE BREAK---

The random feature method for solving the partial differential equation
is proposed in [1]. This method is a natural bridge between traditional and
machine learning-based algorithms. They use random feature functions to
approximate the solution, collocation method to take care of the partial
differential equation, and penalty method to treat the boundary conditions.
A neural networks method which automatically satisfies boundary and initial
conditions is proposed in [15]. A deep mixed residual method for solving
the partial differential equation with high-order derivatives is proposed in

[16]. They rewrite a high-order partial differential equation into a first-order
system, and use the residual of first-order system as the loss function. A
random feature method for solving interface problem is proposed in [2], which
utilizes two sets of random feature functions on each side of the interface.
Although deep neural networks-based methods have achieved significant
progress in solving the partial differential equation, they suffer from some
limitations. The first limitation is that the accuracy of these methods is
unsatisfactory. A survey of related literatures shows that the error of most
deep neural networks-based methods is difficult to fall below the level of
10 _[−]_ [4] . Increasing number of training epochs does not significantly reduce
the error. Another limitation is low efficiency. The computational cost of
solving the partial differential equation with these methods based on deep
neural networks is extremely high. A lot of computational time is needed for
training. For example, some methods based on deep neural networks need
several hours to achieve certain accuracy, while traditional methods such as
finite element methods can achieve similar accuracy in just a few seconds.
Due to low accuracy and high computational cost, it is a challenge for these
methods based on deep neural networks to compete with traditional methods
for low dimensional problems.
The hyperparameter of ELM-based methods has a significant impact on
accuracy. The method with the optimal hyperparameter can achieve high
accuracy, however, the method with an inappropriate hyperparameter results
in very poor accuracy. Selecting an optimal hyperparameter is a challenging
problem.
In this paper, we present a subspace method based on neural networks
for solving the partial differential equation with high accuracy. The basic
idea of our method is to use some functions based on neural networks as
base functions to span a subspace, then find an approximate solution in this
subspace. Our method includes three steps. First, we give the neural networks architecture which includes input layer, hidden layer, subspace layer


4




---PAGE BREAK---

and output layer. Second, we train these base functions of subspace such
that the subspace has effective approximate capability to the solution space
of equation. Third, we find an approximate solution in the subspace to approximate the solution of the equation. We design two special algorithms in
the strong form of partial differential equation. One algorithm enforces the
equation and initial boundary conditions to hold on some collocation points,
we call this algorithm as SNN in discrete form(SNN-D). Another algorithm
enforces _L_ [2] -norm of the residual of the equation and initial boundary conditions to be 0, we call this algorithm as SNN in integral form(SNN-I). Our
method can achieve high accuracy, and the cost of training is low.
Our method is free of parameters (including hyperparameter and penalty
parameter) that need to be artificially adjusted. Different from ELM, we do
not introduce the hyperparameter since we train these parameters of neural
networks. Different from PINN and DGM, we do not use the initial boundary
conditions in the loss function, hence we do not need to introduce the penalty
parameter. When the number of hidden layer reduces to 0 and the number
of training epochs becomes 0, our method degenerates into ELM. When we
use the loss function including both the PDE loss term and initial boundary
loss term in our method, and omit the third step of our method, that is the
least squares method is not be used to update these parameters, with the
training epochs matching those of PINN or DGM, our method degenerates
into PINN or DGM. We use the Adam method to update neural network
parameters, while other methods are viable too. Additionally, we need to
solve an algebraic system by the least squares method.
Numerical examples show that the cost of training these base functions of
subspace is low, and only one hundred to two thousand epochs are needed for
most tests. The error of our method can even fall below the level of 10 _[−]_ [10] for
some tests. In general, the accuracy of SNN-D is higher than that of SNNI. Furthermore, the performance of our method significantly surpasses the
performance of PINN and DGM in terms of the accuracy and computational
cost.
The remainder of this paper is organized as follows. In section 2, we
describe the subspace method based on neural networks for solving the partial
differential equation. In section 3, we present some numerical examples to
test the performance of our method. At last, we give some conclusions.


5




---PAGE BREAK---

**2.** **Subspace** **method** **based** **on** **neural** **networks**


Consider the following equation:


_Au_ ( **x** ) = _f_ ( **x** ) in Ω _,_ (1)

_Bu_ ( **x** ) = _g_ ( **x** ) on _∂_ Ω _,_ (2)


where **x** = ( _x_ 1 _, x_ 2 _, · · ·, xd_ ) _[T]_, Ωis a bounded domain in R _[d]_, _∂_ Ωis the boundary
of Ω, _A_ and _B_ are the differential operators, _f_ and _g_ are given functions.


_2.1._ _Neural_ _networks_ _architecture_

In this section, we describe the neural networks architecture. For simplicity, we only describe the neural networks architecture with one-dimensional
output. Of course, this neural networks architecture can be used in the case
with _k_ -dimensional output.
The neural networks architecture consists of four key components, including an input layer, hidden layers, a subspace layer and an output layer.
The neural network employs some hidden layers to enrich the expressive capability of network. The subspace layer is essential for constructing a finite
dimension space that approximates the solution space of the equation. Figure
1 illustrates this specialized architecture.


Figure 1: The neural networks architecture.


6




---PAGE BREAK---

Let _K_ be the number of hidden layers, _n_ 1, _n_ 2, _· · ·_, _nK_ be the number
of neurons in each hidden layer, respectively. Let _M_ be the dimension of
subspace in the subspace layer, and _φj_ ( _j_ = 1 _,_ 2 _, · · ·, M_ ) be base functions
of subspace, and _ωj_ ( _j_ = 1 _,_ 2 _, · · ·, M_ ) be some coefficients related to base
functions. Denote _φ_ = ( _φ_ 1 _, φ_ 2 _, · · ·, φM_ ) _[T]_ and _ω_ = ( _ω_ 1 _, ω_ 2 _, · · ·, ωM_ ) _[T]_ .
The propagation process can be expressed as follows:

 **y** 0 = **x** _,_


 **y** = _σ_ ( _W_ **y** **b** ) _,_ _k_ = 1 _,_ 2 _, · · ·, K_ + 1 _,_









**y** 0 = **x** _,_
**y** _k_ = _σk_ ( _Wk ·_ **y** _k−_ 1 + **b** _k_ ) _,_ _k_ = 1 _,_ 2 _, · · ·, K_ + 1 _,_
_φ_ = **y** _K_ +1 _,_
_u_ = _φ · ω,_



where _Wk_ _∈_ R _[n][k][×][n][k][−]_ [1] and _bk_ _∈_ R _[n][k]_ are the weight and bias, respectively,
_n_ 0 = _d_ is the dimension of input and _nK_ +1 = _M_ is the dimension of
subspace. **x** _∈_ R _[d]_ is the input and _σ_ ( _·_ ) is the activation function. _θ_ =
_{W_ 1 _, · · ·, WK_ +1 _, b_ 1 _, · · ·, bK_ +1 _}_ is the set of parameters in neural networks,
_u_ ( _x_ ; _θ, ω_ ) is the output with respect to input _x_ with parameters _θ_ and _ω_ .
These weight and bias coefficients are initially randomly generated and
subsequently updated by minimizing the loss function _L_ ( **x** ; _θ, ω_ ). Usually,
this update is achieved by gradient descent method. In each iteration, these
parameters can be updated as follows:

_Wk_ _←_ _Wk −_ _η_ _[∂][L]_ [(] **[x]** [;] _[ θ, ω]_ [)] _,_

_∂Wk_

_bk_ _←_ _bk −_ _η_ _[∂][L]_ [(] **[x]** [;] _[ θ, ω]_ [)] _,_

_∂bk_


where _η_ _>_ 0 is the learning rate. For the gradient descent method, it is
needed to calculate the partial derivative of the loss function with respect to
network parameters, these partial derivatives are implemented through an
automatic differentiation mechanism in Pytorch and Tensorflow.
After the training process, _ω_ is determined by enforcing the equation
and boundary conditions to hold. Although _ω_ can be updated during training, this is not necessary as its final value is obtained by satisfying these
constraints. In fact, we find that there has no significant impact on the
numerical results.


_2.2._ _A_ _general_ _frame_ _of_ _subspace_ _method_ _based_ _on_ _neural_ _networks_
Now, we describe a general frame for the subspace method based on neural
networks for solving the partial differential equation.


7




---PAGE BREAK---

First, we construct the neural network architecture which includes input
layer, hidden layer, subspace layer and output layer. Then, we train the
base functions of subspace such that the subspace has effective approximate
capability to the solution space of equation. At last, we find an approximate
solution in the subspace to approximate the solution of the equation. This
general frame of SNN is as follows:


A general frame of SNN
1. Initialize nerual networks architecture, generate randomly _θ_, and
give _ω_ .
2. Update parameter _θ_ by minimizing the loss function, i.e. training
the base functions of the subspace layer _φ_ 1, _φ_ 2, _· · ·_, _φM_ .
3. Update parameter _ω_ and find an approximate solution in the
subspace to approximate the solution of the equation.


_Remark_ 2.1 _._ Many studies have shown that the imbalance between PDE loss
and initial boundary loss in the training process can lead to lose the accuracy
and increase significantly the cost of training. Weighted techniques have been
used to correct this imbalance[18, 31, 32]. However, how to determine these
weights is a challenging problem. We find that the initial boundary loss is
not important in the training process of base functions for many problems.
In order to overcome this challenge of selecting weights (penalty parameter),
we do not use the initial boundary conditions in step 2, hence we do not
need to introduce the penalty parameter. In fact, the information of PDE
is enough to train the base functions of subspace for many problems, and
the information of initial boundary conditions is not necessary. Of course, if
one does not care about the cost of training, the loss function including both
PDE loss and initial boundary loss can also be used in our method.


_Remark_ 2.2 _._ In step 2, we fix the parameter _ω_, and train the parameter _θ_
by minimizing the loss function. The aim is to derive suitable base functions
such that the subspace spanned by these base functions has effective approximate properties. In order to make a balance for the accuracy and efficiency,
it is not necessary to solve the minimization problem accurately.


_Remark_ 2.3 _._ In step 3, in order to find an approximate solution of the equation, we need to use the information of both PDE and initial boundary
conditions. We can obtain an algebraic system, and solve this system to get
_ω_ . In general, this algebraic system does not form a square matrix, it is
typically solved using the least squares method.


8




---PAGE BREAK---

