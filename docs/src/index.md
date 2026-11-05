# WENDy.jl - weak-form Estimation of Nonlinear Dynamics

This is the documentation of [WENDy.jl](https://github.com/nrummel/WENDy.jl). The work comes from the Applied Math Department at University of Colorado Boulder. For further reading read our paper, [WENDy for Nonlinear-in-Parameter ODEs](https://arxiv.org/abs/2502.08881).

## Current features

- Estimation of parameters for ordinary differential equations
- Supports: 
    - Additive Gaussian Noise and Multiplicative LogNormal Noise
    - Ordinary differential equations that are nonlinear in parameters
    - Ordinary differential equations that are inhomogeneous in time
    - Box constraints for parameter spaces
- Provides acceleration for problems that are linear in parameters 
- Directly calls robust optimization algorithms that are well suited to non-convex problems.
- Creates efficient Julia functions for the likelihood function and its derivatives with minimal inputs from the end user.

## Statement of Need 
Julia has a bountiful ecosystem of numerical solvers that can directly solve [ordinary differential equations](https://github.com/sciml/differentialequations.jl). Furthermore, considerable effort has been made to facilitate [automatic differentiation](https://github.com/JuliaDiff/ForwardDiff.jl). This supports downstream optimization routines that can be used for control, optimal design, and, more pertinently to this work, parameter estimation. However, in spite of these efficiencies, using forward simulation at every iteration of an optimization routine can rapidly increase the computational cost, especially in high-dimensional or stiff systems. For chaotic systems, using forward simulation also can introduce numerical instability. In contrast, weak-form methods do not suffer in this way. Thus, to tackle more challenging systems, alternative approaches such as weak-form methods become attractive. These methods do not rely on forward simulation of the differential equations to accomplish estimation. Instead, by optimizing the *equation error* rather than the *output error*, weak-form methods efficiently optimize the parameters irrespectively of the challenge of forward simulation of the system.

## Formal Problem Statement
WENDy is an algorithm that can estimate unknown parameters for ordinary differential equations given noisy data.

The set up for this algorithm is to assume that a physical system with state variable, $\boldsymbol{u} \in \mathbb{R}^D$, is governed by a system of ordinary differential equation with true parameters, $\mathbf{p}^* \in \mathbb{R}^J$:
```math
    \dot{\boldsymbol{u}}(t) = f(\boldsymbol{u}(t), t, \mathbf{p}^*)
```
The user has observed data of this system on a uniform grid, $$\{t_m, \mathbf{u}_m\}_{m=0}^M$$. The data has been corrupted by noise:
- **Additive Gaussian Case:**
```math
    \begin{align*}
        \{\mathbf{u}_m &= \boldsymbol{u}(t_m, p^*) + \epsilon_m \}_{m=0}^M \\
        \epsilon_m &\stackrel{iid}{\sim} \mathcal{N}(\mathbf{0}, \mathbb{I}_D)\\
    \end{align*}
```
- **Multiplicative LogNormal Case:**
```math
    \begin{align*}
        \{\mathbf{u}_m &= \boldsymbol{u}(t_m, p^*) \circ \eta_m \}_{m=0}^M \\
        \log(\eta) &\stackrel{iid}{\sim} \mathcal{N}(\mathbf{0}, \mathbb{I}_D)\\
    \end{align*}
```
*Note*: The Hadamard product $$\circ$$ is the element-wise multiplication on the two vectors. 

The goal of the algorithm is to recover unknown parameters $\mathbf{p}$. In other words, we hope that if one were to solve the system of differential equations with the estimated parameters then it would match the true state, then 
```math
    \frac{\| \boldsymbol{u}(t; \mathbf{p}) - \boldsymbol{u}(t; \mathbf{p}^*)\|}{\|\boldsymbol{u}(t; \mathbf{p}^*)\|} \ll 1 \\
```
This is done by leveraging an approximate distribution of the weak-form residual, $\mathbf{r}$, and then approximating a maximum likelihood estimate: 
```math
    \mathbf{S}(\mathbf{p})^{-\tfrac{1}{2}} \mathbf{r}(\mathbf{p}) \stackrel{approx}{\sim} \mathcal{N}(0, \mathbb{I})
```
## Comparable Methods
The most common approach to estimate the parameters of a differential equation is to solve a nonlinear least squares problem of the forward solved trajectory. This is referred to as *Output Error Least Squares* (OE-LS). By optimizing over both the parameters $$p$$ and the initial condition $$u_0$$:
```math
    \underset{u_0\in \mathbb{R}^D, p\in \mathbb{R}^J}{\operatorname{argmin}} \frac{1}{2} \sum \|u_m - \hat{u}(t_m; u_0, p  ) \|^2
``` 
one can obtain an estimate for the parameters of interest, but this optimization problem is often multimodal (nonconvex) even in a neighborhood around the true solutions. Thus, other approaches such as weak-form methods are well motivated.
This package provides convenience code to solve OE-LS. See the [example](@ref "Comparing to an Output Error Method") of using OE-LS on the Logistic Growth system.

One also can use Monte Carlo approaches, but unfortunately the underlying sampling will depend on direct simulation of the system. This causes these methods to be much more computationally expensive compared to the weak-form methods or OE-LS
## Acknowledgements 
Functions for the likelihood and its derivatives are formed analytically through symbolic computations using [Symbolics.jl](https://docs.sciml.ai/Symbolics/stable/). These functions are then used in second order optimization methods. While the likelihood is a scalar valued function, its computation relies on the derivatives of vector and matrix valued functions. Building and using efficient data structures to compute these derivative can rely on *vectorization* resulting large matrices with block structure from Kronecker products. In our implementation we instead use multidimensional array and define the operations in Einstein summation notation. These computations are then evaluated efficiently with [Tullio.jl](https://github.com/mcabbott/Tullio.jl?tab=readme-ov-file). Trust region solvers are provided by [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) and [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) for the constrained and unconstrained cases respectively. We note that our code also supports using the Adaptive Regularization Cubics variant (ARCqK) in the unconstrained case provide by [AdaptiveRegularization.jl](https://jso.dev/AdaptiveRegularization.jl/stable/). The trust region solvers and ARCqK usually produce similar results, but in our limited testing we found the trust region solvers work better in general. 

The [logistic growth example](@ref "Manually Specifying The Weak-form Method") demonstrates how to specify a specific weak-form method. Furthermore, for the WENDy-MLE method, one can further specify the optimization solver used to maximize the approximate likelihood function.

# Contributing or Getting Support
This code base is developed primarily by Nicholas Rummel who can be reached via [email](mailto:nicholas.rummel@colorado.edu). If you are interested in contributing please reach out directly. For support, we suggest you open a [GitHub issue](https://github.com/nrummel/WENDy.jl/issues).