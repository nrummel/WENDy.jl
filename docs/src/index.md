# WENDy.jl - Weak Form Estimation of Nonlinear Dynamics

This is the documentation of [WENDy.jl](https://github.com/nrummel/WENDy.jl). The work comes from the the MathBio Group at University of Colorado Boulder. For further reading find our paper at [arxiv link](https://arxiv.org/).

## Current features

- Estimation of parameters for ordinary differential equations
- Supports: 
    - Additive Gaussian Noise and Multiplicative LogNormal Noise.
    - Ordinary differential equations that are nonlinear in parameters
    - Ordinary differential equations that are inhomogeneous in time
    - Box constraints for parameter spaces
- Provides acceleration for problems that are linear in parameters 
- Directly calls robust optimization algorithms that are well suited to an non-convex problems.
- Creates efficient Julia functions for the likelihood function and its derivatives with minimal inputs from the end user.

## Formal Problem Statement
WENDy is an algorithm that can estimate unknown parameters for ordinary differential equations given noisy data.

The set up for this algorithm is to assume that a physical system with state variable, $\boldsymbol{u} \in \mathbb{R}^D$, is governed by a system of ordinary differential equation with true parameters, $\mathbf{p}^* \in \mathbb{R}^J$:
```math
    \dot{\boldsymbol{u}}(t) = f(\boldsymbol{u}(t), t, \mathbf{p}^*)
```
The user has observed data of this system on a uniform grid, $$\{t_m \mathbf{u}_m\}_{m=0}^M$$. The data has been corrupted by noise:
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

The goal of the algorithm is that recover unknown parameters $\mathbf{p}$. In other words, we hope that if one were to solve the system of differential equations with the estimated parameters then it would match the true state, then 
```math
    \frac{\| \boldsymbol{u}(t; \mathbf{p}) - \boldsymbol{u}(t; \mathbf{p}^*)\|}{\|\boldsymbol{u}(t; \mathbf{p}^*)\|} \ll 1 \\
```
This is done by leveraging a estimated distribution of the weak form residual, $\mathbf{r}$, and then approximating a maximum likelihood estimate: 
```math
    \mathbf{S}(\mathbf{p})^{-\tfrac{1}{2}} \mathbf{r}(\mathbf{p}) \sim \mathcal{N}(0, \mathbb{I})
```

## Aknowledgements 
Functions for the likelihood and its derivatives are formed analytically through symbolic computations using [Symbolics.jl](https://docs.sciml.ai/Symbolics/stable/). These functions are then used in second order optimization methods. While the likelihood is a scalar valued function, its computation relies on the derivatives of vector and matrix valued functions. Building and using efficient data structures to compute these derivative can rely on ``vectorization'' resulting large matrices with block structure from Kronecker products. In our implementation we instead use multidimensional array and define the operations in Einstein summation notation. These computations are then evaluated efficiently with [Tullio.jl](https://github.com/mcabbott/Tullio.jl?tab=readme-ov-file). Trust region solvers are provided by [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) and [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) for the constrained and unconstrained cases respectively. We note that our code also supports using the Adaptive Regularization Cubics variant (ARCqK) in the unconstrained case provide by [AdaptiveRegularization.jl](https://jso.dev/AdaptiveRegularization.jl/stable/). The trust region solvers and ARCqK usually produce similar results, but in our limited testing we found the trust region solvers work better in general. 