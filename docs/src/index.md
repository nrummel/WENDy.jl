# WENDy.jl - Weak Form Estimation of Nonlinear Dynamics

This is the documentation of [WENDy].jl](https://github.com/nrummel/WENDy.jl).
WENDy is an algorithm that can estimate unknown parameters for ordinary differential equations given noisey data.

The set up for this algorithm is to assume that a physical system with state variable, $\boldsymbol{u} \in \mathbb{R}^D$, is governed by a system of ordinary differential equation with true parameters, $\mathbf{p}^* \in \mathbb{R}^J$:
```math
    \dot{\boldsymbol{u}}(t) = f(\boldsymbol{u}(t), t, \mathbf{p}^*)
```
The user has observed data of this system. The data has been corrupted by that measurement error:

```math
    \begin{align*}
    \text{\textbf{Additive Gaussian Case: } }\\ 
    \{\mathbf{u}_m &= \boldsymbol{u}(t_m, p^*) + \epsilon_m \}_{m=0}^M \\
    \epsilon &\sim \mathcal{N}(\mathbf{0}, \mathbb{I}_D)\\

    \text{\textbf{Multiplicative LogNormal Case: } }\\ 
    \{\mathbf{u}_m &= \boldsymbol{u}(t_m, p^*) \eta_m \}_{m=0}^M \\
    \log(\eta) &\sim \mathcal{N}(\mathbf{0}, \mathbb{I}_D)\\
    \end{align*}
```
The goal of the algorithm is that recover parameters $\hat{\mathbf{p}}$. In other words, we hope that if the one were to solve the system of differential equations with the estimated parameters then it would match the true state, then 
```math
    \frac{\| \boldsymbol{u}(t; \hat{\mathbf{p}}) - \boldsymbol{u}(t; \mathbf{p}^*)\|}{\|\boldsymbol{u}(t; \mathbf{p}^*)\|} \ll 1 \\
```
This is done by leveraging a estimated distribution of the weak form residual, $\mathbf{r}$, and then approximating a maximum likelihood estimate: 
```math
    \mathbf{S}(\mathbf{p})^{-\tfrac{1}{2}} \mathbf{r}(\mathbf{p}) \sim \mathcal{N}(0, \mathbb{I})
```
The work comes from the the MathBio Group at University of Colorado Boulder. For further reading find our paper at [arxiv link](https://arxiv.org/).

**Current features**

- Estimation of parameters for ordinary differential equations
- Supports: 
    - Additive Gaussian Noise and Multiplicative LogNormal Noise.
    - Ordinary differential equations that are nonlinear in parameters
    - Ordinary differential equations that are inhomogeneous 
    - Box constraints for parameter spaces
- Provides acceleration for problems that are linear in parameters 
- Directly calls robust optimization algorithms that are well suited to an non-convex problems.
- Creates efficient Julia functions for the likelihood function and its derivatives with minimal inputs from the end user.

## Index

```@contents
Pages = [
    "index.md",
    "getting_started.md"
]