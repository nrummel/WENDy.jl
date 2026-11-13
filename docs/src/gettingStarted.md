# Getting Started 

The first step to using WENDy.jl is to build the [`WENDyProblem`](@ref) struct. The constructor will precompute all the necessary data and functions to run different optimizers to estimate the unknown parameters $$p$$.
The algorithm only requires a few inputs 
- `tt` : vector of equispaced times $$\{t_m\}_{m=0}^M$$.
- `U` : data matrix where each row is a state vector at each time $$\{\mathbf{u}_m = \boldsymbol{u}(t_m) + \epsilon_m\}_{m=0}^M$$. This makes the size of this matrix $$M+1\times D$$. 
- `f!` : right hand side function for the ordinary system of equations of the form `f!(du, u, p, t)`
- `J` : An integer specifying the length of the unknown parameters vector $$p$$.

```julia
wendyProb = WENDyProblem(tt, U, f!, J)
```

It is important to note that there are optional inputs that are keyword arguments that can help get the best performance from the algorithm. 
- `linearInParameters` : (Default `false`) If the ODE is linear-in-parameters (lip) then one can accelerate our weak-form algorithms by setting this to `true`. This precomputes the parts of the weak form that do not depend on the parameters, which makes function evaluations and the following optimization much faster. 
- `noiseDist` : (Default `Normal`) This specifies whether the noise is additive Gaussian `Normal` or multiplicative LogNormal `LogNormal`. With `LogNormal` the data are log-transformed (nonpositive data are dropped) and the right hand side is transformed accordingly. 
- `params` : (Default `WENDyParameters()`) One can update the hyper-parameters for WENDy via the [`WENDyParameters`](@ref) struct. Of particular interest are the parameters that inform the test function selection algorithm (see [Test Function Selection](@ref)) and those that inform the optimization algorithms (see [Optimization Parameters](@ref))
- `constraints` : (Default `Nothing`) WENDy supports linear box constraints for each parameter, ∀j ∈ [1, ⋯,J], ℓⱼ ≤ pⱼ ≤ uⱼ. Accepts constraints as a list of tuples, [(ℓ₁,u₁), ⋯]. Note: this only is compatible with the TrustRegion solver.
- `ll` : (Default `Warn`) Logging level 

The next step is to solve the WENDy problem. This is done by calling the [`solve`](@ref) function. An initial guess for the parameters, $$p_0$$, is necessary for the local optimization routines to run. It may be necessary to run the WENDy algorithm with multiple initial guesses for best results. 

```julia 
solve(wendyProb, p₀)
```

## Test Function Selection 
WENDy works by transforming the system of equations into its [weak-form](https://en.wikipedia.org/wiki/Weak_formulation). For this purpose, we select test functions of the form:
```math
	\varphi(t ; a)=C \exp \left(-\frac{\eta}{\left[1-(\tfrac{t}{m_t\Delta t} )^2\right]_{+}}\right)
```
where $m_t$ can be thought of as the radius of the support, the constant $C$ normalizes the test function such that $\|\varphi\|_2=1$, and $\eta$ is a shape parameter, and $[\cdot]_{+} = \max (\cdot, 0)$, so that $\varphi(t ; m_t\Delta t)$ is supported only on $[-m_t\Delta t, m_t\Delta t]$.

WENDy.jl attempts to automatically select multiple test functions with radii that balance preserving information while minimizing the effects due to noise. One may want to adjust the following in the `WENDyParameters`:
- `Kmax` - maximum number of test functions allow. This is a hard threshold, and may be aggressive in some cases. Set the Logging Level `ll` in the `WENDyProblem` constructor to `Info`. If the information lost in the SVD reduction gets below 50% then consider increasing the value of `Kmax`. In most cases, the radii of the test functions is more likely cause of poor results. 
- `radiusMinTime` which is a lower bound on the smallest $$m_t$$ considered by WENDy. This parameters has the same units as the `tt` passed when building a `WENDyProblem`. 
- `radiusMaxTime`which is a upper bound on the smallest $$m_t$$ considered by WENDy. This parameters has the same units as the `tt` passed when building a `WENDyProblem`. 

A good idea is to visualize the data yourself, and see how quickly the state variable is changing. It is ok to give WENDy a large range for the test function radii, but some adjustment may be necessary to get the best results. 

The other test function parameters are specific to the heuristic sub-algorithm used to build the test function matrix and in most cases are best left alone. 

## Optimization Parameters
WENDy makes use of several different optimization routines. For the default solver, one may want to adjust the following in the `WENDyParameters` for runtime or accuracy:
- `optimAbstol` : (Default 1e-8) Absolute Error Tolerance
- `optimReltol` : (Default 1e-8) Relative Error Tolerance
- `optimMaxiters` : (Default 500) Maximum Number of Iterations
- `optimTimelimit` : (Default 200.0) Time limit in seconds

## Solver Selection 
WENDy.jl allows for the user to make use of several different solvers. This is done through the keyword argument `solve(wendyProblem, p₀; alg = ...)` in [`solve`](@ref). 
- For the maximum likelihood estimator one should select either:
    - `TrustRegion()` : (Default) Allows for both constrained and unconstrained problems
    - `ARCqK()` : Only for the unconstrained, but may lead to a different solution
- `IRLS()` : This is (in general) a more computationally efficient solver that still incorporates covariance information from the weak-form residual, but frames a generalized least squares problem instead of a maximum likelihood estimator. 
- `WLS()` : This simple minimizes the norm of the weak-form residual by posing either linear or nonlinear least squares problem. This is the most appropriate if there is no noise in the system. 
- `HybridTrustRegionOELS(oe)` : Runs `TrustRegion()` and then uses the result to initialize the output error least squares problem `oe` (see below).

## Output Error Least Squares
A common technique is to minimize the output error, i.e. the difference between the data and a forward solve of the ODE. WENDy.jl provides a simple forward solve least squares solver for comparison as its own problem type. The forward solves use [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), which WENDy.jl only loads on request, so it must be loaded first (`using OrdinaryDiffEqRosenbrock` is enough for the default solver, Rodas5P):
```julia
using OrdinaryDiffEq
oe = OutputErrorProblem(tt, U, f!, J; params=params)
sol = solve(oe, p₀)
sol.p̂  # estimated parameters
sol.û₀ # estimated initial condition (nothing unless params.fsU0Free, the default)
```

## Evaluating the Weak Form Likelihood
The weak form negative log-likelihood and its derivatives can be evaluated directly with `ℓ(prob, p)`, `∇ℓ!(g, prob, p)` and `∇²ℓ!(H, prob, p)`, and the weak residual with `weakResidual!(r, prob, p)`, `weakJacobian!(∇r, prob, p)` and `weakCovariance(prob, p)`.
