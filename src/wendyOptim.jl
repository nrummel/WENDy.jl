## Solvers
abstract type AbstractWENDySolver<:Function end
## IRLS: iteratively reweighted least squares
# With Cholesky S(pⁿ⁻¹) = CCᵀ held fixed, pⁿ minimizes ½‖C⁻¹r(p)‖²
function _irlsStep(prob::WENDyProblem, pnm1::AbstractVector{<:Real}, params::WENDyParameters; ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr,ll)) do
        ws = try
            _update!(prob.ws, pnm1)
        catch
            nothing
        end
        if isnothing(ws) || !(ws.Sfact isa Cholesky)
            @warn "The covariance calculation failed most likely because S ⊁ 0 "
            return NaN .* ones(size(pnm1))
        end
        _irlsStep(ws.wform, prob, ws.Sfact.L, pnm1, params)
    end
end
# r(p) = Gp - b, so this is a linear least squares problem
function _irlsStep(wform::LinearWeakForm, prob::WENDyProblem, C::AbstractMatrix{<:Real}, pnm1::AbstractVector{<:Real}, params::WENDyParameters)
    pn = (C \ wform.G) \ (C \ wform.b)
    @info "  objective_value = $(norm(C \ (wform.G*pn - wform.b))^2/2)"
    pn
end
function _irlsStep(wform::NonlinearWeakForm, prob::WENDyProblem, C::AbstractMatrix{<:Real}, pnm1::AbstractVector{<:Real}, params::WENDyParameters)
    @info "  Running local optimization method"
    pn, iter = nonlinearLeastSquares(
        (r, p) -> r .= C \ weakResidual(prob, p),
        (∇r, p) -> ∇r .= C \ weakJacobian(prob, p),
        prob.K*prob.D, pnm1, params;
        reltol=params.nlsReltol, abstol=params.nlsAbstol, maxiters=params.nlsMaxiters, return_iters=true
    )
    @info "  iterations = $iter"
    pn
end
struct IRLS<:AbstractWENDySolver end
function (alg::IRLS)(prob::WENDyProblem, p₀::AbstractVector{<:AbstractFloat}, params::WENDyParameters;
    ll::LogLevel=Warn, iterll::LogLevel=Warn, maxIt::Int=1000, return_iterates::Bool=false, kwargs...
)
    with_logger(ConsoleLogger(stderr,ll)) do
        reltol,abstol = params.optimReltol, params.optimAbstol
        J = length(p₀)
        iterates = zeros(J, 0)
        pnm1 = p₀
        for n = 1:maxIt
            @info "Iteration $n"
            pn = _irlsStep(prob, pnm1, params; ll=iterll)
            if any(isnan.(pn))
                @warn "Optimization method failed"
                return return_iterates ? (pn, n, hcat(p₀, iterates)) : pn
            end
            iterates = hcat(iterates, pn)
            resNorm = norm(pnm1 - pn)
            if resNorm / norm(pnm1) < reltol || resNorm < abstol
                @info "  Convergence Criterion met: ‖pⁿ - pⁿ⁻¹‖ = $resNorm"
                return return_iterates ? (pn, n, hcat(p₀, iterates)) : pn
            end
            pnm1 = pn
        end
        @warn "Maxiteration met for IRLS"
        return return_iterates ? (pnm1, maxIt, hcat(p₀, iterates)) : pnm1
    end
end
## Second order solvers over ℓ(p): objective(p) = ℓ(p), gradient!(gradℓ, p) = ∇ℓ(p), hessian!(hessℓ, p) = ∇²ℓ(p)
_secondOrder(prob::WENDyProblem, params::WENDyParameters) = (
    p -> ℓ(prob, p),
    (gradℓ, p) -> ∇ℓ!(gradℓ, prob, p),
    (hessℓ, p) -> ∇²ℓ!(hessℓ, prob, p; approx=params.approxHessian)
)
# These solvers accept a hessian vector product, so we build one that only recomputes H when p changes
function _hessianVectorProduct(hessian!::Function, J::Int)
    pH = fill(NaN, J)
    H = zeros(J, J)
    function Hv!(Hv, p, v; obj_weight=1.0)
        if pH != p
            hessian!(H, p)
            pH .= p
        end
        Hv .= H*v * obj_weight
        Hv
    end
end
## unconstrained
function _trustRegion_unconstrained(
    objective::Function, gradient!::Function, hessian!::Function, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_iterates::Bool=false, kwargs...
)
    res = optimize(
        objective, gradient!, hessian!,
        p₀, NewtonTrustRegion(),
        Optim_Options(
            x_reltol=params.optimReltol, x_abstol=params.optimAbstol, iterations=params.optimMaxiters, time_limit=params.optimTimelimit,
            store_trace=return_iterates, extended_trace=return_iterates
        )
    )
    p̂ = res.minimizer
    return return_iterates ? (p̂, res.iterations, reduce(hcat, t.metadata["x"] for t in res.trace)) : p̂
end
## constrained
function _trustRegion_constrained(
    objective::Function, gradient!::Function, hessian!::Function, p₀::AbstractVector{<:Real}, params::WENDyParameters, constraints::AbstractVector{<:Tuple{<:Real,<:Real}};
    return_iterates::Bool=false, kwargs...
)
    J = length(p₀)
    # this solver expects the gradient in this form
    objgrad!(gradℓ, p) = (gradient!(gradℓ, p); (objective(p), gradℓ))
    # store iterates in a callback
    iterates = zeros(J, 0)
    _clbk(nlp, slvr, stats) = (iterates = hcat(iterates, slvr.x); nothing)
    lower = [Float64(r[1]) for r in constraints]
    upper = [Float64(r[2]) for r in constraints]
    nlp = NLPModel(p₀, lower, upper, objective; objgrad=objgrad!, hprod=_hessianVectorProduct(hessian!, J))
    out = tron(
        nlp,
        verbose=:show_trace in keys(kwargs) ? 1 : 0,
        callback=return_iterates ? _clbk : (::Any, ::Any,::Any) -> nothing,
        atol=params.optimAbstol,
        rtol=params.optimReltol,
        max_iter=params.optimMaxiters,
        max_time=params.optimTimelimit
    )
    return return_iterates ? (out.solution, out.iter, iterates) : out.solution
end
struct TrustRegion<:AbstractWENDySolver end
function (alg::TrustRegion)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_iterates::Bool=false, kwargs...
)
    objective, gradient!, hessian! = _secondOrder(prob, params)
    if isnothing(prob.constraints)
        return _trustRegion_unconstrained(objective, gradient!, hessian!, p₀, params; return_iterates=return_iterates, kwargs...)
    end
    return _trustRegion_constrained(objective, gradient!, hessian!, p₀, params, prob.constraints; return_iterates=return_iterates, kwargs...)
end
## adaptive regularization with cubics
struct ARCqK<:AbstractWENDySolver end
function (alg::ARCqK)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_iterates::Bool=false, kwargs...
)
    objective, gradient!, hessian! = _secondOrder(prob, params)
    J = length(p₀)
    objgrad!(gradℓ, p) = (gradient!(gradℓ, p); (objective(p), gradℓ))
    iterates = zeros(J, 0)
    _clbk(nlp, solver, stats) = (iterates = hcat(iterates, stats.solution); nothing)
    nlp = NLPModel(p₀, objective; objgrad=objgrad!, hprod=_hessianVectorProduct(hessian!, J))
    out = ARCqKOp(
        nlp,
        verbose=:show_trace in keys(kwargs) ? kwargs.show_trace : false,
        callback=return_iterates ? _clbk : (::Any, ::Any,::Any) -> nothing,
        atol=params.optimAbstol,
        rtol=params.optimReltol,
        max_iter=params.optimMaxiters,
        max_time=params.optimTimelimit,
    )
    return return_iterates ? (out.solution, out.iter, iterates) : out.solution
end
## Levenberg-Marquardt for min ½‖r(p)‖² given r!(r, p), ∇r!(∇r, p) and the length of r
function nonlinearLeastSquares(r!::Function, ∇r!::Function, n::Int, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_iterates::Bool=false, return_iters::Bool=false,
    reltol::Real=params.optimReltol, abstol::Real=params.optimAbstol, maxiters::Int=params.optimMaxiters, kwargs...
)
    iterates = zeros(length(p₀), 0)
    function _r!(r, p, ::Any)
        return_iterates && (iterates = hcat(iterates, p))
        r!(r, p)
        nothing
    end
    _∇r!(∇r, p, ::Any) = (∇r!(∇r, p); nothing)
    sol = solve_lsq(
        NonlinearLeastSquaresProblem(NonlinearFunction(_r!; jac=_∇r!, resid_prototype=zeros(n)), p₀),
        LevenbergMarquardt();
        abstol=abstol,
        reltol=reltol,
        maxiters=maxiters,
        maxtime=params.optimTimelimit,
        verbose=false
    )
    p̂, iter = sol.u, sol.stats.nsteps
    return return_iterates ? (p̂, iter, iterates) : return_iters ? (p̂, iter) : p̂
end
## weak form least squares: min ½‖r(p)‖²
struct WLS<:AbstractWENDySolver end
(alg::WLS)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; return_iterates::Bool=false, kwargs...) =
    nonlinearLeastSquares(
        (r, p) -> weakResidual!(r, prob, p), (∇r, p) -> weakJacobian!(∇r, prob, p), prob.K*prob.D,
        p₀, params; return_iterates=return_iterates
    )
## hybrid : MLE (trust region) -> output error least squares
"""
    HybridTrustRegionOELS(oe::OutputErrorProblem)

Solver for a WENDyProblem that first optimizes the weak form negative log-likelihood with the trust region solver, then
uses the result to initialize the output error least squares problem oe (see OutputErrorProblem).
"""
struct HybridTrustRegionOELS<:AbstractWENDySolver
    oe::OutputErrorProblem
end
function (alg::HybridTrustRegionOELS)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_iterates::Bool=false, kwargs...
)
    if !return_iterates
        return solve(alg.oe, TrustRegion()(prob, p₀, params), params)
    end
    p̂_wendy, iter_wendy, iterates_wendy = TrustRegion()(prob, p₀, params; return_iterates=true)
    p̂_oe, iter_oe, iterates_oe = solve(alg.oe, p̂_wendy, params; return_iterates=true)
    # the output error iterates also contain u₀ when it is estimated
    if params.fsU0Free
        iterates_wendy = vcat(iterates_wendy, reduce(hcat, alg.oe.u₀ for _ in 1:size(iterates_wendy,2)))
    end
    return p̂_oe, iter_wendy + iter_oe, hcat(iterates_wendy, iterates_oe)
end
"""
Solve the inverse problem for the unknown parameters
    solve(
        prob::WENDyProblem,
        p₀::AbstractVector{<:Real},
        params::WENDyParameters=WENDyParameters();
        alg::AbstractWENDySolver=TrustRegion(),
        return_iterates::Bool=false,
        kwargs...
    )

# Arguments
- prob::WENDyProblem : An instance of a WENDyProblem for the ODE that you wish to estimate parameters for
- p₀::AbstractVector{<:Real} : Inital guess for the parameters
- params::WENDyParameters : hyperparameters for the WENDy Algorithm
- alg::AbstractWENDySolver=TrustRegion() : (optional) Choice of solver
    - WLS : weak form least squares
    - IRLS : WENDy generalized least squares solver via iterative reweighted least squares
    - TrustRegion : (default) optimize over the weak form negative log-likelihood with a trust region solver. This approximates the maximum likelhood estimator. Note: this is the only solver that will respect the constraints
    - ARCqK : optimize over the weak form negative log-likelihood with adaptive regularized cubics algorithm
    - HybridTrustRegionOELS(oe) : hybrid solver that first optimizes with the trust region solver then passes the result as an intialization to the output error least squares problem oe (an OutputErrorProblem)
- return_iterates::Bool=false : (optional) also return the number of iterations and the iterates, (p̂, iters, iterates)

For output error least squares on its own see OutputErrorProblem.
"""
function solve(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters=WENDyParameters(); alg::AbstractWENDySolver=TrustRegion(), forceAlg::Bool=false, kwargs...)
    if !forceAlg
        maxSizeOfData = sqrt(maximum(prob.X.^2))
        minSig = minimum(prob.σ)
        if minSig/maxSizeOfData < 1e-8
            @info "Switching the Weak Form Least Squares Problem, Noise is considered negligible..."
            alg = WLS()
        end
    end
    alg(prob, p₀, params; kwargs... )
end
