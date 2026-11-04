## Solvers
abstract type AbstractWENDySolver<:Function end
## IRLS: iteratively reweighted least squares
# With Cholesky S(pⁿ⁻¹) = CCᵀ held fixed, pⁿ minimizes ½‖C⁻¹r(p)‖²
function _irlsStep(prob::WENDyProblem{lip}, pnm1::AbstractVector{<:Real}, params::WENDyParameters; ll::LogLevel=Warn) where lip
    with_logger(ConsoleLogger(stderr,ll)) do
        ws = try
            _update!(prob.ws, pnm1)
        catch
            nothing
        end
        if isnothing(ws) || !(ws.F isa Cholesky)
            @warn "The covariance calculation failed most likely because S ⊁ 0 "
            return NaN .* ones(size(pnm1))
        end
        C = ws.F.L
        KD = length(ws.r)
        if lip
            # r(p) = Gp - b₀ so this is a linear least squares problem
            pn = (C \ prob.data.G) \ (C \ prob.data.b₀)
            @info "  objective_value = $(norm(C \ (prob.data.G*pn - prob.data.b₀))^2/2)"
            return pn
        end
        @info "  Running local optimization method"
        pn, iter = nonlinearLeastSquares(
            (r, p) -> ldiv!(C, weakResidual!(r, prob, p)),
            (∇r, p) -> ldiv!(C, weakJacobian!(∇r, prob, p)),
            KD, pnm1, params;
            reltol=params.nlsReltol, abstol=params.nlsAbstol, maxiters=params.nlsMaxiters, return_iters=true
        )
        @info "  iterations = $iter"
        return pn
    end
end
struct IRLS<:AbstractWENDySolver end
function (m::IRLS)(prob::WENDyProblem, p₀::AbstractVector{<:AbstractFloat}, params::WENDyParameters;
    ll::LogLevel=Warn, iterll::LogLevel=Warn, maxIt::Int=1000, return_wits::Bool=false, kwargs...
)
    with_logger(ConsoleLogger(stderr,ll)) do
        reltol,abstol = params.optimReltol, params.optimAbstol
        J = length(p₀)
        wits = zeros(J, 0)
        pnm1 = p₀
        for n = 1:maxIt
            @info "Iteration $n"
            pn = _irlsStep(prob, pnm1, params; ll=iterll)
            if any(isnan.(pn))
                @warn "Optimization method failed"
                return return_wits ? (pn, n, hcat(p₀, wits)) : pn
            end
            wits = hcat(wits, pn)
            resNorm = norm(pnm1 - pn)
            if resNorm / norm(pnm1) < reltol || resNorm < abstol
                @info "  Convergence Criterion met: ‖pⁿ - pⁿ⁻¹‖ = $resNorm"
                return return_wits ? (pn, n, hcat(p₀, wits)) : pn
            end
            pnm1 = pn
        end
        @warn "Maxiteration met for IRLS"
        return return_wits ? (pnm1, maxIt, hcat(p₀, wits)) : pnm1
    end
end
## Second order solvers over m(p): f(p) = m(p), g!(g, p) = ∇m(p), h!(H, p) = Hm(p)
_secondOrder(prob::WENDyProblem, params::WENDyParameters) = (
    p -> wnll(prob, p),
    (g, p) -> ∇wnll!(g, prob, p),
    (H, p) -> Hwnll!(H, prob, p; approx=params.approxHessian)
)
# These solvers accept a hessian vector product, so we build one that only recomputes H when p changes
function _hessianVectorProduct(h!::Function, J::Int)
    pH = fill(NaN, J)
    H = zeros(J, J)
    function Hv!(Hv, p, v; obj_weight=1.0)
        if pH != p
            h!(H, p)
            pH .= p
        end
        Hv .= H*v * obj_weight
        Hv
    end
end
## unconstrained
function _trustRegion_unconstrained(
    f::Function, g!::Function, h!::Function, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_wits::Bool=false, kwargs...
)
    res = optimize(
        f, g!, h!,
        p₀, NewtonTrustRegion(),
        Optim_Options(
            x_reltol=params.optimReltol, x_abstol=params.optimAbstol, iterations=params.optimMaxiters, time_limit=params.optimTimelimit,
            store_trace=return_wits, extended_trace=return_wits
        )
    )
    what = res.minimizer
    return return_wits ? (what, res.iterations, reduce(hcat, t.metadata["x"] for t in res.trace)) : what
end
## constrained
function _trustRegion_constrained(
    f::Function, g!::Function, h!::Function, p₀::AbstractVector{<:Real}, params::WENDyParameters, constraints::AbstractVector{<:Tuple{<:Real,<:Real}};
    return_wits::Bool=false, kwargs...
)
    J = length(p₀)
    # this solver expects the gradient in this form
    fg!(g, p) = (g!(g, p); (f(p), g))
    # store iterates in a callback
    wits = zeros(J, 0)
    _clbk(nlp, slvr, stats) = (wits = hcat(wits, slvr.x); nothing)
    ℓ = [Float64(r[1]) for r in constraints]
    u = [Float64(r[2]) for r in constraints]
    nlp = NLPModel(p₀, ℓ, u, f; objgrad=fg!, hprod=_hessianVectorProduct(h!, J))
    out = tron(
        nlp,
        verbose=:show_trace in keys(kwargs) ? 1 : 0,
        callback=return_wits ? _clbk : (::Any, ::Any,::Any) -> nothing,
        atol=params.optimAbstol,
        rtol=params.optimReltol,
        max_iter=params.optimMaxiters,
        max_time=params.optimTimelimit
    )
    return return_wits ? (out.solution, out.iter, wits) : out.solution
end
struct TrustRegion<:AbstractWENDySolver end
function (m::TrustRegion)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_wits::Bool=false, kwargs...
)
    f, g!, h! = _secondOrder(prob, params)
    if isnothing(prob.constraints)
        return _trustRegion_unconstrained(f, g!, h!, p₀, params; return_wits=return_wits, kwargs...)
    end
    return _trustRegion_constrained(f, g!, h!, p₀, params, prob.constraints; return_wits=return_wits, kwargs...)
end
## adaptive regularization with cubics
struct ARCqK<:AbstractWENDySolver end
function (m::ARCqK)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_wits::Bool=false, kwargs...
)
    f, g!, h! = _secondOrder(prob, params)
    J = length(p₀)
    fg!(g, p) = (g!(g, p); (f(p), g))
    wits = zeros(J, 0)
    _clbk(nlp, solver, stats) = (wits = hcat(wits, stats.solution); nothing)
    nlp = NLPModel(p₀, f; objgrad=fg!, hprod=_hessianVectorProduct(h!, J))
    out = ARCqKOp(
        nlp,
        verbose=:show_trace in keys(kwargs) ? kwargs.show_trace : false,
        callback=return_wits ? _clbk : (::Any, ::Any,::Any) -> nothing,
        atol=params.optimAbstol,
        rtol=params.optimReltol,
        max_iter=params.optimMaxiters,
        max_time=params.optimTimelimit,
    )
    return return_wits ? (out.solution, out.iter, wits) : out.solution
end
## Levenberg-Marquardt for min ½‖r(p)‖² given r!(r, p), ∇r!(∇r, p) and the length of r
function nonlinearLeastSquares(r!::Function, ∇r!::Function, n::Int, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_wits::Bool=false, return_iters::Bool=false,
    reltol::Real=params.optimReltol, abstol::Real=params.optimAbstol, maxiters::Int=params.optimMaxiters, kwargs...
)
    wits = zeros(length(p₀), 0)
    function _r!(r, p, ::Any)
        return_wits && (wits = hcat(wits, p))
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
    what, iter = sol.u, sol.stats.nsteps
    return return_wits ? (what, iter, wits) : return_iters ? (what, iter) : what
end
## weak form least squares: min ½‖r(p)‖²
struct WLS<:AbstractWENDySolver end
(m::WLS)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; return_wits::Bool=false, kwargs...) =
    nonlinearLeastSquares(
        (r, p) -> weakResidual!(r, prob, p), (∇r, p) -> weakJacobian!(∇r, prob, p), prob.K*prob.D,
        p₀, params; return_wits=return_wits
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
function (m::HybridTrustRegionOELS)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_wits::Bool=false, kwargs...
)
    if !return_wits
        return solve(m.oe, TrustRegion()(prob, p₀, params), params)
    end
    what_wendy, iter_wendy, wits_wendy = TrustRegion()(prob, p₀, params; return_wits=true)
    what_oe, iter_oe, wits_oe = solve(m.oe, what_wendy, params; return_wits=true)
    # the output error iterates also contain u₀ when it is estimated
    if params.fsU0Free
        wits_wendy = vcat(wits_wendy, reduce(hcat, m.oe.u₀ for _ in 1:size(wits_wendy,2)))
    end
    return what_oe, iter_wendy + iter_oe, hcat(wits_wendy, wits_oe)
end
## hybrid : weak form least squares -> MLE (trust region)
struct HybridWLSTrustRegion<:AbstractWENDySolver end
function (m::HybridWLSTrustRegion)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    return_wits::Bool=false, kwargs...
)
    if !return_wits
        return TrustRegion()(prob, WLS()(prob, p₀, params), params)
    end
    what_wlsq, iter_wlsq, wits_wlsq = WLS()(prob, p₀, params; return_wits=true)
    what_wendy, iter_wendy, wits_wendy = TrustRegion()(prob, what_wlsq, params; return_wits=true)
    return what_wendy, iter_wlsq + iter_wendy, hcat(wits_wlsq, wits_wendy)
end
"""
Solve the inverse problem for the unknown parameters
    solve(
        prob::WENDyProblem,
        p₀::AbstractVector{<:Real},
        params::WENDyParameters=WENDyParameters();
        alg::AbstractWENDySolver=TrustRegion(),
        return_wits::Bool=false,
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
    - HybridWLSTrustRegion : hybrid solver that first optimizes with the weak form least squares solver then passes the result as an intialization to the trust region weak form negative log-likelihood
- return_wits::Bool=false : (optional) also return the number of iterations and the iterates, (what, iters, wits)

For output error least squares on its own see OutputErrorProblem.
"""
function solve(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters=WENDyParameters(); alg::AbstractWENDySolver=TrustRegion(), forceAlg::Bool=false, kwargs...)
    if !forceAlg
        maxSizeOfData = sqrt(maximum(prob.data.X.^2))
        minSig = minimum(prob.data.sig)
        if minSig/maxSizeOfData < 1e-8
            @info "Switching the Weak Form Least Squares Problem, Noise is considered negligible..."
            alg = WLS()
        end
    end
    alg(prob, p₀, params; kwargs... )
end
