## Solvers
abstract type AbstractWENDySolver<:Function end
# the iterates always start at p₀
_fromStart(p₀, iterates) = !isempty(iterates) && iterates[:,1] == p₀ ? iterates : hcat(p₀, iterates)
_solution(alg, p₀, p̂, iterations, iterates, retcode, start; û₀=nothing) =
    WENDySolution(p̂, û₀, iterations, _fromStart(p₀, iterates), retcode, time() - start, string(nameof(typeof(alg))))
# statuses of the underlying packages
_retcode(r::Optim.MultivariateOptimizationResults) = Optim.converged(r) ? :success :
    Optim.iteration_limit_reached(r) ? :maxiters : r.stopped_by.time_limit ? :timelimit : :failed
_retcode(status::Symbol) = status in (:first_order, :acceptable) ? :success : # JSOSolvers / AdaptiveRegularization
    status == :max_iter ? :maxiters : status == :max_time ? :timelimit : :failed
_retcode(rc::NonlinearSolve.ReturnCode.T) = NonlinearSolve.SciMLBase.successful_retcode(rc) ? :success :
    rc == NonlinearSolve.ReturnCode.MaxIters ? :maxiters : rc == NonlinearSolve.ReturnCode.MaxTime ? :timelimit : :failed
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
            return fill(NaN, prob.dims.J)
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
    out = nonlinearLeastSquares(
        (r, p) -> r .= C \ weakResidual(prob, p),
        (∇r, p) -> ∇r .= C \ weakJacobian(prob, p),
        prob.dims.KD, pnm1, params;
        reltol=params.nlsReltol, abstol=params.nlsAbstol, maxiters=params.nlsMaxiters
    )
    @info "  iterations = $(out.iterations)"
    out.p̂
end
struct IRLS<:AbstractWENDySolver end
function (alg::IRLS)(prob::WENDyProblem, p₀::AbstractVector{<:AbstractFloat}, params::WENDyParameters;
    ll::LogLevel=Warn, iterll::LogLevel=Warn, maxIt::Int=1000, kwargs...
)
    start = time()
    with_logger(ConsoleLogger(stderr,ll)) do
        reltol,abstol = params.optimReltol, params.optimAbstol
        (; J) = prob.dims
        iterates = zeros(J, 0)
        pnm1 = p₀
        for n = 1:maxIt
            @info "Iteration $n"
            pn = _irlsStep(prob, pnm1, params; ll=iterll)
            if any(isnan.(pn))
                @warn "Optimization method failed"
                return _solution(alg, p₀, pn, n, iterates, :failed, start)
            end
            iterates = hcat(iterates, pn)
            resNorm = norm(pnm1 - pn)
            if resNorm / norm(pnm1) < reltol || resNorm < abstol
                @info "  Convergence Criterion met: ‖pⁿ - pⁿ⁻¹‖ = $resNorm"
                return _solution(alg, p₀, pn, n, iterates, :success, start)
            end
            pnm1 = pn
        end
        @warn "Maxiteration met for IRLS"
        return _solution(alg, p₀, pnm1, maxIt, iterates, :maxiters, start)
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
    kwargs...
)
    res = optimize(
        objective, gradient!, hessian!,
        p₀, NewtonTrustRegion(),
        Optim_Options(
            x_reltol=params.optimReltol, x_abstol=params.optimAbstol, iterations=params.optimMaxiters, time_limit=params.optimTimelimit,
            store_trace=true, extended_trace=true
        )
    )
    (; p̂=res.minimizer, iterations=res.iterations, iterates=reduce(hcat, t.metadata["x"] for t in res.trace), retcode=_retcode(res))
end
## constrained
function _trustRegion_constrained(
    objective::Function, gradient!::Function, hessian!::Function, p₀::AbstractVector{<:Real}, params::WENDyParameters, constraints::AbstractVector{<:Tuple{<:Real,<:Real}},
    dims::WENDyDims; kwargs...
)
    (; J) = dims
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
        callback=_clbk,
        atol=params.optimAbstol,
        rtol=params.optimReltol,
        max_iter=params.optimMaxiters,
        max_time=params.optimTimelimit
    )
    (; p̂=out.solution, iterations=out.iter, iterates, retcode=_retcode(out.status))
end
struct TrustRegion<:AbstractWENDySolver end
function (alg::TrustRegion)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; kwargs...)
    start = time()
    objective, gradient!, hessian! = _secondOrder(prob, params)
    out = isnothing(prob.constraints) ?
        _trustRegion_unconstrained(objective, gradient!, hessian!, p₀, params; kwargs...) :
        _trustRegion_constrained(objective, gradient!, hessian!, p₀, params, prob.constraints, prob.dims; kwargs...)
    _solution(alg, p₀, out.p̂, out.iterations, out.iterates, out.retcode, start)
end
## adaptive regularization with cubics
struct ARCqK<:AbstractWENDySolver end
function (alg::ARCqK)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; kwargs...)
    start = time()
    objective, gradient!, hessian! = _secondOrder(prob, params)
    (; J) = prob.dims
    objgrad!(gradℓ, p) = (gradient!(gradℓ, p); (objective(p), gradℓ))
    iterates = zeros(J, 0)
    _clbk(nlp, solver, stats) = (iterates = hcat(iterates, stats.solution); nothing)
    nlp = NLPModel(p₀, objective; objgrad=objgrad!, hprod=_hessianVectorProduct(hessian!, J))
    out = ARCqKOp(
        nlp,
        verbose=:show_trace in keys(kwargs) ? kwargs.show_trace : false,
        callback=_clbk,
        atol=params.optimAbstol,
        rtol=params.optimReltol,
        max_iter=params.optimMaxiters,
        max_time=params.optimTimelimit,
    )
    _solution(alg, p₀, out.solution, out.iter, iterates, _retcode(out.status), start)
end
## Levenberg-Marquardt for min ½‖r(p)‖² given r!(r, p), ∇r!(∇r, p) and the length of r.
## Gives (; p̂, iterations, iterates, retcode), the iterates being every point the residual was evaluated at.
function nonlinearLeastSquares(r!::Function, ∇r!::Function, n::Int, p₀::AbstractVector{<:Real}, params::WENDyParameters;
    reltol::Real=params.optimReltol, abstol::Real=params.optimAbstol, maxiters::Int=params.optimMaxiters, kwargs...
)
    iterates = zeros(length(p₀), 0) # p₀ is p or [p; u₀]: this solver is not tied to one problem
    function _r!(r, p, ::Any)
        iterates = hcat(iterates, p)
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
    (; p̂=sol.u, iterations=sol.stats.nsteps, iterates, retcode=_retcode(sol.retcode))
end
## weak form least squares: min ½‖r(p)‖²
struct WLS<:AbstractWENDySolver end
function (alg::WLS)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; kwargs...)
    start = time()
    out = nonlinearLeastSquares(
        (r, p) -> weakResidual!(r, prob, p), (∇r, p) -> weakJacobian!(∇r, prob, p), prob.dims.KD, p₀, params
    )
    _solution(alg, p₀, out.p̂, out.iterations, out.iterates, out.retcode, start)
end
## hybrid : MLE (trust region) -> output error least squares
"""
    HybridTrustRegionOELS(oe::OutputErrorProblem)

Solver for a WENDyProblem that first optimizes the weak form negative log-likelihood with the trust region solver, then
uses the result to initialize the output error least squares problem oe (see OutputErrorProblem).
"""
struct HybridTrustRegionOELS<:AbstractWENDySolver
    oe::OutputErrorProblem
end
function (alg::HybridTrustRegionOELS)(prob::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; kwargs...)
    start = time()
    wendy = TrustRegion()(prob, p₀, params)
    oe = solve(alg.oe, wendy.p̂, params)
    # the output error iterates start where the trust region ended
    WENDySolution(oe.p̂, oe.û₀, wendy.iterations + oe.iterations, hcat(wendy.iterates, oe.iterates[:,2:end]),
        oe.retcode, time() - start, string(nameof(typeof(alg))))
end
"""
Solve the inverse problem for the unknown parameters
    solve(
        prob::WENDyProblem,
        p₀::AbstractVector{<:Real},
        params::WENDyParameters=WENDyParameters();
        alg::AbstractWENDySolver=TrustRegion(),
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
Returns a WENDySolution (the estimate p̂, iterations, iterates, retcode, time and the solver). For uncertainty
in the estimate see Ĉ, C_cons and C_∇²ℓ. For output error least squares on its own see OutputErrorProblem.
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
