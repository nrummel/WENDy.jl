##
function Linear_IRWLS_Iter(prob::WENDyProblem, params::WENDyParameters;ll::LogLevel=Warn)
    D = prob.D
    J = prob.J
    K = prob.K 
    G0 = zeros(K*D, J)
    ∇r! = JacobianResidual(prob.data, params)
    ∇r!(G0, zeros(J))
    Rᵀ! = Covariance(prob.data, params)
    Linear_IRWLS_Iter(prob.data.b₀, G0, Rᵀ!)
end
##
function (m::Linear_IRWLS_Iter)(wnm1::AbstractVector{<:AbstractFloat};ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr,ll)) do 
        KD = length(m.b₀)
        Rᵀ = zeros(KD,KD)
        m.Rᵀ!(Rᵀ, wnm1)
        dt = @elapsed a = @allocations wn = (Rᵀ \ m.G0) \ (Rᵀ \ m.b₀)
        fHat = 1/2 * norm(Rᵀ \ (m.G0*wn - m.b₀))^2
        @info """ 
            iteration time  = $dt
            allocations     = $a
            objective_value = $(fHat)
        """
        return wn
    end
end
## constructor
function NLS_iter(prob::WENDyProblem, params::WENDyParameters)
    Rᵀ! = Covariance(prob.data, params)
    r! = Residual(prob.data, params)
    ∇r! = JacobianResidual(prob.data, params)

    NLS_iter(prob.data.b₀, Rᵀ!, r!, ∇r!, params.nlsReltol,params.nlsAbstol,  params.nlsMaxiters)
end
# method
function (m::NLS_iter)(wnm1::AbstractVector{<:AbstractFloat};ll::LogLevel=Warn, _ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr,ll)) do 
        @info "  Running local optimization method"
        # compute the covariance 
        try 
            m.Rᵀ!(wnm1)
        catch 
            @warn "The covariance calculation failed most likely because S ⊁ 0 "
            return NaN .* ones(size(wnm1))
        end
        b = m.Rᵀ!.R \ m.b₀ 
        KD = length(b)

        resn!(
            r::AbstractVector,
            w::AbstractVector, 
            ::Any; 
            ll::LogLevel=_ll
        ) = m.r!(r, w, b, m.Rᵀ!.R; ll=ll) 

        jacn!(
            jac::AbstractMatrix, 
            w::AbstractVector, 
            ::Any; 
            ll::LogLevel=_ll
        ) = m.∇r!(jac, w, m.Rᵀ!.R; ll=ll)
        # Solve nonlinear Least squares problem         
        dt = @elapsed a = @allocations sol = solve_lsq(
            NonlinearLeastSquaresProblem(
                NonlinearFunction(
                    resn!; 
                    jac=jacn!, 
                    resid_prototype=zeros(KD)
                ),
                wnm1
            ),
            LevenbergMarquardt();
            reltol=m.reltol,
            maxiters=m.maxiters,
            verbose=false
        )
        wn = sol.u
        res = zeros(KD)
        resn!(res, wn, nothing)
        fHat = 1/2*norm(res)^2
        iter = sol.stats.nsteps
        @info """ 
            iteration time  = $dt
            allocations     = $a
            iterations      = $(iter)
            ret code        = $(sol.retcode)
            objective_value = $(fHat)
        """
        return wn
    end
end
##
abstract type AbstractWENDySolver<:Function end 
struct IRWLS<:AbstractWENDySolver end 

function (m::IRWLS)(prob::WENDyProblem{lip}, w0::AbstractVector{<:AbstractFloat}, params::WENDyParameters; 
    ll::LogLevel=Warn, iterll::LogLevel=Warn, compareIters::Bool=false, 
    maxIt::Int=1000, return_wits::Bool=false
) where lip
    with_logger(ConsoleLogger(stderr,ll)) do 
        reltol,abstol = params.optimReltol, params.optimAbstol
        @info "Building Iteration "
        iter = lip ? Linear_IRWLS_Iter(prob, params) : NLS_iter(prob, params)
        trueIter =compareIters ? Linear_IRWLS_Iter(prob, params) : nothing
        @info "Initializing the linearization least squares solution  ..."
        J = length(w0)
        wit = zeros(J,maxIt)
        resit = zeros(J,maxIt)
        wnm1 = w0 
        wn = similar(w0)
        for n = 1:maxIt 
            @info "Iteration $n"
            dtNl = @elapsed aNl = @allocations wn = iter(wnm1;ll=iterll)
            if ! (typeof(wn)<:AbstractVector) || any(isnan.(wn))
                @warn "Optimization method failed"
                return return_wits ? (wn, n, hcat(w0, wit[:,1:n-1])) : (wn,n)
            end
            resn = wnm1-wn
            resit[:,n] .= resn
            wit[:,n] .= wn
            if !isnothing(trueIter)
                dtL = @elapsed aL = @allocations w_star = trueIter(wnm1)
                relErr = norm(w_star-wn) / norm(wn)
                @info """  Comparing to altnerate iteration
                    relative Error  = $relErr
                    this iteration 
                        $dtNl s, $aNl allocations
                    other iteration 
                        $dtL s, $aL allocations
                """
            end
            resNorm = norm(resn)
            relResNorm = resNorm / norm(wnm1)
            if relResNorm < reltol
                resit = resit[:,1:n] 
                wit = wit[:,1:n] 
                @info """  
                  Convergence Criterion met: 
                    reltol: $relRes < $reltol
                """
                return return_wits ? (wn, n, hcat(w0,wit) ) : (wn,n)
            elseif resNorm < abstol
                resit = resit[:,1:n] 
                wit = wit[:,1:n] 
                @info """  
                  Convergence Criterion met: 
                    abstol: $resNorm < $abstol
                """
                return return_wits ? (wn, n, hcat(w0,wit)) : (wn,n)
            end
            wnm1 = wn
        end
        @warn "Maxiteration met for IRWLS"
        return return_wits ? (wn, maxIt, hcat(w0,wit)) : (wn,maxIt)
    end
end 
## unconstrained
function _trustRegion_unconstrained(
    costFun::SecondOrderCostFunction, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    # Unpack optimization params
    maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
    # Call algorithm
    J = length(w0)
    res = optimize(
        costFun.f, costFun.∇f!, costFun.Hf!, # f, g, h
        w0, NewtonTrustRegion(), 
        Optim_Options(
            x_reltol=reltol, x_abstol=abstol, iterations=maxIt, time_limit=timelimit, 
            store_trace=return_wits, extended_trace=return_wits
        )
    )
    # unpack results
    what = res.minimizer
    iter = res.iterations
    return return_wits ? (what, iter, reduce(hcat, t.metadata["x"] for t in res.trace)) : what
end
## constrained
function _trustRegion_constrained(
    costFun::SecondOrderCostFunction, w0::AbstractVector{<:Real}, params::WENDyParameters, constraints::AbstractVector{Tuple{<:Real,<:Real}}; 
    return_wits::Bool=false, kwargs...
)
    J = length(w0)
    # this solver expects the an explicit gradient in this form
    function fg!(g, w)
        costFun.∇f!(g, w)
        costFun.f(w), g
    end
    # This solver accepts Hvp operator, so we build one with memory
    mem_w = zeros(J)
    _H = zeros(J,J)
    function Hv!(h, w, v; obj_weight=1.0)
        if !all(mem_w .== w)
            # @show norm(w - mem_w)
            costFun.Hf!(_H, w)
            mem_w .= w
        end
        h .= _H*v * obj_weight
        h
    end
    # store iteratios in a callback
    wits_tr = zeros(J, 0)
    function _clbk(nlp, slvr, stats)
        wits_tr
        wits_tr = hcat(wits_tr, slvr.x)
        nothing
    end 
    ℓ = [Float64(r[1]) for r in constraints]
    u = [Float64(r[2]) for r in constraints]
    # build the model to optimize then call solver
    nlp = NLPModel(
        w0,
        ℓ,
        u,
        costFun.f;
        objgrad = fg!,
        hprod = Hv!,
    )

    verbose = :show_trace in keys(kwargs) ? 1 : 0
    out = tron(
        nlp,
        verbose=verbose,
        callback= return_wits ? _clbk : (::Any, ::Any,::Any) -> nothing,
        atol=params.optimAbstol,
        rtol=params.optimReltol,
        max_iter=params.optimMaxiters,
        max_time=params.optimTimelimit
    )

    return return_wits ? (out.solution, out.iter, wits_tr) : (out.solution, out.iter) 
end 
#
struct TrustRegion<:AbstractWENDySolver end 
function (m::TrustRegion)(wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    if isnothing(wendyProb.constraints) 
        return _trustRegion_unconstrained(wendyProb.wnll, w0, params;return_wits=return_wits,kwargs...)
    end 
    return _trustRegion_constrained(wendyProb.wnll, w0, params, wendyProb.constraints; return_wits=return_wits,kwargs...)
end
##
struct ARCqK<:AbstractWENDySolver end 
function (m::ARCqK)(
    wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    costFun = wendyProb.wnll
    J = length(w0)
    # this solver expects the an explicit gradient in this form
    function fg!(g, w)
        costFun.∇f!(g, w)
        costFun.f(w), g
    end
    # This solver accepts Hvp operator, so we build one with memory
    mem_w = zeros(J)
    _H = zeros(J,J)
    function Hv!(h, w, v; obj_weight=1.0)
        if !all(mem_w .== w)
            # @show norm(w - mem_w)
            costFun.Hf!(_H, w)
            mem_w .= w
        end
        h .= _H*v * obj_weight
        h
    end
    # store iteratios in a callback
    wits_arc = zeros(J, 0)
    function _clbk(nlp, solver, stats)
        wits_arc = hcat(wits_arc, nlp.x)
        nothing
    end 
    # build the model to optimize then call solver
    nlp = NLPModel(
        w0,
        costFun.f;
        objgrad = fg!,
        hprod = Hv!
    )
    verbose = :show_trace in keys(kwargs) ? kwargs.show_trace : false
    out = ARCqKOp(
        nlp, 
        verbose = verbose, 
        callback = return_wits ? _clbk : (::Any, ::Any,::Any) -> nothing,
        atol=params.optimAbstol,
        rtol=params.optimReltol,
        max_iter=params.optimMaxiters,
        max_time=params.optimTimelimit,
    )
    return return_wits ? (out.solution, out.iter, wits_arc) : out.solution
end
## solver for nonlinear least squares problems
function nonlinearLeastSquares(costFun::LeastSquaresCostFunction,
    w0::AbstractVector{<:Real}, 
    params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)   
    J = length(w0)
    wits = zeros(J,0)
    function r!(r,w, ::Any)
        if return_wits 
            wits = hcat(wits, w)
        end
        costFun.r!(r,w)
        nothing 
    end

    function ∇r!(J, w, ::Any)
        costFun.∇r!(J,w)
        nothing
    end

    sol = solve_lsq(
        NonlinearLeastSquaresProblem(
            NonlinearFunction( # {iip, specialize} 
                r!;
                jac=∇r!, 
                resid_prototype=zeros(costFun.KD)
            ),
            w0
        ),
        LevenbergMarquardt();
        abstol=params.optimAbstol,
        reltol=params.optimReltol,
        maxiters=params.optimMaxiters,
        maxtime=params.optimTimelimit,
        verbose=false
    )
    what = sol.u
    iter = sol.stats.nsteps
    return return_wits ? (what, iter, wits) : what
end
# forward solve nonlinear least squares
struct FSNLS<:AbstractWENDySolver end 
(m::FSNLS)(wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters; 
return_wits::Bool=false, kwargs...) = nonlinearLeastSquares(wendyProb.fslsq, vcat(w0, wendyProb.u₀), params; return_wits=return_wits, kwargs...)
# weak form nonlinear least squares
struct WLSQ<:AbstractWENDySolver end 
(m::WLSQ)(wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters; 
return_wits::Bool=false, kwargs...) = nonlinearLeastSquares(wendyProb.wlsq, w0, params; return_wits=return_wits, kwargs...)
# hybrid : tr -> fsnls
struct HybridTrustRegionFSNLS<:AbstractWENDySolver end 
function (m::HybridTrustRegionFSNLS)(
    wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    u₀ = wendyProb.u₀
    what_wendy, iter_wendy, wits_wendy = if return_wits
        TrustRegion()(wendyProb, w0, params, return_wits=true)
    else 
        what_wendy, iter_wendy = TrustRegion()(wendyProb, w0, params, return_wits=false)
        what_wendy, iter_wendy, nothing
    end 

    what_fslsq, iter_fslsq, wits_fslsq = if return_wits 
        FSNLS()(wendyProb, what_wendy, params, return_wits=true)
    else 
        what_fslsq, iter_fslsq = FSNLS()(wendyProb, vcat(what_wendy,u₀), params, return_wits=false)
        what_fslsq, iter_fslsq, nothing
    end 
    return return_wits ?  (what_fslsq, iter_wendy+iter_fslsq, hcat(vcat(wits_wendy, reduce(hcat, u₀ for _ in 1:size(wits_wendy,2))), wits_fslsq )) : what_fslsq
end
# hybrid : wlsq -> tr
struct HybridWLSQTrustRegion<:AbstractWENDySolver end 
function (m::HybridWLSQTrustRegion)(
    wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    what_wlsq, iter_wlsq, wits_wlsq = if return_wits 
        WLSQ()(wendyProb, w0, params, return_wits=true)
    else 
        what_wlsq, iter_wlsq = WLSQ()(wendyProb, what_wendy, params, return_wits=false)
        what_wlsq, iter_wlsq, nothing
    end 
    what_wendy, iter_wendy, wits_wendy = if return_wits
        TrustRegion()(wendyProb, what_wlsq, params, return_wits=true)
    else 
        what_wendy, iter_wendy = TrustRegion()(wendyProb, w0, params, return_wits=false)
        what_wendy, iter_wendy, nothing
    end 

    return return_wits ?  (what_wendy, iter_wlsq + iter_wendy, hcat(wits_wlsq,wits_wendy )) : what_wendy
end
""" 
    solve(wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters=WENDyParameters(); alg::Symbol=:trustRegion, kwargs...)

# Arguments 
- wendyProblem::WENDyProblem : An instance of a WENDyProblem for the ODE that you wish to estimate parameters for 
- w0::AbstractVector{<:Real} : Inital guess for the parameters
- params::WENDyParameters : hyperparameters for the WENDy Algorithm 
- alg::AbstractWENDySolver=TrustRegion() : Choice of solver 
    - FSNLS : forward solve nonlinear least squares 
    - WLSQ : weak form least squares 
    - IRWLS : WENDy generalized least squares solver via iterative reweighted east squares 
    - TrustRegion : (default) optimize over the weak form negative log-likelihood with a trust region solver (not this is the only solver that will respect the constraints)
    - ARCqK : optimize over the weak form negative log-likelihood with adaptive regularized cubics algorithm
    - HybridTrustRegionFSNLS : hybrid solver that first optimizes with the trust region solver then passes the result as an intialization to the forward solve least squares solver 
    - HybridWLSQTrustRegion : hybrid solver that first optimizes with the weak form least squares solver then passes the result as an intialization to the trust region weak form negative log-likelihood solver
"""
function solve(wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters=WENDyParameters(); alg::AbstractWENDySolver=TrustRegion(), kwargs...)
    alg(wendyProb, w0, params; kwargs... )
end
