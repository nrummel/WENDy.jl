# constructor 
function IRLSIter(prob::WENDyProblem, params::WENDyParameters)
    return lip ? LinearIRLSIter(prob, params) : NonLinearIRLSIter(prob, params)
end
# constructor
function LinearIRLSIter(prob::WENDyProblem, params::WENDyParameters; ll::LogLevel=Warn)
    D = prob.D
    J = prob.J
    K = prob.K 
    G0 = zeros(K*D, J)
    ∇r! = JacobianResidual(prob.data, params)
    ∇r!(G0, zeros(J))
    Rᵀ! = Covariance(prob.data, params)
    LinearIRLSIter(prob.data.b₀, G0, Rᵀ!)
end
# method
function (m::LinearIRLSIter)(wnm1::AbstractVector{<:AbstractFloat};ll::LogLevel=Warn)
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
# constructor
function NonLinearIRLSIter(prob::WENDyProblem, params::WENDyParameters)
    Rᵀ! = Covariance(prob.data, params)
    r! = Residual(prob.data, params)
    ∇r! = JacobianResidual(prob.data, params)

    NonLinearIRLSIter(prob.data.b₀, Rᵀ!, r!, ∇r!, params.nlsReltol,params.nlsAbstol,  params.nlsMaxiters)
end
# method
function (m::NonLinearIRLSIter)(wnm1::AbstractVector{<:AbstractFloat};ll::LogLevel=Warn)
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
            p::AbstractVector, 
            ::Any
        ) = m.r!(r, p, b, m.Rᵀ!.R) 

        jacn!(
            jac::AbstractMatrix, 
            p::AbstractVector, 
            ::Any
        ) = m.∇r!(jac, p, m.Rᵀ!.R)
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
struct IRLS<:AbstractWENDySolver end 

function (m::IRLS)(prob::WENDyProblem{lip}, p₀::AbstractVector{<:AbstractFloat}, params::WENDyParameters; 
    ll::LogLevel=Warn, iterll::LogLevel=Warn, compareIters::Bool=false, 
    maxIt::Int=1000, return_wits::Bool=false
) where lip
    with_logger(ConsoleLogger(stderr,ll)) do 
        reltol,abstol = params.optimReltol, params.optimAbstol
        @info "Building Iteration "
        iter = IRLSIter(prob, params)
        trueIter =compareIters ? LinearIRLSIter(prob, params) : nothing
        @info "Initializing the linearization least squares solution  ..."
        J = length(p₀)
        wit = zeros(J,maxIt)
        resit = zeros(J,maxIt)
        wnm1 = p₀ 
        wn = similar(p₀)
        for n = 1:maxIt 
            @info "Iteration $n"
            dtNl = @elapsed aNl = @allocations wn = iter(wnm1;ll=iterll)
            if ! (typeof(wn)<:AbstractVector) || any(isnan.(wn))
                @warn "Optimization method failed"
                return return_wits ? (wn, n, hcat(p₀, wit[:,1:n-1])) : (wn,n)
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
                return return_wits ? (wn, n, hcat(p₀,wit) ) : (wn,n)
            elseif resNorm < abstol
                resit = resit[:,1:n] 
                wit = wit[:,1:n] 
                @info """  
                  Convergence Criterion met: 
                    abstol: $resNorm < $abstol
                """
                return return_wits ? (wn, n, hcat(p₀,wit)) : (wn,n)
            end
            wnm1 = wn
        end
        @warn "Maxiteration met for IRLS"
        return return_wits ? (wn, maxIt, hcat(p₀,wit)) : (wn,maxIt)
    end
end 
## unconstrained
function _trustRegion_unconstrained(
    costFun::SecondOrderCostFunction, p₀::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    # Unpack optimization params
    maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
    # Call algorithm
    J = length(p₀)
    res = optimize(
        costFun.f, costFun.∇f!, costFun.Hf!, # f, g, h
        p₀, NewtonTrustRegion(), 
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
    costFun::SecondOrderCostFunction, p₀::AbstractVector{<:Real}, params::WENDyParameters, constraints::AbstractVector{Tuple{<:Real,<:Real}}; 
    return_wits::Bool=false, kwargs...
)
    J = length(p₀)
    # this solver expects the an explicit gradient in this form
    function fg!(g, p)
        costFun.∇f!(g, p)
        costFun.f(p), g
    end
    # This solver accepts Hvp operator, so we build one with memory
    mem_w = zeros(J)
    _H = zeros(J,J)
    function Hv!(h, p, v; obj_weight=1.0)
        if !all(mem_w .== p)
            # @show norm(p - mem_w)
            costFun.Hf!(_H, p)
            mem_w .= p
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
        p₀,
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

    return return_wits ? (out.solution, out.iter, wits_tr) : out.solution
end 
#
struct TrustRegion<:AbstractWENDySolver end 
function (m::TrustRegion)(wendyProb::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    if isnothing(wendyProb.constraints) 
        return _trustRegion_unconstrained(wendyProb.wnll, p₀, params;return_wits=return_wits,kwargs...)
    end 
    return _trustRegion_constrained(wendyProb.wnll, p₀, params, wendyProb.constraints; return_wits=return_wits,kwargs...)
end
##
struct ARCqK<:AbstractWENDySolver end 
function (m::ARCqK)(
    wendyProb::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    costFun = wendyProb.wnll
    J = length(p₀)
    # this solver expects the an explicit gradient in this form
    function fg!(g, p)
        costFun.∇f!(g, p)
        costFun.f(p), g
    end
    # This solver accepts Hvp operator, so we build one with memory
    mem_w = zeros(J)
    _H = zeros(J,J)
    function Hv!(h, p, v; obj_weight=1.0)
        if !all(mem_w .== p)
            # @show norm(p - mem_w)
            costFun.Hf!(_H, p)
            mem_w .= p
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
        p₀,
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
    p₀::AbstractVector{<:Real}, 
    params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)   
    J = length(p₀)
    wits = zeros(J,0)
    function r!(r,p, ::Any)
        if return_wits 
            wits = hcat(wits, p)
        end
        costFun.r!(r,p)
        nothing 
    end

    function ∇r!(J, p, ::Any)
        costFun.∇r!(J,p)
        nothing
    end

    sol = solve_lsq(
        NonlinearLeastSquaresProblem(
            NonlinearFunction( # {iip, specialize} 
                r!;
                jac=∇r!, 
                resid_prototype=zeros(costFun.KD)
            ),
            p₀
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
# Output Error Least Squares
struct OELS<:AbstractWENDySolver end 
(m::OELS)(wendyProb::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; 
return_wits::Bool=false, kwargs...) = nonlinearLeastSquares(wendyProb.oels, vcat(p₀, wendyProb.u₀), params; return_wits=return_wits, kwargs...)
# weak form least squares
struct WLS<:AbstractWENDySolver end 
(m::WLS)(wendyProb::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; 
return_wits::Bool=false, kwargs...) = nonlinearLeastSquares(wendyProb.wlsq, p₀, params; return_wits=return_wits, kwargs...)
# hybrid : MLE (trustRegion) -> oe-ls
struct HybridTrustRegionOELS<:AbstractWENDySolver end 
function (m::HybridTrustRegionOELS)(
    wendyProb::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    u₀ = wendyProb.u₀
    what_wendy, iter_wendy, wits_wendy = if return_wits
        TrustRegion()(wendyProb, p₀, params, return_wits=true)
    else 
        what_wendy, iter_wendy = TrustRegion()(wendyProb, p₀, params, return_wits=false)
        what_wendy, iter_wendy, nothing
    end 

    what_oels, iter_oels, wits_oels = if return_wits 
        OELS()(wendyProb, what_wendy, params, return_wits=true)
    else 
        what_oels, iter_oels = OELS()(wendyProb, vcat(what_wendy,u₀), params, return_wits=false)
        what_oels, iter_oels, nothing
    end 
    return return_wits ?  (what_oels, iter_wendy+iter_oels, hcat(vcat(wits_wendy, reduce(hcat, u₀ for _ in 1:size(wits_wendy,2))), wits_oels )) : what_oels
end
# hybrid : wls -> mle(trustRegion)
struct HybridWLSTrustRegion<:AbstractWENDySolver end 
function (m::HybridWLSTrustRegion)(
    wendyProb::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    what_wlsq, iter_wlsq, wits_wlsq = if return_wits 
        WLS()(wendyProb, p₀, params, return_wits=true)
    else 
        what_wlsq, iter_wlsq = WLS()(wendyProb, what_wendy, params, return_wits=false)
        what_wlsq, iter_wlsq, nothing
    end 
    what_wendy, iter_wendy, wits_wendy = if return_wits
        TrustRegion()(wendyProb, what_wlsq, params, return_wits=true)
    else 
        what_wendy, iter_wendy = TrustRegion()(wendyProb, p₀, params, return_wits=false)
        what_wendy, iter_wendy, nothing
    end 

    return return_wits ?  (what_wendy, iter_wlsq + iter_wendy, hcat(wits_wlsq,wits_wendy )) : what_wendy
end
""" 
Solve the inverse problem for the unknown parameters
    solve(
        wendyProb::WENDyProblem, 
        p₀::AbstractVector{<:Real},
        params::WENDyParameters=WENDyParameters(); 
        alg::Symbol=:trustRegion, 
        kwargs...
    )

# Arguments 
- wendyProblem::WENDyProblem : An instance of a WENDyProblem for the ODE that you wish to estimate parameters for 
- p₀::AbstractVector{<:Real} : Inital guess for the parameters
- params::WENDyParameters : hyperparameters for the WENDy Algorithm 
- alg::AbstractWENDySolver=TrustRegion() : Choice of solver 
    - OELS : output error least squares
    - WLS : weak form least squares 
    - IRLS : WENDy generalized least squares solver via iterative reweighted least squares 
    - TrustRegion : (default) optimize over the weak form negative log-likelihood with a trust region solver. This approximates the maximum likelhood estimator. Note: this is the only solver that will respect the constraints
    - ARCqK : optimize over the weak form negative log-likelihood with adaptive regularized cubics algorithm
    - HybridTrustRegionOELS : hybrid solver that first optimizes with the trust region solver then passes the result as an intialization to the output error least squares problem
    - HybridWLSTrustRegion : hybrid solver that first optimizes with the weak form least squares solver then passes the result as an intialization to the trust region weak form negative log-likelihood solver
"""
function solve(wendyProb::WENDyProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters=WENDyParameters(); alg::AbstractWENDySolver=TrustRegion(), kwargs...)
    alg(wendyProb, p₀, params; kwargs... )
end
