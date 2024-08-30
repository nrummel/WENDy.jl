## Iterative Reweighted Least Squares
abstract type IRWLS_Iter end 
struct Linear_IRWLS_Iter <: IRWLS_Iter
    b₀::AbstractVector{<:AbstractFloat}
    G0::AbstractMatrix{<:AbstractFloat}
    Rᵀ!::Function 
end 
function Linear_IRWLS_Iter(prob::WENDyProblem, params::WENDyParameters;ll::LogLevel=Warn)
    D = prob.D
    J = prob.J
    K = prob.K 
    G0 = zeros(K*D, J)
    ∇r! = GradientResidual(prob, params)
    ∇r!(G0, zeros(J))
    Rᵀ! = Covariance(prob, params)
    Linear_IRWLS_Iter(prob.b₀, G0, Rᵀ!)
end
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
## using NonlinearSolve to solve the NLS problem
# struct
struct NLS_iter <: IRWLS_Iter
    b₀::AbstractVector
    Rᵀ!::Covariance 
    r!::Residual
    ∇r!::GradientResidual
    reltol::AbstractFloat
    abstol::AbstractFloat
    maxiters::Int
end 
# constructor
function NLS_iter(prob::WENDyProblem, params::WENDyParameters)
    Rᵀ! = Covariance(prob, params)
    r! = Residual(prob, params)
    ∇r! = GradientResidual(prob, params)

    NLS_iter(prob.b₀, Rᵀ!, r!, ∇r!, params.nlsReltol,params.nlsAbstol,  params.nlsMaxiters)
end
# method
function (m::NLS_iter)(wnm1::AbstractVector{<:AbstractFloat};ll::LogLevel=Warn, _ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr,ll)) do 
        @info "  Running local optimization method"
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
        ) = m.r!(r, b, w; ll=ll, Rᵀ=m.Rᵀ!.R,) 

        jacn!(
            jac::AbstractMatrix, 
            w::AbstractVector, 
            ::Any; 
            ll::LogLevel=_ll
        ) = m.∇r!(jac, w; ll=ll, Rᵀ=m.Rᵀ!.R,)
                
        prob = NonlinearLeastSquaresProblem(
            NonlinearFunction(
                resn!; 
                jac=jacn!, 
                resid_prototype=zeros(KD)
            ),
            wnm1
        )
        dt = @elapsed a = @allocations sol = NonlinearSolve.solve(
            prob,
            NonlinearSolve.TrustRegion();
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
function IRWLS(prob::WENDyProblem{lip}, w0::AbstractVector{<:AbstractFloat}, params::WENDyParameters; 
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
##
abstract type CostFunction end

struct FirstOrderCostFunction <: CostFunction
    f::Function 
    ∇f!::Function 
end

struct SecondOrderCostFunction <: CostFunction
    f::Function 
    ∇f!::Function 
    Hf!::Function 
end
##
function bfgs_Optim(
    costFun::CostFunction, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)

    maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
    
    res = Optim.optimize(
        costFun.f, costFun.∇f!, w0, Optim.BFGS(),
        Optim.Options(
            x_reltol=reltol, x_abstol=abstol, iterations=maxIt, time_limit=timelimit, 
            store_trace=return_wits, extended_trace=return_wits,
            kwargs...
        )
    )
    what = res.minimizer
    iter = res.iterations
    return return_wits ? (what, iter, reduce(hcat, t.metadata["x"] for t in res.trace)) : (what, iter) 
end
##
function tr_Optim(
    costFun::SecondOrderCostFunction, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    # Unpack optimization params
    maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
    # Call algorithm
    res = Optim.optimize(
        costFun.f, costFun.∇f!, costFun.Hf!, 
        w0, Optim.NewtonTrustRegion(),
        Optim.Options(
            x_reltol=reltol, x_abstol=abstol, iterations=maxIt, time_limit=timelimit,
            store_trace=return_wits, extended_trace=return_wits
        )
    )
    # unpack results
    what = res.minimizer
    iter = res.iterations
    return return_wits ? (what, iter, reduce(hcat, t.metadata["x"] for t in res.trace)) : (what, iter) 
end
## 
function arc_SFN(
    costFun::SecondOrderCostFunction, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false, kwargs...
)
    maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
    function fg!(grads, w)
        costFun.∇f!(grads, w)
        costFun.f(w)
    end
    function Hm(w)
        costFun.Hf!(w)
        costFun.Hf!.H
    end
    ##
    opt = SFN.ARCOptimizer(
        length(w0);
        atol=abstol, 
        rtol=reltol
    )
    stats, what, wits = SFN.minimize!(
        opt, copy(w0), costFun.f, fg!, Hm;
        itmax=maxIt, time_limit=timelimit,
        kwargs...
    )

    return return_wits ? (what, size(wits,2), wits) : (what, size(wits,2))
end
##
function tr_JSO(
    costFun::SecondOrderCostFunction, w0::AbstractVector{<:Real}, params::WENDyParameters; 
    return_wits::Bool=false,kwargs...
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
    function _clbk(nlp, slvr::JSOSolvers.TrunkSolver, stats)
        wits_tr
        wits_tr = hcat(wits_tr, slvr.x)
        nothing
    end 
    # build the model to optimize then call solver
    nlp = NLPModel(
        w0,
        costFun.f;
        objgrad = fg!,
        hprod = Hv!
    )
    verbose = :show_trace in keys(kwargs) ? 1 : 0
    out = trunk(
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
function arc_JSO(
    costFun::SecondOrderCostFunction, w0::AbstractVector{<:Real}, params::WENDyParameters; 
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
    wits_arc = zeros(J, 0)
    function _clbk(nlp, slvr, stats)
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
    return return_wits ? (out.solution, out.iter, wits_arc) : (out.solution, out.iter) 
end