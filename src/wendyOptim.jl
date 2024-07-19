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
function IRWLS(prob::WENDyProblem{LinearInParameters}, params::WENDyParameters, w0::AbstractVector{<:AbstractFloat}; ll::LogLevel=Warn, iterll::LogLevel=Warn, compareIters::Bool=false, maxIt::Int=1000, return_wits::Bool=false) where LinearInParameters
    with_logger(ConsoleLogger(stderr,ll)) do 
        reltol,abstol = params.optimReltol, params.optimAbstol
        @info "Building Iteration "
        iter = LinearInParameters ? Linear_IRWLS_Iter(prob, params) : NLS_iter(prob, params)
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
## Forward Solve Nonlinear Least Squares 
#
function FSNLS(l2::Function, ∇l2!::Function,Hl2!::Function, prob::WENDyProblem, params::WENDyParameters, w0::AbstractVector{<:Real}; fwdSlvAlg::OrdinaryDiffEqAlgorithm=Rosenbrock23(), OptAlg=OptimizationOptimJL.NewtonTrustRegion(), ll::LogLevel=Warn, return_wits::Bool=false, kwargs...)
    maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
    res = Optim.optimize(
        l2, ∇l2!, Hl2!, w0, OptAlg,
        Optim.Options(
            x_reltol=reltol, x_abstol=abstol, iterations=maxIt, time_limit=timelimit,
            store_trace=return_wits,extended_trace=return_wits
        );
    )
    what = res.minimizer
    iter = res.iterations
    return return_wits ? (what, iter, reduce(hcat, t.metadata["x"] for t in res.trace)) : (what, iter) 
end
## 
# function BFGS(m::Function, ∇m!::Function, w0::AbstractVector{<:Real}, params::WENDyParameters; kwargs...)
#     maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
#     res = Optim.optimize(
#         m, ∇m!, w0, Optim.BFGS(),
#         Optim.Options(
#             x_reltol=reltol, x_abstol=abstol, iterations=maxIt, time_limit=timelimit, kwargs...
#         )
#     )
#     what = res.minimizer
#     iter = res.iterations
#     return what, iter, res 
# end
## 
function trustRegion(m::Function, ∇m!::Function, Hm!::Function, w0::AbstractVector{<:Real}, params::WENDyParameters; return_wits::Bool=false, kwargs...)
    maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
    res = Optim.optimize(
        m, ∇m!, Hm!, w0, Optim.NewtonTrustRegion(),
        Optim.Options(
            x_reltol=reltol, x_abstol=abstol, iterations=maxIt, time_limit=timelimit,
            store_trace=return_wits,extended_trace=return_wits
        )
    )
    what = res.minimizer
    iter = res.iterations
    return return_wits ? (what, iter, reduce(hcat, t.metadata["x"] for t in res.trace)) : (what, iter) 
end
## 
function adaptiveCubicRegularization(m::Function, ∇m!::Function, Hm!::Function, w0::AbstractVector{<:Real}, params::WENDyParameters; return_wits::Bool=false,kwargs...)
    maxIt,reltol,abstol,timelimit = params.optimMaxiters, params.optimReltol, params.optimAbstol, params.optimTimelimit
    function fg!(grads, w)
        ∇m!(grads,w)
        m(w)
    end
    function Hm(w)
        Hm!(w)
        Hm!.H
    end
    ##
    opt = SFN.ARCOptimizer(
        length(w0);
        atol=abstol, rtol=reltol)
    stats, what, wits = minimize!(
        opt, copy(w0), m, fg!, Hm;
        itmax=maxIt, time_limit=timelimit
        # show_trace=true, show_every=10,
        # extended_trace=true
    )

    return return_wits ? (what, size(wits,2), wits) : (what, size(wits,2))
end
## 
# function saddleFreeNewton(m::Function, ∇m!::Function, Hm!::Function, w0::AbstractVector{<:Real}; reltol::AbstractFloat=1e-8, abstol::AbstractFloat=1e-8, kwargs...)
#     # opt = SFNOptimizer(length(w0),Symbol("EigenSolver"), linesearch=true)
#     opt = SFNOptimizer(length(w0),Symbol("GLKSolver"), linesearch=true)
#     stats, what = minimize!(
#         opt, copy(w0), _m_, fg!, _Hm_;
#         show_trace=true, show_every=10, extended_trace=true
#     )
#     return what, stat.itererations
# end
"Don't use auto diff doesnt play nice "
function _FSNLS_DiffEqParamEstim(prob::WENDyProblem, params::WENDyParameters, w0::AbstractVector{<:Real}; fwdSlvAlg::OrdinaryDiffEqAlgorithm=Rosenbrock23(), OptAlg=OptimizationOptimJL.NewtonTrustRegion(), kwargs...)
    maxIt,reltol,abstol = params.optimMaxiters, params.optimReltol, params.optimAbstol
    tRng = (prob.tt[1], prob.tt[end])
    odeprob = ODEProblem(prob.data.f!, prob.data.initCond, prob.data.tRng)
    # TODO: maybe set these tolerances in WENDyParameters
    cost_function = build_loss_objective(
        odeprob, fwdSlvAlg, L2Loss(prob.tt, prob.U),
        Optimization.AutoForwardDiff(),
        maxiters = maxIt, reltol=1e-12, abstol=1e-12, # these are passed to the ode solver
        verbose = false, 
    )
    optprob = Optimization.OptimizationProblem(cost_function, w0;)
    optsol = solve(
        optprob, OptAlg; 
        reltol=reltol, maxiters=maxIt, 
        # abstol=abstol, # got a warning Abstol not used by trust region here...
        kwargs...
    )
    return optsol.u, optsol.stats.iterations, optsol
end
## Wrap solvers so they can all be called with the same inputs
algos = (
    irwls=function _irwls(wendyProb, params, w0,m, ∇m!, Hm!,l2,∇l2!,Hl2!;return_wits::Bool=false )
        what, iters, wits = IRWLS(
            wendyProb, params, w0; return_wits=return_wits
        )
    end,
    tr=function _tr(wendyProb, params, w0,m, ∇m!, Hm!,l2,∇l2!,Hl2!;return_wits::Bool=false)
        trustRegion(m,∇m!,Hm!, w0,params; return_wits=return_wits)
    end,
    arc=function _arc(wendyProb, params, w0,m, ∇m!, Hm!,l2,∇l2!,Hl2!;return_wits::Bool=false)
        adaptiveCubicRegularization(
            m, ∇m!, Hm!, w0, params; 
            return_wits=return_wits
        )
    end,
    fsnls_tr=function _fsnls_tr(wendyProb, params, w0,m, ∇m!, Hm!,l2,∇l2!,Hl2!;return_wits::Bool=false)
        FSNLS(
            l2,∇l2!,Hl2!,
            wendyProb, params, w0;
            ll=Warn, OptAlg=Optim.NewtonTrustRegion(), 
            return_wits=return_wits
        )
    end,
)
# bfgs=function _bfgs(wendyProb, params, w0,m, ∇m!, Hm!,l2,∇l2!,Hl2! )
#     w_bfgs, iter_bfgs,_ = BFGS(m,∇m!, w0,params)
#     w_bfgs, iter_bfgs
# end,
# fsnls_bfgs=function _fsnls_bfgs(wendyProb, params, w0,m, ∇m!, Hm!,l2,∇l2!,Hl2!)
#     w_fsnls, iter_fsnls, res_fsnls  = FSNLS(
#         l2,∇l2!,Hl2!,
#         wendyProb, params, w0,
#         ll=Warn, OptAlg=Optim.LBFGS()
#     )
#     w_fsnls, iter_fsnls
# end,
