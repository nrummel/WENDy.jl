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
    Rᵀ! = Rw(prob, params)
    r! = rw(prob, params)
    ∇r! = ∇rw(prob, params)

    NLS_iter(prob.b₀, Rᵀ!, r!, ∇r!, prob.nlsReltol,prob.nlsAbstol,  nlsMaxiters)
end
# method
function (m::NLS_iter)(wnm1::AbstractVector{<:AbstractFloat};ll::LogLevel=Warn, _ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr,ll)) do 
        @info "  Running local optimization method"
        try 
            m.Rᵀ!(m.Rᵀ!.R, wnm1)
        catch e 
            @show wnm1
            throw(e)
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
            maxiters=m.maxiters
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
function IRWLS(prob::WENDyProblem, params::WENDyParameters, w0::AbstractVector{<:AbstractFloat}; ll::LogLevel=Warn, iterll::LogLevel=Warn, compareIters::Bool=false)
    with_logger(ConsoleLogger(stderr,ll)) do 
        maxIt,reltol,abstol = params.optimMaxiters, params.optimReltol, params.optimAbstol
        @info "Building Iteration "
        iter = NLS_iter(prob, params)
        trueIter =compareIters ? Linear_IRWLS_Iter(prob, params) : nothing
        @info "Initializing the linearization least squares solution  ..."
        wit = zeros(J,maxIt)
        resit = zeros(J,maxIt)
        wnm1 = w0 
        wn = similar(w0)
        for n = 1:maxIt 
            @info "Iteration $n"
            dtNl = @elapsed aNl = @allocations wn = iter(wnm1;ll=iterll)
            if ! (typeof(wn)<:AbstractVector)
                @warn "Optimization method failed"
                return wn, hcat(w0, wit[:,1:n-1]), resit[:,1:n-1]
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
                return wn, hcat(w0,wit), resit 
            elseif resNorm < absTol
                resit = resit[:,1:n] 
                wit = wit[:,1:n] 
                @info """  
                  Convergence Criterion met: 
                    abstol: $resNorm < $abstol
                """
                return wn, hcat(w0,wit), resit 
            end
            wnm1 = wn
        end
        @warn "Maxiteration met for IRWLS"
        return wn, hcat(w0,wit), resit 
    end
end 
## Forward Solve Nonlinear Least Squares 
# we use [DiffEqParamEstim.jl]https://docs.sciml.ai/DiffEqParamEstim/stable
function FSNLS(prob::WENDyProblem, ode::ODESystem, w0::AbstractVector{<:Real}; fwdSlvAlg=FBDF(), OptAlg=BFGS(), kwargs...)
    init_cond = prob.U[:,1]
    t_rng = (prob.tt[1], prob.tt[end])
    params = [p => w0[j] for (j,p) in enumerate(parameters(ode))]
    p = ODEProblem(ode, init_cond, t_rng, params)

    cost_function = build_loss_objective(
        p, fwdSlvAlg, L2Loss(prob.tt, prob.U),
        Optimization.AutoForwardDiff(),
        maxiters = params.optimMaxiters, verbose = false
    )
    # Calling Optimization Routine
    optprob = Optimization.OptimizationProblem(cost_function, w0;)
    optsol = solve(optprob, OptAlg; kwargs...)
    return optsol.u, optsol
end
## 
function trustRegion(m::Function, ∇m!::Function, Hm!::Function, w0::AbstractMatrix{<:Real};reltol::AbstractFloat=1e-8, abstol::AbstractFloat=1e-8, kwargs...)
    res = Optim.optimize(
        m, ∇m!, Hm!, w0, Optim.NewtonTrustRegion(),
        Optim.Options(
            x_reltol=reltol, x_abstol=abstol, kwargs...
        )
    )
    what = res.minimizer
    iter = res.iterations
    return what, iter 
end
## 
function adaptiveCubicRegularization(m::Function, ∇m!::Function, Hm!::Function, w0::AbstractMatrix{<:Real}; reltol::AbstractFloat=1e-8, abstol::AbstractFloat=1e-8, kwargs...)
    function fg!(grads, w)
        _∇m!_(grads,w)
        _m_(w)
    end
    function _Hm_(w)
        _Hm!_(_Hm!_.H, w)
        _Hm!_.H
    end
    ##
    opt = SFN.ARCOptimizer(
        length(w0);
        atol=abstol ,rtol=reltol)
    @info "Calling ARC"
    @time stats, warc = minimize!(
        opt, copy(w0), _m_, fg!, _Hm_;
        show_trace=true, show_every=10,
        # extended_trace=true
    )
end
## 
function saddleFreeNewton(m::Function, ∇m!::Function, Hm!::Function, w0::AbstractMatrix{<:Real}; reltol::AbstractFloat=1e-8, abstol::AbstractFloat=1e-8, kwargs...)
    # opt = SFNOptimizer(length(w0),Symbol("EigenSolver"), linesearch=true)
    opt = SFNOptimizer(length(w0),Symbol("GLKSolver"), linesearch=true)
    stats, what = minimize!(
        opt, copy(w0), _m_, fg!, _Hm_;
        show_trace=true, show_every=10, extended_trace=true
    )
    return what, stat.itererations
end
## Code found here https://discourse.julialang.org/t/simple-timeout-of-function/99578/2
# usage @timeout sec begin code... end NaN 
macro timeout(seconds, expr, fail)
    quote
        tsk = @task $expr
        schedule(tsk)
        Timer($seconds) do timer
            istaskdone(tsk) || Base.throwto(tsk, InterruptException())
        end
        try
            fetch(tsk)
        catch _
            $fail
        end
    end
end