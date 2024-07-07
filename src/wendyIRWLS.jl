# external dependencies
using Optimization, OptimizationNLopt
using NonlinearSolve
abstract type IRWLS_Iter end 
struct Linear_IRWLS_Iter <: IRWLS_Iter
    b₀::AbstractVector{<:AbstractFloat}
    G0::AbstractMatrix{<:AbstractFloat}
    RT::Function 
end 
function Linear_IRWLS_Iter(prob::AbstractWENDyProblem, params::WENDyParameters;ll::Logging.LogLevel=Logging.Warn)
    D = prob.D
    J = prob.J
    K = prob.K 
    G0 = zeros(K*D, J)
    _∇r = ∇rw(prob, params)
    _∇r(G0,zeros(J))
    _RT = Rw(prob, params)
    Linear_IRWLS_Iter(prob.b₀, G0, (Rᵀ,w)->_RT(Rᵀ,w))
end
function (m::Linear_IRWLS_Iter)(wnm1::AbstractVector{<:AbstractFloat};ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr,ll)) do 
        KD = length(m.b₀)
        Rᵀ = zeros(KD,KD)
        m.RT(Rᵀ, wnm1)
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
    _RT::Rw 
    _res!::rw
    _jac!::∇rw
    reltol::AbstractFloat
    maxiters::Int
end 
# constructor
function NLS_iter(prob::AbstractWENDyProblem, params::WENDyParameters; reltol::AbstractFloat=1e-8,maxiters::Int=10)
    _RT = Rw(prob, params)
    _res = rw(prob, params)
    _jac = ∇rw(prob, params)
    NLS_iter(prob.b₀, _RT, _res, _jac, reltol, maxiters)
end
# method
function (m::NLS_iter)(wnm1::AbstractVector{<:AbstractFloat};ll::Logging.LogLevel=Logging.Warn, _ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr,ll)) do 
        @info "  Running local optimization method"
        RT = m._RT(wnm1)
        b = RT \ m.b₀ 
        KD = length(b)

        resn!(
            r::AbstractVector,
            w::AbstractVector, 
            ::Any; 
            ll::Logging.LogLevel=_ll
        ) = m._res!(r, RT, b, w; ll=ll) 

        jacn!(
            jac::AbstractMatrix, 
            w::AbstractVector, 
            ::Any; 
            ll::Logging.LogLevel=_ll
        ) = m._jac!(jac, RT, w; ll=ll)
                
        prob = NonlinearLeastSquaresProblem(
            NonlinearFunction(
                resn!; 
                jac=jacn!, 
                resid_prototype=zeros(KD)
            ),
            wnm1
        )
        dt = @elapsed a = @allocations sol = solve(
            prob,
            LevenbergMarquardt();
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
function IRWLS(prob::AbstractWENDyProblem, p::WENDyParameters, iter::IRWLS_Iter, w0::AbstractVector{<:AbstractFloat}; ll::Logging.LogLevel=Logging.Warn, iterll::Logging.LogLevel=Logging.Warn, maxIt::Int=100, relTol::AbstractFloat=1e-10, trueIter::Union{IRWLS_Iter, Nothing}=nothing)
    with_logger(ConsoleLogger(stderr,ll)) do 
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
            if norm(resn) / norm(wnm1) < relTol
                resit = resit[:,1:n] 
                wit = wit[:,1:n] 
                @info "  Convergence Criterion met!"
                return wn, hcat(w0,wit), resit 
            end
            wnm1 = wn
        end
        @warn "Maxiteration met for IRWLS"
        return wn, hcat(w0,wit), resit 
    end
end 