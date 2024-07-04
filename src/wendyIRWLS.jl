# external dependencies
using Optimization, OptimizationNLopt
using NonlinearSolve
abstract type IRWLS_Iter end 
struct Linear_IRWLS_Iter <: IRWLS_Iter
    b0::AbstractVector{<:AbstractFloat}
    G0::AbstractMatrix{<:AbstractFloat}
    RT::Function 
end 
function Linear_IRWLS_Iter(prob::AbstractWENDyProblem, params::WENDyParameters;ll::Logging.LogLevel=Logging.Warn)
    D = prob.D
    J = prob.J
    K = prob.K 
    G0 = ∇res(Matrix{Float64}(I,K*D,K*D),prob.U,prob.V,prob.jacwf!,zeros(J);ll=ll)
    _RT = RTFun(prob, params)
    Linear_IRWLS_Iter(prob.b0, G0, w->_RT(w))
end
function (m::Linear_IRWLS_Iter)(wnm1::AbstractVector{<:AbstractFloat};ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr,ll)) do 
        RT = m.RT(wnm1)
        dt = @elapsed a = @allocations wn = (RT \ m.G0) \ (RT \ m.b0)
        fHat = 1/2 * norm(RT \ (m.G0*wn - m.b0))^2
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
    b0::AbstractVector
    _RT::RTFun 
    _res!::ResFun
    _jac!::∇resFun
    reltol::AbstractFloat
    maxiters::Int
end 
# constructor
function NLS_iter(prob::AbstractWENDyProblem, params::WENDyParameters; reltol::AbstractFloat=1e-8,maxiters::Int=10)
    _RT = RTFun(prob, params)
    _res = ResFun(prob, params)
    _jac = ∇resFun(prob, params)
    NLS_iter(prob.b0, _RT, _res, _jac, reltol, maxiters)
end
# method
function (m::NLS_iter)(wnm1::AbstractVector{<:AbstractFloat};ll::Logging.LogLevel=Logging.Warn, _ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr,ll)) do 
        @info "  Running local optimization method"
        RT = m._RT(wnm1)
        b = RT \ m.b0 
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