function _solve_ode(prob::ODEProblem, t_rng::Tuple, M::Int;
alg=Rosenbrock23(), reltol::Real=1e-8, abstol::Real=1e-8)
    t_step = (t_rng[end]-t_rng[1]) / (M-1)
    return solve(prob, alg, reltol=reltol, abstol = abstol, saveat=t_step)
end
## Function used to odes problems in the format used here
function _solve_ode(
    ode::ODESystem, 
    t_rng::Tuple, 
    M::Int, 
    w::AbstractVector{<:Real}, 
    u0::AbstractVector{<:Real}; 
    alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=Rosenbrock23(), 
    reltol::Real=1e-12, 
    abstol::Real=1e-12
)
    # Build parameter dictionary from the "true params" 
    @assert length(w) == length(parameters(ode)) "Parameter vector must be the same length as parameters in ode"
    params = [p => wj for (p,wj) in zip(parameters(ode), w)]
    @assert length(u0) == length(unknowns(ode)) "Initial condition vector must be the same length as unknowns in ode"
    init_cond = [ic=>u0k for (ic,u0k) in zip(unknowns(ode), u0)]
    p = ODEProblem(ode, init_cond, t_rng, params)
    return _solve_ode(p, t_rng, M; alg=alg, reltol=reltol, abstol=abstol)
end
##
function forwardSolve(prob::WENDyProblem, ex::NamedTuple, w::AbstractVector{<:Real}; kwargs...)
    sol = _solve_ode(ex.ode, (prob.tt[1], prob.tt[end]), prob.M; w=w,
    kwargs...)
    return reduce(hcat, sol.u)
end
## compute forward solve relative error
function forwardSolveRelErr(prob::WENDyProblem, ex::NamedTuple, w::AbstractVector{<:Real}; kwargs...)
    sol = _solve_ode(ex, (prob.tt[1], prob.tt[end]), prob.M; w=w,
    kwargs...)
    Uhat = reduce(hcat, sol.u)
    norm(Uhat-prob.U)/norm(prob.U)
end

##
function _getData(ode::ODESystem, tRng::NTuple{2,<:AbstractFloat}, M::Int, trueParameters::AbstractVector{<:Real}, initCond::AbstractVector{<:Real}, file::String; forceOdeSolve::Bool=false, ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stdout, ll)) do 
        if forceOdeSolve || !isfile(file)
            @info "  Generating data by solving ode"
            sol = _solve_ode(ode, tRng, M, trueParameters, initCond)
            u = reduce(hcat, sol.u)
            t = sol.t
            BSON.@save file t u
            return t,u, sol
        end
        @info "Loading from file"
        data = BSON.load(file) 
        tt_full = data[:t] 
        U_full = data[:u] 
        return tt_full, U_full
    end
end

