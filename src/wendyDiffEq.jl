function _solve_ode(p::ODEProblem, t_rng::Tuple, M::Int;
alg=Rosenbrock23(), reltol::Real=1e-8, abstol::Real=1e-8)
    t_step = (t_rng[end]-t_rng[1]) / (M-1)
    return solve(p, alg, reltol=reltol, abstol = abstol, saveat=t_step)
end
## Function used to odes problems in the format used here
function _solve_ode(ex::NamedTuple, t_rng::Tuple, M::Int; w::Union{AbstractVector{<:Real}, Nothing}=nothing,
    alg=Rosenbrock23(), reltol::Real=1e-8, abstol::Real=1e-8)
    ode = ex.ode
    # Build parameter dictionary from the "true params" 
   params = if !isnothing(w)
        @assert length(w) == length(parameters(ode)) "Parameter vector must be the same length as parameters in ode"
        [p => w[i] for (i,p) in enumerate(parameters(ode))]
    elseif :params in keys(ex) 
        [p => ex.params[j] for (j, p) in enumerate(parameters(ode))]
    else # try to get defauls
        [p => ModelingToolkit.getdefault(p) for p in parameters(ode)]
    end
    # Build the initial condition from the true initial condition
    init_cond = if :init_cond in keys(ex)
        [ic=>ex.init_cond[i] for (i,ic) in enumerate(unknowns(ode))]
    else 
        [ic=>ModelingToolkit.getdefault(ic) for ic in unknowns(ode)]
    end
    p = ODEProblem(ode, init_cond, t_rng, params)
    return _solve_ode(p, t_rng, M; alg=alg, reltol=reltol, abstol=abstol)
end

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
function _saveSol(sol::ODESolution, saveFile::String)
    u = reduce(hcat, sol.u)
    t = sol.t 
    BSON.@save saveFile t u
end
##
function getData(ex::NamedTuple; forceOdeSolve::Bool=false)
    if forceOdeSolve || !isfile(ex.file)
        @info "  Generating data by solving ode"
        sol = _solve_ode(ex,ex.tRng,ex.M)
        _saveSol(sol, ex.file)
        return sol.t, reduce(hcat, sol.u), sol
    end
    @info "Loading from file"
    data = BSON.load(ex.file) 
    tt_full = data[:t] 
    U_full = data[:u] 
    return tt_full, U_full
end

