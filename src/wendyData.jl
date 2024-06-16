using OrdinaryDiffEq, BSON, ModelingToolkit, Logging
using OrdinaryDiffEq: ODESolution
using ModelingToolkit: t_nounits as t, D_nounits
using ModelingToolkit: @mtkmodel, @mtkbuild, ODESystem

## Function used to odes problems in the format used here
function _solve_ode(ode::ODESystem, t_rng::Tuple, M::Int;
    alg=FBDF(), reltol::Real= 1e-8,abstol::Real=1e-8)
    # Build parameter dictionary from the "true params" 
    params = [p => ModelingToolkit.getdefault(p) for p in parameters(ode)]
    # Build the initial condition from the true initial condition
    init_cond = [ic=>ModelingToolkit.getdefault(ic) for ic in unknowns(ode)]
    p = ODEProblem(ode,init_cond , t_rng, params)

    t_step = (t_rng[end]-t_rng[1]) / (M-1)
    return solve(p, alg,reltol=reltol, abstol = abstol, saveat=t_step)
end
##
function _saveSol(sol::ODESolution, saveFile::String)
    
    u = hcat(sol.u...)
    t = sol.t 
    BSON.@save saveFile t u
end
##
function getData(ex::NamedTuple; forceOdeSolve::Bool=false)
    if forceOdeSolve 
        sol = _solve_ode(ex.ode,ode.tRng,ode.M)
        _saveSol(sol, ex.file)
        return sol.t, sol.u
    end
    data = BSON.load(ex.file) 
    tt_full = data[:t] 
    U_full = data[:u] 
    return tt_full, U_full
end

