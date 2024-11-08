## Function used to odes problems in the format used here
function _solve_ode(
    f!::Function, 
    tRng::Tuple, 
    Mp1::Int, 
    w::AbstractVector{<:Real}, 
    u0::AbstractVector{<:Real}; 
    alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=Rosenbrock23(), 
    # reltol::Real=1e-12, 
    # abstol::Real=1e-12,
    verbose::Bool=false,
    kwargs...
)
    odeprob = ODEProblem{true, SciMLBase.FullSpecialize}(f!, u0, tRng, w)
    t_step = (tRng[end]-tRng[1]) / (Mp1-1)
    solve(odeprob, alg; 
        # reltol=reltol, abstol = abstol, 
        saveat=t_step,
        verbose=verbose, kwargs...
    )
end
##
function forwardSolve(prob::WENDyProblem, w::AbstractVector{<:Real}; kwargs...)
    sol = _solve_ode(prob.f!, (prob.tt[1], prob.tt[end]), prob.Mp1, w, prob.U[1,:]; kwargs...)
    return reduce(vcat, sol.u')
end
