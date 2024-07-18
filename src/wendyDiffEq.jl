## Function used to odes problems in the format used here
function _solve_ode(
    f!::Function, 
    tRng::Tuple, 
    M::Int, 
    w::AbstractVector{<:Real}, 
    u0::AbstractVector{<:Real}; 
    alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=Rosenbrock23(), 
    reltol::Real=1e-12, 
    abstol::Real=1e-12,
    kwargs...
)
    odeprob = ODEProblem{true, SciMLBase.FullSpecialize}(f!, u0, tRng, w)
    t_step = (tRng[end]-tRng[1]) / (M-1)
    solve(odeprob, alg; 
        reltol=reltol, abstol = abstol, saveat=t_step,
        verbose=false, kwargs...
    )
end
##
function forwardSolve(prob::WENDyProblem, w::AbstractVector{<:Real}; kwargs...)
    sol = _solve_ode(prob.data.f!, prob.data.tRng, prob.M, w, prob.data.initCond; kwargs...)
    return reduce(hcat, sol.u)
end
## compute forward solve relative error
function forwardSolveRelErr(prob::WENDyProblem, w::AbstractVector{<:Real}; kwargs...)
    Uhat = forwardSolve(prob, w; verbose=false)
    norm(Uhat-prob.U) / norm(prob.U)
end
## 
function _l2(w::AbstractVector{<:Real}, U::AbstractMatrix, ex::WENDyData)
    try 
        odeprob = ODEProblem{true, SciMLBase.FullSpecialize}(
            ex.f!, 
            ex.initCond,
            ex.tRng,
            w
        )
        M = size(U,2)
        t_step = abs(ex.tRng[end] - ex.tRng[1]) / (M-1)
        sol = solve(
            odeprob,
            Rosenbrock23(); 
            reltol=1e-12, abstol=1e-12, 
            saveat=t_step, 
            # saveat=ex.tRng[1]:t_step:ex.tRng[end], 
            verbose=true
        )
        Uhat = reduce(hcat, sol.u)
        sum((Uhat[:] - U[:]).^2) 
    catch
        NaN
    end
end

