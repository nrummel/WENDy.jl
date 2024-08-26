## Function used to odes problems in the format used here
function _solve_ode(
    f!::Function, 
    tRng::Tuple, 
    M::Int, 
    w::AbstractVector{<:Real}, 
    u0::AbstractVector{<:Real}; 
    alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=Rosenbrock23(), 
    # reltol::Real=1e-12, 
    # abstol::Real=1e-12,
    verbose::Bool=false,
    kwargs...
)
    odeprob = ODEProblem{true, SciMLBase.FullSpecialize}(f!, u0, tRng, w)
    t_step = (tRng[end]-tRng[1]) / (M-1)
    solve(odeprob, alg; 
        # reltol=reltol, abstol = abstol, 
        saveat=t_step,
        verbose=verbose, kwargs...
    )
end
##
function forwardSolve(prob::WENDyProblem, w::AbstractVector{<:Real}; kwargs...)
    sol = _solve_ode(prob.data.f!, prob.data.tRng, prob.M, w, prob.data.initCond; kwargs...)
    return reduce(hcat, sol.u)
end
## compute forward solve relative error
function forwardSolveRelErr(prob::WENDyProblem, w::AbstractVector{<:Real}; kwargs...)
    Uhat = forwardSolve(prob, w; kwargs...)
    # norm(Uhat-prob.U_exact) / norm(prob.U_exact)
    Ustar = prob.U_exact
    _, M = size(Ustar)
    @views avg_rel_err = sum(norm(Uhat[:,m] - Ustar[:,m]) for m in 1:M) / sum(norm(Ustar[:,m]) for m in 1:M)
    @views final_rel_err = norm(Uhat[:,end] - Ustar[:,end]) / norm(Ustar[:,end])
    avg_rel_err, final_rel_err
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
            # reltol=1e-12, abstol=1e-12, 
            # saveat=t_step, 
            saveat=ex.tRng[1]:t_step:ex.tRng[end], 
            verbose=false
        )
        Uhat = reduce(hcat, sol.u)
        _, M = size(U)
        @views sum(norm(Uhat[:,m] - U[:,m]) for m in 1:M)
    catch
        NaN
    end
end

