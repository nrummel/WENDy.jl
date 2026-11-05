## Output error (forward solve) least squares
"""
    OutputErrorProblem(tt, U, f!, J; params=WENDyParameters())

The output error least squares problem: find p (and optionally u₀) that minimize ½‖û(tt; p, u₀) - U‖² where û solves
the ODE du/dt = f(u, p, t). This is the common alternative to the weak form, provided for comparison and for the
hybrid solver HybridTrustRegionOELS.

## Arguments
- tt::AbstractVector{<:Real} : vector of times (equispaced)
- U::AbstractVecOrMat{<:Real} : Corrupted state variable data
- f!::Function : Right hand-side of the differential equation, f!(du, u, p, t)
- J::Int : number of parameters (to be estimated)
- params::WENDyParameters : (optional) the forward solve settings fsAlg, fsReltol, fsAbstol and fsU0Free are used

# Fields
- tt, U, f!, J : as above
- D::Int : number of state variables
- u₀::AbstractVector{<:Real} : initial condition taken from the data, U[1,:] (the initial guess for u₀ when fsU0Free)
- params::WENDyParameters : default parameters for solve
"""
struct OutputErrorProblem
    tt::AbstractVector{<:Real}
    U::AbstractMatrix{<:Real}
    f!::Function
    J::Int
    D::Int
    u₀::AbstractVector{<:Real}
    params::WENDyParameters
end
function OutputErrorProblem(tt::AbstractVector{<:Real}, U::AbstractVecOrMat{<:Real}, f!::Function, J::Int; params::WENDyParameters=WENDyParameters())
    U = U isa AbstractVector ? reshape(U, length(U), 1) : U
    _, D = size(U)
    OutputErrorProblem(tt, U, f!, J, D, U[1,:], params)
end
# û(tt; p, u₀) - U, where w = [p; u₀] when params.fsU0Free and w = p otherwise (then u₀ = U[1,:])
function _forwardSolveResidual(oe::OutputErrorProblem, w::AbstractVector, params::WENDyParameters)
    Mp1, D = size(oe.U)
    J = oe.J
    try
        p, u₀ = params.fsU0Free ? (w[1:J], w[J+1:end]) : (w[1:J], oe.U[1,:])
        tRng = (oe.tt[1], oe.tt[end])
        dt = (oe.tt[end] - oe.tt[1]) / (length(oe.tt) - 1)
        sol = solve_ode(ODEProblem(oe.f!, u₀, tRng, p), params.fsAlg;
            reltol=params.fsReltol, abstol=params.fsAbstol,
            saveat=dt, verbose=DEVerbosity(SciMLLogging.None())
        )
        Û = reduce(vcat, um' for um in sol.u)
        return (Û - oe.U)[:]
    catch
        NaN*ones(Mp1*D)
    end
end
"""
    oeResidual!(r, oe::OutputErrorProblem, w, params=oe.params)

Output error residual û(tt; p, u₀) - U (flattened), written into r. Here w = [p; u₀] when params.fsU0Free and w = p otherwise.
"""
oeResidual!(r::AbstractVector{<:Real}, oe::OutputErrorProblem, w::AbstractVector{<:Real}, params::WENDyParameters=oe.params) =
    r .= _forwardSolveResidual(oe, w, params)
"""
    oeJacobian!(∇r, oe::OutputErrorProblem, w, params=oe.params)

Jacobian of the output error residual wrt w (by forward mode automatic differentiation through the ODE solve), written into ∇r
"""
oeJacobian!(∇r::AbstractMatrix{<:Real}, oe::OutputErrorProblem, w::AbstractVector{<:Real}, params::WENDyParameters=oe.params) =
    ForwardDiff.jacobian!(∇r, w -> _forwardSolveResidual(oe, w, params), w)
"""
    solve(oe::OutputErrorProblem, p₀, params=oe.params; return_wits=false)

Solve the output error least squares problem with Levenberg-Marquardt starting from p₀. When params.fsU0Free the
initial condition is estimated as well (starting from oe.u₀) and the result is [p̂; û₀], otherwise it is p̂.
With return_wits=true this returns (what, iterations, iterates).
"""
function solve(oe::OutputErrorProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters=oe.params; return_wits::Bool=false, kwargs...)
    w₀ = params.fsU0Free ? vcat(p₀, oe.u₀) : p₀
    nonlinearLeastSquares(
        (r, w) -> oeResidual!(r, oe, w, params),
        (∇r, w) -> oeJacobian!(∇r, oe, w, params),
        length(oe.U), w₀, params;
        return_wits=return_wits
    )
end
