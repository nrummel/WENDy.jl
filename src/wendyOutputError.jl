## Output error (forward solve) least squares
"""
    OutputErrorProblem(tt, U, f!, J; params=WENDyParameters())

The output error least squares problem: find p (and optionally u₀) that minimize ½‖û(tt; p, u₀) - U‖² where û solves
the ODE du/dt = f(u, p, t). This is the common alternative to the weak form, provided for comparison and for the
hybrid solver HybridTrustRegionOELS.

The forward solves need OrdinaryDiffEq.jl, which WENDy only loads on request: run `using OrdinaryDiffEq` first.

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
    isnothing(Base.get_extension(@__MODULE__, :WENDyOrdinaryDiffEqExt)) &&
        error("OutputErrorProblem solves the ODE with OrdinaryDiffEq.jl: run `using OrdinaryDiffEq` first")
    U = U isa AbstractVector ? reshape(U, length(U), 1) : U
    _, D = size(U)
    OutputErrorProblem(tt, U, f!, J, D, U[1,:], params)
end
# Û(tt; p, u₀), the (Mp1 x D) forward solve; defined in ext/WENDyOrdinaryDiffEqExt.jl
function _forwardSolve end
# û(tt; p, u₀) - U, where pu₀ = [p; u₀] when params.fsU0Free and pu₀ = p otherwise (then u₀ = U[1,:])
function _forwardSolveResidual(oe::OutputErrorProblem, pu₀::AbstractVector, params::WENDyParameters)
    Mp1, D = size(oe.U)
    J = oe.J
    try
        p, u₀ = params.fsU0Free ? (pu₀[1:J], pu₀[J+1:end]) : (pu₀[1:J], oe.U[1,:])
        return (_forwardSolve(oe, p, u₀, params) - oe.U)[:]
    catch
        NaN*ones(Mp1*D)
    end
end
"""
    oeResidual!(r, oe::OutputErrorProblem, pu₀, params=oe.params)

Output error residual û(tt; p, u₀) - U (flattened), written into r. Here pu₀ = [p; u₀] when params.fsU0Free and pu₀ = p otherwise.
"""
oeResidual!(r::AbstractVector{<:Real}, oe::OutputErrorProblem, pu₀::AbstractVector{<:Real}, params::WENDyParameters=oe.params) =
    r .= _forwardSolveResidual(oe, pu₀, params)
"""
    oeJacobian!(∇r, oe::OutputErrorProblem, pu₀, params=oe.params)

Jacobian of the output error residual wrt pu₀ (by forward mode automatic differentiation through the ODE solve), written into ∇r
"""
oeJacobian!(∇r::AbstractMatrix{<:Real}, oe::OutputErrorProblem, pu₀::AbstractVector{<:Real}, params::WENDyParameters=oe.params) =
    ForwardDiff.jacobian!(∇r, pu₀ -> _forwardSolveResidual(oe, pu₀, params), pu₀)
"""
    solve(oe::OutputErrorProblem, p₀, params=oe.params; return_iterates=false)

Solve the output error least squares problem with Levenberg-Marquardt starting from p₀. When params.fsU0Free the
initial condition is estimated as well (starting from oe.u₀) and the result is [p̂; û₀], otherwise it is p̂.
With return_iterates=true this returns (p̂, iterations, iterates).
"""
function solve(oe::OutputErrorProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters=oe.params; return_iterates::Bool=false, kwargs...)
    w₀ = params.fsU0Free ? vcat(p₀, oe.u₀) : p₀
    nonlinearLeastSquares(
        (r, pu₀) -> oeResidual!(r, oe, pu₀, params),
        (∇r, pu₀) -> oeJacobian!(∇r, oe, pu₀, params),
        length(oe.U), w₀, params;
        return_iterates=return_iterates
    )
end
