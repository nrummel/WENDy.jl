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
- Mp1::Int : number of time points
- u₀::AbstractVector{<:Real} : initial condition taken from the data, U[1,:] (the initial guess for u₀ when fsU0Free)
- params::WENDyParameters : default parameters for solve
- ∇ₓf!, ∂ₜf! : symbolic state jacobian ∇ₓf!(out, u, p, t) (D x D) and time gradient ∂ₜf!(out, u, p, t) (length D) of f!,
  handed to the forward solver instead of its internal autodiff; built only when WENDy._OE_SYMBOLIC[] (else nothing)
"""
struct OutputErrorProblem
    tt::AbstractVector{<:Real}
    U::AbstractMatrix{<:Real}
    f!::Function
    J::Int
    D::Int
    Mp1::Int
    u₀::AbstractVector{<:Real}
    params::WENDyParameters
    ∇ₓf!::Union{Nothing,Function}
    ∂ₜf!::Union{Nothing,Function}
end
""" (private) Build the time gradient ∂f/∂t of the rhs f!(du, u, p, t), in place ∂ₜf!(out, u, p, t) """
function _get∂ₜf(f!::Function, D::Int, J::Int)
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∂ₜf = [Symbolics.derivative(dxd, t) for dxd in dx]
    return _build(∂ₜf, x, p, t)[end]
end
function OutputErrorProblem(tt::AbstractVector{<:Real}, U::AbstractVecOrMat{<:Real}, f!::Function, J::Int; params::WENDyParameters=WENDyParameters())
    isnothing(Base.get_extension(@__MODULE__, :WENDyOrdinaryDiffEqExt)) &&
        error("OutputErrorProblem solves the ODE with OrdinaryDiffEq.jl: run `using OrdinaryDiffEq` first")
    U = U isa AbstractVector ? reshape(U, length(U), 1) : U
    Mp1, D = size(U)
    ∇ₓf!, ∂ₜf! = _OE_SYMBOLIC[] ? (_get∇ₓf(f!, D, J), _get∂ₜf(f!, D, J)) : (nothing, nothing)
    OutputErrorProblem(tt, U, f!, J, D, Mp1, U[1,:], params, ∇ₓf!, ∂ₜf!)
end
# Û(tt; p, u₀), the (Mp1 x D) forward solve; defined in ext/WENDyOrdinaryDiffEqExt.jl
function _forwardSolve end
# û(tt; p, u₀) - U, where pu₀ = [p; u₀] when params.fsU0Free and pu₀ = p otherwise (then u₀ = U[1,:])
function _forwardSolveResidual(oe::OutputErrorProblem, pu₀::AbstractVector, params::WENDyParameters)
    (; J, D, Mp1) = oe
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
    solve(oe::OutputErrorProblem, p₀, params=oe.params)

Solve the output error least squares problem with Levenberg-Marquardt starting from p₀. When params.fsU0Free the
initial condition is estimated as well, starting from oe.u₀. Returns a WENDySolution with the estimate p̂, the
estimated initial condition û₀ (nothing unless params.fsU0Free), and the parameter part of the iterates.
"""
function solve(oe::OutputErrorProblem, p₀::AbstractVector{<:Real}, params::WENDyParameters=oe.params; kwargs...)
    start = time()
    pu₀_init = params.fsU0Free ? vcat(p₀, oe.u₀) : p₀
    out = nonlinearLeastSquares(
        (r, pu₀) -> oeResidual!(r, oe, pu₀, params),
        (∇r, pu₀) -> oeJacobian!(∇r, oe, pu₀, params),
        oe.Mp1*oe.D, pu₀_init, params
    )
    J = oe.J
    û₀ = params.fsU0Free ? out.p̂[J+1:end] : nothing
    WENDySolution(out.p̂[1:J], û₀, out.iterations, _fromStart(p₀, out.iterates[1:J,:]), out.retcode, time() - start,
        "OutputErrorLeastSquares")
end
