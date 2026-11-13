## Output error (forward solve) least squares
## Forward sensitivities. With S = ∂u/∂[p; u₀] (D × (J+D)), the augmented state z = [u; vec(S)] solves
##     du/dt = f(u, p, t),    dS/dt = ∇ₓf(u, p, t) S + [∇ₚf(u, p, t)  0],    u(0) = u₀, S(0) = [0  I],
## so one Float64 solve gives û(tt) and its jacobian wrt p and u₀ (the u₀ columns are dropped unless fsU0Free).
## The rhs of this system, its jacobian wrt z and its time gradient are generated symbolically from f! and kept behind
## FunctionWrappers with fixed Float64 signatures, so the forward solver sees the same function types for every f! and
## is compiled once (rather than once per f! with ForwardDiff Duals, as differentiating through the solver does).
const _ZVec = FunctionWrapper{Nothing, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}
const _ZMat = FunctionWrapper{Nothing, Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Float64}}
"""
The forward sensitivity system of an OutputErrorProblem (see above): rhs!(dz, z, p, t), jac!(Jz, z, p, t) and
tgrad!(dzdt, z, p, t) for z = [u; vec(S)] of length D(1+J+D), and the last solve (at pu₀, with the given fsU0Free).
"""
mutable struct OESensitivity
    D::Int
    J::Int
    rhs!::_ZVec
    jac!::_ZMat
    tgrad!::_ZVec
    last::Any # (key, r, ∇r) of the last solve, key = (copy of pu₀, fsU0Free)
end
_nothing(f!) = (o, z, p, t) -> (f!(o, z, p, t); nothing)
function OESensitivity(f!::Function, D::Int, J::Int)
    n = D*(1 + J + D)
    @variables z[1:n] p[1:J] t
    zz = collect(z)
    u = zz[1:D]
    S = reshape(zz[D+1:end], D, J + D)
    fu = Vector{Any}(undef, D)
    f!(fu, u, collect(p), t)
    ∇ₓf = jacobian(fu, u)
    ∇ₚf = jacobian(fu, collect(p))
    dS = ∇ₓf * S + hcat(∇ₚf, zeros(Num, D, D))
    rhs = [fu; vec(dS)]
    jac = jacobian(rhs, zz)
    tgrad = [Symbolics.derivative(e, t) for e in rhs]
    build(ex) = _build(ex, z, p, t)[end]
    OESensitivity(D, J, _ZVec(_nothing(build(rhs))), _ZMat(_nothing(build(jac))), _ZVec(_nothing(build(tgrad))), nothing)
end
"""
    OutputErrorProblem(tt, U, f!, J; params=WENDyParameters())

The output error least squares problem: find p (and optionally u₀) that minimize ½‖û(tt; p, u₀) - U‖² where û solves
the ODE du/dt = f(u, p, t). This is the common alternative to the weak form, provided for comparison and for the
hybrid solver HybridTrustRegionOELS.

The forward solves need OrdinaryDiffEq.jl, which WENDy only loads on request: run `using OrdinaryDiffEq` first
(`using OrdinaryDiffEqRosenbrock` is enough for the default solver, Rodas5P).

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
- sens::OESensitivity : the forward sensitivity system that gives the jacobian (see OESensitivity)
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
    sens::OESensitivity
end
function OutputErrorProblem(tt::AbstractVector{<:Real}, U::AbstractVecOrMat{<:Real}, f!::Function, J::Int; params::WENDyParameters=WENDyParameters())
    isnothing(Base.get_extension(@__MODULE__, :WENDyOrdinaryDiffEqExt)) &&
        error("OutputErrorProblem solves the ODE with OrdinaryDiffEq.jl: run `using OrdinaryDiffEq` (or `using OrdinaryDiffEqRosenbrock`) first")
    U = U isa AbstractVector ? reshape(U, length(U), 1) : U
    Mp1, D = size(U)
    OutputErrorProblem(tt, U, f!, J, D, Mp1, U[1,:], params, OESensitivity(f!, D, J))
end
# (Û, S) at tt: Û (Mp1 x D) and S (D x (J+D) x Mp1), or nothing when the solve fails; defined in the extension
function _sensitivitySolve end
# r = vec(Û - U) and ∇r = ∂r/∂pu₀ from one sensitivity solve, cached for the next call at the same pu₀. Only the
# jacobian comes from here: the residual alone is a plain forward solve, which is much cheaper (D states rather than
# D(1+J+D)) and, with Float64 arguments, also compiled once by OrdinaryDiffEq's default AutoSpecialize.
function _sensitivityResidual(oe::OutputErrorProblem, pu₀::AbstractVector{<:Real}, params::WENDyParameters)
    (; J, D, Mp1, sens) = oe
    key = (Vector{Float64}(pu₀), params.fsU0Free)
    !isnothing(sens.last) && sens.last[1] == key && return sens.last[2], sens.last[3]
    p, u₀ = params.fsU0Free ? (pu₀[1:J], pu₀[J+1:end]) : (pu₀[1:J], oe.U[1,:])
    nc = params.fsU0Free ? J + D : J
    out = try
        _sensitivitySolve(oe, Vector{Float64}(p), Vector{Float64}(u₀), params)
    catch
        nothing
    end
    r, ∇r = if isnothing(out)
        fill(NaN, Mp1*D), fill(NaN, Mp1*D, nc)
    else
        Û, S = out
        # rows of r are (m, d) with m fastest, as vec of the Mp1 x D residual
        vec(Û - oe.U), reshape(permutedims(S[:, 1:nc, :], (3, 1, 2)), Mp1*D, nc)
    end
    sens.last = (key, r, ∇r)
    r, ∇r
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

Jacobian of the output error residual wrt pu₀, written into ∇r, from the forward sensitivity equations (see OESensitivity)
"""
oeJacobian!(∇r::AbstractMatrix{<:Real}, oe::OutputErrorProblem, pu₀::AbstractVector{<:Real}, params::WENDyParameters=oe.params) =
    ∇r .= _sensitivityResidual(oe, pu₀, params)[2]
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
