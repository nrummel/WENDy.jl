## How S(p) = (1-δ)L(p)L(p)ᵀ + δI is formed
abstract type CovarianceStrategy end
"""
S(p) = (1-δ)∑ₐᵦ qₐqᵦ L̃ₐL̃ᵦᵀ + δI from the precomputed blocks P[:,a,b] = vec(L̃ₐL̃ᵦᵀ) of a linear weak form,
so nothing per p touches the Mp1*D dimension (see _LLᵀBlocks)
"""
struct BlockCovariance <: CovarianceStrategy
    P::Array{Float64,3}
end
"""S(p) = (1-δ)L(p)L(p)ᵀ + δI from L(p) (any weak form)"""
struct FactorCovariance <: CovarianceStrategy end
# Linear weak forms use the blocks unless they would take more than this much memory
const MAX_LLᵀ_BLOCK_BYTES = 2^30
covarianceStrategy(wform::NonlinearWeakForm) = FactorCovariance()
function covarianceStrategy(wform::LinearWeakForm)
    (; J, KD) = wform.dims
    (J+1)^2*KD^2*sizeof(Float64) <= MAX_LLᵀ_BLOCK_BYTES ? BlockCovariance(_LLᵀBlocks(wform.L₀, wform.L₁, wform.dims)) : FactorCovariance()
end
# (L, S) at p with q = [1; p], L = nothing when it is not needed
function _covariance(::FactorCovariance, wform::WeakForm, q::AbstractVector{<:Real}, p::AbstractVector{<:Real}, diagReg::Real)
    L = _L(wform, p)
    S = L * L'
    S .*= 1-diagReg
    S[diagind(S)] .+= diagReg
    L, S
end
_covariance(c::BlockCovariance, wform::WeakForm, q::AbstractVector{<:Real}, p::AbstractVector{<:Real}, diagReg::Real) =
    nothing, _S(q, c.P, diagReg, wform.dims)
## Workspace for the weak form negative log-likelihood ℓ(p), its gradient ∇ℓ(p) and hessian ∇²ℓ(p)
# The optimizers usually evaluate ℓ, ∇ℓ and ∇²ℓ at the same p, so they share this state. Whenever p changes
# (compared by value) we recompute S(p), its factorization, r(p) and y = S⁻¹r. The remaining pieces are only
# computed the first time something asks for them at that p.
@kwdef mutable struct WENDyWorkspace
    wform::WeakForm
    cov::CovarianceStrategy
    diagReg::Float64
    # the parameters everything below was computed for
    p::Vector{Float64} = Float64[]
    # computed whenever p changes
    q::Vector{Float64} = Float64[] # [1; p]
    L::Union{Nothing,Matrix{Float64}} = nothing
    S::Matrix{Float64} = zeros(0,0)
    Sfact::Factorization = cholesky(ones(1,1))
    logDet::Float64 = 0.0
    r::Vector{Float64} = Float64[]
    y::Vector{Float64} = Float64[] # S⁻¹r
    # computed on request
    S⁻¹::Union{Nothing,Matrix{Float64}} = nothing
    ∇r::Union{Nothing,Matrix{Float64}} = nothing
    ∇L::Union{Nothing,Array{Float64,3}} = nothing
    ∇Lℓ::Union{Nothing,Matrix{Float64}} = nothing # S⁻¹L - y(Lᵀy)ᵀ, so ∂ℓ/∂L = (1-δ)∇Lℓ
end
WENDyWorkspace(wform::WeakForm, params::WENDyParameters; cov::CovarianceStrategy=covarianceStrategy(wform)) =
    WENDyWorkspace(wform=wform, cov=cov, diagReg=params.diagReg)
# Bring the workspace up to date with p: S(p), its factorization, log|S|, r(p) and y = S⁻¹r
function _update!(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    ws.p == p && return ws
    ws.p = Float64[] # marks the state as invalid until everything below is done
    ws.S⁻¹ = ws.∇r = ws.∇L = ws.∇Lℓ = nothing
    ws.q = [1; p]
    ws.L, ws.S = _covariance(ws.cov, ws.wform, ws.q, p, ws.diagReg)
    ws.r = _r(ws.wform, p)
    ws.Sfact, ws.logDet = try
        Sfact = cholesky(Symmetric(ws.S))
        Sfact, 2*sum(log, diag(Sfact.U))
    catch
        Sfact = lu(ws.S)
        Sfact, sum(log.(filter(x-> x >0, diag(Sfact.U))))
    end
    ws.y = ws.Sfact \ ws.r
    ws.p = copy(p)
    ws
end
# pieces computed on request
_S⁻¹!(ws::WENDyWorkspace) = isnothing(ws.S⁻¹) ? (ws.S⁻¹ = inv(ws.Sfact)) : ws.S⁻¹
_∇r!(ws::WENDyWorkspace) = isnothing(ws.∇r) ? (ws.∇r = _∇r(ws.wform, ws.p)) : ws.∇r
_∇L!(ws::WENDyWorkspace) = isnothing(ws.∇L) ? (ws.∇L = _∇L(ws.wform, ws.p)) : ws.∇L
_∇Lℓ!(ws::WENDyWorkspace) = isnothing(ws.∇Lℓ) ? (ws.∇Lℓ = ws.Sfact \ ws.L - ws.y * (ws.L'ws.y)') : ws.∇Lℓ
## ℓ(p), ∇ℓ(p), ∇²ℓ(p) on the workspace (see wendyProblems.jl for the versions on a WENDyProblem)
function ℓ(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    _update!(ws, p)
    (dot(ws.r, ws.y) + ws.logDet)/2 + ws.wform.dims.KD/2*log(2*pi)
end
∇ℓ(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = (_update!(ws, p); ∇ℓ(ws.cov, ws))
∇ℓ(c::BlockCovariance, ws::WENDyWorkspace) = _∇ℓ(ws.q, c.P, ws.wform.G, ws.y, _S⁻¹!(ws), ws.diagReg, ws.wform.dims)
∇ℓ(::FactorCovariance, ws::WENDyWorkspace) = _∇ℓ(_∇L!(ws), _∇r!(ws), ws.y, _∇Lℓ!(ws), ws.diagReg, ws.wform.dims)
∇²ℓ(ws::WENDyWorkspace, p::AbstractVector{<:Real}; approx::Bool=false) = (_update!(ws, p); ∇²ℓ(ws.cov, ws; approx=approx))
∇²ℓ(c::BlockCovariance, ws::WENDyWorkspace; approx::Bool=false) =
    _∇²ℓ(ws.q, c.P, ws.wform.G, ws.Sfact, approx ? nothing : _S⁻¹!(ws), ws.y, ws.diagReg, ws.wform.dims)
function ∇²ℓ(::FactorCovariance, ws::WENDyWorkspace; approx::Bool=false)
    H = _∇²ℓ(ws.Sfact, _∇L!(ws), ws.L, _∇r!(ws), ws.y, ws.diagReg, ws.wform.dims; approx=approx) + _Hsecond(ws.wform, ws)
    (H + H') / 2
end
