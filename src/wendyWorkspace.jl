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
covarianceStrategy(w::NonlinearWeakForm) = FactorCovariance()
function covarianceStrategy(w::LinearWeakForm)
    KD, _, J = size(w.L₁)
    (J+1)^2*KD^2*sizeof(Float64) <= MAX_LLᵀ_BLOCK_BYTES ? BlockCovariance(_LLᵀBlocks(w.L₀, w.L₁)) : FactorCovariance()
end
# (L, S) at p with q = [1; p], L = nothing when it is not needed
function _covariance(::FactorCovariance, w::WeakForm, q::AbstractVector{<:Real}, p::AbstractVector{<:Real}, diagReg::Real)
    L = _L(w, p)
    S = L * L'
    S .*= 1-diagReg
    S[diagind(S)] .+= diagReg
    L, S
end
_covariance(c::BlockCovariance, w::WeakForm, q::AbstractVector{<:Real}, p::AbstractVector{<:Real}, diagReg::Real) =
    nothing, _S(q, c.P, diagReg)
## Workspace for the weak form negative log-likelihood m(p), its gradient ∇m(p) and hessian Hm(p)
# The optimizers usually evaluate m, ∇m and Hm at the same p, so they share this state. Whenever p changes
# (compared by value) we recompute S(p), its factorization, r(p) and y = S⁻¹r. The remaining pieces are only
# computed the first time something asks for them at that p.
@kwdef mutable struct WENDyWorkspace
    form::WeakForm
    cov::CovarianceStrategy
    diagReg::Float64
    # the parameters everything below was computed for
    p::Vector{Float64} = Float64[]
    # computed whenever p changes
    q::Vector{Float64} = Float64[] # [1; p]
    L::Union{Nothing,Matrix{Float64}} = nothing
    S::Matrix{Float64} = zeros(0,0)
    F::Factorization = cholesky(ones(1,1))
    logDet::Float64 = 0.0
    r::Vector{Float64} = Float64[]
    y::Vector{Float64} = Float64[] # S⁻¹r
    # computed on request
    S⁻¹::Union{Nothing,Matrix{Float64}} = nothing
    ∇r::Union{Nothing,Matrix{Float64}} = nothing
    ∇L::Union{Nothing,Array{Float64,3}} = nothing
    M::Union{Nothing,Matrix{Float64}} = nothing   # S⁻¹L - y(Lᵀy)ᵀ
end
WENDyWorkspace(form::WeakForm, params::WENDyParameters; cov::CovarianceStrategy=covarianceStrategy(form)) =
    WENDyWorkspace(form=form, cov=cov, diagReg=params.diagReg)
# Bring the workspace up to date with p: S(p), its factorization, log|S|, r(p) and y = S⁻¹r
function _update!(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    ws.p == p && return ws
    ws.p = Float64[] # marks the state as invalid until everything below is done
    ws.S⁻¹ = ws.∇r = ws.∇L = ws.M = nothing
    ws.q = [1; p]
    ws.L, ws.S = _covariance(ws.cov, ws.form, ws.q, p, ws.diagReg)
    ws.r = _r(ws.form, p)
    ws.F, ws.logDet = try
        F = cholesky(Symmetric(ws.S))
        F, 2*sum(log, diag(F.U))
    catch
        F = lu(ws.S)
        F, sum(log.(filter(x-> x >0, diag(F.U))))
    end
    ws.y = ws.F \ ws.r
    ws.p = copy(p)
    ws
end
# pieces computed on request
_S⁻¹!(ws::WENDyWorkspace) = isnothing(ws.S⁻¹) ? (ws.S⁻¹ = inv(ws.F)) : ws.S⁻¹
_∇r!(ws::WENDyWorkspace) = isnothing(ws.∇r) ? (ws.∇r = _∇r(ws.form, ws.p)) : ws.∇r
_∇L!(ws::WENDyWorkspace) = isnothing(ws.∇L) ? (ws.∇L = _∇L(ws.form, ws.p)) : ws.∇L
_M!(ws::WENDyWorkspace) = isnothing(ws.M) ? (ws.M = ws.F \ ws.L - ws.y * (ws.L'ws.y)') : ws.M
## m(p), ∇m(p), Hm(p) on the workspace (see wendyProblems.jl for the versions on a WENDyProblem)
function wnll(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    _update!(ws, p)
    (dot(ws.r, ws.y) + ws.logDet)/2 + length(ws.r)/2*log(2*pi)
end
∇wnll(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = (_update!(ws, p); ∇wnll(ws.cov, ws))
∇wnll(c::BlockCovariance, ws::WENDyWorkspace) = _∇wnll(ws.q, c.P, ws.form.G, ws.y, _S⁻¹!(ws), ws.diagReg)
∇wnll(::FactorCovariance, ws::WENDyWorkspace) = _∇wnll(_∇L!(ws), _∇r!(ws), ws.y, _M!(ws), ws.diagReg)
Hwnll(ws::WENDyWorkspace, p::AbstractVector{<:Real}; approx::Bool=false) = (_update!(ws, p); Hwnll(ws.cov, ws; approx=approx))
Hwnll(c::BlockCovariance, ws::WENDyWorkspace; approx::Bool=false) =
    _Hwnll(ws.q, c.P, ws.form.G, ws.F, approx ? nothing : _S⁻¹!(ws), ws.y, ws.diagReg)
function Hwnll(::FactorCovariance, ws::WENDyWorkspace; approx::Bool=false)
    H = _Hwnll(ws.F, _∇L!(ws), ws.L, _∇r!(ws), ws.y, ws.diagReg; approx=approx) + _Hsecond(ws.form, ws)
    (H + H') / 2
end
