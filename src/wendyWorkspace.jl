## Workspace for the weak form negative log-likelihood m(p), its gradient ∇m(p) and hessian Hm(p)
# The optimizers usually evaluate m, ∇m and Hm at the same p, so they share this state. Whenever p changes
# (compared by value) we recompute S(p), its factorization, r(p) and y = S⁻¹r. The remaining pieces are only
# computed the first time something asks for them at that p.
#
# When the problem is linear in parameters we precompute the blocks of LLᵀ (see _LLᵀBlocks) so nothing per p
# touches the Mp1*D dimension, unless they would take more than this much memory, then we fall back to L(p).
const MAX_LLᵀ_BLOCK_BYTES = 2^30
@kwdef mutable struct WENDyWorkspace
    # data
    lip::Bool
    data::WENDyInternals
    diagReg::Real
    P::Union{Nothing,AbstractArray{<:Real,3}} = nothing # blocks of LLᵀ (nothing ⇒ build S from L(p))
    L₀::AbstractMatrix{<:Real}                          # p independent part of L
    # the parameters everything below was computed for
    p::AbstractVector{<:Real} = Float64[]
    # computed whenever p changes
    q::AbstractVector{<:Real} = Float64[] # [1; p]
    L::AbstractMatrix{<:Real} = zeros(0,0) # (L(p) only)
    S::AbstractMatrix{<:Real} = zeros(0,0)
    F::Factorization = cholesky(ones(1,1))
    logDet::Real = 0.0
    r::AbstractVector{<:Real} = Float64[]
    y::AbstractVector{<:Real} = Float64[] # S⁻¹r
    # computed on request
    S⁻¹::Union{Nothing,AbstractMatrix{<:Real}} = nothing # (blocks of LLᵀ only)
    ∇r::Union{Nothing,AbstractMatrix{<:Real}} = nothing  # constant when linear in parameters
    ∇L::Union{Nothing,AbstractArray{<:Real,3}} = nothing # constant when linear in parameters (L(p) only)
    M::Union{Nothing,AbstractMatrix{<:Real}} = nothing   # S⁻¹L - y(Lᵀy)ᵀ (L(p) only)
end
# useBlocks = false forces S to be built from L(p) even when the problem is linear in parameters
function WENDyWorkspace(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters; useBlocks::Bool=true) where lip
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    KD, J = K*D, data.J
    ws = WENDyWorkspace(lip=lip, data=data, diagReg=params.diagReg, L₀=_L₀(data.Vp, data.sig))
    if lip
        ws.∇r = data.G
        ws.∇L = _L₁(data)
        if useBlocks && (J+1)^2*KD^2*sizeof(Float64) <= MAX_LLᵀ_BLOCK_BYTES
            ws.P = _LLᵀBlocks(ws.L₀, ws.∇L)
            ws.L₀, ws.∇L = zeros(0,0), nothing # not needed anymore
        end
    end
    ws
end
## L(p), ∇L(p), r(p), ∇r(p) (not cached, so least squares solvers can call these at any p)
_L(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = ws.lip ? _L(p, ws.L₀, ws.∇L) : _L(p, ws.data, ws.L₀)
_∇L(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = ws.lip ? ws.∇L : _∇L(p, ws.data)
_r(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = ws.lip ? ws.data.G*p - ws.data.b₀ : _r(p, ws.data)
_∇r(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = ws.lip ? ws.data.G : _∇r(p, ws.data)
## Cached state
# Bring the workspace up to date with p: S(p), its factorization, log|S|, r(p) and y = S⁻¹r
function _update!(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    ws.p == p && return ws
    ws.p = Float64[] # marks the state as invalid until everything below is done
    ws.S⁻¹ = ws.M = nothing
    ws.lip || (ws.∇r = ws.∇L = nothing)
    ws.q = [1; p]
    if isnothing(ws.P)
        ws.L = _L(ws, p)
        ws.S = ws.L * ws.L'
        ws.S .*= 1-ws.diagReg
        ws.S[diagind(ws.S)] .+= ws.diagReg
    else
        ws.S = _S(ws.q, ws.P, ws.diagReg)
    end
    ws.r = _r(ws, p)
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
# S⁻¹ (blocks of LLᵀ only)
_S⁻¹!(ws::WENDyWorkspace) = isnothing(ws.S⁻¹) ? (ws.S⁻¹ = inv(ws.F)) : ws.S⁻¹
# ∇r(p) and ∇L(p) (L(p) only)
function _jacobians!(ws::WENDyWorkspace)
    isnothing(ws.∇r) && (ws.∇r = _∇r(ws, ws.p))
    isnothing(ws.∇L) && (ws.∇L = _∇L(ws, ws.p))
    nothing
end
# M = S⁻¹L - y(Lᵀy)ᵀ (L(p) only)
_M!(ws::WENDyWorkspace) = isnothing(ws.M) ? (ws.M = ws.F \ ws.L - ws.y * (ws.L'ws.y)') : ws.M
## m(p), ∇m(p), Hm(p) on the workspace (see wendyProblems.jl for the versions on a WENDyProblem)
function wnll(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    _update!(ws, p)
    (dot(ws.r, ws.y) + ws.logDet)/2 + length(ws.r)/2*log(2*pi)
end
function ∇wnll(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    _update!(ws, p)
    isnothing(ws.P) || return _∇wnll(ws.q, ws.P, ws.data.G, ws.y, _S⁻¹!(ws), ws.diagReg)
    _jacobians!(ws)
    _∇wnll(ws.∇L, ws.∇r, ws.y, _M!(ws), ws.diagReg)
end
function Hwnll(ws::WENDyWorkspace, p::AbstractVector{<:Real}; approx::Bool=false)
    _update!(ws, p)
    isnothing(ws.P) || return _Hwnll(ws.q, ws.P, ws.data.G, ws.F, approx ? nothing : _S⁻¹!(ws), ws.y, ws.diagReg)
    _jacobians!(ws)
    d = ws.data
    second = !ws.lip # second order terms only when not linear in parameters
    _Hwnll(
        ws.F, ws.∇L, ws.L, ws.∇r, ws.y, ws.diagReg,
        second ? _HwJuF(p, d) : nothing, second ? _HwF(p, d) : nothing, second ? _M!(ws) : nothing, d.V, d.sig;
        approx=approx
    )
end
