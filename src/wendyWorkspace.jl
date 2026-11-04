## Workspace for the weak form negative log-likelihood m(p), its gradient ∇m(p) and hessian Hm(p)
# The optimizers usually evaluate m, ∇m and Hm at the same p, so they share this state. Whenever p changes
# (compared by value) we recompute S(p), its factorization, r(p) and y = S⁻¹r. The remaining pieces are only
# computed the first time something asks for them at that p. Everything else here is a buffer.
#
# When the problem is linear in parameters we precompute the blocks of LLᵀ (see _LLᵀBlocks!) so nothing per p
# touches the Mp1*D dimension, unless they would take more than this much memory, then we fall back to L(p).
const MAX_LLᵀ_BLOCK_BYTES = 2^30
@kwdef mutable struct WENDyWorkspace
    # data
    lip::Bool
    data::WENDyInternals
    diagReg::Real
    P::Union{Nothing,AbstractArray{<:Real,3}} # blocks of LLᵀ (nothing ⇒ build S from L(p))
    L₀::AbstractMatrix{<:Real}                # p independent part of L
    L₁::Union{Nothing,AbstractArray{<:Real,3}} # ∇L, constant when linear in parameters
    # the parameters everything below was computed for
    p::AbstractVector{<:Real}
    hasP::Bool = false
    # computed whenever p changes
    q::AbstractVector{<:Real} # [1; p]
    L::AbstractMatrix{<:Real}
    S::AbstractMatrix{<:Real}
    F::Factorization
    logDet::Real = 0.0
    r::AbstractVector{<:Real}
    S⁻¹r::AbstractVector{<:Real}
    # computed on request
    hasS⁻¹::Bool = false # (blocks of LLᵀ only)
    S⁻¹::AbstractMatrix{<:Real}
    hasJacobians::Bool = false # (L(p) only)
    ∇r::AbstractMatrix{<:Real}
    ∇L::AbstractArray{<:Real,3}
    hasM::Bool = false # (L(p) only)
    Lᵀy::AbstractVector{<:Real}
    M::AbstractMatrix{<:Real} # S⁻¹L - y(Lᵀy)ᵀ
    # buffers for evaluating L(p), ∇L(p), r(p), ∇r(p) when not linear in parameters
    JuF::AbstractArray{<:Real,3}
    _L₁::AbstractArray{<:Real,4}
    JwJuF::AbstractArray{<:Real,4}
    _∇L::AbstractArray{<:Real,5}
    Fₘ::AbstractMatrix{<:Real}
    Gₖ::AbstractMatrix{<:Real}
    JwF::AbstractArray{<:Real,3}
    __∇r::AbstractArray{<:Real,3}
    _∇r::AbstractArray{<:Real,3}
    # buffers for the gradient and hessian
    Qⱼ::AbstractMatrix{<:Real}
    S⁻¹∇r::AbstractMatrix{<:Real}
    ∇Lᵀy::AbstractMatrix{<:Real}
    ∇Sy::AbstractMatrix{<:Real}
    S⁻¹∇Sy::AbstractMatrix{<:Real}
    ∂ⱼS::AbstractMatrix{<:Real}
    S⁻¹∇S::AbstractArray{<:Real,3}
    S⁻¹∂ⱼL::AbstractMatrix{<:Real}
    S⁻¹_yyᵀ::AbstractMatrix{<:Real}
    # buffers for the second order terms of the hessian when not linear in parameters
    HwF::AbstractArray{<:Real,4}
    HwJuF::AbstractArray{<:Real,5}
    VᵀY::AbstractMatrix{<:Real}
    VᵀM::AbstractArray{<:Real,3}
end
# useBlocks = false forces S to be built from L(p) even when the problem is linear in parameters
function WENDyWorkspace(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters; useBlocks::Bool=true) where lip
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    KD, MD = K*D, Mp1*D
    L₀ = zeros(KD, MD)
    _L₀!(L₀, data.Vp, data.sig, zeros(K,D,D,Mp1), zeros(K,D,Mp1,D))
    L₁ = nothing
    P = nothing
    if lip
        L₁ = zeros(KD, MD, J)
        _L₁!(L₁, data.tt, data.X, data.V, data.sig, data.∇ₓf!, zeros(D,D,Mp1), zeros(K,D,D,Mp1), zeros(K,D,Mp1,D), zeros(J))
        if useBlocks && (J+1)^2*KD^2*sizeof(Float64) <= MAX_LLᵀ_BLOCK_BYTES
            P = zeros(KD*KD, J+1, J+1)
            _LLᵀBlocks!(P, L₀, L₁, zeros(KD, KD))
            L₀, L₁ = zeros(0,0), nothing # not needed anymore
        end
    end
    usesL = isnothing(P)
    # sizes of what the chosen path does not need are 0
    n(x) = (lip ? 0 : x)       # only needed when not linear in parameters
    l(x) = (usesL ? x : 0)     # only needed when S is built from L(p)
    b(x) = (usesL ? 0 : x)     # only needed when S is built from the blocks of LLᵀ
    WENDyWorkspace(
        lip=lip, data=data, diagReg=params.diagReg, P=P, L₀=L₀, L₁=L₁,
        p=fill(NaN, J), q=zeros(J+1),
        L=zeros(l(KD), l(MD)), S=zeros(KD, KD), F=cholesky(Matrix{Float64}(I, KD, KD)), r=zeros(KD), S⁻¹r=zeros(KD),
        S⁻¹=zeros(b(KD), b(KD)),
        ∇r=lip ? data.G : zeros(KD, J), ∇L=lip ? (usesL ? L₁ : zeros(0,0,0)) : zeros(KD, MD, J),
        Lᵀy=zeros(l(MD)), M=zeros(l(KD), l(MD)),
        JuF=zeros(D, D, n(Mp1)), _L₁=zeros(K, D, n(Mp1), D), JwJuF=zeros(D, D, J, n(Mp1)), _∇L=zeros(K, D, n(Mp1), D, J),
        Fₘ=zeros(n(Mp1), D), Gₖ=zeros(n(K), D), JwF=zeros(D, J, n(Mp1)), __∇r=zeros(D, J, n(K)), _∇r=zeros(n(K), D, J),
        Qⱼ=zeros(b(KD), b(KD)), S⁻¹∇r=zeros(KD, J), ∇Lᵀy=zeros(l(MD), J), ∇Sy=zeros(KD, J), S⁻¹∇Sy=zeros(KD, J),
        ∂ⱼS=zeros(KD, KD), S⁻¹∇S=zeros(KD, KD, J), S⁻¹∂ⱼL=zeros(l(KD), l(MD)), S⁻¹_yyᵀ=zeros(b(KD), b(KD)),
        HwF=zeros(D, J, J, n(Mp1)), HwJuF=zeros(D, D, J, J, n(Mp1)), VᵀY=zeros(n(Mp1), D), VᵀM=zeros(D, n(Mp1), D)
    )
end
## Evaluate L(p), ∇L(p), r(p), ∇r(p) into the given arrays (never into the cached state, so least squares
## solvers can call these at any p)
function _evalL!(L::AbstractMatrix{<:Real}, ws::WENDyWorkspace, p::AbstractVector{<:Real})
    d = ws.data
    if ws.lip
        _L!(L, p, ws.L₁, ws.L₀)
    else
        _L!(L, p, d.tt, d.X, d.V, ws.L₀, d.sig, d.∇ₓf!, ws.JuF, ws._L₁)
    end
    L
end
function _eval∇L!(∇L::AbstractArray{<:Real,3}, ws::WENDyWorkspace, p::AbstractVector{<:Real})
    d = ws.data
    ws.lip ? (∇L .= ws.L₁) : _∇L!(∇L, p, d.tt, d.X, d.V, d.sig, d.∇ₚ∇ₓf!, ws.JwJuF, ws._∇L)
    ∇L
end
function _evalr!(r::AbstractVector{<:Real}, ws::WENDyWorkspace, p::AbstractVector{<:Real})
    d = ws.data
    ws.lip ? _r!(r, p, d.G, d.b₀) : _r!(r, p, d.tt, d.X, d.V, d.b₀, d.f!, ws.Fₘ, ws.Gₖ)
    r
end
function _eval∇r!(∇r::AbstractMatrix{<:Real}, ws::WENDyWorkspace, p::AbstractVector{<:Real})
    d = ws.data
    ws.lip ? (∇r .= d.G) : _∇r!(∇r, p, d.tt, d.X, d.V, d.∇ₚf!, ws.JwF, ws.__∇r, ws._∇r)
    ∇r
end
## Cached state
# Bring the workspace up to date with p: S(p), its factorization, log|S|, r(p) and y = S⁻¹r
function _update!(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    ws.hasP && ws.p == p && return ws
    ws.hasP = ws.hasS⁻¹ = ws.hasJacobians = ws.hasM = false
    ws.p .= p
    ws.q[1] = 1
    @views ws.q[2:end] .= p
    if isnothing(ws.P)
        _evalL!(ws.L, ws, p)
        mul!(ws.S, ws.L, ws.L', 1-ws.diagReg, 0)
        @views ws.S[diagind(ws.S)] .+= ws.diagReg
    else
        _S!(ws.S, ws.q, ws.P, ws.diagReg)
    end
    _evalr!(ws.r, ws, p)
    ws.F, ws.logDet = try
        F = cholesky(Symmetric(ws.S))
        F, 2*sum(log, diag(F.U))
    catch
        F = lu(ws.S)
        F, sum(log.(filter!(x-> x >0, diag(F.U))))
    end
    ldiv!(ws.S⁻¹r, ws.F, ws.r)
    ws.hasP = true
    ws
end
# S⁻¹ (blocks of LLᵀ only)
function _S⁻¹!(ws::WENDyWorkspace)
    if !ws.hasS⁻¹
        copyto!(ws.S⁻¹, I)
        ldiv!(ws.F, ws.S⁻¹)
        ws.hasS⁻¹ = true
    end
    ws.S⁻¹
end
# ∇r(p) and ∇L(p) (L(p) only, constant when linear in parameters)
function _jacobians!(ws::WENDyWorkspace)
    if !ws.hasJacobians && !ws.lip
        _eval∇r!(ws.∇r, ws, ws.p)
        _eval∇L!(ws.∇L, ws, ws.p)
    end
    ws.hasJacobians = true
    nothing
end
# M = S⁻¹L - y(Lᵀy)ᵀ (L(p) only)
function _M!(ws::WENDyWorkspace)
    if !ws.hasM
        ldiv!(ws.M, ws.F, ws.L)
        mul!(ws.Lᵀy, ws.L', ws.S⁻¹r)
        mul!(ws.M, ws.S⁻¹r, ws.Lᵀy', -1, 1)
        ws.hasM = true
    end
    ws.M
end
## m(p), ∇m(p), Hm(p) on the workspace (see wendyProblems.jl for the versions on a WENDyProblem)
function wnll(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    _update!(ws, p)
    KD = length(ws.r)
    return (dot(ws.r, ws.S⁻¹r) + ws.logDet)/2 + KD/2*log(2*pi)
end
function ∇wnll!(g::AbstractVector{<:Real}, ws::WENDyWorkspace, p::AbstractVector{<:Real})
    _update!(ws, p)
    if isnothing(ws.P)
        _jacobians!(ws)
        _∇wnll!(g, ws.∇L, ws.∇r, ws.S⁻¹r, _M!(ws), ws.diagReg)
    else
        _∇wnll!(g, ws.q, ws.P, ws.data.G, ws.S⁻¹r, _S⁻¹!(ws), ws.diagReg, ws.Qⱼ)
    end
    g
end
function Hwnll!(H::AbstractMatrix{<:Real}, ws::WENDyWorkspace, p::AbstractVector{<:Real}; approx::Bool=false)
    _update!(ws, p)
    if isnothing(ws.P)
        _jacobians!(ws)
        d = ws.data
        if !ws.lip
            _HwF!(ws.HwF, p, d.tt, d.X, d.Hₚf!)
            _HwJuF!(ws.HwJuF, p, d.tt, d.X, d.Hₚ∇ₓf!)
        end
        _Hwnll!(
            H,
            ws.F, ws.∇L, ws.L, ws.∇r, ws.S⁻¹r, ws.diagReg,
            ws.lip ? nothing : ws.HwJuF, ws.lip ? nothing : ws.HwF, ws.lip ? nothing : _M!(ws), d.V, d.sig,
            ws.S⁻¹∇r, ws.Lᵀy, ws.∇Lᵀy, ws.∇Sy, ws.S⁻¹∇Sy, ws.∂ⱼS, ws.S⁻¹∇S, ws.S⁻¹∂ⱼL,
            ws.VᵀY, ws.VᵀM;
            approx=approx
        )
    else
        _Hwnll!(
            H,
            ws.q, ws.P, ws.data.G, ws.F, approx ? nothing : _S⁻¹!(ws), ws.S⁻¹r, ws.diagReg,
            ws.S⁻¹∇r, ws.Qⱼ, ws.∂ⱼS, ws.∇Sy, ws.S⁻¹∇Sy, ws.S⁻¹∇S, ws.S⁻¹_yyᵀ
        )
    end
    H
end
