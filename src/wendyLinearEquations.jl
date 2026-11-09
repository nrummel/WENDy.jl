## Equations that work if the ode rhs is linear in p
# Then L(p) = L₀ + ∑ⱼpⱼL₁ⱼ and r(p) = Gp - b with G = ∇r constant.
# L₁[:,:,j] = Lₚ(eⱼ), the gradient/jacobian of L(p)
function _L₁(tt, X, Φ, σ, ∇ₓf!, J::Int)
    eⱼ(j) = [i == j ? 1.0 : 0.0 for i in 1:J]
    cat((_Lₚ(eⱼ(j), tt, X, Φ, σ, ∇ₓf!) for j in 1:J)...; dims=3)
end
## L(p)
_L(p::AbstractVector{<:Real}, L₀::AbstractMatrix{<:Real}, L₁::AbstractArray{<:Real,3}) = L₀ + reshape(reshape(L₁, :, length(p)) * p, size(L₀))
## LLᵀ blocks - with q = [1; p] and L̃ = [L₀, L₁₁, ⋯, L₁ⱼ], LLᵀ = ∑ₐᵦ qₐqᵦ L̃ₐL̃ᵦᵀ, so we store
# P[:,a,b] = vec(L̃ₐL̃ᵦᵀ) once and never touch the Mp1*D dimension again
function _LLᵀBlocks(L₀::AbstractMatrix{<:Real}, L₁::AbstractArray{<:Real,3})
    J = size(L₁, 3)
    KD = size(L₀, 1)
    L̃(a) = a == 1 ? L₀ : view(L₁,:,:,a-1)
    P = zeros(KD*KD, J+1, J+1)
    for a in 1:J+1, b in a:J+1
        LₐLᵦᵀ = L̃(a) * L̃(b)'
        P[:,a,b] = vec(LₐLᵦᵀ)
        P[:,b,a] = vec(LₐLᵦᵀ')
    end
    P
end
# S(p) = (1-δ)∑ₐᵦ qₐqᵦ L̃ₐL̃ᵦᵀ + δI
function _S(q::AbstractVector{<:Real}, P::AbstractArray{<:Real,3}, diagReg::Real)
    KD = isqrt(size(P, 1))
    S = reshape(reshape(P, KD*KD, :) * vec(q*q'), KD, KD)
    S .*= 1-diagReg
    S[diagind(S)] .+= diagReg
    S
end
# Qⱼ = ∑ₐqₐL̃ₐL₁ⱼᵀ = LL₁ⱼᵀ
_Q(q::AbstractVector{<:Real}, P::AbstractArray{<:Real,3}, j::Int) = (KD = isqrt(size(P, 1)); reshape(view(P,:,:,j+1) * q, KD, KD))
# Cₐᵦ = ⟨B, L̃ₐL̃ᵦᵀ⟩ for every pair of blocks
_C(B::AbstractMatrix{<:Real}, P::AbstractArray{<:Real,3}) = reshape(reshape(P, size(P,1), :)' * vec(B), size(P,2), size(P,3))
## ∇ℓ(p) - gradient using the LLᵀ blocks. With y = S⁻¹r and ∂ⱼS = (1-δ)(Qⱼ + Qⱼᵀ)
# ∂ⱼℓ = Gⱼᵀy + (1-δ)⟨S⁻¹ - yyᵀ, Qⱼ⟩ = Gⱼᵀy + (1-δ)∑ₐqₐCₐ,ⱼ₊₁ with C = _C(S⁻¹ - yyᵀ, P)
function _∇ℓ(q::AbstractVector{<:Real}, P::AbstractArray{<:Real,3}, G::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, S⁻¹::AbstractMatrix{<:Real}, diagReg::Real)
    C = _C(S⁻¹ - y*y', P)
    G'y + (1-diagReg) * C[:,2:end]'q
end
## ∇²ℓ(p) - hessian of the Maholinobis distance when the ode is linear in parameters ⇒ ∀p  ∂ᵢⱼL = 0 and ∂ᵢⱼr = 0, so ∂ᵢⱼS = ∂ⱼL∂ᵢLᵀ + ∂ᵢL∂ⱼLᵀ is constant in p. Also notice that ∇ₚ[G*p -b] = G
# With the LLᵀ blocks, ∂ⱼS = (1-δ)(Qⱼ + Qⱼᵀ) and ⟨S⁻¹∂ⱼL, ∂ᵢL⟩ - vᵢᵀvⱼ = ⟨S⁻¹ - yyᵀ, L₁ᵢL₁ⱼᵀ⟩, so nothing here touches the Mp1*D dimension.
# With S⁻¹ = nothing (approx) the log-det curvature (1-δ)⟨S⁻¹, L₁ᵢL₁ⱼᵀ⟩ - ½tr(S⁻¹∂ᵢSS⁻¹∂ⱼS) is left out,
# which is exactly the hessian of the Maholinobis distance ½rᵀS⁻¹r, and no KD×KD products are needed
function _∇²ℓ(
    q::AbstractVector{<:Real}, P::AbstractArray{<:Real,3}, G::AbstractMatrix{<:Real}, Sfact::Factorization,
    S⁻¹::Union{Nothing,AbstractMatrix{<:Real}}, y::AbstractVector{<:Real}, diagReg::Real
)
    J = size(G, 2)
    approx = isnothing(S⁻¹)
    δ̃ = 1-diagReg
    S⁻¹∇r = Sfact \ G # ∇r = G
    ∂S = [(Qⱼ = _Q(q, P, j); δ̃ .* (Qⱼ .+ Qⱼ')) for j in 1:J]
    W = reduce(hcat, ∂ⱼS * y for ∂ⱼS in ∂S)
    # ∂ⱼrᵀS⁻¹∂ᵢr + covariance terms
    H = G'S⁻¹∇r + _∇²ℓCovTerms(S⁻¹∇r, W, Sfact \ W, approx ? nothing : [S⁻¹ * ∂ⱼS for ∂ⱼS in ∂S])
    # (1-δ)⟨S⁻¹ - yyᵀ, L₁ᵢL₁ⱼᵀ⟩, or just -(1-δ)⟨yyᵀ, L₁ᵢL₁ⱼᵀ⟩ when approx
    C = _C(approx ? -y*y' : S⁻¹ - y*y', P)
    H += δ̃ * C[2:end,2:end]
    (H + H') / 2
end
