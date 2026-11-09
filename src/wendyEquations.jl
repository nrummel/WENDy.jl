## Equations that do not depend on if the problem is linear or not in parameters
# L(p) = L₀ + Lₚ(p) is the covariance factor of the weak residual, S(p) = (1-δ)LLᵀ + δI. Its rows are indexed
# by (k,d) and its columns by (m,d′), each reshaped to K*D × Mp1*D.
"""L₀ = Φ̇ ∘ diag(σ₁,⋯,σ_D)"""
function _L₀(Φ̇::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    K, Mp1 = size(Φ̇)
    D = length(σ)
    L₀ = zeros(K, D, Mp1, D)
    for d in 1:D
        L₀[:,d,:,d] = Φ̇ * σ[d]
    end
    reshape(L₀, K*D, Mp1*D)
end
# Kernels that loop over time points take the symbolic function as an argument, so each is compiled for the 
# function's concrete type (the weak forms keep it as ::Function).
"""Lₚ(p) = Φ ∘ ∇ₓf(x, p) ∘ diag(σ₁,⋯,σ_D), the part of L that depends on p"""
function _Lₚ(p::AbstractVector{T}, tt, X, Φ, σ, ∇ₓf!) where T<:Real
    Mp1, D = size(X)
    K, _ = size(Φ)
    JuF = zeros(T, D, D, Mp1)
    for m in 1:Mp1
        @views ∇ₓf!(JuF[:,:,m], X[m,:], p, tt[m])
    end
    @tullio Lₚ[k,d2,m,d1] := JuF[d2,d1,m] * Φ[k,m] * σ[d1]
    reshape(Lₚ, K*D, Mp1*D)
end
"""
∇ℓ(p) - Gradient of weak form negative log likelihood in adjoint form. With y = S⁻¹r and
∂ⱼS = (1-δ)(∂ⱼLLᵀ + L∂ⱼLᵀ):
    yᵀ∂ⱼSy     = 2(1-δ) yᵀ∂ⱼL(Lᵀy)
    tr(S⁻¹∂ⱼS) = 2(1-δ) ⟨S⁻¹L, ∂ⱼL⟩
so ∂ⱼℓ = ∂ⱼrᵀy + (1-δ)⟨∂ⱼL, ∇Lℓ⟩ with ∇Lℓ = S⁻¹L - y(Lᵀy)ᵀ (so ∂ℓ/∂L = (1-δ)∇Lℓ). This needs one solve S⁻¹L instead of forming ∂ⱼS
and solving against it for every j.
"""
function _∇ℓ(∇L::AbstractArray{<:Real,3}, ∇r::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, ∇Lℓ::AbstractMatrix{<:Real}, diagReg::Real)
    J = size(∇L, 3)
    ∇r'y + (1-diagReg) * (reshape(∇L, :, J)' * vec(∇Lℓ))
end
"""
∇²ℓ(p) terms that only depend on how S changes, shared by the linear and general Hessians.
With W = [∂₁Sy ⋯ ∂ⱼSy] and A[:,:,j] = S⁻¹∂ⱼS this is
    -(S⁻¹∇r)ᵀW - Wᵀ(S⁻¹∇r) + WᵀS⁻¹W - ½[tr(AᵢAⱼ)]ᵢⱼ
Pass A = nothing to leave out the trace (see approx in _∇²ℓ).
"""
function _∇²ℓCovTerms(S⁻¹∇r::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, S⁻¹W::AbstractMatrix{<:Real}, A::Union{Nothing,AbstractArray{<:Real,3}})
    H = -S⁻¹∇r'W - W'S⁻¹∇r + W'S⁻¹W
    isnothing(A) && return H
    J = size(A, 3)
    H - [dot(transpose(view(A,:,:,i)), view(A,:,:,j)) for i in 1:J, j in 1:J] / 2
end
"""
∇²ℓ(p) - Hessian of weak form negative log likelihood using trace identities. With y = S⁻¹r, Sⱼy = ∂ⱼSy and vⱼ = ∂ⱼLᵀy:
    Hᵢⱼ = ∂ᵢⱼrᵀy + ∂ⱼrᵀ(S⁻¹∇r)ᵢ - (S⁻¹∇r)ⱼᵀSᵢy - (S⁻¹∇r)ᵢᵀSⱼy + SᵢyᵀS⁻¹Sⱼy - ½tr(S⁻¹∂ᵢSS⁻¹∂ⱼS)
          + (1-δ)[⟨S⁻¹∂ⱼL, ∂ᵢL⟩ - vᵢᵀvⱼ + ⟨∂ᵢⱼL, ∇Lℓ⟩],   ∇Lℓ = S⁻¹L - y(Lᵀy)ᵀ
so no solve or KD×KD product happens per pair (i,j). This returns everything but the second order terms 
∂ᵢⱼrᵀy + (1-δ)⟨∂ᵢⱼL, ∇Lℓ⟩, which depend on the weak form (see _Hsecond).

With approx = true the log-det curvature E = (1-δ)⟨S⁻¹∂ⱼL, ∂ᵢL⟩ - ½tr(S⁻¹∂ᵢSS⁻¹∂ⱼS) is left out. What remains is the
exact hessian of the Maholinobis distance ½rᵀS⁻¹r plus (1-δ)⟨∇Lℓ, ∂ᵢⱼL⟩, and Sⱼy = (1-δ)(∂ⱼL(Lᵀy) + L vⱼ) only needs matvecs,
so the per j KD×Mp1D products and solves are skipped.
"""
function _∇²ℓ(
    Sfact::Factorization, ∇L::AbstractArray{<:Real,3}, L::AbstractMatrix{<:Real}, ∇r::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, diagReg::Real;
    approx::Bool=false
)
    J = size(∇L, 3)
    δ̃ = 1-diagReg
    S⁻¹∇r = Sfact \ ∇r
    Lᵀy = L'y
    ∇Lᵀy = reduce(hcat, view(∇L,:,:,j)'y for j in 1:J) # vⱼ
    # ∂ⱼrᵀS⁻¹∂ᵢr - (1-δ)vᵢᵀvⱼ
    H = ∇r'S⁻¹∇r - δ̃ * ∇Lᵀy'∇Lᵀy
    W = zeros(length(y), J)
    A = approx ? nothing : Array{Float64}(undef, length(y), length(y), J)
    for j in 1:J
        ∂ⱼL = @view ∇L[:,:,j]
        if approx
            # Sⱼy = ∂ⱼSy = (1-δ)(∂ⱼL(Lᵀy) + L(∂ⱼLᵀy))
            W[:,j] = δ̃ * (∂ⱼL*Lᵀy + L*∇Lᵀy[:,j])
            continue
        end
        ∂ⱼLLᵀ = ∂ⱼL * L'
        ∂ⱼS = δ̃ .* (∂ⱼLLᵀ .+ ∂ⱼLLᵀ')
        W[:,j] = ∂ⱼS * y
        A[:,:,j] = Sfact \ ∂ⱼS
        # (1-δ)⟨S⁻¹∂ⱼL, ∂ᵢL⟩ for all i
        H[:,j] += δ̃ * (reshape(∇L, :, J)' * vec(Sfact \ ∂ⱼL))
    end
    H + _∇²ℓCovTerms(S⁻¹∇r, W, Sfact \ W, A) # (symmetrized by the caller, after adding the second order terms)
end
