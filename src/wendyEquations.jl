## Equations that do not depend on if the problem is linear or not in parameters
# L(p) = L₀ + Lₚ(p) is the covariance factor of the weak residual, S(p) = (1-δ)LLᵀ + δI. Its rows are indexed
# by (k,d) and its columns by (m,d′), each reshaped to K*D × Mp1*D.
"""L₀ = V′ ∘ diag(σ₁,⋯,σ_D)"""
function _L₀(Vp::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real})
    K, Mp1 = size(Vp)
    D = length(sig)
    L₀ = zeros(K, D, Mp1, D)
    for d in 1:D
        L₀[:,d,:,d] = Vp * sig[d]
    end
    reshape(L₀, K*D, Mp1*D)
end
# The fields of WENDyInternals are abstractly typed, so kernels that loop over time points take its fields as 
# arguments (a function barrier) and get compiled for their concrete types; the data versions just unpack.
"""Lₚ(p) = V ∘ ∇ₓf(x, p) ∘ diag(σ₁,⋯,σ_D), the part of L that depends on p"""
_Lₚ(p::AbstractVector{<:Real}, data::WENDyInternals) = _Lₚ(p, data.tt, data.X, data.V, data.sig, data.∇ₓf!)
function _Lₚ(p::AbstractVector{T}, tt, X, V, sig, ∇ₓf!) where T<:Real
    Mp1, D = size(X)
    K, _ = size(V)
    JuF = zeros(T, D, D, Mp1)
    for m in 1:Mp1
        @views ∇ₓf!(JuF[:,:,m], X[m,:], p, tt[m])
    end
    @tullio Lₚ[k,d2,m,d1] := JuF[d2,d1,m] * V[k,m] * sig[d1]
    reshape(Lₚ, K*D, Mp1*D)
end
"""
∇m(p) - Gradient of weak form negative log likelihood in adjoint form. With y = S⁻¹r and
∂ⱼS = (1-δ)(∂ⱼLLᵀ + L∂ⱼLᵀ):
    yᵀ∂ⱼSy     = 2(1-δ) yᵀ∂ⱼL(Lᵀy)
    tr(S⁻¹∂ⱼS) = 2(1-δ) ⟨S⁻¹L, ∂ⱼL⟩
so ∂ⱼm = ∂ⱼrᵀy + (1-δ)⟨∂ⱼL, M⟩ with M = S⁻¹L - y(Lᵀy)ᵀ. This needs one solve S⁻¹L instead of forming ∂ⱼS
and solving against it for every j.
"""
function _∇wnll(∇L::AbstractArray{<:Real,3}, ∇r::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, M::AbstractMatrix{<:Real}, diagReg::Real)
    J = size(∇L, 3)
    ∇r'y + (1-diagReg) * (reshape(∇L, :, J)' * vec(M))
end
"""
Hm(p) terms that only depend on how S changes, shared by the linear and general Hessians.
With U = S⁻¹∇r, W = [∂₁Sy ⋯ ∂ⱼSy] and Aⱼ = S⁻¹∂ⱼS this is
    -UᵀW - WᵀU + WᵀS⁻¹W - ½[tr(AᵢAⱼ)]ᵢⱼ
Pass A = nothing to leave out the trace (see approx in _Hwnll).
"""
function _HwnllCovTerms(U::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, S⁻¹W::AbstractMatrix{<:Real}, A::Union{Nothing,AbstractVector})
    H = -U'W - W'U + W'S⁻¹W
    isnothing(A) && return H
    J = length(A)
    H - [dot(transpose(A[i]), A[j]) for i in 1:J, j in 1:J] / 2
end
"""
Hm(p) - Hessian of weak form negative log likelihood using trace identities. With y = S⁻¹r, U = S⁻¹∇r,
wⱼ = ∂ⱼSy, ℓ = Lᵀy, vⱼ = ∂ⱼLᵀy:
    Hᵢⱼ = ∂ᵢⱼrᵀy + ∂ⱼrᵀUᵢ - Uⱼᵀwᵢ - Uᵢᵀwⱼ + wᵢᵀS⁻¹wⱼ - ½tr(S⁻¹∂ᵢSS⁻¹∂ⱼS)
          + (1-δ)[⟨S⁻¹∂ⱼL, ∂ᵢL⟩ - vᵢᵀvⱼ + ⟨∂ᵢⱼL, M⟩],   M = S⁻¹L - yℓᵀ
so no solve or KD×KD product happens per pair (i,j). The second order terms are contracted through V without forming ∇²r or ∇²L:
    ∂ᵢⱼrᵀy          = ∑ HwF[d,i,j,m] (VᵀY)[m,d],                  Y = reshape(y, K, D)
    ⟨∂ᵢⱼL, M⟩       = ∑ HwJuF[d₂,d₁,i,j,m] σ[d₁] (VᵀM)[d₂,m,d₁],  M reshaped to K × D × Mp1 × D
Pass HwF = HwJuF = M = nothing when the problem is linear in parameters.

With approx = true the log-det curvature E = (1-δ)⟨S⁻¹∂ⱼL, ∂ᵢL⟩ - ½tr(S⁻¹∂ᵢSS⁻¹∂ⱼS) is left out. What remains is the
exact hessian of the Maholinobis distance ½rᵀS⁻¹r plus (1-δ)⟨M, ∂ᵢⱼL⟩, and wⱼ = (1-δ)(∂ⱼL ℓ + L vⱼ) only needs matvecs,
so the per j KD×Mp1D products and solves are skipped.
"""
function _Hwnll(
    F::Factorization, ∇L::AbstractArray{<:Real,3}, L::AbstractMatrix{<:Real}, ∇r::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, diagReg::Real,
    HwJuF::Union{Nothing,AbstractArray{<:Real,5}}, HwF::Union{Nothing,AbstractArray{<:Real,4}}, M::Union{Nothing,AbstractMatrix{<:Real}},
    V::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real};
    approx::Bool=false
)
    J = size(∇L, 3)
    K, Mp1 = size(V)
    D = length(sig)
    δ̃ = 1-diagReg
    U = F \ ∇r
    ℓ = L'y
    ∇Lᵀy = reduce(hcat, view(∇L,:,:,j)'y for j in 1:J) # vⱼ
    # ∂ⱼrᵀS⁻¹∂ᵢr - (1-δ)vᵢᵀvⱼ
    H = ∇r'U - δ̃ * ∇Lᵀy'∇Lᵀy
    W = zeros(length(y), J)
    A = approx ? nothing : Vector{Matrix{Float64}}(undef, J)
    for j in 1:J
        ∂ⱼL = @view ∇L[:,:,j]
        if approx
            # wⱼ = ∂ⱼSy = (1-δ)(∂ⱼL(Lᵀy) + L(∂ⱼLᵀy))
            W[:,j] = δ̃ * (∂ⱼL*ℓ + L*∇Lᵀy[:,j])
            continue
        end
        ∂ⱼLLᵀ = ∂ⱼL * L'
        ∂ⱼS = δ̃ .* (∂ⱼLLᵀ .+ ∂ⱼLLᵀ')
        W[:,j] = ∂ⱼS * y
        A[j] = F \ ∂ⱼS
        # (1-δ)⟨S⁻¹∂ⱼL, ∂ᵢL⟩ for all i
        H[:,j] += δ̃ * (reshape(∇L, :, J)' * vec(F \ ∂ⱼL))
    end
    H += _HwnllCovTerms(U, W, F \ W, A)
    # ∂ᵢⱼrᵀy
    if !isnothing(HwF)
        VᵀY = V' * reshape(y, K, D)
        @tullio H2[i,j] := HwF[d,i,j,m] * VᵀY[m,d]
        H += H2
    end
    # (1-δ)⟨∂ᵢⱼL, M⟩
    if !isnothing(HwJuF)
        M₄ = reshape(M, K, D, Mp1, D)
        @tullio VᵀM[d2,m,d1] := V[k,m] * M₄[k,d2,m,d1]
        @tullio H3[i,j] := HwJuF[d2,d1,i,j,m] * sig[d1] * VᵀM[d2,m,d1]
        H += δ̃ * H3
    end
    (H + H') / 2
end
