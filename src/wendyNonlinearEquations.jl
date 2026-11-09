## Equations used when the ode rhs is not linear in p (see NonlinearWeakForm)
# ∇L(p), ∇L[:,:,j] = ∂ⱼL
function _∇L(p::AbstractVector{<:Real}, tt, X, Φ, σ, ∇ₚ∇ₓf!, dims::WENDyDims)
    (; K, D, Mp1, J) = dims
    JwJuF = zeros(D, D, J, Mp1)
    for m in 1:Mp1 
        @views ∇ₚ∇ₓf!(JwJuF[:,:,:,m], X[m,:], p, tt[m])
    end
    @tullio ∇L[k,d2,m,d1,j] := JwJuF[d2,d1,j,m] * Φ[k,m] * σ[d1] 
    reshape(∇L, K*D, Mp1*D, J)
end
# r(p) = g(p) - b where g(p) = vec(Φ F(p)) and F(p)[m,:] = f(xₘ, p, tₘ)
function _r(p::AbstractVector{T}, tt, X, Φ, b, f!, dims::WENDyDims) where T<:Real
    (; D, Mp1) = dims
    F = zeros(T, Mp1, D)
    for m in 1:Mp1
        @views f!(F[m,:], X[m,:], p, tt[m])
    end
    vec(Φ * F) - b
end
# ∇r(p) = ∇g(p)
function _∇r(p::AbstractVector{<:Real}, tt, X, Φ, ∇ₚf!, dims::WENDyDims)
    (; K, D, Mp1, J) = dims
    JwF = zeros(D, J, Mp1)
    for m in 1:Mp1
        @views ∇ₚf!(JwF[:,:,m], X[m,:], p, tt[m])
    end
    @tullio ∇r[k,d,j] := Φ[k,m] * JwF[d,j,m] 
    reshape(∇r, K*D, J)
end
# Hessian of the rhs wrt p at each time point: HwF[:,:,:,m] = ∇ₚ²f(xₘ) (∇²r = Φ∘HwF is never formed)
function _HwF(p::AbstractVector{<:Real}, tt, X, Hₚf!, dims::WENDyDims)
    (; D, Mp1, J) = dims
    HwF = zeros(D, J, J, Mp1)
    for m in 1:Mp1 
        @views Hₚf!(HwF[:,:,:,m], X[m,:], p, tt[m])
    end
    HwF
end
# Hessian of the state jacobian of the rhs wrt p at each time point: HwJuF[:,:,:,:,m] = ∇ₚ²∇ₓf(xₘ) (∇²L = Φ∘HwJuF∘diag(σ) is never formed)
function _HwJuF(p::AbstractVector{<:Real}, tt, X, Hₚ∇ₓf!, dims::WENDyDims)
    (; D, Mp1, J) = dims
    HwJuF = zeros(D, D, J, J, Mp1)
    for m in 1:Mp1 
        @views Hₚ∇ₓf!(HwJuF[:,:,:,:,m], X[m,:], p, tt[m])
    end
    HwJuF
end
"""
Second order terms of the hessian, ∂ᵢⱼrᵀy + (1-δ)⟨∂ᵢⱼL, ∇Lℓ⟩ with y = S⁻¹r and ∇Lℓ = S⁻¹L - y(Lᵀy)ᵀ, contracted through 
Φ without forming ∇²r or ∇²L:
    ∂ᵢⱼrᵀy    = ∑ HwF[d,i,j,m] (ΦᵀY)[m,d],                  Y = reshape(y, K, D)
    ⟨∂ᵢⱼL, ∇Lℓ⟩ = ∑ HwJuF[d₂,d₁,i,j,m] σ[d₁] (Φᵀ∇Lℓ)[d₂,m,d₁],  ∇Lℓ reshaped to K × D × Mp1 × D
"""
function _Hsecond(p::AbstractVector{<:Real}, tt, X, Φ, σ, Hₚf!, Hₚ∇ₓf!, y::AbstractVector{<:Real}, ∇Lℓ::AbstractMatrix{<:Real}, diagReg::Real, dims::WENDyDims)
    (; K, D, Mp1) = dims
    HwF, HwJuF = _HwF(p, tt, X, Hₚf!, dims), _HwJuF(p, tt, X, Hₚ∇ₓf!, dims)
    ΦᵀY = Φ' * reshape(y, K, D)
    @tullio H2[i,j] := HwF[d,i,j,m] * ΦᵀY[m,d]
    ∇Lℓ₄ = reshape(∇Lℓ, K, D, Mp1, D)
    @tullio Φᵀ∇Lℓ[d2,m,d1] := Φ[k,m] * ∇Lℓ₄[k,d2,m,d1]
    @tullio H3[i,j] := HwJuF[d2,d1,i,j,m] * σ[d1] * Φᵀ∇Lℓ[d2,m,d1]
    H2 + (1-diagReg) * H3
end
