## Equations used when the ode rhs is not linear in p (see NonlinearWeakForm)
# ∇L(p), ∇L[:,:,j] = ∂ⱼL
function _∇L(p::AbstractVector{<:Real}, tt, X, V, sig, ∇ₚ∇ₓf!)
    Mp1, D = size(X)
    K, _ = size(V)
    J = length(p)
    JwJuF = zeros(D, D, J, Mp1)
    for m in 1:Mp1 
        @views ∇ₚ∇ₓf!(JwJuF[:,:,:,m], X[m,:], p, tt[m])
    end
    @tullio ∇L[k,d2,m,d1,j] := JwJuF[d2,d1,j,m] * V[k,m] * sig[d1] 
    reshape(∇L, K*D, Mp1*D, J)
end
# r(p) = G(p) - b₀ where G(p) = V F(p) and F(p)[m,:] = f(xₘ, p, tₘ)
function _r(p::AbstractVector{T}, tt, X, V, b₀, f!) where T<:Real
    Mp1, D = size(X)
    F = zeros(T, Mp1, D)
    for m in 1:Mp1
        @views f!(F[m,:], X[m,:], p, tt[m])
    end
    vec(V * F) - b₀
end
# ∇r(p) = ∇G(p)
function _∇r(p::AbstractVector{<:Real}, tt, X, V, ∇ₚf!)
    Mp1, D = size(X)
    K, _ = size(V)
    J = length(p)
    JwF = zeros(D, J, Mp1)
    for m in 1:Mp1
        @views ∇ₚf!(JwF[:,:,m], X[m,:], p, tt[m])
    end
    @tullio ∇r[k,d,j] := V[k,m] * JwF[d,j,m] 
    reshape(∇r, K*D, J)
end
# Hessian of the rhs wrt p at each time point: HwF[:,:,:,m] = ∇ₚ²f(xₘ) (∇²r = V∘HwF is never formed)
function _HwF(p::AbstractVector{<:Real}, tt, X, Hₚf!)
    Mp1, D = size(X)
    J = length(p)
    HwF = zeros(D, J, J, Mp1)
    for m in 1:Mp1 
        @views Hₚf!(HwF[:,:,:,m], X[m,:], p, tt[m])
    end
    HwF
end
# Hessian of the state jacobian of the rhs wrt p at each time point: HwJuF[:,:,:,:,m] = ∇ₚ²∇ₓf(xₘ) (∇²L = V∘HwJuF∘diag(σ) is never formed)
function _HwJuF(p::AbstractVector{<:Real}, tt, X, Hₚ∇ₓf!)
    Mp1, D = size(X)
    J = length(p)
    HwJuF = zeros(D, D, J, J, Mp1)
    for m in 1:Mp1 
        @views Hₚ∇ₓf!(HwJuF[:,:,:,:,m], X[m,:], p, tt[m])
    end
    HwJuF
end
"""
Second order terms of the hessian, ∂ᵢⱼrᵀy + (1-δ)⟨∂ᵢⱼL, M⟩ with y = S⁻¹r and M = S⁻¹L - y(Lᵀy)ᵀ, contracted through 
V without forming ∇²r or ∇²L:
    ∂ᵢⱼrᵀy    = ∑ HwF[d,i,j,m] (VᵀY)[m,d],                  Y = reshape(y, K, D)
    ⟨∂ᵢⱼL, M⟩ = ∑ HwJuF[d₂,d₁,i,j,m] σ[d₁] (VᵀM)[d₂,m,d₁],  M reshaped to K × D × Mp1 × D
"""
function _Hsecond(p::AbstractVector{<:Real}, tt, X, V, sig, Hₚf!, Hₚ∇ₓf!, y::AbstractVector{<:Real}, M::AbstractMatrix{<:Real}, diagReg::Real)
    Mp1, D = size(X)
    K, _ = size(V)
    HwF, HwJuF = _HwF(p, tt, X, Hₚf!), _HwJuF(p, tt, X, Hₚ∇ₓf!)
    VᵀY = V' * reshape(y, K, D)
    @tullio H2[i,j] := HwF[d,i,j,m] * VᵀY[m,d]
    M₄ = reshape(M, K, D, Mp1, D)
    @tullio VᵀM[d2,m,d1] := V[k,m] * M₄[k,d2,m,d1]
    @tullio H3[i,j] := HwJuF[d2,d1,i,j,m] * sig[d1] * VᵀM[d2,m,d1]
    H2 + (1-diagReg) * H3
end
