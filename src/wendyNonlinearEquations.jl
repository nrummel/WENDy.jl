## Equations used when the ode rhs is not linear in p (see NonlinearWeakForm)
# ∇L(p), ∇L[:,:,j] = ∂ⱼL (row block d₂ uses the test functions V_{d₂} of state d₂, see _Lₚ)
function _∇L(p::AbstractVector{<:Real}, tt, X, V, sig, ∇ₚ∇ₓf!)
    Mp1, D = size(X)
    J = length(p)
    JwJuF = zeros(D, D, J, Mp1)
    for m in 1:Mp1 
        @views ∇ₚ∇ₓf!(JwJuF[:,:,:,m], X[m,:], p, tt[m])
    end
    cat((begin
        A, Vd = JwJuF[d2,:,:,:], V[d2]
        @tullio ∇L[k,m,d1,j] := A[d1,j,m] * Vd[k,m] * sig[d1]
        reshape(∇L, size(Vd, 1), Mp1*D, J)
    end for d2 in 1:D)...; dims=1)
end
# r(p) = G(p) - b₀ where row block d of G(p) is V_d F(p)[:,d] and F(p)[m,:] = f(xₘ, p, tₘ)
function _r(p::AbstractVector{T}, tt, X, V, b₀, f!) where T<:Real
    Mp1, D = size(X)
    F = zeros(T, Mp1, D)
    for m in 1:Mp1
        @views f!(F[m,:], X[m,:], p, tt[m])
    end
    reduce(vcat, V[d] * F[:,d] for d in 1:D) - b₀
end
# ∇r(p) = ∇G(p), row block d is V_d ∇ₚf_d
function _∇r(p::AbstractVector{<:Real}, tt, X, V, ∇ₚf!)
    Mp1, D = size(X)
    J = length(p)
    JwF = zeros(D, J, Mp1)
    for m in 1:Mp1
        @views ∇ₚf!(JwF[:,:,m], X[m,:], p, tt[m])
    end
    reduce(vcat, begin
        A, Vd = JwF[d,:,:], V[d]
        @tullio ∇r[k,j] := Vd[k,m] * A[j,m]
    end for d in 1:D)
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
    ∂ᵢⱼrᵀy    = ∑ HwF[d,i,j,m] (VᵀY)[m,d],                  (VᵀY)[:,d] = V_dᵀ y_d
    ⟨∂ᵢⱼL, M⟩ = ∑ HwJuF[d₂,d₁,i,j,m] σ[d₁] (VᵀM)[d₂,m,d₁],  (VᵀM)[d₂,m,d₁] = ∑ₖ V_{d₂}[k,m] M_{d₂}[k,(m,d₁)]
where y_d and M_d are the row blocks of state d.
"""
function _Hsecond(p::AbstractVector{<:Real}, tt, X, V, sig, Hₚf!, Hₚ∇ₓf!, y::AbstractVector{<:Real}, M::AbstractMatrix{<:Real}, diagReg::Real)
    Mp1, D = size(X)
    K = size.(V, 1)
    HwF, HwJuF = _HwF(p, tt, X, Hₚf!), _HwJuF(p, tt, X, Hₚ∇ₓf!)
    VᵀY = reduce(hcat, V[d]' * y[_rows(K, d)] for d in 1:D)
    @tullio H2[i,j] := HwF[d,i,j,m] * VᵀY[m,d]
    VᵀM = permutedims(stack(begin
        M₃, Vd = reshape(M[_rows(K, d2), :], K[d2], Mp1, D), V[d2]
        @tullio VᵀMd[m,d1] := Vd[k,m] * M₃[k,m,d1]
    end for d2 in 1:D), (3, 1, 2))
    @tullio H3[i,j] := HwJuF[d2,d1,i,j,m] * sig[d1] * VᵀM[d2,m,d1]
    H2 + (1-diagReg) * H3
end
