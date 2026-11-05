## Equations used when the ode rhs is not linear in p
# (see _Lₚ for why each kernel has a version that takes the fields of WENDyInternals)
# L(p)
_L(p::AbstractVector{<:Real}, data::WENDyInternals, L₀::AbstractMatrix{<:Real}) = L₀ + _Lₚ(p, data)
# ∇L(p), ∇L[:,:,j] = ∂ⱼL
_∇L(p::AbstractVector{<:Real}, data::WENDyInternals) = _∇L(p, data.tt, data.X, data.V, data.sig, data.∇ₚ∇ₓf!)
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
_r(p::AbstractVector{<:Real}, data::WENDyInternals) = _r(p, data.tt, data.X, data.V, data.b₀, data.f!)
function _r(p::AbstractVector{T}, tt, X, V, b₀, f!) where T<:Real
    Mp1, D = size(X)
    F = zeros(T, Mp1, D)
    for m in 1:Mp1
        @views f!(F[m,:], X[m,:], p, tt[m])
    end
    vec(V * F) - b₀
end
# ∇r(p) = ∇G(p)
_∇r(p::AbstractVector{<:Real}, data::WENDyInternals) = _∇r(p, data.tt, data.X, data.V, data.∇ₚf!)
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
_HwF(p::AbstractVector{<:Real}, data::WENDyInternals) = _HwF(p, data.tt, data.X, data.Hₚf!)
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
_HwJuF(p::AbstractVector{<:Real}, data::WENDyInternals) = _HwJuF(p, data.tt, data.X, data.Hₚ∇ₓf!)
function _HwJuF(p::AbstractVector{<:Real}, tt, X, Hₚ∇ₓf!)
    Mp1, D = size(X)
    J = length(p)
    HwJuF = zeros(D, D, J, J, Mp1)
    for m in 1:Mp1 
        @views Hₚ∇ₓf!(HwJuF[:,:,:,:,m], X[m,:], p, tt[m])
    end
    HwJuF
end
