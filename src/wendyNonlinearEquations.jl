# L(p)
function _L!(
    L::AbstractMatrix{<:Real},p::AbstractVector{<:Real}, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, L₀::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, # data
    ∇ₓf!::Function, # functions
    JuF::AbstractArray{<:Real, 3}, _L₁::AbstractArray{<:Real, 4} # buffers
) 
    Mp1, D = size(X)
    K, _ = size(V)
    @inbounds for m in 1:Mp1
        @views ∇ₓf!(JuF[:,:,m], X[m,:], p, tt[m])
    end
    @tullio _L₁[k,d2,m,d1] = JuF[d2,d1,m] * V[k,m]* sig[d1] # increases allocation from 4 to 45 
    # permutedims!(_L₁,__L₁,(1,2,4,3))
    @views L .= reshape(_L₁,K*D,Mp1*D) + L₀
    nothing
end
# ∇L(p)
function _∇L!(
    ∇L::AbstractArray{<:Real,3}, p::AbstractVector{<:Real},
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},V::AbstractMatrix{<:Real},sig::AbstractVector{<:Real},
    ∇ₚ∇ₓf!::Function,
    JwJuF::AbstractArray{<:Real,4}, _∇L::AbstractArray{<:Real,5})
    Mp1, D = size(X)
    K, _ = size(V)
    J = length(p)
    # compute ∇L
    @inbounds for m in 1:Mp1 
        @views ∇ₚ∇ₓf!(JwJuF[:,:,:,m], X[m,:], p, tt[m])
    end
    @tullio _∇L[k,d2,m,d1,j] = JwJuF[d2,d1,j,m] * V[k,m] * sig[d1] 
    @views ∇L .= reshape(_∇L,K*D,Mp1*D,J)
    nothing
end
# G(p)
function _g!(g::AbstractVector, p::AbstractVector, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix, V::AbstractMatrix, # data
    f!::Function, # function
    F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real} # buffers
)
    Mp1, D = size(X)
    K, _ = size(V)
    for m in 1:Mp1
        @views f!(F[m,:], X[m,:], p, tt[m])
    end
    mul!(G, V, F)
    @views g .= reshape(G, K*D)
    nothing
end
# r(p) = G(p) - b₀
function _r!(
    r::AbstractVector,p::AbstractVector, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix, V::AbstractMatrix, b₀::AbstractVector, # data
    f!::Function, # function
    F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real} # buffers
) 
    _g!(
        r, p, 
        tt, X, V, 
        f!,
        F, G
    )
    @views r .-= b₀
    nothing
end
# Weighted residual (Rᵀ)⁻¹(G(p)) - b, where b = (Rᵀ)⁻¹b₀
function _Rᵀr!(r::AbstractVector, p::AbstractVector, # output/input
     tt::AbstractVector{<:Real}, X::AbstractMatrix, V::AbstractMatrix, Rᵀ::AbstractMatrix,b::AbstractVector, # Data
     f!::Function, # functions
     F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}, g::AbstractVector # buffeers   
) 
    _g!(g, p, tt, X, V, f!, F, G)
    ldiv!(r, LowerTriangular(Rᵀ), g)
    @views r .-= b
    nothing
end
# ∇r = ∇G
function _∇r!(
    ∇r::AbstractMatrix{<:Real}, p::AbstractVector{<:Real}, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, 
    ∇ₚf!::Function, # functions
    JwF::AbstractArray{<:Real, 3}, __∇r::AbstractArray{<:Real, 3}, _∇r::AbstractArray{<:Real, 3} # buffers
) 
    Mp1, D = size(X)
    K, _ = size(V)
    _, J = size(∇r)
    @assert length(p) == J "p must be of length $J"

    @inbounds for m in 1:Mp1
        @views ∇ₚf!(JwF[:,:,m], X[m,:], p, tt[m])
    end
    
    @tullio __∇r[d,j,k] = V[k,m] * JwF[d,j,m] 
    permutedims!(_∇r, __∇r,(3,1,2))
    @views ∇r .= reshape(_∇r, K*D, J)
    nothing
end
# Hessian of the rhs wrt p at each time point: HwF[:,:,:,m] = ∇ₚ²f(xₘ) (∇²r = V∘HwF is never formed)
function _HwF!(
    HwF::AbstractArray{<:Real,4}, p::AbstractVector{<:Real}, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, # data
    Hₚf!::Function # functions
)
    Mp1, _ = size(X)
    @inbounds for m in 1:Mp1 
        @views Hₚf!(HwF[:,:,:,m], X[m,:], p, tt[m] )
    end
    nothing
end
# Hessian of the state jacobian of the rhs wrt p at each time point: HwJuF[:,:,:,:,m] = ∇ₚ²∇ₓf(xₘ) (∇²L = V∘HwJuF∘diag(σ) is never formed)
function _HwJuF!(
    HwJuF::AbstractArray{<:Real,5}, p::AbstractVector{<:Real}, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, # data
    Hₚ∇ₓf!::Function # functions
)
    Mp1, _ = size(X)
    @inbounds for m in 1:Mp1 
        @views Hₚ∇ₓf!(HwJuF[:,:,:,:,m], X[m,:], p, tt[m])
    end
    nothing
end
