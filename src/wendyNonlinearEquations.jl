# L(w)
function _L!(
    L::AbstractMatrix{<:Real},w::AbstractVector{<:Real}, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, L₀::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, # data
    ∇ₓf!::Function, # functions
    JuF::AbstractArray{<:Real, 3}, __L₁::AbstractArray{<:Real, 4}, _L₁::AbstractArray{<:Real, 4}; # buffers
    ll::LogLevel=Warn # kwargs
) 
    Mp1, D = size(X)
    K, _ = size(V)
    @inbounds for m in 1:Mp1
        @views ∇ₓf!(JuF[:,:,m], X[m,:], w, tt[m])
    end
    @tullio _L₁[k,d2,m,d1] = JuF[d2,d1,m] * V[k,m]* sig[d1] # increases allocation from 4 to 45 
    # permutedims!(_L₁,__L₁,(1,2,4,3))
    @views L .= reshape(_L₁,K*D,Mp1*D) + L₀
    nothing
end
# ∇L(w)
function _∇L!(
    ∇L::AbstractArray{<:Real,3}, w::AbstractVector{<:Real},
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},V::AbstractMatrix{<:Real},sig::AbstractVector{<:Real},
    ∇ₚ∇ₓf!::Function,
    JwJuF::AbstractArray{<:Real,4}, __∇L::AbstractArray{<:Real,5}, _∇L::AbstractArray{<:Real,5})
    Mp1, D = size(X)
    K, _ = size(V)
    J = length(w)
    # compute ∇L
    @inbounds for m in 1:Mp1 
        @views ∇ₚ∇ₓf!(JwJuF[:,:,:,m], X[m,:], w, tt[m])
    end
    @tullio _∇L[k,d2,m,d1,j] = JwJuF[d2,d1,j,m] * V[k,m] * sig[d1] 
    # @tullio __∇L[k,d2,d1,j,m] = JwJuF[d2,d1,j,m] * V[k,m] * sig[d1] 
    # permutedims!(_∇L,__∇L,(1,2,5,3,4))
    @views ∇L .= reshape(_∇L,K*D,Mp1*D,J)
    nothing
end
# G(w)
function _g!(g::AbstractVector, w::AbstractVector, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix, V::AbstractMatrix, # data
    f!::Function, # function
    F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}; # buffers
    ll::LogLevel=Warn #kwargs
)
    Mp1, D = size(X)
    K, _ = size(V)
    for m in 1:Mp1
        @views f!(F[m,:], X[m,:], w, tt[m])
    end
    mul!(G, V, F)
    @views g .= reshape(G, K*D)
    nothing
end
# r(w) = G(w) - b₀
function _r!(
    r::AbstractVector,w::AbstractVector, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix, V::AbstractMatrix, b₀::AbstractVector, # data
    f!::Function, # function
    F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}; # buffers
    ll::LogLevel=Warn #kwargs
) 
    _g!(
        r, w, 
        tt, X, V, 
        f!,
        F, G; 
        ll=ll
    )
    @views r .-= b₀
    nothing
end
# Weighted residual (Rᵀ)⁻¹(G(w)) - b, where b = (Rᵀ)⁻¹b₀
function _Rᵀr!(r::AbstractVector, w::AbstractVector, # output/input
     tt::AbstractVector{<:Real}, X::AbstractMatrix, V::AbstractMatrix, Rᵀ::AbstractMatrix,b::AbstractVector, # Data
     f!::Function, # functions
     F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}, g::AbstractVector; # buffeers   
     ll::LogLevel=Warn #kwargs
) 
    _g!(g, w, tt, X, V, f!, F, G; ll=ll)
    ldiv!(r, LowerTriangular(Rᵀ), g)
    @views r .-= b
    nothing
end
# ∇r = ∇G
function _∇r!(
    ∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}, # output/input
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, 
    ∇ₚf!::Function, # functions
    JwF::AbstractArray{<:Real, 3}, __∇r::AbstractArray{<:Real, 3}, _∇r::AbstractArray{<:Real, 3}; # buffers
    ll::LogLevel=Warn # kwargs
) 
    Mp1, D = size(X)
    K, _ = size(V)
    _, J = size(∇r)
    @assert length(w) == J "w must be of length $J"

    @inbounds for m in 1:Mp1
        @views ∇ₚf!(JwF[:,:,m], X[m,:], w, tt[m])
    end
    # TODO maybe make the dimensions slightly different _JG[k,d,j] to minimize permutedims
    @tullio __∇r[d,j,k] = V[k,m] * JwF[d,j,m] 
    permutedims!(_∇r, __∇r,(3,1,2))
    @views ∇r .= reshape(_∇r, K*D, J)
    nothing
end
function _Hwnll!(
    H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real},
    ∇L::AbstractArray{<:Real, 3}, tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, _Y::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, L::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}, ∇r::AbstractMatrix{<:Real}, b₀::AbstractVector{<:Real}, sig::AbstractVector{<:Real},
    r::AbstractVector{<:Real},  
    Hₚf!::Function, Hₚ∇ₓf!::Function,  S⁻¹r::AbstractVector{<:Real}, 
    S⁻¹∇r::AbstractMatrix{<:Real}, ∂ⱼLLᵀ::AbstractMatrix{<:Real}, ∇S::AbstractArray{<:Real, 3}, HwF::AbstractArray{<:Real, 4}, _∇²r::AbstractArray{<:Real, 4}, ∇²r::AbstractArray{<:Real, 3}, HwJuF::AbstractArray{<:Real, 5}, __∇²L::AbstractArray{<:Real, 6}, _∇²L::AbstractArray{<:Real, 6}, ∇²L::AbstractArray{<:Real, 4}, ∂ⱼL∂ᵢLᵀ::AbstractMatrix{<:Real}, ∂ᵢⱼLLᵀ::AbstractMatrix{<:Real}, ∂ᵢⱼS::AbstractMatrix{<:Real}, S⁻¹∂ⱼS::AbstractMatrix{<:Real}, ∂ᵢSS⁻¹∂ⱼS::AbstractMatrix{<:Real}
)
    Mp1, D = size(X)
    K, _ = size(V)
    J = length(w)
    # Hm(w) - Hessian of Maholinobis distance 
    F = svd(S)
    ## Precompute S⁻¹(G(w)-b) and S⁻¹∂ⱼG(w)
    begin
        ldiv!(S⁻¹r, F, r)
        ldiv!(S⁻¹∇r, F, ∇r)
    end
    ## Compute ∇S 
    tmp = similar(∂ⱼLLᵀ)
    @inbounds for j = 1:J 
        @views mul!(∂ⱼLLᵀ, ∇L[:,:,j], L')
        tmp .= ∂ⱼLLᵀ
        @views ∇S[:,:,j] .= ∂ⱼLLᵀ + (tmp)'
    end
    ## compute ∇²r
    begin
        @inbounds for m in 1:Mp1 
            @views Hₚf!(HwF[:,:,:,m], X[m,:], w, tt[m] )
        end
        @tullio _∇²r[k,d,j1,j2] = V[k,m] * HwF[d,j1,j2,m] 
        ∇²r = reshape(_∇²r,K*D,J,J)
    end
    ## compute ∇²L
    begin
        @inbounds for m in 1:Mp1 
            @views Hₚ∇ₓf!(HwJuF[:,:,:,:,m], _Y[m,:], w, tt[m])
        end
        # turns out this is slower
        # @time "outerprod" @tullio __∇²L[k,d2,d1,j1,j2,m] = V[k,m] * HwJuF[d2,d1,j1,j2,m] * sig[d1] 
        # @time "permutedims" permutedims!(_∇²L,__∇²L, (1,2,6,3,4,5))
        @tullio _∇²L[k,d2,m,d1,j1,j2] = V[k,m] * HwJuF[d2,d1,j1,j2,m] * sig[d1] 
        ∇²L = reshape(_∇²L,K*D,Mp1*D,J,J)
    end
    ## Compute ∇²m
    @inbounds for j = 1:J
        # this only depends on j so we do it once
        @views ldiv!(S⁻¹∂ⱼS, F, ∇S[:,:,j])
        for i = j:J  
            # Commpute ∂ᵢⱼS   
            @views mul!(∂ⱼL∂ᵢLᵀ, ∇L[:,:,j], (∇L[:,:,i])')
            @views mul!(∂ᵢⱼLLᵀ, ∇²L[:,:,i,j], L')
            @views ∂ᵢⱼS .= ∂ⱼL∂ᵢLᵀ + (∂ⱼL∂ᵢLᵀ)' + ∂ᵢⱼLLᵀ + (∂ᵢⱼLLᵀ)'
            # compute ∂ᵢSS⁻¹∂ⱼS
            @views mul!(∂ᵢSS⁻¹∂ⱼS, ∇S[:,:,i], S⁻¹∂ⱼS) 
            # make the building blocks
            @views prt0 = dot(∇²r[:,j,i], S⁻¹r)                # ∂ᵢⱼrᵀS⁻¹r
            @views prt1 = - dot(S⁻¹∇r[:,j], ∇S[:,:,i], S⁻¹r)   # ∂ⱼrᵀ∂ᵢS⁻¹r = -(S⁻¹∂ⱼr)ᵀ∂ᵢS(S⁻¹r)
            @views prt2 = dot(∇r[:,j], S⁻¹∇r[:,i])             # ∂ⱼrᵀS⁻¹∂ᵢr
            @views prt3 = - 2*dot(S⁻¹∇r[:,i], ∇S[:,:,j], S⁻¹r) # 2∂ᵢrᵀ∂ⱼS⁻¹r = -(S⁻¹∂ᵢr)ᵀ∂ⱼSS⁻¹r
            @views prt4 = - dot(S⁻¹r, ∂ᵢⱼS, S⁻¹r)              # rᵀ∂ᵢⱼS⁻¹r  = -(S⁻¹r)ᵀ∂ᵢⱼSS⁻¹r
            @views prt5 = 2*dot(S⁻¹r, ∂ᵢSS⁻¹∂ⱼS, S⁻¹r)         #            + 2(S⁻¹r)ᵀ∂ᵢSS⁻¹∂ⱼSS⁻¹r
            logDetTerm = - tr(F \ ∂ᵢSS⁻¹∂ⱼS) + tr(F \ ∂ᵢⱼS)
            # Put everything together
            H[j,i] = (
                1/2*(
                    2*(prt0+prt1+prt2) 
                    + prt3+prt4+prt5
                    + logDetTerm
                )
            )
        end
    end
    # Only compute the upper triangular part of the hessian
    # and because it is symmetric, all computations are done
    H .= Symmetric(H)
    nothing 
end 