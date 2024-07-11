# L(w)
function _L!(
    L::AbstractMatrix{<:Real},w::AbstractVector{<:Real}, # output/input
    U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, L₀::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, # data
    jacuf!::Function, # functions
    JuF::AbstractArray{<:Real, 3}, __L₁::AbstractArray{<:Real, 4}, _L₁::AbstractArray{<:Real, 4}; # buffers
     ll::LogLevel=Warn # kwargs
) 
    K,D,M,_ = size(_L₁)
    @inbounds for m in 1:M
        jacuf!(view(JuF,:,:,m), w, view(U,:,m))
    end
    @tullio __L₁[k,d2,d1,m] = JuF[d2,d1,m] * V[k,m]* sig[d1] # increases allocation from 4 to 45 
    permutedims!(_L₁,__L₁,(1,2,4,3))
    @views L .= reshape(_L₁,K*D,M*D) + L₀
    nothing
end
# ∇L(w)
function _∇L!(
    ∇L::AbstractArray{<:Real,3}, w::AbstractVector{<:Real},
    jacwjacuf!::Function,
    JwJuF::AbstractArray{<:Real,4}, __∇L::AbstractArray{<:Real,5}, _∇L::AbstractArray{<:Real,5})
    # compute ∇L
    @inbounds for m in 1:M 
        @views jacwjacuf!(JwJuF[:,:,:,m], w, U[:,m])
    end
    @tullio __∇L[k,d2,d1,j,m] = JwJuF[d2,d1,j,m] * V[k,m] * sig[d1] 
    permutedims!(_∇L,__∇L,(1,2,5,3,4))
    @views ∇L .= reshape(_∇L,K*D,M*D,J)
    nothing
end
# G(w)
function _g!(g::AbstractVector, w::AbstractVector, # output/input
    U::AbstractMatrix, V::AbstractMatrix, # data
    f!::Function, # function
    F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}; # buffers
    ll::LogLevel=Warn #kwargs
)
    K, M = size(V)
    D, _ = size(U)
    for m in 1:M
        f!(view(F,:,m), w, view(U,:,m))
    end
    mul!(G, V, F')
    @views g .= reshape(G, K*D,1)
    nothing
end
# r(w) = G(w) - b₀
function _r!(
    r::AbstractVector,w::AbstractVector, # output/input
    U::AbstractMatrix, V::AbstractMatrix, b₀::AbstractVector, # data
    f!::Function, # function
    F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}; # buffers
    ll::LogLevel=Warn #kwargs
) 
    _g!(r, w, U, V, f!, F, G,; ll=ll)
    @views r .-= b₀
    nothing
end
# Weighted residual (Rᵀ)⁻¹(G(w)) - b, where b = (Rᵀ)⁻¹b₀
function _Rᵀr!(r::AbstractVector, w::AbstractVector, # output/input
     U::AbstractMatrix, V::AbstractMatrix, Rᵀ::AbstractMatrix,b::AbstractVector, # Data
     f!::Function, # functions
     F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}, g::AbstractVector; # buffeers   
     ll::LogLevel=Warn #kwargs
) 
    _g!(g,w, U, V, f!, F, G; ll=ll)
    ldiv!(r, LowerTriangular(Rᵀ), g)
    @views r .-= b
    nothing
end
# ∇r = ∇G
function _∇r!(
    ∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}, # output/input
    U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, 
    jacwf!::Function, # functions
    JwF::AbstractArray{<:Real, 3}, __∇r::AbstractArray{<:Real, 3}, _∇r::AbstractArray{<:Real, 3}; # buffers
    ll::LogLevel=Warn # kwargs
) 
    K, M = size(V)
    D, _ = size(U)
    J = length(w)
    @inbounds for m in 1:M
        jacwf!(view(JwF,:,:,m), w, view(U,:,m))
    end
    # TODO maybe make the dimensions slightly different _JG[k,d,j] to minimize permutedims
    @tullio __∇r[d,j,k] = V[k,m] * JwF[d,j,m] 
    # @inbounds for d = 1:D 
    #     mul!(view(_JG,d,:,:), V, view(JwF,d,:,:)') 
    # end
    permutedims!(_∇r, __∇r,(3,1,2))
    @views ∇r .= reshape(_∇r, K*D, J)
    nothing
end
# (Rᵀ)⁻¹∇r = (Rᵀ)⁻¹∇G
function _Rᵀ⁻¹∇r!(
    Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}, # output/input
    U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, Rᵀ::AbstractMatrix{<:Real}, # data
    jacwf!::Function, # functions
    JwF::AbstractArray{<:Real, 3}, __∇r::AbstractArray{<:Real, 3}, _∇r::AbstractArray{<:Real, 3}, ∇r::AbstractMatrix{<:Real}; # buffers
    ll::LogLevel=Warn # kwargs
) 
    _∇r!(
        ∇r,w,
        U,V,
        jacwf!,
        JwF,__∇r,_∇r;
        ll=ll
    )
    ldiv!(Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), ∇r)
    nothing
end
function _Hm!(
    H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real},
    ∇L::AbstractArray{<:Real, 3}, U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, L::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}, ∇r::AbstractMatrix{<:Real}, b₀::AbstractVector{<:Real}, sig::AbstractVector{<:Real},
    r::AbstractVector{<:Real},  heswf!::Function, heswjacuf!::Function,  S⁻¹r::AbstractVector{<:Real}, 
    S⁻¹∇r::AbstractMatrix{<:Real}, ∂ⱼLLᵀ::AbstractMatrix{<:Real}, ∇S::AbstractArray{<:Real, 3}, HwF::AbstractArray{<:Real, 4}, _∇²r::AbstractArray{<:Real, 4}, ∇²r::AbstractArray{<:Real, 3}, HwJuF::AbstractArray{<:Real, 5}, __∇²L::AbstractArray{<:Real, 6}, _∇²L::AbstractArray{<:Real, 6}, ∇²L::AbstractArray{<:Real, 4}, ∂ⱼL∂ᵢLᵀ::AbstractMatrix{<:Real}, ∂ᵢⱼLLᵀ::AbstractMatrix{<:Real}, ∂ᵢⱼS::AbstractMatrix{<:Real}, S⁻¹∂ⱼS::AbstractMatrix{<:Real}, ∂ᵢSS⁻¹∂ⱼS::AbstractMatrix{<:Real}
)
    D,M = size(U)
    K,_ = size(V)
    J = length(w)
    # Hm(w) - Hessian of Maholinobis distance 
    F = svd(S)
    ## Precompute S⁻¹(G(w)-b) and S⁻¹∂ⱼG(w)
    ldiv!(S⁻¹r, F, r)
    ldiv!(S⁻¹∇r, F, ∇r)
    ## Compute ∇S 
    @inbounds for j = 1:J 
        @views mul!(∂ⱼLLᵀ, ∇L[:,:,j], L')
        @views ∇S[:,:,j] .= ∂ⱼLLᵀ + (∂ⱼLLᵀ)'
    end
    ## compute ∇²r
    @inbounds for m in 1:M 
        heswf!(view(HwF,:,:,:,m), w, view(U,:,m))
    end
    # TODO remove: dummmy check for our particular example
    @tullio _∇²r[k,d,j1,j2] = V[k,m] * HwF[d,j1,j2,m] 
    ∇²r = reshape(_∇²r,K*D,J,J)
    ## compute ∇²L
    @inbounds for m in 1:M 
        heswjacuf!(view(HwJuF,:,:,:,:,m), w, view(U,:,m))
    end
    @tullio __∇²L[k,d2,d1,j1,j2,m] = V[k,m] * HwJuF[d2,d1,j1,j2,m] * sig[d1] 
    permutedims!(_∇²L,__∇²L, (1,2,6,3,4,5))
    ∇²L = reshape(_∇²L,K*D,M*D,J,J)
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
            # Put everything together
            H[j,i] = 1/2*(2*(prt0+prt1+prt2) + prt3+prt4+prt5)
        end
    end
    # Only compute the upper triangular part of the hessian
    # and because it is symmetric, all computations are done
    H .= Symmetric(H)
    nothing 
end 