## Equations that work if the ode rhs is linear in p
# L₁ - this is the gradient/jacobian of L(p) when L is linear 
function _L₁!(
    L₁::AbstractArray{<:Real,3}, 
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, 
    ∇ₓf!::Function, 
    JuF::AbstractArray{<:Real, 3}, _∂ⱼL::AbstractArray{<:Real, 4}, ∂ⱼL::AbstractArray{<:Real, 4}, eⱼ::AbstractVector{<:Real}
)
∂ⱼL
    Mp1,D = size(X)
    K,_ = size(V)
    J = length(eⱼ)
    KD = K*D
    MD = Mp1*D
    for j = 1:J 
        eⱼ .= 0
        eⱼ[j] = 1
        @inbounds for m in 1:Mp1
            @views ∇ₓf!(JuF[:,:,m], X[m,:], eⱼ, tt[m])
        end
        @tullio _∂ⱼL[k,d2,d1,m] = JuF[d2,d1,m] * V[k,m]* sig[d1] # increases allocation from 4 to 45 
        permutedims!(∂ⱼL,_∂ⱼL,(1,2,4,3))
        @views L₁[:,:,j] .= reshape(∂ⱼL,KD,MD) 
    end
    nothing
end
## L(p)
function _L!(
    L::AbstractMatrix{<:Real},p::AbstractVector{<:Real}, # output/input
    L₁::AbstractArray{<:Real}, L₀::AbstractMatrix{<:Real} # data
) 
    @tullio L[kd,md] = L₁[kd,md,j]*p[j] 
    L .+= L₀
    nothing
end
## G(p)
function _g!(g::AbstractVector, p::AbstractVector, # output/input
    G::AbstractMatrix{<:Real} # data
)
    mul!(g, G, p)
    nothing
end
# r(p) = G*p - b₀
function _r!(
    r::AbstractVector, p::AbstractVector, # output/input
    G::AbstractMatrix, b₀::AbstractVector # data
) 
    _g!(r, p, G )
    @views r .-= b₀
    nothing
end
# Weighted residual (Rᵀ)⁻¹(G(p)) - b, where b = (Rᵀ)⁻¹b₀
function _Rᵀr!(r::AbstractVector, p::AbstractVector, # output/input
     G::AbstractMatrix{<:Real}, Rᵀ::AbstractMatrix, b::AbstractVector, # Data
     g::AbstractVector # buffeers   
) 
    _g!(g, p, G)
    ldiv!(r, LowerTriangular(Rᵀ), g)
    @views r .-= b
    nothing
end
## Hm(p) - hessian of the Maholinobis distance when the ode is linear in parameters ⇒ ∀w  ∂ᵢⱼS = 0 and ∂ᵢⱼr = 0. Also notice that ∇_w[G*p -b₀] = G
function _Hwnll!(
    H::AbstractMatrix{<:Real}, p::AbstractVector{<:Real},
    ∇L::AbstractArray{<:Real,3}, G::AbstractMatrix{<:Real}, L::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}, 
    r::AbstractVector{<:Real}, 
    S⁻¹r::AbstractVector{<:Real}, S⁻¹∇r::AbstractMatrix{<:Real}, 
    ∂ⱼLLᵀ::AbstractMatrix{<:Real}, ∇S::AbstractArray{<:Real, 3}, 
    ∂ⱼL∂ᵢLᵀ::AbstractMatrix{<:Real}, ∂ᵢⱼS::AbstractMatrix{<:Real}, S⁻¹∂ⱼS::AbstractMatrix{<:Real}, ∂ᵢSS⁻¹∂ⱼS::AbstractMatrix{<:Real}
)
    @assert !all(0 .== ∇L)
    @assert !all(0 .== G)
    @assert !all(0 .== L)
    @assert !all(0 .== S)
    @assert !all(0 .== r)
    J = length(p)
    # Hm(p) - Hessian of Maholinobis distance 
    F = try 
        cholesky(S) 
    catch 
        lu(S)
    end
    ## Precompute S⁻¹(G(p)-b) and S⁻¹∂ⱼG(p)
    ldiv!(S⁻¹r, F, r)
    ldiv!(S⁻¹∇r, F, G)
    ## Compute ∇S 
    @inbounds for j = 1:J 
        @views mul!(∂ⱼLLᵀ, ∇L[:,:,j], L')
        @views ∇S[:,:,j] .= ∂ⱼLLᵀ + (∂ⱼLLᵀ)'
    end
    ## Compute ∇²m
    @inbounds for j = 1:J 
        # this only depends on j so we do it once
        @views ldiv!(S⁻¹∂ⱼS, F, ∇S[:,:,j])
        for i = j:J 
            # Commpute ∂ᵢⱼS   
            @views mul!(∂ⱼL∂ᵢLᵀ, ∇L[:,:,j], (∇L[:,:,i])')
            @views ∂ᵢⱼS .= ∂ⱼL∂ᵢLᵀ + (∂ⱼL∂ᵢLᵀ)'
            # compute ∂ᵢSS⁻¹∂ⱼS
            @views mul!(∂ᵢSS⁻¹∂ⱼS, ∇S[:,:,i], S⁻¹∂ⱼS) 
            # make the building blocks
            @views prt0 = - dot(S⁻¹∇r[:,j], ∇S[:,:,i], S⁻¹r)   # ∂ⱼrᵀ∂ᵢS⁻¹r = -(S⁻¹∂ⱼr)ᵀ∂ᵢS(S⁻¹r)
            @views prt1 = dot(G[:,j], S⁻¹∇r[:,i])             # ∂ⱼrᵀS⁻¹∂ᵢr
            @views prt2 = - 2*dot(S⁻¹∇r[:,i], ∇S[:,:,j], S⁻¹r) # 2∂ᵢrᵀ∂ⱼS⁻¹r = -(S⁻¹∂ᵢr)ᵀ∂ⱼSS⁻¹r
            @views prt3 = - dot(S⁻¹r, ∂ᵢⱼS, S⁻¹r)              # rᵀ∂ᵢⱼS⁻¹r  = -(S⁻¹r)ᵀ∂ᵢⱼSS⁻¹r
            @views prt4 = 2*dot(S⁻¹r, ∂ᵢSS⁻¹∂ⱼS, S⁻¹r)         #             + 2(S⁻¹r)ᵀ∂ᵢSS⁻¹∂ⱼSS⁻¹r
            logDetTerm = - tr(F \ ∂ᵢSS⁻¹∂ⱼS) + tr(F \ ∂ᵢⱼS)
            # Put everything together
            H[j,i] = 1/2*(
                2*(prt0+prt1) 
                + prt2+prt3+prt4
                + logDetTerm
            )
        end
    end
    # Only compute the upper triangular part of the hessian
    # and because it is symmetric, all computations are done
    H .= Symmetric(H)
    nothing 
end 