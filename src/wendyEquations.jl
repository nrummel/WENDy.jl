## Equations that do not depend on if the problem is linear or not in parameters
"""L₀ = V′ ∘ diag(σ₁,⋯,σ_D)"""
function _L₀!(
    L₀::AbstractMatrix, # output
    Vp::AbstractMatrix, sig::AbstractVector, # data
    __L₀::AbstractArray{<:Real,4}, _L₀::AbstractArray{<:Real,4} # buffers
)
    # TODO should we just use _L₀[k,d,m,d]?
    @tullio __L₀[k,d,d,m] = Vp[k,m]*sig[d]
    permutedims!(_L₀,__L₀,(1,2,4,3))
    K,D,Mp1,_ = size(_L₀)
    @views L₀ .= reshape(_L₀,K*D,Mp1*D)
    nothing 
end
""" 
Performs diagonal regularization and then  computed the Cholesky factorization of the weak residual's covariance.
"""
function _R!(
    R::AbstractMatrix{<:Real}, # output/input
    L::AbstractMatrix{<:Real}, diagReg::AbstractFloat, # data
    thisI::AbstractMatrix{<:Real}, Sreg::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}; # buffers
    doChol::Bool=true, ll::LogLevel=Warn #kwargs
) 
    mul!(S, L, L')
    @views Sreg .= thisI
    mul!(Sreg, S, I, (1-diagReg), diagReg)
    @views R .= Sreg
    if !doChol 
        return nothing 
    end 
    cholesky!(Symmetric(R))
    @views R .= UpperTriangular(R)
    nothing
end
""" weak form negative log likelihood """
function _wnll(S::AbstractMatrix, r::AbstractVector, S⁻¹r::AbstractVector, constTerm::AbstractFloat)
    F, logDet = try
        F = cholesky(S)
        logDet = 2*sum(log.(diag(F.U)))
        F, logDet 
    catch
        F = lu(S)
        logDet = sum(log.(filter!(x-> x >0, diag(F.U))))
        F, logDet 
    end
    ldiv!(S⁻¹r, F, r)
    mdist = dot(r, S⁻¹r)
    #constTerm
    1/2*(
        mdist 
        + logDet 
    ) + constTerm
end
"""∇m(p) - Gradient of weak form negative log likelihood"""
function _∇wnll!(
    ∇m::AbstractVector{<:Real},
    ∇L::AbstractArray{<:Real,3}, L::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}, ∇r::AbstractMatrix{<:Real}, r::AbstractVector{<:Real},
    S⁻¹r::AbstractVector{<:Real}, ∂ⱼLLᵀ::AbstractMatrix{<:Real}, ∇S::AbstractArray{<:Real,3}
)
    J = length(∇m)
    # TODO: perhaps do this inplace?
    F = try 
        cholesky(S) 
    catch  
        qr(S) 
    end
    # precompute the S⁻¹r 
    ldiv!(S⁻¹r, F, r)
    # compute ∇S
    @inbounds for j = 1:J 
        @views mul!(∂ⱼLLᵀ, ∇L[:,:,j], L')
        @views ∇S[:,:,j] .= ∂ⱼLLᵀ + (∂ⱼLLᵀ)'
    end
    # compute ∇m
    @inbounds for j in 1:J 
        @views prt0 = 2*dot(∇r[:,j], S⁻¹r)         # 2∂ⱼrᵀS⁻¹r
        @views prt1 = - dot(S⁻¹r, ∇S[:,:,j], S⁻¹r) # rᵀ∂ⱼS⁻¹r = -(S⁻¹r)ᵀ∂ⱼSS⁻¹r
        @views logDetPrt = tr(F \ ∇S[:,:,j])
        ∇m[j] = 1/2 * (
            prt0 
            + prt1 
            + logDetPrt
            )
    end 
    nothing
end