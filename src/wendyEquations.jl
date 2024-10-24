## Equations that do not depend on if the problem is linear or not in w
# L₀ = V′ ∘ diag(σ₁,⋯,σ_D)
function _L₀!(
    L₀::AbstractMatrix, # output
    Vp::AbstractMatrix, sig::AbstractVector, # data
    __L₀::AbstractArray{<:Real,4}, _L₀::AbstractArray{<:Real,4} # buffers
)
    @tullio __L₀[k,d,d,m] = Vp[k,m]*sig[d]
    permutedims!(_L₀,__L₀,(1,2,4,3))
    K,D,Mp1,_ = size(_L₀)
    @views L₀ .= reshape(_L₀,K*D,Mp1*D)
    nothing 
end
# R(w) - Cholesky factorization of the Covariance/just compute and regularizee the covariance
function _R!(
    R::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}, # output/input
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
# m(w) - Maholinobis distance
function _m(S::AbstractMatrix, r::AbstractVector, S⁻¹r::AbstractVector, constTerm::AbstractFloat)
    F = svd(S)
    ldiv!(S⁻¹r, F, r)
    mdist = dot(r, S⁻¹r)
    logDet = sum(log.(F.S))
    #constTerm
    1/2*(
        mdist 
        + logDet 
    ) + constTerm
end
# function _m(Rᵀ⁻¹r::AbstractVector, )
#     1/2*(dot(Rᵀ⁻¹r, Rᵀ⁻¹r) + sum(log.()))
# end
# ∇m(w) - Gradient of Maholinobis distance
function _∇m!(
    ∇m::AbstractVector{<:Real},w::AbstractVector{<:Real},
    ∇L::AbstractArray{<:Real,3}, L::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}, ∇r::AbstractMatrix{<:Real}, r::AbstractVector{<:Real},
    S⁻¹r::AbstractVector{<:Real}, ∂ⱼLLᵀ::AbstractMatrix{<:Real}, ∇S::AbstractArray{<:Real,3};
    ll::LogLevel=Warn
)
    J = length(∇m)
    # TODO: perhaps do this inplace?
    F = svd(S) 
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