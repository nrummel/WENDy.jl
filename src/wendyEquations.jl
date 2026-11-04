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
"""
∇m(p) - Gradient of weak form negative log likelihood in adjoint form. With y = S⁻¹r and
∂ⱼS = (1-δ)(∂ⱼLLᵀ + L∂ⱼLᵀ):
    yᵀ∂ⱼSy     = 2(1-δ) yᵀ∂ⱼL(Lᵀy)
    tr(S⁻¹∂ⱼS) = 2(1-δ) ⟨S⁻¹L, ∂ⱼL⟩
so ∂ⱼm = ∂ⱼrᵀy + (1-δ)⟨∂ⱼL, S⁻¹L - y(Lᵀy)ᵀ⟩. This needs one solve S⁻¹L instead of forming ∂ⱼS
and solving against it for every j.
"""
function _∇wnll!(
    ∇m::AbstractVector{<:Real},
    ∇L::AbstractArray{<:Real,3}, L::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}, ∇r::AbstractMatrix{<:Real}, r::AbstractVector{<:Real}, diagReg::Real,
    S⁻¹r::AbstractVector{<:Real}, S⁻¹L::AbstractMatrix{<:Real}, Lᵀy::AbstractVector{<:Real}
)
    J = length(∇m)
    F = try
        cholesky(S)
    catch
        qr(S)
    end
    # y = S⁻¹r
    ldiv!(S⁻¹r, F, r)
    # M = S⁻¹L - y(Lᵀy)ᵀ, stored in S⁻¹L
    ldiv!(S⁻¹L, F, L)
    mul!(Lᵀy, L', S⁻¹r)
    mul!(S⁻¹L, S⁻¹r, Lᵀy', -1, 1)
    # ∇m = ∇rᵀy + (1-δ)[⟨∂ⱼL, M⟩]ⱼ
    mul!(∇m, ∇r', S⁻¹r)
    mul!(∇m, reshape(∇L, :, J)', vec(S⁻¹L), 1-diagReg, 1)
    nothing
end
"""
Hm(p) terms that only depend on how S changes, shared by the linear and general Hessians. 
With U = S⁻¹∇r, wⱼ = ∂ⱼSy and Aⱼ = S⁻¹∂ⱼS this adds
    -Uⱼᵀwᵢ - Uᵢᵀwⱼ + wᵢᵀS⁻¹wⱼ - ½tr(AᵢAⱼ)
"""
function _HwnllCovTerms!(
    H::AbstractMatrix{<:Real}, # output (accumulated into)
    S⁻¹∇r::AbstractMatrix{<:Real}, ∇Sy::AbstractMatrix{<:Real}, S⁻¹∇Sy::AbstractMatrix{<:Real}, S⁻¹∇S::AbstractArray{<:Real,3} # data
)
    J = size(H,1)
    mul!(H, S⁻¹∇r', ∇Sy, -1, 1)
    mul!(H, ∇Sy', S⁻¹∇r, -1, 1)
    mul!(H, ∇Sy', S⁻¹∇Sy, 1, 1)
    @inbounds for j in 1:J, i in 1:j
        @views trAᵢAⱼ = dot(transpose(S⁻¹∇S[:,:,i]), S⁻¹∇S[:,:,j])
        H[i,j] -= trAᵢAⱼ/2
        i != j && (H[j,i] -= trAᵢAⱼ/2)
    end
    nothing
end
"""
Hm(p) - Hessian of weak form negative log likelihood using trace identities. With y = S⁻¹r, U = S⁻¹∇r, 
wⱼ = ∂ⱼSy, ℓ = Lᵀy, vⱼ = ∂ⱼLᵀy:
    Hᵢⱼ = ∂ᵢⱼrᵀy + ∂ⱼrᵀUᵢ - Uⱼᵀwᵢ - Uᵢᵀwⱼ + wᵢᵀS⁻¹wⱼ - ½tr(S⁻¹∂ᵢSS⁻¹∂ⱼS)
          + (1-δ)[⟨S⁻¹∂ⱼL, ∂ᵢL⟩ - vᵢᵀvⱼ + ⟨∂ᵢⱼL, S⁻¹L - yℓᵀ⟩]
so no solve or KD×KD product happens per pair (i,j). The second order terms are contracted through V without forming ∇²r or ∇²L:
    ∂ᵢⱼrᵀy          = ∑ HwF[d,i,j,m] (VᵀY)[m,d],                  Y = reshape(y, K, D)
    ⟨∂ᵢⱼL, M⟩       = ∑ HwJuF[d₂,d₁,i,j,m] σ[d₁] (VᵀM)[d₂,m,d₁],  M = reshape(S⁻¹L - yℓᵀ, K, D, Mp1, D)
Pass HwF = HwJuF = nothing when the problem is linear in parameters.
"""
function _Hwnll!(
    H::AbstractMatrix{<:Real}, # output
    ∇L::AbstractArray{<:Real,3}, L::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}, ∇r::AbstractMatrix{<:Real}, r::AbstractVector{<:Real}, diagReg::Real, # data
    HwJuF::Union{Nothing,AbstractArray{<:Real,5}}, HwF::Union{Nothing,AbstractArray{<:Real,4}}, V::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, # data (second order terms only)
    S⁻¹r::AbstractVector{<:Real}, S⁻¹∇r::AbstractMatrix{<:Real}, S⁻¹L::AbstractMatrix{<:Real}, Lᵀy::AbstractVector{<:Real}, ∇Lᵀy::AbstractMatrix{<:Real}, # buffers
    ∇Sy::AbstractMatrix{<:Real}, S⁻¹∇Sy::AbstractMatrix{<:Real}, ∂ⱼS::AbstractMatrix{<:Real}, S⁻¹∇S::AbstractArray{<:Real,3}, S⁻¹∂ⱼL::AbstractMatrix{<:Real}, # buffers
    VᵀY::AbstractMatrix{<:Real}, VᵀM::AbstractArray{<:Real,3} # buffers (second order terms only)
)
    J = size(H,1)
    K, Mp1 = size(V)
    D = length(sig)
    δ̃ = 1-diagReg
    F = try 
        cholesky(S) 
    catch  
        qr(S) 
    end
    ldiv!(S⁻¹r, F, r)
    ldiv!(S⁻¹∇r, F, ∇r)
    mul!(Lᵀy, L', S⁻¹r)
    # ∂ⱼrᵀS⁻¹∂ᵢr
    mul!(H, ∇r', S⁻¹∇r)
    @inbounds for j in 1:J
        @views mul!(∇Lᵀy[:,j], ∇L[:,:,j]', S⁻¹r)
        # ∂ⱼS = (1-δ)(∂ⱼLLᵀ + L∂ⱼLᵀ)
        @views mul!(∂ⱼS, ∇L[:,:,j], L', 1-diagReg, 0)
        ∂ⱼS .+= ∂ⱼS'
        @views mul!(∇Sy[:,j], ∂ⱼS, S⁻¹r)
        @views ldiv!(S⁻¹∇S[:,:,j], F, ∂ⱼS)
        # (1-δ)⟨S⁻¹∂ⱼL, ∂ᵢL⟩ for all i
        @views ldiv!(S⁻¹∂ⱼL, F, ∇L[:,:,j])
        @views mul!(H[:,j], reshape(∇L, :, J)', vec(S⁻¹∂ⱼL), 1-diagReg, 1)
    end
    ldiv!(S⁻¹∇Sy, F, ∇Sy)
    _HwnllCovTerms!(H, S⁻¹∇r, ∇Sy, S⁻¹∇Sy, S⁻¹∇S)
    # -(1-δ)vᵢᵀvⱼ
    mul!(H, ∇Lᵀy', ∇Lᵀy, -(1-diagReg), 1)
    # ∂ᵢⱼrᵀy
    if !isnothing(HwF)
        Y = reshape(S⁻¹r, K, D)
        @tullio VᵀY[m,d] = V[k,m] * Y[k,d]
        @tullio H[i,j] += HwF[d,i,j,m] * VᵀY[m,d]
    end
    # (1-δ)⟨∂ᵢⱼL, S⁻¹L - yℓᵀ⟩
    if !isnothing(HwJuF)
        ldiv!(S⁻¹L, F, L)
        mul!(S⁻¹L, S⁻¹r, Lᵀy', -1, 1)
        M = reshape(S⁻¹L, K, D, Mp1, D)
        @tullio VᵀM[d2,m,d1] = V[k,m] * M[k,d2,m,d1]
        @tullio H[i,j] += δ̃ * HwJuF[d2,d1,i,j,m] * sig[d1] * VᵀM[d2,m,d1]
    end
    H .= (H .+ H') ./ 2
    nothing
end
