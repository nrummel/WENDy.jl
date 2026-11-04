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
## LLᵀ blocks - because L(p) = L₀ + ∑ⱼpⱼL₁ⱼ, with q = [1; p] and L̃ = [L₀, L₁₁, ⋯, L₁ⱼ]
# LLᵀ = ∑ₐᵦ qₐqᵦ L̃ₐL̃ᵦᵀ, so we store P[:,a,b] = vec(L̃ₐL̃ᵦᵀ) once and never touch the Mp1*D dimension again
function _LLᵀBlocks!(
    P::AbstractArray{<:Real,3}, # output
    L₀::AbstractMatrix{<:Real}, L₁::AbstractArray{<:Real,3}, # data
    LaLbᵀ::AbstractMatrix{<:Real} # buffers
)
    J = size(L₁,3)
    L̃(a) = a == 1 ? L₀ : view(L₁,:,:,a-1)
    for a in 1:J+1, b in a:J+1
        mul!(LaLbᵀ, L̃(a), L̃(b)')
        @views P[:,a,b] .= vec(LaLbᵀ)
        @views P[:,b,a] .= vec(LaLbᵀ')
    end
    nothing
end
# S(p) = (1-δ)∑ₐᵦ qₐqᵦ L̃ₐL̃ᵦᵀ + δI
function _S!(
    S::AbstractMatrix{<:Real}, q::AbstractVector{<:Real}, # output/input (q = [1; p])
    P::AbstractArray{<:Real,3}, diagReg::Real # data
)
    mul!(vec(S), reshape(P, size(P,1), :), vec(q*q'))
    S .*= (1-diagReg)
    @views S[diagind(S)] .+= diagReg
    nothing
end
## ∇m(p) - gradient using the LLᵀ blocks. With y = S⁻¹r and Qⱼ = ∑ₐqₐL̃ₐL₁ⱼᵀ = LL₁ⱼᵀ, ∂ⱼS = (1-δ)(Qⱼ + Qⱼᵀ) so
# ∂ⱼm = Gⱼᵀy - (1-δ)yᵀQⱼy + (1-δ)⟨S⁻¹, Qⱼ⟩
function _∇wnll!(
    ∇m::AbstractVector{<:Real}, # output
    q::AbstractVector{<:Real}, P::AbstractArray{<:Real,3}, G::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, S⁻¹::AbstractMatrix{<:Real}, diagReg::Real, # data (q = [1; p], y = S⁻¹r)
    Qⱼ::AbstractMatrix{<:Real} # buffers
)
    J = length(∇m)
    mul!(∇m, G', y)
    @inbounds for j in 1:J 
        @views mul!(vec(Qⱼ), P[:,:,j+1], q)
        ∇m[j] += (1-diagReg)*(dot(S⁻¹, Qⱼ) - dot(y, Qⱼ, y))
    end
    nothing
end
## Hm(p) - hessian of the Maholinobis distance when the ode is linear in parameters ⇒ ∀w  ∂ᵢⱼL = 0 and ∂ᵢⱼr = 0, so ∂ᵢⱼS = ∂ⱼL∂ᵢLᵀ + ∂ᵢL∂ⱼLᵀ is constant in p. Also notice that ∇_w[G*p -b₀] = G
# With the LLᵀ blocks, ∂ⱼS = (1-δ)(Qⱼ + Qⱼᵀ) and ⟨S⁻¹∂ⱼL, ∂ᵢL⟩ - vᵢᵀvⱼ = ⟨S⁻¹ - yyᵀ, L₁ᵢL₁ⱼᵀ⟩, so nothing here touches the Mp1*D dimension
function _Hwnll!(
    H::AbstractMatrix{<:Real}, # output
    q::AbstractVector{<:Real}, P::AbstractArray{<:Real,3}, G::AbstractMatrix{<:Real}, F::Factorization, S⁻¹::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, diagReg::Real, # data (q = [1; p], F factors S, y = S⁻¹r)
    S⁻¹G::AbstractMatrix{<:Real}, Qⱼ::AbstractMatrix{<:Real}, ∂ⱼS::AbstractMatrix{<:Real}, ∇Sy::AbstractMatrix{<:Real}, S⁻¹∇Sy::AbstractMatrix{<:Real}, S⁻¹∇S::AbstractArray{<:Real,3}, S⁻¹_yyᵀ::AbstractMatrix{<:Real} # buffers
)
    J = size(H,1)
    ldiv!(S⁻¹G, F, G)
    # ∂ⱼrᵀS⁻¹∂ᵢr
    mul!(H, G', S⁻¹G)
    @inbounds for j in 1:J
        @views mul!(vec(Qⱼ), P[:,:,j+1], q)
        ∂ⱼS .= (1-diagReg) .* (Qⱼ .+ Qⱼ')
        @views mul!(∇Sy[:,j], ∂ⱼS, y)
        @views mul!(S⁻¹∇S[:,:,j], S⁻¹, ∂ⱼS)
    end
    ldiv!(S⁻¹∇Sy, F, ∇Sy)
    _HwnllCovTerms!(H, S⁻¹G, ∇Sy, S⁻¹∇Sy, S⁻¹∇S)
    # (1-δ)⟨S⁻¹ - yyᵀ, L₁ᵢL₁ⱼᵀ⟩
    copyto!(S⁻¹_yyᵀ, S⁻¹)
    mul!(S⁻¹_yyᵀ, y, y', -1, 1)
    @inbounds for j in 1:J, i in 1:J
        @views H[i,j] += (1-diagReg)*dot(vec(S⁻¹_yyᵀ), P[:,i+1,j+1])
    end
    H .= (H .+ H') ./ 2
    nothing 
end
