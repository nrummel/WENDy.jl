function f!(dum::AbstractArray{<:Real}, w::AbstractVector{<:Real}, um::AbstractVector{<:Real})
    dum[1] = (*)(um[1], (^)(w[1], 2))
end

function jacuf!(dum::AbstractArray{<:Real}, w::AbstractVector{<:Real}, um::AbstractVector{<:Real})
    dum[1] = (^)(w[1], 2)
end

function jacwf!(dum::AbstractArray{<:Real}, w::AbstractVector{<:Real}, um::AbstractVector{<:Real})
    dum[1] = (*)((*)(2, um[1]), w[1])
end

function jacwjacuf!(dum::AbstractArray{<:Real}, w::AbstractVector{<:Real}, um::AbstractVector{<:Real})
    dum[1] = (*)(2, w[1])
end 

function heswf!(dum::AbstractArray{<:Real}, w::AbstractVector{<:Real}, um::AbstractVector{<:Real})
    dum[1] = (*)(2, um[1])
end

function heswjacuf!(dum::AbstractArray{<:Real}, w::AbstractVector{<:Real}, um::AbstractVector{<:Real})
    dum[1] = 2
end

function m( 
    w::AbstractVector{<:Real}, 
    U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real},Vp::AbstractMatrix{<:Real}, b₀::AbstractVector{<:Real}, sig::AbstractVector{<:Real}, diagReg::AbstractFloat, ::Val{T}=Val(Float64)
) where T<:Real
    D, M = size(U)
    K,_ = size(V)
    J = length(w)
    ## Compute the covariance
    __L₀ = zeros(T,K,D,D,M)
    _L₀ = zeros(T,K,D,M,D)
    L₀ = zeros(T,K*D,M*D)
    _L₀!(
        L₀,
        Vp,sig,
        __L₀,_L₀
    )
    JuF = zeros(T,D,D,M)
    __L₁ = zeros(T,K,D,D,M)
    _L₁ = zeros(T,K,D,M,D)
    L = zeros(T,K*D,M*D)
    _L!(
        L,w,
        U,V,L₀,sig,
        jacuf!,
        JuF,__L₁,_L₁
    )
    Sreg = L*L'*(1-diagReg) + diagReg*I
    ## Compute the residual 
    r = zeros(T, K*D)
    F = zeros(T, D, M)
    G = zeros(T, K, D)
    g = zeros(T, K*D)
    _r!(
        r,w,
        U, V, b₀,
        f!, 
        F, G
    )
    return 1/2 * dot(r, (Sreg \ r))
end

function ∇m!(
    ∇m::AbstractVector{<:Real}, w::AbstractVector{<:Real}, 
    U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real},Vp::AbstractMatrix{<:Real}, b₀::AbstractVector{<:Real}, sig::AbstractVector{<:Real}, diagReg::AbstractFloat, ::Val{T}=Val(Float64)
) where T<:Real
    D, M = size(U)
    K,_ = size(V)
    J = length(w)
    ## Compute the covariance
    __L₀ = zeros(T,K,D,D,M)
    _L₀ = zeros(T,K,D,M,D)
    L₀ = zeros(T,K*D,M*D)
    _L₀!(
        L₀,
        Vp,sig,
        __L₀,_L₀
    )
    JuF = zeros(T,D,D,M)
    __L₁ = zeros(T,K,D,D,M)
    _L₁ = zeros(T,K,D,M,D)
    L = zeros(T,K*D,M*D)
    _L!(
        L,w,
        U,V,L₀,sig,
        jacuf!,
        JuF,__L₁,_L₁
    )
    Sreg = L*L'*(1-diagReg) + diagReg*I
    ## Compute the residual 
    r = zeros(T, K*D)
    F = zeros(T, D, M)
    G = zeros(T, K, D)
    g = zeros(T, K*D)
    _r!(
        r,w,
        U, V, b₀,
        f!, 
        F, G
    )
    ## Compue the gradient of the residual 
    ∇r = zeros(K*D,J)
    JwF = zeros(T,D,J,M)
    __∇r = zeros(T,D,J,K)
    _∇r = zeros(T,K,D,J)
    ∇r = zeros(T,K*D, J)
    _∇r!(
        ∇r,w, 
        U,V,
        jacwf!,
        JwF, __∇r, _∇r
    )
    # 
    JwJuF = zeros(T, D, D, J, M)
    __∇L = zeros(T, K, D, D, J, M)
    _∇L = zeros(T, K, D, M, D, J)
    ∇L = zeros(T, K*D, M*D, J)
    _∇L!(
        ∇L, w,
        U,V,sig,
        jacwjacuf!,
        JwJuF, __∇L, _∇L
    )
    # preallocate buffers
    S⁻¹r = zeros(T, K*D)
    ∂ⱼLLᵀ = zeros(T, K*D,K*D)
    ∇S = zeros(T,K*D,K*D,J)
    _∇m!(
        ∇m, w, # ouput, input
        ∇L, L, Sreg, ∇r, r, # data
        S⁻¹r, ∂ⱼLLᵀ, ∇S; # buffers
    )
end
function Hm!(
    H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}, 
    U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real},Vp::AbstractMatrix{<:Real}, b₀::AbstractVector{<:Real}, sig::AbstractVector{<:Real}, diagReg::AbstractFloat, ::Val{T}=Val(Float64)
) where T<:Real
    D, M = size(U)
    K,_ = size(V)
    J = length(w)
    ## Compute the covariance
    __L₀ = zeros(T,K,D,D,M)
    _L₀ = zeros(T,K,D,M,D)
    L₀ = zeros(T,K*D,M*D)
    _L₀!(
        L₀,
        Vp,sig,
        __L₀,_L₀
    )
    JuF = zeros(T,D,D,M)
    __L₁ = zeros(T,K,D,D,M)
    _L₁ = zeros(T,K,D,M,D)
    L = zeros(T,K*D,M*D)
    _L!(
        L,w,
        U,V,L₀,sig,
        jacuf!,
        JuF,__L₁,_L₁
    )
    Sreg = L*L'*(1-diagReg) + diagReg*I
    ## Compute the residual 
    r = zeros(T, K*D)
    F = zeros(T, D, M)
    G = zeros(T, K, D)
    g = zeros(T, K*D)
    _r!(
        r,w,
        U, V, b₀,
        f!, 
        F, G
    )
    ## Compue the gradient of the residual 
    ∇r = zeros(K*D,J)
    JwF = zeros(T,D,J,M)
    __∇r = zeros(T,D,J,K)
    _∇r = zeros(T,K,D,J)
    ∇r = zeros(T,K*D, J)
    _∇r!(
        ∇r,w, 
        U,V,
        jacwf!,
        JwF,__∇r,_∇r;
    )
    # compute 
    JwJuF = zeros(T, D, D, J, M)
    __∇L = zeros(T, K, D, D, J, M)
    _∇L = zeros(T, K, D, M, D, J)
    ∇L = zeros(T, K*D, M*D, J)
    _∇L!(
        ∇L, w,
        U,V,sig,
        jacwjacuf!,
        JwJuF, __∇L, _∇L
    )
    # Compute the rest of the hessian
    S⁻¹r = zeros(T, K*D)
    S⁻¹∇r = zeros(T, K*D, J)
    ∂ⱼLLᵀ = zeros(T, K*D, K*D)
    ∇S = zeros(T, K*D, K*D, J)
    HwF = zeros(T, D, J, J, M)
    _∇²r = zeros(T, K, D, J, J)
    ∇²r = zeros(T, K*D, J, J)
    HwJuF = zeros(T, D, D, J, J, M)
    __∇²L = zeros(T, K, D, D, J, J, M)
    _∇²L = zeros(T, K, D, M, D, J, J)
    ∇²L = zeros(T, K*D, M*D, J, J)
    ∂ⱼL∂ᵢLᵀ = zeros(T, K*D, K*D)
    ∂ᵢⱼLLᵀ = zeros(T, K*D, K*D)
    ∂ᵢⱼS = zeros(T, K*D, K*D)
    S⁻¹∂ⱼS = zeros(T, K*D, K*D)
    ∂ᵢSS⁻¹∂ⱼS = zeros(T, K*D, K*D)
    _Hm!(
        H, w,
        ∇L, U, V, L, Sreg, ∇r, b₀, sig,
        r, 
        heswf!, heswjacuf!,  
        S⁻¹r, S⁻¹∇r, ∂ⱼLLᵀ, ∇S, HwF, _∇²r, ∇²r, HwJuF, __∇²L, _∇²L, ∇²L, ∂ⱼL∂ᵢLᵀ, ∂ᵢⱼLLᵀ, ∂ᵢⱼS, S⁻¹∂ⱼS, ∂ᵢSS⁻¹∂ⱼS
    )
end