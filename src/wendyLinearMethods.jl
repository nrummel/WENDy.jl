## L(w)
# struct
struct LinearCovarianceFactor<:CovarianceFactor
    # output
    L::AbstractMatrix{<:Real}
    # data
    L₁::AbstractArray{<:Real,3} 
    L₀::AbstractMatrix{<:Real}
end 
# constructor
function LinearCovarianceFactor(prob::WENDyProblem{true}, params::Union{Nothing, WENDyParameters}, # Function
    ::Val{T}=Val(Float64) #optional type
) where T<:Real
    U, V, Vp, sig, jacuf! = prob.U, prob.V, prob.Vp, prob.sig, prob.jacuf!
    D, M = size(U)
    K, _ = size(V)
    J = prob.J
    # preallocate output
    L = zeros(T,K*D,M*D)
    # precompute L₀ because it does not depend on w
    __L₀ = zeros(T,K,D,D,M)
    _L₀ = zeros(T,K,D,M,D)
    L₀ = zeros(T,K*D,M*D)
    _L₀!(
        L₀,
        Vp, sig,
        __L₀,_L₀
    )
    # precompute L₁ because it is constant wrt w 
    L₁ = zeros(T, K*D, M*D,J)
    JuF = zeros(T,D,D,M)
    _∂Lⱼ = zeros(T,K,D,D,M)
    ∂Lⱼ = zeros(T,K,D,M,D)
    eⱼ = zeros(T,J)
    _L₁!(
        L₁, 
        prob._Y, prob.V, prob.sig,
        prob.jacuf!, 
        JuF, _∂Lⱼ, ∂Lⱼ, eⱼ
    )

   return LinearCovarianceFactor(
        L,
        L₁, L₀
    )
end
# method inplace 
function (m::LinearCovarianceFactor)(L::AbstractMatrix, w::AbstractVector{<:Real}; ll::LogLevel=Info) 
    _L!(
        L,w,
        m.L₁,m.L₀;
        ll=ll
    )
    nothing
end
# method mutate internal data 
function (m::LinearCovarianceFactor)(w::AbstractVector{<:Real}; ll::LogLevel=Info) 
    _L!(
        m.L,w,
        m.L₁,m.L₀;
        ll=ll
    )
    nothing
end
## ∇L(w)
struct LinearGradientCovarianceFactor<:GradientCovarianceFactor 
    # output
    ∇L::AbstractArray{<:Real,3}
end 
function LinearGradientCovarianceFactor(prob, params, ::Val{T}=Val(Float64)) where T<:Real
    K,M,D,J = prob.K, prob.M, prob.D, prob.J
    L₁ = zeros(T, K*D, M*D,J)
    JuF = zeros(T,D,D,M)
    _∂Lⱼ = zeros(T,K,D,D,M)
    ∂Lⱼ = zeros(T,K,D,M,D)
    eⱼ = zeros(T,J)
    _L₁!(L₁, prob._Y, prob.V, prob.sig, prob.jacuf!, JuF, _∂Lⱼ, ∂Lⱼ, eⱼ)
   
    LinearGradientCovarianceFactor(L₁)
end
# method inplace 
function (m::LinearGradientCovarianceFactor)(∇L::AbstractArray{3, <:Real}, ::AbstractVector{<:Real}; ll::LogLevel=Warn)
    @views ∇L .= m.∇L
end
(m::LinearGradientCovarianceFactor)(::AbstractVector{<:Real}; ll::LogLevel=Warn) = nothing
## r(w) - Residual
# struct
struct LinearResidual<:Residual
    # ouput
    r::AbstractVector{<:Real}
    # data
    G::AbstractMatrix{<:Real}
    # buffer
    g::AbstractVector{<:Real} 
end
# constructors 
function LinearResidual(prob::WENDyProblem{true}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    D, M = size(prob.U)
    K, _ = size(prob.V)
    # ouput
    r = zeros(T,K*D)
    # buffers
    g = zeros(T, K*D)
    LinearResidual(r,prob.G,g)
end
# method inplace 
function (m::LinearResidual)(r::AbstractVector{<:Real}, b::AbstractVector{<:Real}, w::AbstractVector{T}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where T<:Real 
    if isnothing(Rᵀ)
        _r!(
            r, w, 
            m.G, b; # assume here that b = b₀ 
            ll=ll
        )
    else 
        _Rᵀr!(
            r, w, 
            m.G, Rᵀ, b, 
            m.g; 
            ll=ll 
        )
    end
    nothing
end
# method mutate internal data 
function (m::LinearResidual)(b::AbstractVector{<:Real}, w::AbstractVector{T}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where T<:Real 
    if isnothing(Rᵀ)
        _r!(
            m.r, w, 
            m.G, b; ## assumes that b = b₀
            ll=ll
        )
    else 
        _Rᵀr!(
            m.r, w, 
            m.G, Rᵀ, b, 
            m.g; 
            ll=ll 
        )
    end
    nothing
end
struct LinearGradientResidual<:GradientResidual
    Rᵀ⁻¹∇r::AbstractMatrix{<:Real}
    ∇r::AbstractMatrix{<:Real}
    G::AbstractMatrix{<:Real}
    g::AbstractVector{<:Real}
end
# constructors
function LinearGradientResidual(prob::WENDyProblem{true}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    LinearGradientResidual(similar(prob.G), similar(prob.G), prob.G, similar(prob.b₀))
end
# method inplace 
function (m::LinearGradientResidual)(Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    if isnothing(Rᵀ)
        @views Rᵀ⁻¹∇r .= m.G
        return nothing
    end
    ldiv!(Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    nothing
end 
# method mutate internal data 
function (m::LinearGradientResidual)(w::AbstractVector{<:Real}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    if isnothing(Rᵀ)
        @views m.∇r .= m.G
        return nothing
    end
    ldiv!(m.Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.G)
    nothing
end 
## Hm(w) - Hessian of Maholinobis Distance
struct LinearHesianMahalanobisDistance<:HesianMahalanobisDistance
    # output 
    H::AbstractMatrix{<:Real}
    # data 
    b₀::AbstractVector{<:Real}
    # functions 
    R!::Covariance
    r!::Residual
    ∇r!::GradientResidual
    ∇L!::GradientCovarianceFactor
    # buffers 
    S⁻¹r::AbstractVector{<:Real}
    S⁻¹∇r::AbstractMatrix{<:Real}
    ∂ⱼLLᵀ::AbstractMatrix{<:Real}
    ∇S::AbstractArray{<:Real, 3}
    ∂ⱼL∂ᵢLᵀ::AbstractMatrix{<:Real}
    ∂ᵢⱼS::AbstractMatrix{<:Real}
    S⁻¹∂ⱼS::AbstractMatrix{<:Real}
    ∂ᵢSS⁻¹∂ⱼS::AbstractMatrix{<:Real}
end

function LinearHesianMahalanobisDistance(prob::WENDyProblem{true}, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    K,M,D,J = prob.K, prob.M, prob.D, prob.J
    # ouput 
    H = zeros(J,J)
    # functions
    R! = Covariance(prob, params)
    r! = Residual(prob, params)
    ∇r! = GradientResidual(prob, params)
    ∇L! = GradientCovarianceFactor(prob, params)
    # buffers
    S⁻¹r = zeros(T, K*D)
    S⁻¹∇r = zeros(T, K*D, J)
    ∂ⱼLLᵀ = zeros(T, K*D, K*D)
    ∇S = zeros(T, K*D, K*D, J)
    ∂ⱼL∂ᵢLᵀ = zeros(T, K*D, K*D)
    ∂ᵢⱼS = zeros(T, K*D, K*D)
    S⁻¹∂ⱼS = zeros(T, K*D, K*D)
    ∂ᵢSS⁻¹∂ⱼS = zeros(T, K*D, K*D)
    LinearHesianMahalanobisDistance(
        H,
        prob.b₀,
        R!, r!, ∇r!, ∇L!,
        S⁻¹r, S⁻¹∇r, ∂ⱼLLᵀ, ∇S, ∂ⱼL∂ᵢLᵀ, ∂ᵢⱼS, S⁻¹∂ⱼS, ∂ᵢSS⁻¹∂ⱼS
    )
end
# method inplace
function (m::LinearHesianMahalanobisDistance)(H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    # TODO: try letting cholesky factorization back in here
    m.R!(w; transpose=false, doChol=false) 
    m.r!(m.b₀, w) 
    m.∇r!(w)
    m.∇L!(w)
    _Hm!(
        H, w,
        m.∇L!.∇L, m.∇r!.∇r, m.R!.L!.L, m.R!.Sreg, 
        m.r!.r,  
        m.S⁻¹r, m.S⁻¹∇r, m.∂ⱼLLᵀ, m.∇S, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
end
# method mutate internal data
function (m::LinearHesianMahalanobisDistance)(w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    # TODO: try letting cholesky factorization back in here
    m.R!(w; transpose=false, doChol=false) 
    m.r!(m.b₀, w) 
    m.∇r!(w)
    m.∇L!(w)
    _Hm!(
        m.H, w,
        m.∇L!.∇L, m.∇r!.∇r, m.R!.L!.L, m.R!.Sreg,
        m.r!.r,  
        m.S⁻¹r, m.S⁻¹∇r, m.∂ⱼLLᵀ, m.∇S, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
end