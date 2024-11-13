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
function LinearCovarianceFactor(data::WENDyInternals{true,<:Distribution}, params::Union{Nothing, WENDyParameters}, # Function
    ::Val{T}=Val(Float64) #optional type
) where T<:Real
    tt, X, V, Vp, sig, ∇ₓf! = data.tt, data.X, data.V, data.Vp, data.sig,data.∇ₓf!
    Mp1, D = size(X)
    K, Mp1 = size(V)
    J = data.J
    # preallocate output
    L = zeros(T,K*D,Mp1*D)
    # precompute L₀ because it does not depend on w
    __L₀ = zeros(T,K,D,D,Mp1)
    _L₀ = zeros(T,K,D,Mp1,D)
    L₀ = zeros(T,K*D,Mp1*D)
    _L₀!(
        L₀,
        Vp, sig,
        __L₀,_L₀
    )
    # precompute L₁ because it is constant wrt w 
    L₁ = zeros(T, K*D, Mp1*D,J)
    JuF = zeros(T,D,D,Mp1)
    _∂ⱼL = zeros(T,K,D,D,Mp1)
    ∂ⱼL = zeros(T,K,D,Mp1,D)
    eⱼ = zeros(T,J)
    _L₁!(
        L₁, 
        tt, X, V, sig,
        ∇ₓf!, 
        JuF, _∂ⱼL, ∂ⱼL, eⱼ
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
    return m.L
end
## ∇L(w)
struct LinearGradientCovarianceFactor<:GradientCovarianceFactor 
    # output
    ∇L::AbstractArray{<:Real,3}
end 
function LinearGradientCovarianceFactor(data::WENDyInternals{true, <:Distribution}, ::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    L₁ = zeros(T, K*D, Mp1*D,J)
    JuF = zeros(T,D,D,Mp1)
    _∂ⱼL = zeros(T,K,D,D,Mp1)
    ∂ⱼL = zeros(T,K,D,Mp1,D)
    eⱼ = zeros(T,J)
    _L₁!(
        L₁, 
        data.tt, data.X, data.V, data.sig, 
        data.∇ₓf!, 
        JuF, _∂ⱼL, ∂ⱼL, eⱼ
    )
   
    LinearGradientCovarianceFactor(L₁)
end
# method inplace 
function (m::LinearGradientCovarianceFactor)(∇L::AbstractArray{3, <:Real}, ::AbstractVector{<:Real}; ll::LogLevel=Warn)
    ∇L .= m.∇L
end
function (m::LinearGradientCovarianceFactor)(::AbstractVector{<:Real}; ll::LogLevel=Warn) 
    return m.∇L
end
## r(w) - Residual
# struct
struct LinearResidual<:Residual
    # ouput
    r::AbstractVector{<:Real}
    # data
    b₀::AbstractVector{<:Real}
    G::AbstractMatrix{<:Real}
    # buffer
    g::AbstractVector{<:Real} 
end
# constructors 
function LinearResidual(data::WENDyInternals{true,<:Distribution}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    KD = K*D
    # ouput
    r = zeros(T,KD)
    # buffers
    g = zeros(T, KD)
    LinearResidual(r,data.b₀, data.G, g)
end
# method inplace 
function (m::LinearResidual)(r::AbstractVector{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn) 
    _r!(
        r, w, 
        m.G, m.b₀;    
        ll=ll
    )
    nothing 
end
# Inplace: This assumes that b = R⁻ᵀ*b₀
function (m::LinearResidual)(r::AbstractVector{<:Real}, w::AbstractVector{<:Real}, b::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}; ll::LogLevel=Warn) 
    _Rᵀr!(
        r, w, 
        m.∇r, Rᵀ, b, 
        m.g; 
        ll=ll 
    )
    nothing
end
# method mutate internal data 
function (m::LinearResidual)(w::AbstractVector{<:Real}; ll::LogLevel=Warn) 
    _r!(
        m.r, w, 
        m.G, m.b₀; 
        ll=ll
    )
    return m.r
end
# method mutate internal data: This assumes that b = R⁻ᵀ*b₀
function (m::LinearResidual)(w::AbstractVector{<:Real}, b::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}; ll::LogLevel=Warn) 
    _Rᵀr!(
        m.r, w, 
        m.G, Rᵀ, b, 
        m.g; 
        ll=ll 
    )
    return m.r
end
struct LinearJacobianResidual<:JacobianResidual
    Rᵀ⁻¹∇r::AbstractMatrix{<:Real}
    ∇r::AbstractMatrix{<:Real}
    g::AbstractVector{<:Real}
end
# constructors
function LinearJacobianResidual(data::WENDyInternals{true,<:Distribution}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    LinearJacobianResidual(similar(data.G), data.G, similar(data.b₀))
end
# method inplace 
function (m::LinearJacobianResidual)(∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    @views ∇r .= m.∇r
    return nothing
end
# method in place when Rᵀ is given
function (m::LinearJacobianResidual)(Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, ::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}; ll::LogLevel=Warn)
    ldiv!(Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    nothing
end 
# method mutate internal data 
function (m::LinearJacobianResidual)(::AbstractVector{<:Real}; ll::LogLevel=Warn)
    return m.∇r
end
# method mutate internal data when Rᵀ is given
function (m::LinearJacobianResidual)(::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}; ll::LogLevel=Warn)
    ldiv!(m.Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    return m.Rᵀ⁻¹∇r
end 
## Hm(w) - Hessian of Maholinobis Distance
struct LinearHesianWeakNLL<:HesianWeakNLL
    # output 
    H::AbstractMatrix{<:Real}
    # data 
    b₀::AbstractVector{<:Real}
    # functions 
    R!::Covariance
    r!::Residual
    ∇r!::JacobianResidual
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

function LinearHesianWeakNLL(data::WENDyInternals{true,<:Distribution}, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    _, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    # ouput 
    H = zeros(J,J)
    # functions
    R! = Covariance(data, params)
    r! = Residual(data, params)
    ∇r! = JacobianResidual(data, params)
    ∇L! = GradientCovarianceFactor(data, params)
    # buffers
    S⁻¹r = zeros(T, K*D)
    S⁻¹∇r = zeros(T, K*D, J)
    ∂ⱼLLᵀ = zeros(T, K*D, K*D)
    ∇S = zeros(T, K*D, K*D, J)
    ∂ⱼL∂ᵢLᵀ = zeros(T, K*D, K*D)
    ∂ᵢⱼS = zeros(T, K*D, K*D)
    S⁻¹∂ⱼS = zeros(T, K*D, K*D)
    ∂ᵢSS⁻¹∂ⱼS = zeros(T, K*D, K*D)
    LinearHesianWeakNLL(
        H,
        data.b₀,
        R!, r!, ∇r!, ∇L!,
        S⁻¹r, S⁻¹∇r, ∂ⱼLLᵀ, ∇S, ∂ⱼL∂ᵢLᵀ, ∂ᵢⱼS, S⁻¹∂ⱼS, ∂ᵢSS⁻¹∂ⱼS
    )
end
# method inplace
function (m::LinearHesianWeakNLL)(H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    # TODO: try letting cholesky factorization back in here
    m.R!(w; transpose=false, doChol=false) 
    m.r!(w) 
    m.∇r!(w)
    m.∇L!(w)
    _Hwnll!(
        H, w,
        m.∇L!.∇L, m.∇r!.∇r, m.R!.L!.L, m.R!.Sreg, 
        m.r!.r,  
        m.S⁻¹r, m.S⁻¹∇r, m.∂ⱼLLᵀ, m.∇S, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
end
# method mutate internal data
function (m::LinearHesianWeakNLL)(w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    # TODO: try letting cholesky factorization back in here
    m.R!(w; transpose=false, doChol=false) 
    m.r!(w) 
    m.∇r!(w)
    m.∇L!(w)
    _Hwnll!(
        m.H, w,
        m.∇L!.∇L, m.∇r!.∇r, m.R!.L!.L, m.R!.Sreg,
        m.r!.r,  
        m.S⁻¹r, m.S⁻¹∇r, m.∂ⱼLLᵀ, m.∇S, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
    return m.H
end