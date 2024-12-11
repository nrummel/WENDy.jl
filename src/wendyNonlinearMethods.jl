## L(w)
# struct
struct NonlinearCovarianceFactor<:CovarianceFactor
    # output
    L::AbstractMatrix{<:Real}
    # data
    tt::AbstractVector{<:Real} 
    _Y::AbstractMatrix{<:Real} 
    V::AbstractMatrix{<:Real} 
    L₀::AbstractMatrix{<:Real}
    sig::AbstractVector{<:Real}
    # functions
    ∇ₓf!::Function
    # buffers
    JuF::AbstractArray{<:Real, 3}
    __L₁::AbstractArray{<:Real, 4}
    _L₁::AbstractArray{<:Real, 4}
end 
# constructor
function NonlinearCovarianceFactor(data::WENDyInternals{false,<:Distribution}, params::Union{WENDyParameters,Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    # preallocate output
    L = zeros(T,K*D,Mp1*D)
    # precompute L₀ because it does not depend on w
    __L₀ = zeros(T,K,D,D,Mp1)
    _L₀ = zeros(T,K,D,Mp1,D)
    L₀ = zeros(T,K*D,Mp1*D)
    _L₀!(
        L₀,
        data.Vp, data.sig,
        __L₀,_L₀
    )
    # buffers
    JuF = zeros(T,D,D,Mp1)
    __L₁ = zeros(T,K,D,D,Mp1)
    _L₁ = zeros(T,K,D,Mp1,D)

    return NonlinearCovarianceFactor(
        L,
        data.tt, data.X, data.V,L₀,data.sig, 
        data.∇ₓf!, 
        JuF, __L₁, _L₁
    )
end
# method inplace 
function (m::NonlinearCovarianceFactor)(L::AbstractMatrix, w::AbstractVector{<:Real}; ll::LogLevel=Info) 
    _L!(
        L,w,
        m.tt, m._Y,m.V,m.L₀,m.sig,
        m.∇ₓf!,
        m.JuF,m.__L₁,m._L₁;
        ll=ll
    )
    nothing
end
# method mutate internal data 
function (m::NonlinearCovarianceFactor)(w::AbstractVector{<:Real}; ll::LogLevel=Info) 
    _L!(
        m.L,w,
        m.tt, m._Y,m.V,m.L₀,m.sig,
        m.∇ₓf!,
        m.JuF,m.__L₁,m._L₁;
        ll=ll
    )
    return m.L
end
##
struct NonlinearGradientCovarianceFactor<:GradientCovarianceFactor 
    # output
    ∇L::AbstractArray{<:Real,3}
    # data 
    tt::AbstractVector{<:Real}
    _Y::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    sig::AbstractVector{<:Real}
    # functions
    ∇ₚ∇ₓf!::Function
    # buffers
    JwJuF::AbstractArray{<:Real,4}
    __∇L::AbstractArray{<:Real,5}
    _∇L::AbstractArray{<:Real,5}
end 
function NonlinearGradientCovarianceFactor(data, params, ::Val{T}=Val(Float64)) where T<:Real
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    ∇L = zeros(T,K*D,Mp1*D,J)
    JwJuF = zeros(T,D,D,J,Mp1)
    __∇L = zeros(T,K,D,D,J,Mp1)
    _∇L = zeros(T,K,D,Mp1,D,J)
   
    NonlinearGradientCovarianceFactor(
        ∇L,
        data.tt, data.X, data.V, data.sig, 
        data.∇ₚ∇ₓf!,
        JwJuF, __∇L, _∇L
    )
end
# method inplace 
function (m::NonlinearGradientCovarianceFactor)(∇L::AbstractArray{3, <:Real}, w::AbstractVector{<:Real};ll::LogLevel=Warn)
    _∇L!(
        ∇L, w,
        m.tt,m._Y,m.V,m.sig,
        m.∇ₚ∇ₓf!,
        m.JwJuF, m.__∇L, m._∇L
    )
end
# method mutating interal storage
function (m::NonlinearGradientCovarianceFactor)(w::AbstractVector{<:Real};ll::LogLevel=Warn)
    _∇L!(
        m.∇L, w,
        m.tt,m._Y,m.V,m.sig,
        m.∇ₚ∇ₓf!,
        m.JwJuF, m.__∇L, m._∇L
    )
end
## r(w) - NonlinearResidual
# struct
struct NonlinearResidual<:Residual
    # ouput
    r::AbstractVector{<:Real}
    # data
    tt::AbstractVector{<:Real}
    b₀::AbstractVector{<:Real}
    X::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real} 
    # functions
    f!::Function
    # buffers
    F::AbstractMatrix{<:Real} 
    G::AbstractMatrix{<:Real} 
    g::AbstractVector{<:Real} 
end
# constructors 
function NonlinearResidual(data::WENDyInternals{false,<:Distribution}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    # ouput
    r = zeros(T,K*D)
    # buffers
    F = zeros(T, Mp1, D)
    G = zeros(T, K, D)
    g = zeros(T, K*D)
    NonlinearResidual(
        r,
        data.tt,data.b₀,data.X, data.V, 
        data.f!,
        F,G,g
    )
end
# method inplace 
function (m::NonlinearResidual)(r::AbstractVector{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn) 
    _r!(
        r,w, 
        m.tt, m.X, m.V, m.b₀, 
        m.f!, 
        m.F, m.G; 
        ll=ll
    )
end
# method inplace: assume b = R⁻ᵀ*b₀
function (m::NonlinearResidual)(r::AbstractVector{<:Real}, w::AbstractVector{<:Real}, b::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}; ll::LogLevel=Warn) 
    _Rᵀr!(
        r, w, 
        m.tt, m.X, m.V, Rᵀ, b, 
        m.f!, 
        m.F, m.G, m.g; 
        ll=ll 
    )
end
# method mutate internal data 
function (m::NonlinearResidual)(w::AbstractVector{<:Real}; ll::LogLevel=Warn) 
    _r!(
        m.r,w, 
        m.tt, m.X, m.V, m.b₀, 
        m.f!, 
        m.F, m.G; 
        ll=ll
    )
    return m.r
end
# method mutate internal data: assume b = R⁻ᵀ*b₀
function (m::NonlinearResidual)(w::AbstractVector{<:Real}, b::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}; ll::LogLevel=Warn) 
    _Rᵀr!(
        m.r, w, 
        m.tt, m.X, m.V, Rᵀ, b, 
        m.f!, 
        m.F, m.G, m.g; 
        ll=ll 
    )
    return m.r
end
## ∇r & Rᵀ∇r
# struct
struct NonlinearJacobianResidual<:JacobianResidual
    # ouput 
    Rᵀ⁻¹∇r::AbstractMatrix{<:Real}
    # data 
    tt::AbstractVector{<:Real}
    X::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    # functions 
    ∇ₚf!::Function
    #buffers
    JwF::AbstractArray{<:Real,3}
    __∇r::AbstractArray{<:Real, 3}
    _∇r::AbstractArray{<:Real, 3}
    ∇r::AbstractMatrix{<:Real} 
end
# constructors
function NonlinearJacobianResidual(data::WENDyInternals{false,<:Distribution}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    J = data.J
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    Rᵀ⁻¹∇r = zeros(K*D,J)
    JwF = zeros(T,D,J,Mp1)
    __∇r = zeros(T,D,J,K)
    _∇r = zeros(T,K,D,J)
    ∇r = zeros(T,K*D, J)
    NonlinearJacobianResidual(
        Rᵀ⁻¹∇r, 
        data.tt, data.X, data.V, 
        data.∇ₚf!, 
        JwF, __∇r, _∇r, ∇r
    )
end
# method inplace 
function (m::NonlinearJacobianResidual)(∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    
    _∇r!(
        ∇r,w, 
        m.tt,m.X,m.V,
        m.∇ₚf!,
        m.JwF,m.__∇r,m._∇r;
        ll=ll
    )
    
end
# method inplace 
function (m::NonlinearJacobianResidual)(Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}; ll::LogLevel=Warn)
    m(w) #compute ∇r
    ldiv!(Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    nothing
end 
# method mutate internal data 
function (m::NonlinearJacobianResidual)(w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    _∇r!(
        m.∇r,w, 
        m.tt,m.X,m.V,
        m.∇ₚf!,
        m.JwF,m.__∇r,m._∇r;
        ll=ll
    )
    return m.∇r
end
# method mutate internal data 
function (m::NonlinearJacobianResidual)(w::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}; ll::LogLevel=Warn)
    m(w) #compute ∇r
    ldiv!(m.Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    return m.Rᵀ⁻¹∇r
end 
## Hm(w) - Hessian of Maholinobis Distance
struct NonlinearHesianWeakNLL<:HesianWeakNLL 
    # output 
    H::AbstractMatrix{<:Real}
    # data 
    tt::AbstractVector{<:Real}
    X::AbstractMatrix{<:Real}
    _Y::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    b₀::AbstractVector{<:Real}
    sig::AbstractVector{<:Real}
    # functions 
    R!::Covariance
    r!::Residual
    ∇r!::JacobianResidual
    ∇L!::GradientCovarianceFactor
    Hₚf!::Function
    Hₚ∇ₓf!::Function
    # buffers 
    S⁻¹r::AbstractVector{<:Real}
    S⁻¹∇r::AbstractMatrix{<:Real}
    ∂ⱼLLᵀ::AbstractMatrix{<:Real}
    ∇S::AbstractArray{<:Real, 3}
    HwF::AbstractArray{<:Real, 4}
    _∇²r::AbstractArray{<:Real, 4}
    ∇²r::AbstractArray{<:Real, 3}
    HwJuF::AbstractArray{<:Real, 5}
    __∇²L::AbstractArray{<:Real, 6}
    _∇²L::AbstractArray{<:Real, 6}
    ∇²L::AbstractArray{<:Real, 4}
    ∂ⱼL∂ᵢLᵀ::AbstractMatrix{<:Real}
    ∂ᵢⱼLLᵀ::AbstractMatrix{<:Real}
    ∂ᵢⱼS::AbstractMatrix{<:Real}
    S⁻¹∂ⱼS::AbstractMatrix{<:Real}
    ∂ᵢSS⁻¹∂ⱼS::AbstractMatrix{<:Real}
end

function NonlinearHesianWeakNLL(data::WENDyInternals{false,<:Distribution}, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J =  data.J

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
    HwF = zeros(T, D, J, J, Mp1)
    _∇²r = zeros(T, K, D, J, J)
    ∇²r = zeros(T, K*D, J, J)
    HwJuF = zeros(T, D, D, J, J, Mp1)
    __∇²L = zeros(T, K, D, D, J, J, Mp1)
    _∇²L = zeros(T, K, D, Mp1, D, J, J)
    ∇²L = zeros(T, K*D, Mp1*D, J, J)
    ∂ⱼL∂ᵢLᵀ = zeros(T, K*D, K*D)
    ∂ⱼᵢLLᵀ = zeros(T, K*D, K*D)
    ∂ᵢⱼS = zeros(T, K*D, K*D)
    S⁻¹∂ⱼS = zeros(T, K*D, K*D)
    ∂ᵢSS⁻¹∂ⱼS = zeros(T, K*D, K*D)
    NonlinearHesianWeakNLL(
        H,
        data.tt,data.X, data.X, data.V, data.b₀, data.sig,
        R!, r!, ∇r!, ∇L!, data.Hₚf!, data.Hₚ∇ₓf!,  
        S⁻¹r, S⁻¹∇r,  ∂ⱼLLᵀ, ∇S, HwF, _∇²r, ∇²r, HwJuF, __∇²L, _∇²L, ∇²L, ∂ⱼL∂ᵢLᵀ, ∂ⱼᵢLLᵀ, ∂ᵢⱼS, S⁻¹∂ⱼS, ∂ᵢSS⁻¹∂ⱼS
    )
end
## method inplace
function (m::NonlinearHesianWeakNLL)(H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    # TODO: try letting cholesky factorization back in here
    m.R!(w; transpose=false, doChol=false) 
    m.r!(w) 
    m.∇r!(w)
    m.∇L!(w)
    _Hwnll!(
        H, w,
        m.∇L!.∇L, m.tt, m.X, m._Y, m.V, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.b₀, m.sig,
        m.r!.r, 
        m.Hₚf!, m.Hₚ∇ₓf!,  
        m.S⁻¹r, m.S⁻¹∇r, m.∂ⱼLLᵀ, m.∇S, m.HwF, m._∇²r, m.∇²r, m.HwJuF, m.__∇²L, m._∇²L, m.∇²L, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼLLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
end
# method mutate internal data
function (m::NonlinearHesianWeakNLL)(w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    # TODO: try letting cholesky factorization back in here
    m.R!(w; transpose=false, doChol=false) 
    m.r!(w) 
    m.∇r!(w)
    m.∇L!(w)
    _Hwnll!(
        m.H, w,
        m.∇L!.∇L, m.tt, m.X, m._Y, m.V, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.b₀, m.sig,
        m.r!.r, 
        m.Hₚf!, m.Hₚ∇ₓf!,  
        m.S⁻¹r, m.S⁻¹∇r, m.∂ⱼLLᵀ, m.∇S, m.HwF, m._∇²r, m.∇²r, m.HwJuF, m.__∇²L, m._∇²L, m.∇²L, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼLLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
    return m.H
end