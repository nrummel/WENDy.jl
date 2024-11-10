## L(w)
abstract type CovarianceFactor<:Function end 
function CovarianceFactor(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip
    return lip ? LinearCovarianceFactor(data, params) : NonlinearCovarianceFactor(data, params)
end
## ∇L(w)
abstract type GradientCovarianceFactor<:Function end 
function GradientCovarianceFactor(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters, ::Val{T}=Val(Float64)) where {lip, T<:Real}
    return lip ? LinearGradientCovarianceFactor(data, params, Val(T)) : NonlinearGradientCovarianceFactor(data, params, Val(T))
end
## R(w)
# struct/method
struct Covariance<:Function
    #output
    R::AbstractMatrix{<:Real} 
    # data 
    diagReg::Real
    # functions
    L!::CovarianceFactor 
    # buffers
    thisI::AbstractMatrix{<:Real}
    S::AbstractMatrix{<:Real}
    Sreg::AbstractMatrix{<:Real}
end
# constructors
function Covariance(diagReg::AbstractFloat, L!::CovarianceFactor, K::Int, D::Int, ::Val{T}=Val(Float64)) where T<:Real
    KD,MD = size(L!.L)
    # ouput
    R = zeros(T, KD, KD)
    # buffers 
    thisI = Matrix{T}(I, KD, KD)
    S = zeros(T, KD, KD)
    Sreg = zeros(T, KD, KD)
    Covariance(R, diagReg, L!, thisI, S, Sreg)
end
function Covariance(data::WENDydatalem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    L! = CovarianceFactor(data, params)
    Covariance(params.diagReg, L!, data.K, data.D, Val(T))
end
# method inplace 
function (m::Covariance)(R::AbstractMatrix{<:Real}, w::AbstractVector{W};ll::LogLevel=Warn, transpose::Bool=true, doChol::Bool=true) where W<:Real
    m.L!(w; ll=ll) 
    _R!(
        R,w,
        m.L!.L, m.diagReg,
        m.thisI,m.Sreg,m.S;
        doChol=doChol, ll=ll
    )
    if transpose 
        @views R .= R'
    end
    nothing
end
# method mutate internal data 
function (m::Covariance)(w::AbstractVector{W}; ll::LogLevel=Warn, transpose::Bool=true, doChol::Bool=true) where W<:Real
    m.L!(w;ll=ll) 
    _R!(
        m.R, w,
        m.L!.L, m.diagReg,
        m.thisI,m.Sreg,m.S;
        doChol=doChol, ll=ll
    )
    if transpose 
        @views m.R .= m.R'
    end
    m.R
end
## G(w) - b / G*w - b 
abstract type Residual<:Function end 
function Residual(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters, ::Val{T}=Val(Float64)) where {lip,T<:Real} 
    return lip ? LinearResidual(data, params, Val(T)) :  NonlinearResidual(data, params, Val(T))
end
abstract type JacobianResidual<:Function end
function JacobianResidual(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters, ::Val{T}=Val(Float64)) where {lip,T<:Real}
    return lip ? LinearJacobianResidual(data, params, Val(T)) : NonlinearJacobianResidual(data, params, Val(T))
end
## Maholinobis distance 
# struct
struct WeakNLL<:Function
    # functions
    R!::Covariance
    r!::Residual
    # buffer
    S::AbstractMatrix{<:Real}
    Rᵀ::AbstractMatrix{<:Real}
    r::AbstractVector{<:Real}
    S⁻¹r::AbstractVector{<:Real}
    Rᵀ⁻¹r::AbstractVector{<:Real}
end
# constructor
function WeakNLL(data::WENDydatalem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    # functions
    R! = Covariance(data, params, Val(T))
    r! = Residual(data, params, Val(T))
    # buffer
    K,D = data.K,data.D
    S = zeros(T, K*D, K*D)
    Rᵀ= zeros(T, K*D, K*D)
    r = zeros(T, K*D)
    S⁻¹r = zeros(T, K*D)
    Rᵀ⁻¹r = zeros(T, K*D)
    WeakNLL(
        R!, r!, 
        S, Rᵀ, r, S⁻¹r,Rᵀ⁻¹r
    )
end
# method
function (m::WeakNLL)(w::AbstractVector{T}; ll::LogLevel=Warn) where T<:Real
    KD,_ = size(m.S) 
    constTerm = KD/2*log(2*pi)
    m.r!(m.r,w;ll=ll)
    m.R!(m.S,w;ll=ll, doChol=false)
    return _m(m.S, m.r, m.S⁻¹r, constTerm)
end
## ∇m(w) - gradient of Maholinobis distance
struct GradientWeakNLL<:Function
    # output
    ∇m::AbstractVector{<:Real}
    # functions
    R!::Covariance
    r!::Residual
    ∇r!::JacobianResidual
    ∇L!::GradientCovarianceFactor
    # Buffers
    S⁻¹r::AbstractVector{<:Real}
    ∂ⱼLLᵀ::AbstractMatrix{<:Real}
    ∇S::AbstractArray{<:Real,3}
end
# constructor
function GradientWeakNLL(data::WENDydatalem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T
    K,D,J  = data.K, data.D, data.J
    # output
    ∇m = zeros(T,J)
    # methods 
    R!  = Covariance(data, params, Val(T))
    r!  = Residual(data, params, Val(T))
    ∇r! = JacobianResidual(data, params, Val(T))
    ∇L! = GradientCovarianceFactor(data, params, Val(T))
    # preallocate buffers
    S⁻¹r = zeros(T, K*D)
    ∂ⱼLLᵀ = zeros(T, K*D,K*D)
    ∇S = zeros(T,K*D,K*D,J)
    GradientWeakNLL(
        ∇m,
        R!, r!, ∇r!, ∇L!,
        S⁻¹r, ∂ⱼLLᵀ, ∇S
    )
end
# method inplace
function (m::GradientWeakNLL)(∇m::AbstractVector{<:Real}, w::AbstractVector{W}; ll::LogLevel=Warn) where W<:Real
    # Compute L(w) & S(w)
    m.R!(w; ll=ll, transpose=false, doChol=false) 
    # Compute residal
    m.r!(w; ll=ll)
    # Compute jacobian of residual
    m.∇r!(w; ll=ll)
    # Compute jacobian of covariance factor
    m.∇L!(w; ll=ll)
    
    _∇m!(
        ∇m, w, # ouput, input
        m.∇L!.∇L, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.r!.r, # data
        m.S⁻¹r,  m.∂ⱼLLᵀ, m.∇S; # buffers
        ll=ll # kwargs
    )
end
# method mutate internal data
function (m::GradientWeakNLL)(w::AbstractVector{W}; ll::LogLevel=Warn) where W<:Real
    # Compute L(w) & S(w)
    m.R!(w; ll=ll, transpose=false, doChol=false) 
    # Compute residal
    m.r!(w; ll=ll)
    # Compute jacobian of residual
    m.∇r!(w; ll=ll)
    # Compute jacobian of covariance factor
    m.∇L!(w; ll=ll)
    _∇m!(
        m.∇m, w, # ouput, input
        m.∇L!.∇L, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.r!.r, # data
        m.S⁻¹r,  m.∂ⱼLLᵀ, m.∇S; # buffers
        ll=ll # kwargs
    )
end
## Because of the complications we separate these into separate types
abstract type HesianWeakNLL<:Function end
function HesianWeakNLL(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip
    return lip ? LinearHesianWeakNLL(data, params) : NonlinearHesianWeakNLL(data, params)
end