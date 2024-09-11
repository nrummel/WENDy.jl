## L(w)
abstract type CovarianceFactor<:Function end 
function CovarianceFactor(prob::WENDyProblem{lip}, params::WENDyParameters) where lip
    return lip ? LinearCovarianceFactor(prob, params) : NonlinearCovarianceFactor(prob, params)
end
## ∇L(w)
abstract type GradientCovarianceFactor<:Function end 
function GradientCovarianceFactor(prob::WENDyProblem{lip}, params::WENDyParameters, ::Val{T}=Val(Float64)) where {lip, T<:Real}
    return lip ? LinearGradientCovarianceFactor(prob, params, Val(T)) : NonlinearGradientCovarianceFactor(prob, params, Val(T))
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
function Covariance(prob::WENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    L! = CovarianceFactor(prob, params)
    Covariance(params.diagReg, L!, prob.K, prob.D, Val(T))
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
function Residual(prob::WENDyProblem{lip}, params::WENDyParameters, ::Val{T}=Val(Float64)) where {lip,T<:Real} 
    return lip ? LinearResidual(prob, params, Val(T)) :  NonlinearResidual(prob, params, Val(T))
end
abstract type GradientResidual<:Function end
function GradientResidual(prob::WENDyProblem{lip}, params::WENDyParameters, ::Val{T}=Val(Float64)) where {lip,T<:Real}
    return lip ? LinearGradientResidual(prob, params, Val(T)) : NonlinearGradientResidual(prob, params, Val(T))
end
## Maholinobis distance 
# struct
struct WeakNLL<:Function
    b₀::AbstractVector{<:Real}
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
function WeakNLL(prob::WENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    # functions
    R! = Covariance(prob, params, Val(T))
    r! = Residual(prob, params, Val(T))
    # buffer
    K,D = prob.K,prob.D
    S = zeros(T, K*D, K*D)
    Rᵀ= zeros(T, K*D, K*D)
    r = zeros(T, K*D)
    S⁻¹r = zeros(T, K*D)
    Rᵀ⁻¹r = zeros(T, K*D)
    WeakNLL(
        prob.b₀, 
        R!, r!, 
        S, Rᵀ, r, S⁻¹r,Rᵀ⁻¹r
    )
end
# method
function (m::WeakNLL)(w::AbstractVector{T}; ll::LogLevel=Warn, efficient::Bool=false) where T<:Real
    # if efficient
    #     # Can be unstable because in worst case S(w) ⊁ 0
    #     # m(w) = r(w)ᵀS⁻¹r(w) = ((Rᵀ)⁻¹r)ᵀ((Rᵀ)⁻¹r)
    #     m.R!(
    #         m.Rᵀ,w;
    #         ll=ll
    #     )
    #     b = similar(m.b₀)
    #     ldiv!(b, LowerTriangular(m.Rᵀ), m.b₀) # b = (Rᵀ)⁻¹b₀
    #     m.r!(
    #         m.Rᵀ⁻¹r, b, w; 
    #         ll=ll, Rᵀ=m.Rᵀ
    #     ) # b = (Rᵀ)⁻¹G(w)
    #     return _m(m.Rᵀ⁻¹r) # ((Rᵀ)⁻¹r)^T(Rᵀ)⁻¹r
    # end 
    KD,_ = size(m.S) 
    constTerm = KD/2*log(2*pi)
    m.r!(m.r,m.b₀,w;ll=ll)
    m.R!(m.S,w;ll=ll, doChol=false)
    return _m(m.S, m.r, m.S⁻¹r, constTerm)
end
## ∇m(w) - gradient of Maholinobis distance
struct GradientWeakNLL<:Function
    # output
    ∇m::AbstractVector{<:Real}
    # data 
    b₀::AbstractVector{<:Real}
    # functions
    R!::Covariance
    r!::Residual
    ∇r!::GradientResidual
    ∇L!::GradientCovarianceFactor
    # Buffers
    S⁻¹r::AbstractVector{<:Real}
    ∂ⱼLLᵀ::AbstractMatrix{<:Real}
    ∇S::AbstractArray{<:Real,3}
end
# constructor
function GradientWeakNLL(prob::WENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T
    K,D,J  = prob.K, prob.D, prob.J
    # output
    ∇m = zeros(T,J)
    # methods 
    R!  = Covariance(prob, params, Val(T))
    r!  = Residual(prob, params, Val(T))
    ∇r! = GradientResidual(prob, params, Val(T))
    ∇L! = GradientCovarianceFactor(prob, params, Val(T))
    # preallocate buffers
    S⁻¹r = zeros(T, K*D)
    ∂ⱼLLᵀ = zeros(T, K*D,K*D)
    ∇S = zeros(T,K*D,K*D,J)
    GradientWeakNLL(
        ∇m,
        prob.b₀,
        R!, r!, ∇r!, ∇L!,
        S⁻¹r, ∂ⱼLLᵀ, ∇S
    )
end
# method inplace
function (m::GradientWeakNLL)(∇m::AbstractVector{<:Real}, w::AbstractVector{W}; ll::LogLevel=Warn) where W<:Real
    # Compute L(w) & S(w)
    m.R!(w; ll=ll, transpose=false, doChol=false) 
    # Compute residal
    m.r!(m.b₀, w; ll=ll)
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
    m.r!(m.b₀, w; ll=ll)
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
function HesianWeakNLL(prob::WENDyProblem{lip}, params::WENDyParameters) where lip
    return lip ? LinearHesianWeakNLL(prob, params) : NonlinearHesianWeakNLL(prob, params)
end