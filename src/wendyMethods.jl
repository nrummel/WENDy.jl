## L(w)
abstract type CovarianceFactor<:Function end 
function CovarianceFactor(prob::T, params::WENDyProblem) where T<:WENDyProblem 
    if T<:NonlinearWENDyProblem || params.forceNonLinear
        return NonlinearCovarianceFactor(prob, params)
    end
    return LinearCovarianceFactor(prob, params)
end
## ∇L(w)
abstract type GradientCovarianceFactor<:Function end 
function GradientCovarianceFactor(prob::T, params::WENDyProblem) where T<:WENDyProblem 
    if T<:NonlinearWENDyProblem || params.forceNonLinear
        return NonlinearGradientCovarianceFactor(prob, params)
    end
    return LinearGradientCovarianceFactor(prob, params)
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
    K,D,_,_ = size(L!._L₁)
    # ouput
    R = zeros(T, K*D, K*D)
    # buffers 
    thisI = Matrix{T}(I, K*D, K*D)
    S = zeros(T, K*D, K*D)
    Sreg = zeros(T, K*D, K*D)
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
        @views R .= R'
    end
    nothing
end
## G(w) - b / G*w - b 
abstract type Residual<:Function end 
function Residual(prob::T, params::WENDyProblem) where T<:WENDyProblem 
    if T<:NonlinearWENDyProblem || params.forceNonLinear
        return NonlinearResidual(prob, params)
    end
    return LinearResidual(prob, params)
end
abstract type GradientResidual<:Function end
function GradientResidual(prob::T, params::WENDyProblem) where T<:WENDyProblem 
    if T<:NonlinearWENDyProblem || params.forceNonLinear
        return NonlinearGradientResidual(prob, params)
    end
    return LinearGradientResidual(prob, params)
end
## Maholinobis distance 
# struct
struct MahalanobisDistance<:Function
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
function MahalanobisDistance(prob::WENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
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
    MahalanobisDistance(
        prob.b₀, 
        R!, r!, 
        S, Rᵀ, r, S⁻¹r,Rᵀ⁻¹r
    )
end
# method
function (m::MahalanobisDistance)(w::AbstractVector{T}; ll::LogLevel=Warn, efficient::Bool=false) where T<:Real
    if efficient
        # Can be unstable because in worst case S(w) ⊁ 0
        # m(w) = r(w)ᵀS⁻¹r(w) = ((Rᵀ)⁻¹r)ᵀ((Rᵀ)⁻¹r)
        m.R!(
            m.Rᵀ,w;
            ll=ll
        )
        b = similar(m.b₀)
        ldiv!(b, LowerTriangular(m.Rᵀ), m.b₀)
        m.r!(
            m.Rᵀ⁻¹r, b, w; 
            ll=ll, Rᵀ=m.Rᵀ
        )
        return _m(m.Rᵀ⁻¹r)
    end 
    m.r!(m.r,w;ll=ll)
    m.R!(m.S,w;ll=ll, doChol=false)
    return _m(m.S,m.r,m.S⁻¹r)
end
## ∇m(w) - gradient of Maholinobis distance
struct GradientMahalanobisDistance<:Function
    # output
    ∇m::AbstractVector{<:Real}
    # data
    U::AbstractMatrix{<:Real} 
    V::AbstractMatrix{<:Real} 
    b₀::AbstractVector{<:Real} 
    sig::AbstractVector{<:Real} 
    # functions
    R!::Covariance
    r!::Residual
    ∇r!::GradientResidual
    jacwjacuf!::Function
    # Buffers
    S⁻¹r::AbstractVector{<:Real}
    JwJuF::AbstractArray{<:Real,4}
    __∇L::AbstractArray{<:Real,5}
    _∇L::AbstractArray{<:Real,5}
    ∇L::AbstractArray{<:Real,3}
    ∂ⱼLLᵀ::AbstractMatrix{<:Real}
    ∇S::AbstractArray{<:Real,3}
end
# constructor
function GradientMahalanobisDistance(prob::WENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T
    K,M,D,J  = prob.K, prob.M, prob.D, prob.J
    # output
    ∇m = zeros(T,J)
    # methods 
    R!  = Covariance(prob, params, Val(T))
    r!  = Residual(prob, params, Val(T))
    ∇r! = GradientResidual(prob, params, Val(T))
    # preallocate buffers
    S⁻¹r = zeros(T, K*D)
    JwJuF = zeros(T,D,D,J,M)
    __∇L = zeros(T,K,D,D,J,M)
    _∇L = zeros(T,K,D,M,D,J)
    ∇L = zeros(T,K*D,M*D,J)
    ∂ⱼLLᵀ = zeros(T, K*D,K*D)
    ∇S = zeros(T,K*D,K*D,J)
    GradientMahalanobisDistance(
        ∇m,
        prob.U, prob.V, prob.b₀, prob.sig,
        R!, r!, ∇r!, prob.jacwjacuf!,
        S⁻¹r, JwJuF, __∇L, _∇L, ∇L, ∂ⱼLLᵀ, ∇S
    )
end
# method inplace
function (m::GradientMahalanobisDistance)(∇m::AbstractVector{<:Real}, w::AbstractVector{W}; ll::LogLevel=Warn) where W<:Real
    # Compute L(w) & S(w)
    m.R!(w; ll=ll, transpose=false, doChol=false) 
    # Compute residal
    m.r!(m.b₀, w; ll=ll)
    # Compute jacobian of residual
    m.∇r!(w; ll=ll)

    _∇m!(
        ∇m, w, # ouput, input
        m.U, m.V, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.sig, m.r!.r, # data
        m.jacwjacuf!, # functions
        m.S⁻¹r, m.JwJuF, m.__∇L, m._∇L, m.∇L, m.∂ⱼLLᵀ, m.∇S; # buffers
        ll=ll # kwargs
    )
    nothing
end
# method mutate internal data
function (m::GradientMahalanobisDistance)(w::AbstractVector{W}; ll::LogLevel=Warn) where W<:Real
    # Compute L(w) & S(w)
    m.R!(w; ll=ll, transpose=false, doChol=false) 
    # Compute residal
    m.r!(m.b₀, w; ll=ll)
    # Compute jacobian of residual
    m.∇r!(w; ll=ll)

    _∇m!(
        m.∇m, w, # ouput, input
        m.U, m.V, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.sig, m.r!.r, # data
        m.jacwjacuf!, # functions
        m.S⁻¹r, m.JwJuF, m.__∇L, m._∇L, m.∇L, m.∂ⱼLLᵀ, m.∇S; # buffers
        ll=ll # kwargs
    )
    nothing
end