""" 
    For computational efficiency, it is advantageous to compute the matrix L(p) where LLᵀ = S(p) the covariance of the weak residual. 
    Depending on whether the problem is linear in parameters or not a LinearCovarianceFactor or a NonLinearCovarianceFactor will be created. 
"""
abstract type CovarianceFactor<:Function end 
function CovarianceFactor(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip
    return lip ? LinearCovarianceFactor(data, params) : NonlinearCovarianceFactor(data, params)
end
""" 
    Jacobian of the covariance factor with respect to the parameters of interest p, ∇ₚL. 
"""
abstract type JacobianCovarianceFactor<:Function end 
function JacobianCovarianceFactor(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip
    return lip ? LinearJacobianCovarianceFactor(data, params) : NonlinearJacobianCovarianceFactor(data, params)
end
## R(p)
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
# constructor
function Covariance(data::WENDyInternals, params::WENDyParameters) 
    L! = CovarianceFactor(data, params)
    KD, _ = size(L!.L)
    # ouput
    R = zeros(KD, KD)
    # buffers 
    thisI = Matrix(I, KD, KD)
    S = zeros(KD, KD)
    Sreg = zeros(KD, KD)
    Covariance(R, params.diagReg, L!, thisI, S, Sreg)
end
# method inplace 
function (m::Covariance)(R::AbstractMatrix{<:Real}, p::AbstractVector{<:Real}; transpose::Bool=true, doChol::Bool=true)
    m.L!(p) 
    _R!(
        R,
        m.L!.L, m.diagReg,
        m.thisI,m.Sreg,m.S;
        doChol=doChol 
    )
    if transpose 
        @views R .= R'
    end
    nothing
end
# method mutate internal data 
function (m::Covariance)(p::AbstractVector{<:Real}; transpose::Bool=true, doChol::Bool=true)
    m.L!(p) 
    _R!(
        m.R,
        m.L!.L, m.diagReg,
        m.thisI,m.Sreg,m.S;
        doChol=doChol
    )
    if transpose 
        @views m.R .= m.R'
    end
    m.R
end
## G(p) - b / G*p - b 
abstract type Residual<:Function end 
function Residual(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip 
    return lip ? LinearResidual(data, params) :  NonlinearResidual(data, params)
end
abstract type JacobianResidual<:Function end
function JacobianResidual(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip
    return lip ? LinearJacobianResidual(data, params) : NonlinearJacobianResidual(data, params)
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
function WeakNLL(data::WENDyInternals, params::WENDyParameters)
    # functions
    R! = Covariance(data, params)
    r! = Residual(data, params)
    # buffer
    _, D = size(data.X)
    K, _ = size(data.V)
    S = zeros(K*D, K*D)
    Rᵀ= zeros(K*D, K*D)
    r = zeros(K*D)
    S⁻¹r = zeros(K*D)
    Rᵀ⁻¹r = zeros(K*D)
    WeakNLL(
        R!, r!, 
        S, Rᵀ, r, S⁻¹r,Rᵀ⁻¹r
    )
end
# method
function (m::WeakNLL)(p::AbstractVector{<:Real})
    KD,_ = size(m.S) 
    constTerm = KD/2*log(2*pi)
    m.r!(m.r,p)
    m.R!(m.S,p, doChol=false)
    return _wnll(m.S, m.r, m.S⁻¹r, constTerm)
end
## ∇m(p) - gradient of Maholinobis distance
struct GradientWeakNLL<:Function
    # output
    ∇m::AbstractVector{<:Real}
    # functions
    R!::Covariance
    r!::Residual
    ∇r!::JacobianResidual
    ∇L!::JacobianCovarianceFactor
    # Buffers
    S⁻¹r::AbstractVector{<:Real}
    ∂ⱼLLᵀ::AbstractMatrix{<:Real}
    ∇S::AbstractArray{<:Real,3}
end
# constructor
function GradientWeakNLL(data::WENDyInternals, params::WENDyParameters) 
    _, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    # output
    ∇m = zeros(J)
    # methods 
    R!  = Covariance(data, params)
    r!  = Residual(data, params)
    ∇r! = JacobianResidual(data, params)
    ∇L! = JacobianCovarianceFactor(data, params)
    # preallocate buffers
    S⁻¹r = zeros( K*D)
    ∂ⱼLLᵀ = zeros( K*D,K*D)
    ∇S = zeros(K*D,K*D,J)
    GradientWeakNLL(
        ∇m,
        R!, r!, ∇r!, ∇L!,
        S⁻¹r, ∂ⱼLLᵀ, ∇S
    )
end
# method inplace
function (m::GradientWeakNLL)(∇m::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    # Compute L(p) & S(p)
    m.R!(p, transpose=false, doChol=false) 
    # Compute residal
    m.r!(p)
    # Compute jacobian of residual
    m.∇r!(p)
    # Compute jacobian of covariance factor
    m.∇L!(p)
    
    _∇wnll!(
        ∇m, # ouput
        m.∇L!.∇L, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.r!.r, # data
        m.S⁻¹r,  m.∂ⱼLLᵀ, m.∇S; # buffers
    )
end
# method mutate internal data
function (m::GradientWeakNLL)(p::AbstractVector{<:Real})
    # Compute L(p) & S(p)
    m.R!(p, transpose=false, doChol=false) 
    # Compute residal
    m.r!(p)
    # Compute jacobian of residual
    m.∇r!(p)
    # Compute jacobian of covariance factor
    m.∇L!(p)
    _∇wnll!(
        m.∇m,  # ouput
        m.∇L!.∇L, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.r!.r, # data
        m.S⁻¹r,  m.∂ⱼLLᵀ, m.∇S
    )
    return m.∇m
end
## Because of the complications we separate these into separate types
abstract type HesianWeakNLL<:Function end
function HesianWeakNLL(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip
    return lip ? LinearHesianWeakNLL(data, params) : NonlinearHesianWeakNLL(data, params)
end