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
## Blocks of LLᵀ
# When the problem is linear in parameters we precompute the blocks of LLᵀ (see _LLᵀBlocks!) for the gradient and hessian,
# unless they would take more than this much memory, then we fall back to the adjoint forms
const MAX_LLᵀ_BLOCK_BYTES = 2^30
function _LLᵀBlocks(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip
    _, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    KD = K*D
    if !lip || (J+1)^2*KD^2*sizeof(Float64) > MAX_LLᵀ_BLOCK_BYTES
        return nothing
    end
    L! = LinearCovarianceFactor(data, params)
    P = zeros(KD*KD, J+1, J+1)
    _LLᵀBlocks!(P, L!.L₀, L!.L₁, zeros(KD, KD))
    return P
end
## ∇m(p) - gradient of Maholinobis distance
abstract type GradientWeakNLL<:Function end
function GradientWeakNLL(data::WENDyInternals, params::WENDyParameters; P::Union{Nothing,AbstractArray{<:Real,3}}=_LLᵀBlocks(data, params))
    return isnothing(P) ? AdjointGradientWeakNLL(data, params) : LinearGradientWeakNLL(data, params, P)
end
# Adjoint form: works whether or not the problem is linear in parameters
struct AdjointGradientWeakNLL<:GradientWeakNLL
    # output
    ∇m::AbstractVector{<:Real}
    # data
    diagReg::Real
    # functions
    R!::Covariance
    r!::Residual
    ∇r!::JacobianResidual
    ∇L!::JacobianCovarianceFactor
    # Buffers
    S⁻¹r::AbstractVector{<:Real}
    S⁻¹L::AbstractMatrix{<:Real}
    Lᵀy::AbstractVector{<:Real}
end
# constructor
function AdjointGradientWeakNLL(data::WENDyInternals, params::WENDyParameters)
    Mp1, D = size(data.X)
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
    S⁻¹r = zeros(K*D)
    S⁻¹L = zeros(K*D, Mp1*D)
    Lᵀy = zeros(Mp1*D)
    AdjointGradientWeakNLL(
        ∇m,
        params.diagReg,
        R!, r!, ∇r!, ∇L!,
        S⁻¹r, S⁻¹L, Lᵀy
    )
end
# method inplace
function (m::AdjointGradientWeakNLL)(∇m::AbstractVector{<:Real}, p::AbstractVector{<:Real})
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
        m.∇L!.∇L, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.r!.r, m.diagReg, # data
        m.S⁻¹r, m.S⁻¹L, m.Lᵀy # buffers
    )
end
# method mutate internal data
function (m::AdjointGradientWeakNLL)(p::AbstractVector{<:Real})
    m(m.∇m, p)
    return m.∇m
end
## Hm(p) - Hessian of Maholinobis distance
abstract type HesianWeakNLL<:Function end
function HesianWeakNLL(data::WENDyInternals, params::WENDyParameters; P::Union{Nothing,AbstractArray{<:Real,3}}=_LLᵀBlocks(data, params))
    return isnothing(P) ? AdjointHesianWeakNLL(data, params) : LinearHesianWeakNLL(data, params, P)
end
# Trace identity form: works whether or not the problem is linear in parameters
struct AdjointHesianWeakNLL<:HesianWeakNLL
    # output
    H::AbstractMatrix{<:Real}
    # data
    lip::Bool
    diagReg::Real
    tt::AbstractVector{<:Real}
    X::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    sig::AbstractVector{<:Real}
    # functions
    R!::Covariance
    r!::Residual
    ∇r!::JacobianResidual
    ∇L!::JacobianCovarianceFactor
    Hₚf!::Function
    Hₚ∇ₓf!::Function
    # buffers
    S⁻¹r::AbstractVector{<:Real}
    S⁻¹∇r::AbstractMatrix{<:Real}
    S⁻¹L::AbstractMatrix{<:Real}
    Lᵀy::AbstractVector{<:Real}
    ∇Lᵀy::AbstractMatrix{<:Real}
    ∇Sy::AbstractMatrix{<:Real}
    S⁻¹∇Sy::AbstractMatrix{<:Real}
    ∂ⱼS::AbstractMatrix{<:Real}
    S⁻¹∇S::AbstractArray{<:Real,3}
    S⁻¹∂ⱼL::AbstractMatrix{<:Real}
    # buffers for second order terms (empty when linear in parameters)
    HwF::AbstractArray{<:Real,4}
    HwJuF::AbstractArray{<:Real,5}
    VᵀY::AbstractMatrix{<:Real}
    VᵀM::AbstractArray{<:Real,3}
end
function AdjointHesianWeakNLL(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters) where lip
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    KD, MD = K*D, Mp1*D
    # output
    H = zeros(J,J)
    # functions
    R! = Covariance(data, params)
    r! = Residual(data, params)
    ∇r! = JacobianResidual(data, params)
    ∇L! = JacobianCovarianceFactor(data, params)
    # second derivatives vanish when linear in parameters
    Mp1₂ = lip ? 0 : Mp1
    HwF = zeros(D, J, J, Mp1₂)
    HwJuF = zeros(D, D, J, J, Mp1₂)
    VᵀY = zeros(Mp1₂, D)
    VᵀM = zeros(D, Mp1₂, D)
    AdjointHesianWeakNLL(
        H,
        lip, params.diagReg, data.tt, data.X, data.V, data.sig,
        R!, r!, ∇r!, ∇L!, data.Hₚf!, data.Hₚ∇ₓf!,
        zeros(KD), zeros(KD, J), zeros(KD, MD), zeros(MD), zeros(MD, J),
        zeros(KD, J), zeros(KD, J), zeros(KD, KD), zeros(KD, KD, J), zeros(KD, MD),
        HwF, HwJuF, VᵀY, VᵀM
    )
end
# method inplace
function (m::AdjointHesianWeakNLL)(H::AbstractMatrix{<:Real}, p::AbstractVector{<:Real})
    m.R!(p; transpose=false, doChol=false)
    m.r!(p)
    m.∇r!(p)
    m.∇L!(p)
    if !m.lip
        _HwF!(m.HwF, p, m.tt, m.X, m.Hₚf!)
        _HwJuF!(m.HwJuF, p, m.tt, m.X, m.Hₚ∇ₓf!)
    end
    _Hwnll!(
        H,
        m.∇L!.∇L, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.r!.r, m.diagReg,
        m.lip ? nothing : m.HwJuF, m.lip ? nothing : m.HwF, m.V, m.sig,
        m.S⁻¹r, m.S⁻¹∇r, m.S⁻¹L, m.Lᵀy, m.∇Lᵀy,
        m.∇Sy, m.S⁻¹∇Sy, m.∂ⱼS, m.S⁻¹∇S, m.S⁻¹∂ⱼL,
        m.VᵀY, m.VᵀM
    )
end
# method mutate internal data
function (m::AdjointHesianWeakNLL)(p::AbstractVector{<:Real})
    m(m.H, p)
    return m.H
end