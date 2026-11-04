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
## Shared state for m(p), ∇m(p) and Hm(p) 
# The optimizers usually evaluate the weak negative log-likelihood, its gradient and its hessian at the same p, so 
# they share this cache. Whenever p changes (compared by value) we recompute S(p), its factorization, r(p) and 
# y = S⁻¹r. The remaining pieces are only computed the first time something asks for them at that p.
mutable struct WeakNLLCache
    # the parameters everything below was computed for 
    p::AbstractVector{<:Real}
    hasP::Bool
    # data 
    diagReg::Real
    P::Union{Nothing,AbstractArray{<:Real,3}} # blocks of LLᵀ (nothing ⇒ build S from L(p))
    # functions (L!, ∇r! and ∇L! are nothing when using the blocks of LLᵀ)
    L!::Union{Nothing,CovarianceFactor}
    r!::Residual
    ∇r!::Union{Nothing,JacobianResidual}
    ∇L!::Union{Nothing,JacobianCovarianceFactor}
    # computed whenever p changes
    q::AbstractVector{<:Real} # [1; p]
    S::AbstractMatrix{<:Real}
    F::Factorization
    logDet::Real
    r::AbstractVector{<:Real}
    S⁻¹r::AbstractVector{<:Real}
    # computed on request (blocks of LLᵀ only)
    hasS⁻¹::Bool
    S⁻¹::AbstractMatrix{<:Real}
    # computed on request (L(p) only)
    hasJacobians::Bool
    hasM::Bool
    Lᵀy::AbstractVector{<:Real}
    M::AbstractMatrix{<:Real} # S⁻¹L - y(Lᵀy)ᵀ
end
function WeakNLLCache(data::WENDyInternals, params::WENDyParameters; P::Union{Nothing,AbstractArray{<:Real,3}}=_LLᵀBlocks(data, params))
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    KD, MD = K*D, Mp1*D
    r! = Residual(data, params)
    L!, ∇r!, ∇L! = if isnothing(P)
        CovarianceFactor(data, params), JacobianResidual(data, params), JacobianCovarianceFactor(data, params)
    else
        nothing, nothing, nothing
    end
    S⁻¹ = isnothing(P) ? zeros(0,0) : zeros(KD, KD)
    Lᵀy, M = isnothing(P) ? (zeros(MD), zeros(KD, MD)) : (zeros(0), zeros(0,0))
    WeakNLLCache(
        fill(NaN, J), false,
        params.diagReg, P, 
        L!, r!, ∇r!, ∇L!,
        zeros(J+1), zeros(KD, KD), cholesky(Matrix{Float64}(I, KD, KD)), 0.0, zeros(KD), zeros(KD),
        false, S⁻¹,
        false, false, Lᵀy, M
    )
end
# Bring the cache up to date with p: S(p), its factorization, log|S|, r(p) and y = S⁻¹r
function _update!(c::WeakNLLCache, p::AbstractVector{<:Real})
    c.hasP && c.p == p && return c
    c.hasP = c.hasS⁻¹ = c.hasJacobians = c.hasM = false
    c.p .= p
    c.q[1] = 1
    @views c.q[2:end] .= p
    if isnothing(c.P)
        c.L!(p)
        mul!(c.S, c.L!.L, c.L!.L', 1-c.diagReg, 0)
        @views c.S[diagind(c.S)] .+= c.diagReg
    else
        _S!(c.S, c.q, c.P, c.diagReg)
    end
    c.r!(c.r, p)
    c.F, c.logDet = try
        F = cholesky(Symmetric(c.S))
        F, 2*sum(log, diag(F.U))
    catch
        F = lu(c.S)
        F, sum(log.(filter!(x-> x >0, diag(F.U))))
    end
    ldiv!(c.S⁻¹r, c.F, c.r)
    c.hasP = true
    c
end
# S⁻¹ (blocks of LLᵀ only)
function _S⁻¹!(c::WeakNLLCache)
    if !c.hasS⁻¹
        copyto!(c.S⁻¹, I)
        ldiv!(c.F, c.S⁻¹)
        c.hasS⁻¹ = true
    end
    c.S⁻¹
end
# ∇r(p) and ∇L(p), stored in c.∇r!.∇r and c.∇L!.∇L (L(p) only)
function _jacobians!(c::WeakNLLCache)
    if !c.hasJacobians
        c.∇r!(c.p)
        c.∇L!(c.p)
        c.hasJacobians = true
    end
    nothing
end
# M = S⁻¹L - y(Lᵀy)ᵀ (L(p) only)
function _M!(c::WeakNLLCache)
    if !c.hasM
        L = c.L!.L
        ldiv!(c.M, c.F, L)
        mul!(c.Lᵀy, L', c.S⁻¹r)
        mul!(c.M, c.S⁻¹r, c.Lᵀy', -1, 1)
        c.hasM = true
    end
    c.M
end
## m(p) - Maholinobis distance
struct WeakNLL<:Function
    cache::WeakNLLCache
end
WeakNLL(data::WENDyInternals, params::WENDyParameters; cache::WeakNLLCache=WeakNLLCache(data, params)) = WeakNLL(cache)
function (m::WeakNLL)(p::AbstractVector{<:Real})
    c = _update!(m.cache, p)
    KD = length(c.r)
    return (dot(c.r, c.S⁻¹r) + c.logDet)/2 + KD/2*log(2*pi)
end
## ∇m(p) - gradient of Maholinobis distance
abstract type GradientWeakNLL<:Function end
function GradientWeakNLL(data::WENDyInternals, params::WENDyParameters; cache::WeakNLLCache=WeakNLLCache(data, params))
    return isnothing(cache.P) ? AdjointGradientWeakNLL(data, params, cache) : LinearGradientWeakNLL(data, params, cache)
end
# Adjoint form: works whether or not the problem is linear in parameters
struct AdjointGradientWeakNLL<:GradientWeakNLL
    # output
    ∇m::AbstractVector{<:Real}
    # shared state
    cache::WeakNLLCache
end
AdjointGradientWeakNLL(data::WENDyInternals, params::WENDyParameters, cache::WeakNLLCache) = AdjointGradientWeakNLL(zeros(data.J), cache)
# method inplace
function (m::AdjointGradientWeakNLL)(∇m::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    c = _update!(m.cache, p)
    _jacobians!(c)
    _∇wnll!(
        ∇m, # ouput
        c.∇L!.∇L, c.∇r!.∇r, c.S⁻¹r, _M!(c), c.diagReg # data
    )
end
# method mutate internal data
function (m::AdjointGradientWeakNLL)(p::AbstractVector{<:Real})
    m(m.∇m, p)
    return m.∇m
end
## Hm(p) - Hessian of Maholinobis distance
abstract type HesianWeakNLL<:Function end
function HesianWeakNLL(data::WENDyInternals, params::WENDyParameters; cache::WeakNLLCache=WeakNLLCache(data, params))
    return isnothing(cache.P) ? AdjointHesianWeakNLL(data, params, cache) : LinearHesianWeakNLL(data, params, cache)
end
# Trace identity form: works whether or not the problem is linear in parameters
struct AdjointHesianWeakNLL<:HesianWeakNLL
    # output
    H::AbstractMatrix{<:Real}
    # shared state
    cache::WeakNLLCache
    # data
    lip::Bool
    tt::AbstractVector{<:Real}
    X::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    sig::AbstractVector{<:Real}
    # functions
    Hₚf!::Function
    Hₚ∇ₓf!::Function
    # buffers
    S⁻¹∇r::AbstractMatrix{<:Real}
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
function AdjointHesianWeakNLL(data::WENDyInternals{lip, <:Distribution}, params::WENDyParameters, cache::WeakNLLCache) where lip
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    KD, MD = K*D, Mp1*D
    # second derivatives vanish when linear in parameters
    Mp1₂ = lip ? 0 : Mp1
    AdjointHesianWeakNLL(
        zeros(J,J),
        cache,
        lip, data.tt, data.X, data.V, data.sig,
        data.Hₚf!, data.Hₚ∇ₓf!,
        zeros(KD, J), zeros(MD), zeros(MD, J), zeros(KD, J), zeros(KD, J), zeros(KD, KD), zeros(KD, KD, J), zeros(KD, MD),
        zeros(D, J, J, Mp1₂), zeros(D, D, J, J, Mp1₂), zeros(Mp1₂, D), zeros(D, Mp1₂, D)
    )
end
# method inplace: approx = true leaves out the log-det curvature (see _Hwnll!)
function (m::AdjointHesianWeakNLL)(H::AbstractMatrix{<:Real}, p::AbstractVector{<:Real}; approx::Bool=false)
    c = _update!(m.cache, p)
    _jacobians!(c)
    if !m.lip
        _HwF!(m.HwF, p, m.tt, m.X, m.Hₚf!)
        _HwJuF!(m.HwJuF, p, m.tt, m.X, m.Hₚ∇ₓf!)
    end
    _Hwnll!(
        H,
        c.F, c.∇L!.∇L, c.L!.L, c.∇r!.∇r, c.S⁻¹r, c.diagReg,
        m.lip ? nothing : m.HwJuF, m.lip ? nothing : m.HwF, m.lip ? nothing : _M!(c), m.V, m.sig,
        m.S⁻¹∇r, m.Lᵀy, m.∇Lᵀy, m.∇Sy, m.S⁻¹∇Sy, m.∂ⱼS, m.S⁻¹∇S, m.S⁻¹∂ⱼL,
        m.VᵀY, m.VᵀM;
        approx=approx
    )
end
# method mutate internal data
function (m::AdjointHesianWeakNLL)(p::AbstractVector{<:Real}; approx::Bool=false)
    m(m.H, p; approx=approx)
    return m.H
end
