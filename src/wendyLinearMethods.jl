## L(p)
# struct
struct LinearCovarianceFactor<:CovarianceFactor
    # output
    L::AbstractMatrix{<:Real}
    # data
    L₁::AbstractArray{<:Real,3} 
    L₀::AbstractMatrix{<:Real}
end 

# constructor
function LinearCovarianceFactor(data::WENDyInternals{true,<:Distribution}, ::Union{Nothing, WENDyParameters}=nothing) 
    tt, X, V, Vp, sig, ∇ₓf! = data.tt, data.X, data.V, data.Vp, data.sig,data.∇ₓf!
    Mp1, D = size(X)
    K, Mp1 = size(V)
    J = data.J
    # preallocate output
    L = zeros(K*D,Mp1*D)
    # precompute L₀ because it does not depend on p
    __L₀ = zeros(K,D,D,Mp1)
    _L₀ = zeros(K,D,Mp1,D)
    L₀ = zeros(K*D,Mp1*D)
    _L₀!(
        L₀,
        Vp, sig,
        __L₀,_L₀
    )
    # precompute L₁ because it is constant wrt p 
    L₁ = zeros(K*D, Mp1*D,J)
    JuF = zeros(D,D,Mp1)
    _∂ⱼL = zeros(K,D,D,Mp1)
    ∂ⱼL = zeros(K,D,Mp1,D)
    eⱼ = zeros(J)
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
function (m::LinearCovarianceFactor)(L::AbstractMatrix, p::AbstractVector{<:Real}) 
    _L!(
        L,p,
        m.L₁,m.L₀
    )
    nothing
end
# method mutate internal data 
function (m::LinearCovarianceFactor)(p::AbstractVector{<:Real}) 
    _L!(
        m.L,p,
        m.L₁,m.L₀
    )
    return m.L
end
## ∇L(p)
struct LinearJacobianCovarianceFactor<:JacobianCovarianceFactor 
    # output
    ∇L::AbstractArray{<:Real,3}
end 
function LinearJacobianCovarianceFactor(data::WENDyInternals{true, <:Distribution}, ::Union{WENDyParameters,Nothing}=nothing)
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    L₁ = zeros(K*D, Mp1*D,J)
    JuF = zeros(D,D,Mp1)
    _∂ⱼL = zeros(K,D,D,Mp1)
    ∂ⱼL = zeros(K,D,Mp1,D)
    eⱼ = zeros(J)
    _L₁!(
        L₁, 
        data.tt, data.X, data.V, data.sig, 
        data.∇ₓf!, 
        JuF, _∂ⱼL, ∂ⱼL, eⱼ
    )
   
    LinearJacobianCovarianceFactor(L₁)
end
# method inplace 
function (m::LinearJacobianCovarianceFactor)(∇L::AbstractArray{3, <:Real}, ::AbstractVector{<:Real})
    ∇L .= m.∇L
end
function (m::LinearJacobianCovarianceFactor)(::AbstractVector{<:Real}) 
    return m.∇L
end
## r(p) - Residual
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
function LinearResidual(data::WENDyInternals{true,<:Distribution}, params::Union{WENDyParameters, Nothing}=nothing)
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    KD = K*D
    # ouput
    r = zeros(KD)
    # buffers
    g = zeros(KD)
    LinearResidual(r,data.b₀, data.G, g)
end
# method inplace 
function (m::LinearResidual)(r::AbstractVector{<:Real}, p::AbstractVector{<:Real}) 
    _r!(
        r, p, 
        m.G, m.b₀
    )
    nothing 
end
# Inplace: This assumes that b = R⁻ᵀ*b₀
function (m::LinearResidual)(r::AbstractVector{<:Real}, p::AbstractVector{<:Real}, b::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}) 
    _Rᵀr!(
        r, p, 
        m.∇r, Rᵀ, b, 
        m.g
    )
    nothing
end
# method mutate internal data 
function (m::LinearResidual)(p::AbstractVector{<:Real}) 
    _r!(
        m.r, p, 
        m.G, m.b₀
    )
    return m.r
end
# method mutate internal data: This assumes that b = R⁻ᵀ*b₀
function (m::LinearResidual)(p::AbstractVector{<:Real}, b::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}) 
    _Rᵀr!(
        m.r, p, 
        m.G, Rᵀ, b, 
        m.g
    )
    return m.r
end
struct LinearJacobianResidual<:JacobianResidual
    Rᵀ⁻¹∇r::AbstractMatrix{<:Real}
    ∇r::AbstractMatrix{<:Real}
    g::AbstractVector{<:Real}
end
# constructors
function LinearJacobianResidual(data::WENDyInternals{true,<:Distribution}, params::Union{WENDyParameters, Nothing}=nothing)
    LinearJacobianResidual(similar(data.G), data.G, similar(data.b₀))
end
# method inplace 
function (m::LinearJacobianResidual)(∇r::AbstractMatrix{<:Real}, ::Union{AbstractVector{<:Real},Nothing}=nothing)
    @views ∇r .= m.∇r
    return nothing
end
# method in place when Rᵀ is given
function (m::LinearJacobianResidual)(Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, ::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real})
    ldiv!(Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    nothing
end 
# method mutate internal data 
function (m::LinearJacobianResidual)(::AbstractVector{<:Real})
    return m.∇r
end
# method mutate internal data when Rᵀ is given
function (m::LinearJacobianResidual)(::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real})
    ldiv!(m.Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    return m.Rᵀ⁻¹∇r
end 
## ∇m(p) - gradient of Maholinobis distance from the precomputed blocks of LLᵀ
struct LinearGradientWeakNLL<:GradientWeakNLL
    # output
    ∇m::AbstractVector{<:Real}
    # data
    P::AbstractArray{<:Real,3}
    G::AbstractMatrix{<:Real}
    b₀::AbstractVector{<:Real}
    diagReg::Real
    # buffers
    S::AbstractMatrix{<:Real}
    S⁻¹::AbstractMatrix{<:Real}
    r::AbstractVector{<:Real}
    S⁻¹r::AbstractVector{<:Real}
    Qⱼ::AbstractMatrix{<:Real}
    q::AbstractVector{<:Real}
end
# constructor: P are the precomputed blocks of LLᵀ (see _LLᵀBlocks)
function LinearGradientWeakNLL(data::WENDyInternals{true,<:Distribution}, params::WENDyParameters, P::AbstractArray{<:Real,3})
    _, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    KD = K*D
    # output
    ∇m = zeros(J)
    LinearGradientWeakNLL(
        ∇m,
        P, data.G, data.b₀, params.diagReg,
        zeros(KD, KD), zeros(KD, KD), zeros(KD), zeros(KD), zeros(KD, KD), zeros(J+1)
    )
end
# method inplace
function (m::LinearGradientWeakNLL)(∇m::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    _∇wnll!(
        ∇m, p,
        m.P, m.G, m.b₀, m.diagReg,
        m.S, m.S⁻¹, m.r, m.S⁻¹r, m.Qⱼ, m.q
    )
end
# method mutate internal data
function (m::LinearGradientWeakNLL)(p::AbstractVector{<:Real})
    m(m.∇m, p)
    return m.∇m
end
## Hm(p) - Hessian of Maholinobis Distance from the precomputed blocks of LLᵀ
struct LinearHesianWeakNLL<:HesianWeakNLL
    # output 
    H::AbstractMatrix{<:Real}
    # data 
    P::AbstractArray{<:Real,3}
    G::AbstractMatrix{<:Real}
    b₀::AbstractVector{<:Real}
    diagReg::Real
    # buffers 
    S::AbstractMatrix{<:Real}
    S⁻¹::AbstractMatrix{<:Real}
    r::AbstractVector{<:Real}
    S⁻¹r::AbstractVector{<:Real}
    S⁻¹G::AbstractMatrix{<:Real}
    Qⱼ::AbstractMatrix{<:Real}
    ∂ⱼS::AbstractMatrix{<:Real}
    ∇Sy::AbstractMatrix{<:Real}
    S⁻¹∇Sy::AbstractMatrix{<:Real}
    S⁻¹∇S::AbstractArray{<:Real,3}
    q::AbstractVector{<:Real}
end
# constructor: P are the precomputed blocks of LLᵀ (see _LLᵀBlocks)
function LinearHesianWeakNLL(data::WENDyInternals{true,<:Distribution}, params::WENDyParameters, P::AbstractArray{<:Real,3}) 
    _, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    KD = K*D
    # ouput 
    H = zeros(J,J)
    LinearHesianWeakNLL(
        H,
        P, data.G, data.b₀, params.diagReg,
        zeros(KD, KD), zeros(KD, KD), zeros(KD), zeros(KD), zeros(KD, J), 
        zeros(KD, KD), zeros(KD, KD), zeros(KD, J), zeros(KD, J), zeros(KD, KD, J), zeros(J+1)
    )
end
# method inplace
function (m::LinearHesianWeakNLL)(H::AbstractMatrix{<:Real}, p::AbstractVector{<:Real})
    _Hwnll!(
        H, p,
        m.P, m.G, m.b₀, m.diagReg,
        m.S, m.S⁻¹, m.r, m.S⁻¹r, m.S⁻¹G, 
        m.Qⱼ, m.∂ⱼS, m.∇Sy, m.S⁻¹∇Sy, m.S⁻¹∇S, m.q
    )
end
# method mutate internal data
function (m::LinearHesianWeakNLL)(p::AbstractVector{<:Real})
    m(m.H, p)
    return m.H
end
