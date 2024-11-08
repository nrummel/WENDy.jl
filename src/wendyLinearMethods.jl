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
    Mp1, D = size(U)
    K, Mp1 = size(V)
    J = prob.J
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
    _∂Lⱼ = zeros(T,K,D,D,Mp1)
    ∂Lⱼ = zeros(T,K,D,Mp1,D)
    eⱼ = zeros(T,J)
    _L₁!(
        L₁, 
        prob.tt, prob._Y, prob.V, prob.sig,
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
    return m.L
end
## ∇L(w)
struct LinearGradientCovarianceFactor<:GradientCovarianceFactor 
    # output
    ∇L::AbstractArray{<:Real,3}
end 
function LinearGradientCovarianceFactor(prob, params, ::Val{T}=Val(Float64)) where T<:Real
    K,Mp1,D,J = prob.K, prob.Mp1, prob.D, prob.J
    L₁ = zeros(T, K*D, Mp1*D,J)
    JuF = zeros(T,D,D,Mp1)
    _∂Lⱼ = zeros(T,K,D,D,Mp1)
    ∂Lⱼ = zeros(T,K,D,Mp1,D)
    eⱼ = zeros(T,J)
    _L₁!(
        L₁, 
        prob.tt, prob._Y, prob.V, prob.sig, 
        prob.jacuf!, 
        JuF, _∂Lⱼ, ∂Lⱼ, eⱼ
    )
   
    LinearGradientCovarianceFactor(L₁)
end
# method inplace 
function (m::LinearGradientCovarianceFactor)(∇L::AbstractArray{3, <:Real}, ::AbstractVector{<:Real}; ll::LogLevel=Warn)
    s∇L .= m.∇L
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
    Mp1, D = size(prob.U)
    K, _ = size(prob.V)
    KD = K*D
    # ouput
    r = zeros(T,KD)
    # buffers
    g = zeros(T, KD)
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
    return m.r
end
struct LinearJacobianResidual<:JacobianResidual
    Rᵀ⁻¹∇r::AbstractMatrix{<:Real}
    ∇r::AbstractMatrix{<:Real}
    G::AbstractMatrix{<:Real}
    g::AbstractVector{<:Real}
end
# constructors
function LinearJacobianResidual(prob::WENDyProblem{true}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    LinearJacobianResidual(similar(prob.G), similar(prob.G), prob.G, similar(prob.b₀))
end
# method inplace 
function (m::LinearJacobianResidual)(Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    if isnothing(Rᵀ)
        @views Rᵀ⁻¹∇r .= m.G
        return nothing
    end
    ldiv!(Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    nothing
end 
# method mutate internal data and return
function (m::LinearJacobianResidual)(w::AbstractVector{<:Real}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    if isnothing(Rᵀ)
        m.∇r .= m.G
        return m.∇r
    end
    ldiv!(m.Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.G)
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

function LinearHesianWeakNLL(prob::WENDyProblem{true}, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    K,Mp1,D,J = prob.K, prob.Mp1, prob.D, prob.J
    # ouput 
    H = zeros(J,J)
    # functions
    R! = Covariance(prob, params)
    r! = Residual(prob, params)
    ∇r! = JacobianResidual(prob, params)
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
    LinearHesianWeakNLL(
        H,
        prob.b₀,
        R!, r!, ∇r!, ∇L!,
        S⁻¹r, S⁻¹∇r, ∂ⱼLLᵀ, ∇S, ∂ⱼL∂ᵢLᵀ, ∂ᵢⱼS, S⁻¹∂ⱼS, ∂ᵢSS⁻¹∂ⱼS
    )
end
# method inplace
function (m::LinearHesianWeakNLL)(H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn)
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
function (m::LinearHesianWeakNLL)(w::AbstractVector{<:Real}; ll::LogLevel=Warn)
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
    return m.H
end