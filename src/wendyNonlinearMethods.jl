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
    jacuf!::Function
    # buffers
    JuF::AbstractArray{<:Real, 3}
    __L₁::AbstractArray{<:Real, 4}
    _L₁::AbstractArray{<:Real, 4}
end 
# constructor
function NonlinearCovarianceFactor(prob::WENDyProblem{false}, params::Union{WENDyParameters,Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real
    D, M = size(prob.U)
    K, _ = size(prob.V)
    # preallocate output
    L = zeros(T,K*D,M*D)
    # precompute L₀ because it does not depend on w
    __L₀ = zeros(T,K,D,D,M)
    _L₀ = zeros(T,K,D,M,D)
    L₀ = zeros(T,K*D,M*D)
    _L₀!(
        L₀,
        prob.Vp, prob.sig,
        __L₀,_L₀
    )
    # buffers
    JuF = zeros(T,D,D,M)
    __L₁ = zeros(T,K,D,D,M)
    _L₁ = zeros(T,K,D,M,D)

    return NonlinearCovarianceFactor(
        L,
        prob.tt, prob._Y, prob.V,L₀,prob.sig, 
        prob.jacuf!, 
        JuF, __L₁, _L₁
    )
end
# method inplace 
function (m::NonlinearCovarianceFactor)(L::AbstractMatrix, w::AbstractVector{<:Real}; ll::LogLevel=Info) 
    _L!(
        L,w,
        m.tt, m._Y,m.V,m.L₀,m.sig,
        m.jacuf!,
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
        m.jacuf!,
        m.JuF,m.__L₁,m._L₁;
        ll=ll
    )
    nothing
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
    jacwjacuf!::Function
    # buffers
    JwJuF::AbstractArray{<:Real,4}
    __∇L::AbstractArray{<:Real,5}
    _∇L::AbstractArray{<:Real,5}
end 
function NonlinearGradientCovarianceFactor(prob, params, ::Val{T}=Val(Float64)) where T<:Real
    K,M,D,J = prob.K, prob.M, prob.D, prob.J
    ∇L = zeros(T,K*D,M*D,J)
    JwJuF = zeros(T,D,D,J,M)
    __∇L = zeros(T,K,D,D,J,M)
    _∇L = zeros(T,K,D,M,D,J)
   
    NonlinearGradientCovarianceFactor(
        ∇L,
        prob.tt, prob._Y, prob.V, prob.sig, 
        prob.jacwjacuf!,
        JwJuF, __∇L, _∇L
    )
end
# method inplace 
function (m::NonlinearGradientCovarianceFactor)(∇L::AbstractArray{3, <:Real}, w::AbstractVector{<:Real};ll::LogLevel=Warn)
    _∇L!(
        ∇L, w,
        m.tt,m._Y,m.V,m.sig,
        m.jacwjacuf!,
        m.JwJuF, m.__∇L, m._∇L
    )
end
# method mutating interal storage
function (m::NonlinearGradientCovarianceFactor)(w::AbstractVector{<:Real};ll::LogLevel=Warn)
    _∇L!(
        m.∇L, w,
        m.tt,m._Y,m.V,m.sig,
        m.jacwjacuf!,
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
    U::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real} 
    # functions
    f!::Function
    # buffers
    F::AbstractMatrix{<:Real} 
    G::AbstractMatrix{<:Real} 
    g::AbstractVector{<:Real} 
end
# constructors 
function NonlinearResidual(prob::WENDyProblem{false}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real
    D, M = size(prob.U)
    K, _ = size(prob.V)
    # ouput
    r = zeros(T,K*D)
    # buffers
    F = zeros(T, D, M)
    G = zeros(T, K, D)
    g = zeros(T, K*D)
    NonlinearResidual(
        r,
        prob.tt,prob.U, prob.V, 
        prob.f!,
        F,G,g
    )
end
# method inplace 
function (m::NonlinearResidual)(r::AbstractVector{<:Real}, b::AbstractVector{<:Real}, w::AbstractVector{T}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where T<:Real 
    if isnothing(Rᵀ)
        _r!(
            r,w, 
            m.tt, m.U, m.V, b, # assumes that b = b₀ in this case
            m.f!, 
            m.F, m.G; 
            ll=ll
        )
    else 
        _Rᵀr!(
            r, w, 
            m.tt, m.U, m.V, Rᵀ, b, 
            m.f!, 
            m.F, m.G, m.g; 
            ll=ll 
        )
    end
    nothing
end
# method mutate internal data 
function (m::NonlinearResidual)(b::AbstractVector{<:Real}, w::AbstractVector{T}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where T<:Real 
    if isnothing(Rᵀ)
        _r!(
            m.r,w, 
            m.tt, m.U, m.V, b, # assumes that b = b₀ in this case
            m.f!, 
            m.F, m.G; 
            ll=ll
        )
    else 
        _Rᵀr!(
            m.r, w, 
            m.tt, m.U, m.V, Rᵀ, b, 
            m.f!, 
            m.F, m.G, m.g; 
            ll=ll 
        )
    end
    nothing
end
## ∇r & Rᵀ∇r
# struct
struct NonlinearGradientResidual<:GradientResidual
    # ouput 
    Rᵀ⁻¹∇r::AbstractMatrix{<:Real}
    # data 
    tt::AbstractVector{<:Real}
    U::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    # functions 
    jacwf!::Function
    #buffers
    JwF::AbstractArray{<:Real,3}
    __∇r::AbstractArray{<:Real, 3}
    _∇r::AbstractArray{<:Real, 3}
    ∇r::AbstractMatrix{<:Real} 
end
# constructors
function NonlinearGradientResidual(prob::WENDyProblem{false}, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    J = prob.J
    D, M = size(prob.U)
    K, _ = size(prob.V)
    Rᵀ⁻¹∇r = zeros(K*D,J)
    JwF = zeros(T,D,J,M)
    __∇r = zeros(T,D,J,K)
    _∇r = zeros(T,K,D,J)
    ∇r = zeros(T,K*D, J)
    NonlinearGradientResidual(
        Rᵀ⁻¹∇r, 
        prob.tt, prob.U, prob.V, 
        prob.jacwf!, 
        JwF, __∇r, _∇r, ∇r
    )
end
# method inplace 
function (m::NonlinearGradientResidual)(Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    if isnothing(Rᵀ)
        _∇r!(
            Rᵀ⁻¹∇r,w, # in this context Rᵀ⁻¹∇r === ∇r
            m.tt,m.U,m.V,
            m.jacwf!,
            m.JwF,m.__∇r,m._∇r;
            ll=ll
        )
        return nothing
    end
    _Rᵀ⁻¹∇r!(
        Rᵀ⁻¹∇r,w,
        m.tt,m.U,m.V,Rᵀ,
        m.jacwf!,
        m.JwF,m.__∇r,m._∇r,m.∇r;
        ll=ll
    )
    nothing
end 
# method mutate internal data 
function (m::NonlinearGradientResidual)(w::AbstractVector{<:Real}; ll::LogLevel=Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    if isnothing(Rᵀ)
        _∇r!(
            m.∇r,w, 
            m.tt,m.U,m.V,
            m.jacwf!,
            m.JwF,m.__∇r,m._∇r;
            ll=ll
        )
        return nothing
    end
    _Rᵀ⁻¹∇r!(
        m.Rᵀ⁻¹∇r,w,
        m.tt,m.U,m.V,Rᵀ,
        m.jacwf!,
        m.JwF,m.__∇r,m._∇r,m.∇r;
        ll=ll
    )
    nothing
end 
##
## Hm(w) - Hessian of Maholinobis Distance
struct NonlinearHesianWeakNLL<:HesianWeakNLL 
    # output 
    H::AbstractMatrix{<:Real}
    # data 
    tt::AbstractVector{<:Real}
    U::AbstractMatrix{<:Real}
    _Y::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    b₀::AbstractVector{<:Real}
    sig::AbstractVector{<:Real}
    # functions 
    R!::Covariance
    r!::Residual
    ∇r!::GradientResidual
    ∇L!::GradientCovarianceFactor
    heswf!::Function
    heswjacuf!::Function
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

function NonlinearHesianWeakNLL(prob::WENDyProblem{false}, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    K,M,D,J = prob.K, prob.M, prob.D, prob.J
    # ouput 
    H = zeros(J,J)
    # functions
    R! = Covariance(prob, params)
    r! = Residual(prob, params)
    ∇r! = GradientResidual(prob, params)
    ∇L! = GradientCovarianceFactor(prob, params)
    # buffers
    S⁻¹r = zeros(T, K*D)
    S⁻¹∇r = zeros(T, K*D, J)
    ∂ⱼLLᵀ = zeros(T, K*D, K*D)
    ∇S = zeros(T, K*D, K*D, J)
    HwF = zeros(T, D, J, J, M)
    _∇²r = zeros(T, K, D, J, J)
    ∇²r = zeros(T, K*D, J, J)
    HwJuF = zeros(T, D, D, J, J, M)
    __∇²L = zeros(T, K, D, D, J, J, M)
    _∇²L = zeros(T, K, D, M, D, J, J)
    ∇²L = zeros(T, K*D, M*D, J, J)
    ∂ⱼL∂ᵢLᵀ = zeros(T, K*D, K*D)
    ∂ⱼᵢLLᵀ = zeros(T, K*D, K*D)
    ∂ᵢⱼS = zeros(T, K*D, K*D)
    S⁻¹∂ⱼS = zeros(T, K*D, K*D)
    ∂ᵢSS⁻¹∂ⱼS = zeros(T, K*D, K*D)
    NonlinearHesianWeakNLL(
        H,
        prob.tt,prob.U, prob._Y, prob.V, prob.b₀, prob.sig,
        R!, r!, ∇r!, ∇L!, prob.heswf!, prob.heswjacuf!,  
        S⁻¹r, S⁻¹∇r,  ∂ⱼLLᵀ, ∇S, HwF, _∇²r, ∇²r, HwJuF, __∇²L, _∇²L, ∇²L, ∂ⱼL∂ᵢLᵀ, ∂ⱼᵢLLᵀ, ∂ᵢⱼS, S⁻¹∂ⱼS, ∂ᵢSS⁻¹∂ⱼS
    )
end
## method inplace
function (m::NonlinearHesianWeakNLL)(H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    # TODO: try letting cholesky factorization back in here
    m.R!(w; transpose=false, doChol=false) 
    m.r!(m.b₀, w) 
    m.∇r!(w)
    m.∇L!(w)
    _Hm!(
        H, w,
        m.∇L!.∇L, m.tt, m.U, m._Y, m.V, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.b₀, m.sig,
        m.r!.r, 
        m.heswf!, m.heswjacuf!,  
        m.S⁻¹r, m.S⁻¹∇r, m.∂ⱼLLᵀ, m.∇S, m.HwF, m._∇²r, m.∇²r, m.HwJuF, m.__∇²L, m._∇²L, m.∇²L, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼLLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
end
# method mutate internal data
function (m::NonlinearHesianWeakNLL)(w::AbstractVector{<:Real}; ll::LogLevel=Warn)
    # TODO: try letting cholesky factorization back in here
    m.R!(w; transpose=false, doChol=false) 
    m.r!(m.b₀, w) 
    m.∇r!(w)
    m.∇L!(w)
    _Hm!(
        m.H, w,
        m.∇L!.∇L, m.tt, m.U, m._Y, m.V, m.R!.L!.L, m.R!.Sreg, m.∇r!.∇r, m.b₀, m.sig,
        m.r!.r, 
        m.heswf!, m.heswjacuf!,  
        m.S⁻¹r, m.S⁻¹∇r, m.∂ⱼLLᵀ, m.∇S, m.HwF, m._∇²r, m.∇²r, m.HwJuF, m.__∇²L, m._∇²L, m.∇²L, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼLLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
end