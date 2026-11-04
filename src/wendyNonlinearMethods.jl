## L(p)
# struct
struct NonlinearCovarianceFactor<:CovarianceFactor
    # output
    L::AbstractMatrix{<:Real}
    # data
    tt::AbstractVector{<:Real} 
    X::AbstractMatrix{<:Real} 
    V::AbstractMatrix{<:Real} 
    L₀::AbstractMatrix{<:Real}
    sig::AbstractVector{<:Real}
    # functions
    ∇ₓf!::Function
    # buffers
    JuF::AbstractArray{<:Real, 3}
    _L₁::AbstractArray{<:Real, 4}
end 
# constructor
function NonlinearCovarianceFactor(data::WENDyInternals{false,<:Distribution}, params::Union{WENDyParameters,Nothing}=nothing)
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    # preallocate output
    L = zeros(K*D,Mp1*D)
    # precompute L₀ because it does not depend on p
    __L₀ = zeros(K,D,D,Mp1)
    _L₀ = zeros(K,D,Mp1,D)
    L₀ = zeros(K*D,Mp1*D)
    _L₀!(
        L₀,
        data.Vp, data.sig,
        __L₀,_L₀
    )
    # buffers
    JuF = zeros(D,D,Mp1)
    _L₁ = zeros(K,D,Mp1,D)

    return NonlinearCovarianceFactor(
        L,
        data.tt, data.X, data.V,L₀,data.sig, 
        data.∇ₓf!, 
        JuF, _L₁
    )
end
# method inplace 
function (m::NonlinearCovarianceFactor)(L::AbstractMatrix, p::AbstractVector{<:Real}) 
    _L!(
        L,p,
        m.tt, m.X,m.V,m.L₀,m.sig,
        m.∇ₓf!,
        m.JuF, m._L₁
    )
    nothing
end
# method mutate internal data 
function (m::NonlinearCovarianceFactor)(p::AbstractVector{<:Real}) 
    _L!(
        m.L,p,
        m.tt, m.X,m.V,m.L₀,m.sig,
        m.∇ₓf!,
        m.JuF, m._L₁
    )
    return m.L
end
##
struct NonlinearJacobianCovarianceFactor<:JacobianCovarianceFactor 
    # output
    ∇L::AbstractArray{<:Real,3}
    # data 
    tt::AbstractVector{<:Real}
    X::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    sig::AbstractVector{<:Real}
    # functions
    ∇ₚ∇ₓf!::Function
    # buffers
    JwJuF::AbstractArray{<:Real,4}
    _∇L::AbstractArray{<:Real,5}
end 
function NonlinearJacobianCovarianceFactor(data::WENDyInternals{false,<:Distribution}, params::Union{WENDyParameters,Nothing}=nothing)
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    J = data.J
    ∇L = zeros(K*D,Mp1*D,J)
    JwJuF = zeros(D,D,J,Mp1)
    _∇L = zeros(K,D,Mp1,D,J)
   
    NonlinearJacobianCovarianceFactor(
        ∇L,
        data.tt, data.X, data.V, data.sig, 
        data.∇ₚ∇ₓf!,
        JwJuF, _∇L
    )
end
# method inplace 
function (m::NonlinearJacobianCovarianceFactor)(∇L::AbstractArray{3, <:Real}, p::AbstractVector{<:Real})
    _∇L!(
        ∇L, p,
        m.tt,m.X,m.V,m.sig,
        m.∇ₚ∇ₓf!,
        m.JwJuF, m._∇L
    )
end
# method mutating interal storage
function (m::NonlinearJacobianCovarianceFactor)(p::AbstractVector{<:Real})
    _∇L!(
        m.∇L, p,
        m.tt,m.X,m.V,m.sig,
        m.∇ₚ∇ₓf!,
        m.JwJuF, m._∇L
    )
end
## r(p) - NonlinearResidual
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
function NonlinearResidual(data::WENDyInternals{false,<:Distribution}, params::Union{WENDyParameters, Nothing}=nothing)
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    # ouput
    r = zeros(K*D)
    # buffers
    F = zeros(Mp1, D)
    G = zeros(K, D)
    g = zeros(K*D)
    NonlinearResidual(
        r,
        data.tt,data.b₀,data.X, data.V, 
        data.f!,
        F,G,g
    )
end
# method inplace 
function (m::NonlinearResidual)(r::AbstractVector{<:Real}, p::AbstractVector{<:Real}) 
    _r!(
        r,p, 
        m.tt, m.X, m.V, m.b₀, 
        m.f!, 
        m.F, m.G
    )
end
# method inplace: assume b = R⁻ᵀ*b₀
function (m::NonlinearResidual)(r::AbstractVector{<:Real}, p::AbstractVector{<:Real}, b::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}) 
    _Rᵀr!(
        r, p, 
        m.tt, m.X, m.V, Rᵀ, b, 
        m.f!, 
        m.F, m.G, m.g
    )
end
# method mutate internal data 
function (m::NonlinearResidual)(p::AbstractVector{<:Real}) 
    _r!(
        m.r,p, 
        m.tt, m.X, m.V, m.b₀, 
        m.f!, 
        m.F, m.G
    )
    return m.r
end
# method mutate internal data: assume b = R⁻ᵀ*b₀
function (m::NonlinearResidual)(p::AbstractVector{<:Real}, b::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real}) 
    _Rᵀr!(
        m.r, p, 
        m.tt, m.X, m.V, Rᵀ, b, 
        m.f!, 
        m.F, m.G, m.g
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
function NonlinearJacobianResidual(data::WENDyInternals{false,<:Distribution}, params::Union{WENDyParameters, Nothing}=nothing) 
    J = data.J
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    Rᵀ⁻¹∇r = zeros(K*D,J)
    JwF = zeros(D,J,Mp1)
    __∇r = zeros(D,J,K)
    _∇r = zeros(K,D,J)
    ∇r = zeros(K*D, J)
    NonlinearJacobianResidual(
        Rᵀ⁻¹∇r, 
        data.tt, data.X, data.V, 
        data.∇ₚf!, 
        JwF, __∇r, _∇r, ∇r
    )
end
# method inplace 
function (m::NonlinearJacobianResidual)(∇r::AbstractMatrix{<:Real}, p::AbstractVector{<:Real})
    
    _∇r!(
        ∇r,p, 
        m.tt,m.X,m.V,
        m.∇ₚf!,
        m.JwF,m.__∇r,m._∇r
    )
    
end
# method inplace 
function (m::NonlinearJacobianResidual)(Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, p::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real})
    m(p) #compute ∇r
    ldiv!(Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    nothing
end 
# method mutate internal data 
function (m::NonlinearJacobianResidual)(p::AbstractVector{<:Real})
    _∇r!(
        m.∇r,p, 
        m.tt,m.X,m.V,
        m.∇ₚf!,
        m.JwF,m.__∇r,m._∇r
    )
    return m.∇r
end
# method mutate internal data 
function (m::NonlinearJacobianResidual)(p::AbstractVector{<:Real}, Rᵀ::AbstractMatrix{<:Real})
    m(p) #compute ∇r
    ldiv!(m.Rᵀ⁻¹∇r, LowerTriangular(Rᵀ), m.∇r)
    return m.Rᵀ⁻¹∇r
end 
