## L(w)
# struct
struct Lw 
    # output
    L::AbstractMatrix{<:Real}
    # data
    U::AbstractMatrix{<:Real} 
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
function Lw(
    U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, Vp::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, # data (available in prob struct)
    jacuf!::Function, # Function
    ::Val{T}=Val(Float64) #optional type
) where T<:Real
    D, M = size(U)
    K, _ = size(V)
    # preallocate output
    L = zeros(T,K*D,M*D)
    # precompute L₀ because it does not depend on w
    __L₀ = zeros(T,K,D,D,M)
    _L₀ = zeros(T,K,D,M,D)
    L₀ = zeros(T,K*D,M*D)
    _L₀!(
        L₀,
        Vp,sig,
        __L₀,_L₀
    )
    # buffers
    JuF = zeros(T,D,D,M)
    __L₁ = zeros(T,K,D,D,M)
    _L₁ = zeros(T,K,D,M,D)

    return Lw(
        L,
        U,V,L₀,sig, 
        jacuf!, 
        JuF, __L₁, _L₁
    )
end
function Lw(prob::AbstractWENDyProblem, params::Union{WENDyParameters,Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real
    Lw(prob.U, prob.V, prob.Vp, prob.sig, prob.jacuf!, Val(T))
end
# method inplace 
function (m::Lw)(L::AbstractMatrix, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Info) 
    _L!(
        L,w,
        m.U,m.V,m.L₀,m.sig,
        m.jacuf!,
        m.JuF,m.__L₁,m._L₁;
        ll=ll
    )
    nothing
end
# method mutate internal data 
function (m::Lw)(w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Info) 
    _L!(
        m.L,w,
        m.U,m.V,m.L₀,m.sig,
        m.jacuf!,
        m.JuF,m.__L₁,m._L₁;
        ll=ll
    )
    nothing
end
## R(w)
# struct/method
struct Rw
    #output
    R::AbstractMatrix{<:Real} 
    # data 
    diagReg::Real
    # functions
    _L!_::Lw 
    # buffers
    thisI::AbstractMatrix{<:Real}
    S::AbstractMatrix{<:Real}
    Sreg::AbstractMatrix{<:Real}
end
# constructors
function Rw(diagReg::AbstractFloat, _L!_::Lw, K::Int, D::Int, ::Val{T}=Val(Float64)) where T<:Real
    K,D,_,_ = size(_L!_._L₁)
    # ouput
    R = zeros(T, K*D, K*D)
    # buffers 
    thisI = Matrix{T}(I, K*D, K*D)
    S = zeros(T, K*D, K*D)
    Sreg = zeros(T, K*D, K*D)
    Rw(R, diagReg, _L!_, thisI, S, Sreg)
end
function Rw(prob::AbstractWENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    _L!_ = Lw(prob, params)
    Rw(params.diagReg, _L!_, prob.K, prob.D, Val(T))
end
# method inplace 
function (m::Rw)(R::AbstractMatrix{<:Real}, w::AbstractVector{W};ll::Logging.LogLevel=Logging.Warn, transpose::Bool=true, doChol::Bool=true) where W<:Real
    m._L!_(w; ll=ll) 
    _R!(
        R,w,
        m._L!_.L, m.diagReg,
        m.thisI,m.Sreg,m.S;
        doChol=doChol, ll=ll
    )
    if transpose 
        @views R .= R'
    end
    nothing
end
# method mutate internal data 
function (m::Rw)(w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn, transpose::Bool=true, doChol::Bool=true) where W<:Real
    m._L!_(w;ll=ll) 
    _R!(
        m.R, w,
        m._L!_.L, m.diagReg,
        m.thisI,m.Sreg,m.S;
        doChol=doChol, ll=ll
    )
    if transpose 
        @views R .= R'
    end
    nothing
end
## r(w) - Residual
# struct
struct rw 
    # ouput
    r::AbstractVector{<:Real}
    # data
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
function rw(U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, f!::Function, ::Val{T}=Val(Float64)) where T<:Real
    D, M = size(U)
    K, _ = size(V)
    # ouput
    r = zeros(T,K*D)
    # buffers
    F = zeros(T, D, M)
    G = zeros(T, K, D)
    g = zeros(T, K*D)
    rw(r,U,V,f!,F,G,g)
end
function rw(prob::AbstractWENDyProblem, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    rw(prob.U, prob.V, prob.f!, Val(T))
end
# method inplace 
function (m::rw)(r::AbstractVector{<:Real}, b::AbstractVector{<:Real}, w::AbstractVector{T}; ll::Logging.LogLevel=Logging.Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where T<:Real 
    if isnothing(Rᵀ)
        _r!(
            r,w, 
            m.U, m.V, b, # assumes that b = b₀ in this case
            m.f!, 
            m.F, m.G; 
            ll=ll
        )
    else 
        _Rᵀr!(
            r, w, 
            m.U, m.V, Rᵀ, b, 
            m.f!, 
            m.F, m.G, m.g; 
            ll=ll 
        )
    end
    nothing
end
# method mutate internal data 
function (m::rw)(b::AbstractVector{<:Real}, w::AbstractVector{T}; ll::Logging.LogLevel=Logging.Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where T<:Real 
    if isnothing(Rᵀ)
        _r!(
            m.r,w, 
            m.U, m.V, b, # assumes that b = b₀ in this case
            m.f!, 
            m.F, m.G; 
            ll=ll
        )
    else 
        _Rᵀr!(
            m.r, w, 
            m.U, m.V, Rᵀ, b, 
            m.f!, 
            m.F, m.G, m.g; 
            ll=ll 
        )
    end
    nothing
end
## ∇r & Rᵀ∇r
# struct
struct ∇rw
    # ouput 
    Rᵀ⁻¹∇r::AbstractMatrix{<:Real}
    # data 
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
function ∇rw(U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, J::Int, ::Val{T}=Val(Float64)) where T<:Real
    D, M = size(U)
    K, _ = size(V)
    Rᵀ⁻¹∇r = zeros(K*D,J)
    JwF = zeros(T,D,J,M)
    __∇r = zeros(T,D,J,K)
    _∇r = zeros(T,K,D,J)
    ∇r = zeros(T,K*D, J)
    ∇rw(Rᵀ⁻¹∇r, U, V, jacwf!, JwF, __∇r, _∇r, ∇r)
end
function ∇rw(prob::AbstractWENDyProblem, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    ∇rw(prob.U, prob.V, prob.jacwf!, prob.J, Val(T))
end
# method inplace 
function (m::∇rw)(Rᵀ⁻¹∇r::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    if isnothing(Rᵀ)
        _∇r!(
            Rᵀ⁻¹∇r,w, # in this context Rᵀ⁻¹∇r === ∇r
            m.U,m.V,
            m.jacwf!,
            m.JwF,m.__∇r,m._∇r;
            ll=ll
        )
        return nothing
    end
    _Rᵀ⁻¹∇r!(
        Rᵀ⁻¹∇r,w,
        m.U,m.V,Rᵀ,
        m.jacwf!,
        m.JwF,m.__∇r,m._∇r,m.∇r;
        ll=ll
    )
    nothing
end 
# method mutate internal data 
function (m::∇rw)(w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn, Rᵀ::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    if isnothing(Rᵀ)
        _∇r!(
            m.∇r,w, 
            m.U,m.V,
            m.jacwf!,
            m.JwF,m.__∇r,m._∇r;
            ll=ll
        )
        return nothing
    end
    _Rᵀ⁻¹∇r!(
        m.Rᵀ⁻¹∇r,w,
        m.U,m.V,Rᵀ,
        m.jacwf!,
        m.JwF,m.__∇r,m._∇r,m.∇r;
        ll=ll
    )
    nothing
end 
## Maholinobis distance 
# struct
struct mw
    b₀::AbstractVector{<:Real}
    # functions
    _R!_::Rw
    _r!_::rw
    # buffer
    S::AbstractMatrix{<:Real}
    Rᵀ::AbstractMatrix{<:Real}
    r::AbstractVector{<:Real}
    S⁻¹r::AbstractVector{<:Real}
    Rᵀ⁻¹r::AbstractVector{<:Real}
end
# constructor
function mw(prob::AbstractWENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    # functions
    _R!_ = Rw(prob, params, Val(T))
    _r!_ = rw(prob, params, Val(T))
    # buffer
    K,D = prob.K,prob.D
    S = zeros(T, K*D, K*D)
    Rᵀ= zeros(T, K*D, K*D)
    r = zeros(T, K*D)
    S⁻¹r = zeros(T, K*D)
    Rᵀ⁻¹r = zeros(T, K*D)
    mw(
        prob.b₀, 
        _R!_, _r!_, 
        S, Rᵀ, r, S⁻¹r,Rᵀ⁻¹r
    )
end
# method
function (m::mw)(w::AbstractVector{T}; ll::Logging.LogLevel=Logging.Warn,efficient::Bool=false) where T<:Real
    if efficient
        # m(w) = r(w)ᵀS⁻¹r(w) = ((Rᵀ)⁻¹r)ᵀ((Rᵀ)⁻¹r)
        m._R!_(
            m.Rᵀ,w;
            ll=ll
        )
        b = similar(m.b₀)
        ldiv!(b, LowerTriangular(m.Rᵀ), m.b₀)
        m._r!_(
            m.Rᵀ⁻¹r, b, w; 
            ll=ll, Rᵀ=m.Rᵀ
        )
        return _m(m.Rᵀ⁻¹r)
    end 
    m._r!_(m.r,w;ll=ll)
    m._R!_(m.S,w;ll=ll, doChol=false)
    return _m(m.S,m.r,m.S⁻¹r)
end
## ∇m(w) - gradient of Maholinobis distance
struct ∇mw
    # output
    ∇m::AbstractVector{<:Real}
    # data
    U::AbstractMatrix{<:Real} 
    V::AbstractMatrix{<:Real} 
    b₀::AbstractVector{<:Real} 
    sig::AbstractVector{<:Real} 
    # functions
    _R!_::Rw
    _r!_::rw
    _∇r!_::∇rw
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
function ∇mw(prob::AbstractWENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T
    K,M,D,J  = prob.K, prob.M, prob.D, prob.J
    # output
    ∇m = zeros(T,J)
    # methods 
    _R!_  = Rw(prob, params, Val(T))
    _r!_  = rw(prob, params, Val(T))
    _∇r!_ = ∇rw(prob, params, Val(T))
    # preallocate buffers
    S⁻¹r = zeros(T, K*D)
    JwJuF = zeros(T,D,D,J,M)
    __∇L = zeros(T,K,D,D,J,M)
    _∇L = zeros(T,K,D,M,D,J)
    ∇L = zeros(T,K*D,M*D,J)
    ∂ⱼLLᵀ = zeros(T, K*D,K*D)
    ∇S = zeros(T,K*D,K*D,J)
    ∇mw(
        ∇m,
        prob.U, prob.V, prob.b₀, prob.sig,
        _R!_, _r!_, _∇r!_, prob.jacwjacuf!,
        S⁻¹r, JwJuF, __∇L, _∇L, ∇L, ∂ⱼLLᵀ, ∇S
    )
end
# method inplace
function (m::∇mw)(∇m::AbstractVector{<:Real}, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W<:Real
    # Compute L(w) & S(w)
    m._R!_(w; ll=ll, transpose=false, doChol=false) 
    # Compute residal
    m._r!_(m.b₀, w; ll=ll)
    # Compute jacobian of residual
    m._∇r!_(w; ll=ll)

    _∇m!(
        ∇m, w, # ouput, input
        m.U, m.V, m._R!_._L!_.L, m._R!_.Sreg, m._∇r!_.∇r, m.sig, m._r!_.r, # data
        m.jacwjacuf!, # functions
        m.S⁻¹r, m.JwJuF, m.__∇L, m._∇L, m.∇L, m.∂ⱼLLᵀ, m.∇S; # buffers
        ll=ll # kwargs
    )
    nothing
end
# method mutate internal data
function (m::∇mw)(w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W<:Real
    # Compute L(w) & S(w)
    m._R!_(w; ll=ll, transpose=false, doChol=false) 
    # Compute residal
    m._r!_(m.b₀, w; ll=ll)
    # Compute jacobian of residual
    m._∇r!_(w; ll=ll)

    _∇m!(
        m.∇m, w, # ouput, input
        m.U, m.V, m._R!_._L!_.L, m._R!_.Sreg, m._∇r!_.∇r, m.sig, m._r!_.r, # data
        m.jacwjacuf!, # functions
        m.S⁻¹r, m.JwJuF, m.__∇L, m._∇L, m.∇L, m.∂ⱼLLᵀ, m.∇S; # buffers
        ll=ll # kwargs
    )
    nothing
end
##
## Hm(w) - Hessian of Maholinobis Distance
struct Hmw 
    # output 
    H::AbstractMatrix{<:Real}
    # data 
    U::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    b₀::AbstractVector{<:Real}
    sig::AbstractVector{<:Real}
    # functions 
    _R!_::Rw
    _r!_::rw
    _∇r!_::∇rw
    jacwjacuf!::Function
    heswf!::Function
    heswjacuf!::Function
    # buffers 
    S⁻¹r::AbstractVector{<:Real}
    S⁻¹∇r::AbstractMatrix{<:Real}
    JwJuF::AbstractArray{<:Real, 4}
    __∇L::AbstractArray{<:Real, 5}
    _∇L::AbstractArray{<:Real, 5}
    ∇L::AbstractArray{<:Real, 3}
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

function Hmw(prob::AbstractWENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    K,M,D,J = prob.K, prob.M, prob.D, prob.J
    # ouput 
    H = zeros(J,J)
    # functions
    _R!_ = Rw(prob, params)
    _r!_ = rw(prob, params)
    _∇r!_ = ∇rw(prob, params)
    # buffers
    S⁻¹r = zeros(T, K*D)
    S⁻¹∇r = zeros(T, K*D, J)
    JwJuF = zeros(T, D, D, J, M)
    __∇L = zeros(T, K, D, D, J, M)
    _∇L = zeros(T, K, D, M, D, J)
    ∇L = zeros(T, K*D, M*D, J)
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
    Hmw(
        H,
        prob.U, prob.V,prob.b₀,prob.sig,
        _R!_, _r!_, _∇r!_, prob.jacwjacuf!, prob.heswf!, prob.heswjacuf!,  
        S⁻¹r, S⁻¹∇r, JwJuF, __∇L, _∇L, ∇L, ∂ⱼLLᵀ, ∇S, HwF, _∇²r, ∇²r, HwJuF, __∇²L, _∇²L, ∇²L, ∂ⱼL∂ᵢLᵀ, ∂ⱼᵢLLᵀ, ∂ᵢⱼS, S⁻¹∂ⱼS, ∂ᵢSS⁻¹∂ⱼS
    )
end
# method inplace
function (m::Hmw)(H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn)
    # TODO: try letting cholesky factorization back in here
    m._R!_(w; transpose=false, doChol=false) 
    m._r!_(m.b₀, w) 
    m._∇r!_(w)
    _Hm!(
        H, w,
        m.U, m.V, m._R!_._L!_.L, m._R!_.Sreg, m._∇r!_.∇r, m.b₀, m.sig,
        m._r!_.r, 
        m.jacwjacuf!, m.heswf!, m.heswjacuf!,  
        m.S⁻¹r, m.S⁻¹∇r, m.JwJuF, m.__∇L, m._∇L, m.∇L, m.∂ⱼLLᵀ, m.∇S, m.HwF, m._∇²r, m.∇²r, m.HwJuF, m.__∇²L, m._∇²L, m.∇²L, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼLLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
end
# method mutate internal data
function (m::Hmw)(w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn)
    # TODO: try letting cholesky factorization back in here
    m._R!_(w; transpose=false, doChol=false) 
    m._r!_(m.b₀, w) 
    m._∇r!_(w)
    _Hm!(
        m.H, w,
        m.U, m.V, m._R!_._L!_.L, m._R!_.Sreg, m._∇r!_.∇r, m.b₀, m.sig,
        m._r!_.r, 
        m.jacwjacuf!, m.heswf!, m.heswjacuf!,  
        m.S⁻¹r, m.S⁻¹∇r, m.JwJuF, m.__∇L, m._∇L, m.∇L, m.∂ⱼLLᵀ, m.∇S, m.HwF, m._∇²r, m.∇²r, m.HwJuF, m.__∇²L, m._∇²L, m.∇²L, m.∂ⱼL∂ᵢLᵀ, m.∂ᵢⱼLLᵀ, m.∂ᵢⱼS, m.S⁻¹∂ⱼS, m.∂ᵢSS⁻¹∂ⱼS
    )
end