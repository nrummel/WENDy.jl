using Tullio, LinearAlgebra, Logging, LoopVectorization
## Basic Functions
## L(w)
function _L!(L::AbstractMatrix{<:Real}, U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real} , Vp::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, jacuf!::Function, JuF::AbstractArray{<:Real, 3}, _L0::AbstractArray{<:Real, 4}, _Lbuff0::AbstractArray{<:Real, 4}, _Lbuff1::AbstractArray{<:Real, 4}, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn) 
    # with_logger(ConsoleLogger(stderr, ll)) do 
        # @info "++++ Lw Eval ++++"    
        K,D,M,_ = size(_Lbuff1)
        # @info " Compute ∇uF "
        # dt = @elapsed a = @allocations begin  
            @inbounds for m in 1:M
                jacuf!(view(JuF,:,:,m), w, view(U,:,m))
            end
        # end
        # @info "  $dt s, $a allocations"
        # @info " Compute L1 "
        # dt = @elapsed a = @allocations begin 
            @avx for m in 1:M, d1 in 1:D, d2 in 1:D, k in 1:K
                _Lbuff0[k,d2,d1,m] = JuF[d2,d1,m] * V[k,m] * sig[d1]
            end
        # end
        # @info "  $dt s, $a allocations"
        # @info " add L0 "
        # dt = @elapsed a = @allocations begin 
             _Lbuff0 .+= _L0
        # end
        # @info "  $dt s, $a allocations"
        # @info " permutedims Lw "
        # dt = @elapsed a = @allocations begin 
             permutedims!(_Lbuff1,_Lbuff0,(1,2,4,3))
        # end
        # @info "  $dt s, $a allocations"
        # @info " Reshape Lw"
        # dt = @elapsed a = @allocations begin 
            @views L .= reshape(_Lbuff1,K*D,M*D)
        # end
        # @info "  $dt s, $a allocations"
        nothing
    # end
end
# struct
struct LFun 
    L::AbstractMatrix{<:Real}
    U::AbstractMatrix{<:Real} 
    V::AbstractMatrix{<:Real} 
    Vp::AbstractMatrix{<:Real}
    sig::AbstractVector{<:Real}
    jacuf!::Function
    JuF::AbstractArray{<:Real, 3}
    _L0::AbstractArray{<:Real, 4}
    _Lbuff0::AbstractArray{<:Real, 4}
    _Lbuff1::AbstractArray{<:Real, 4}
end 
# constructor
function LFun(U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, Vp::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, jacuf!::Function, ::Val{T}=Val(Float64)) where T<:Real
    D, M = size(U)
    K, _ = size(V)
    L = zeros(K*D,M*D)
    JuF = zeros(T,D,D,M)
    _L0 = zeros(T,K,D,D,M)
    _Lbuff0 = zeros(T,K,D,D,M)
    _Lbuff1  = zeros(T,K,D,M,D)
    @tullio _L0[k,d,d,m] = Vp[k,m]*sig[d]
    return LFun(L,U,V,Vp,sig, jacuf!, JuF, _L0, _Lbuff0, _Lbuff1)
end
function LFun(prob::AbstractWENDyProblem, params::Union{WENDyParameters,Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real
    LFun(prob.U, prob.V, prob.Vp, prob.sig, prob.jacuf!, Val(T))
end
# method
function (m::LFun)(w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Info) 
    _L!(m.L, m.U, m.V , m.Vp, m.sig, m.jacuf!, m.JuF, m._L0, m._Lbuff0, m._Lbuff1, w; ll) 
    m.L
end
## RT
function _R!(R::AbstractMatrix, thisI::AbstractMatrix, S::AbstractMatrix, L::AbstractMatrix, diagReg::AbstractFloat; ll::Logging.LogLevel=Logging.Warn)
    # with_logger(ConsoleLogger(stderr, ll)) do 
        # @info " Compute Sw "
        # dt = @elapsed a = @allocations begin
             mul!(S, L, L')
        # end
        # @info "  $dt s, $a allocations"
        # regularize for possible ill conditioning
        # @info " Initialize R with I"
        # dt = @elapsed a = @allocations begin
             @views R .= thisI
        # end
        # @info "  $dt s, $a allocations"
        # @info " Regularize Sw "
        # dt = @elapsed a = @allocations begin
             mul!(R, S, I, (1-diagReg), diagReg)
        # end
        # @info "  $dt s, $a allocations"
        # compute cholesky for lin solv efficiency
        # @info " Compute cholesky of Sw "
        # dt = @elapsed a = @allocations begin
             cholesky!(Symmetric(R))
        # end
        # @info "  $dt s, $a allocations"
        # @info " UpperTriangular wrapper"
        # dt = @elapsed a = @allocations begin
             @views R .= UpperTriangular(R)
        # end
        # @info "  $dt s, $a allocations"
        nothing
    # end
end
# struct/method
struct RTFun
    S::AbstractMatrix{<:Real}
    R::AbstractMatrix{<:Real}
    thisI::AbstractMatrix{<:Real}
    _L::LFun 
    diagReg::Real
end
# constructors
function RTFun(U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, Vp::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, jacuf!::Function,diagReg::Real, ::Val{T}=Val(Float64)) where T<:Real
    D, _ = size(U)
    K, _ = size(V)
    S = zeros(K*D, K*D)
    R = zeros(K*D, K*D)
    thisI = Matrix{T}(I, K*D, K*D)
    _L = LFun(U,V,Vp,sig, jacuf!)
    RTFun(S, R, thisI, _L, diagReg)
end
function RTFun(prob::AbstractWENDyProblem, params::WENDyParameters, ::Val{T}=Val(Float64)) where T<:Real
    RTFun(prob.U, prob.V, prob.Vp, prob.sig, prob.jacuf!, params.diagReg, Val(T))
end
# method
function (m::RTFun)(w::AbstractVector{W};ll::Logging.LogLevel=Logging.Warn) where W<:Real
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Res Eval ++++"
        L = m._L(w;ll=ll) 
        _R!(m.R, m.thisI, m.S, L, m.diagReg; ll=ll)
        return m.R'
    end
end
# function without struct declaration
function RT(U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, Vp::AbstractMatrix{<:Real}, sig::AbstractVector{<:Real}, jacuf!::Function,diagReg::Real, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn)
    @warn "This allocates storage, please utilize the RTFun struct/method"
    _RT = RTFun(U, V, Vp, sig, jacuf!,diagReg)
    _RT(w,ll=ll)
end
## Residual RT(G(w) - b)
function _res!(r::AbstractVector, RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, F::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}, g::AbstractVector, b::AbstractVector, f!::Function, w::AbstractVector; ll::Logging.LogLevel=Logging.Warn) 
    # with_logger(ConsoleLogger(stderr, ll)) do 
    #     @info "++++ Res Eval ++++"
        K, M = size(V)
        D, _ = size(U)
        # @info " Evaluate F "
        # dt = @elapsed a = @allocations begin 
            for m in 1:M
                f!(view(F,:,m), w, view(U,:,m))
            end
        # end
        # @info "  Is Real? $(eltype(F)<:Real), $(eltype(F))"
        # @info "  $dt s, $a allocations"
        # @info " Mat Mat mult "
        # dt = @elapsed a = @allocations begin
            mul!(G, V, F')
        # end
        # @info "  $dt s, $a allocations"
        # @info " Reshape "
        # dt = @elapsed a = @allocations begin 
            @views g .= reshape(G, K*D,1)
        # end
        # @info "  Is Real? $(eltype(g)<:Real), $(eltype(g))"
        # @info "  $dt s, $a allocations"
        # @info " Linear Solve "
        # dt = @elapsed a = @allocations begin 
            ldiv!(r, LowerTriangular(RT), g)
        # end
        # @info "  Is Real? $(eltype(res)<:Real), $(eltype(res))"
        # @info "  $dt s, $a allocations"
        # @info " Vec Vec add "
        # dt = @elapsed a = @allocations begin 
            @views r .-= b
        # end
        # @info "  Is Real? $(eltype(res)<:Real), $(eltype(res))"
        # @info "  $dt s, $a allocations"
        # @info "++++++++++++++++++"
        nothing
    # end
end
# struct
struct ResFun 
    r::AbstractVector{<:Real}
    U::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real} 
    F::AbstractMatrix{<:Real} 
    G::AbstractMatrix{<:Real} 
    g::AbstractVector{<:Real} 
    f!::Function
end
# constructors 
function ResFun(U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, f!::Function, ::Val{T}=Val(Float64)) where T<:Real
    D, M = size(U)
    K, _ = size(V)
    r = zeros(K*D)
    F = zeros(T, D, M)
    G = zeros(T, K, D)
    g = zeros(T, K*D)
    ResFun(r,U,V,F,G,g,f!)
end
function ResFun(prob::AbstractWENDyProblem, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    ResFun(prob.U, prob.V, prob.f!, Val(T))
end
# alloacting (kinda)
function (m::ResFun)(RT::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, w::AbstractVector{T}; ll::Logging.LogLevel=Logging.Warn, allocate::Bool=false) where T<:Real 
    KD, _ = size(RT)
    if allocate 
        r = zeros(KD)
        _res!(r, RT, m.U, m.V, b, m.f!, w; ll=ll)
        return r
    end
    _res!(m.r, RT, m.U, m.V, m.F, m.G, m.g, b, m.f!, w; ll=ll)
    return m.r
end
# inplace
function (m::ResFun)(r::AbstractVector{<:Real}, RT::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, w::AbstractVector{T}; ll::Logging.LogLevel=Logging.Warn) where T<:Real 
    KD, _ = size(RT)
    _res!(r, RT, m.U, m.V, m.F, m.G, m.g, b, m.f!, w; ll=ll)
    nothing
end
# 
function res(RT::AbstractMatrix{<:Real}, U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, f!::Function, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn)
    @warn "This allocates storage, please utilize the ResFun struct/method"
    _r = ResFun(U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, f!::Function)
    return _r(RT, b, w; ll=ll)
end
## Jacobian of the residual 
function _∇res!(∇res::AbstractMatrix{<:Real}, RT::AbstractMatrix{<:Real}, U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, JwF::AbstractArray{<:Real, 3}, _JG::AbstractArray{<:Real, 3}, JG::AbstractArray{<:Real, 3}, jacG::AbstractMatrix{<:Real}, jacwf!::Function, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn) 
    # with_logger(ConsoleLogger(stderr, ll)) do 
        # @info "++++ Jac Res Eval ++++"
        K, M = size(V)
        D, _ = size(U)
        J = length(w)
        # @info " Evaluating jacobian of F"
        # dt = @elapsed a = @allocations begin
            @inbounds for m in 1:M
                jacwf!(view(JwF,:,:,m), w, view(U,:,m))
            end
        # end
        # @info "  $dt s, $a allocations"
        # @info " Computing ∇G = V ∘ ∇F"
        # dt = @elapsed a = @allocations begin 
            # @tullio _JG[d,j,k] = V[k,m] * JwF[d,j,m] 
            @inbounds for d = 1:D 
                mul!(view(_JG,d,:,:), view(JwF,d,:,:), V') 
            end
        # end
        # @info "  $dt s, $a allocations"
        # @info " Permutedims for ∇G"
        # dt = @elapsed a = @allocations begin 
            permutedims!(JG, _JG,(3,1,2))
        # end
        # @info "  $dt s, $a allocations"
        # @info " Reshape ∇G"
        # dt = @elapsed a = @allocations begin 
            jacG = reshape(JG, K*D, J)
        # end
        # @info "  $dt s, $a allocations"
        # @info " RT \\ ∇G"
        # dt = @elapsed a = @allocations begin 
            ldiv!(∇res, LowerTriangular(RT), jacG)
        # end
        # @info "  $dt s, $a allocations"
        nothing
    # end
end
# struct
struct ∇resFun
    jac::AbstractMatrix{<:Real}
    U::AbstractMatrix{<:Real}
    V::AbstractMatrix{<:Real}
    JwF::AbstractArray{<:Real,3}
    _JG::AbstractArray{<:Real, 3}
    JG::AbstractArray{<:Real, 3}
    jacG::AbstractMatrix{<:Real} 
    jacwf!::Function
end
# constructors
function ∇resFun(U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, J::Int, ::Val{T}=Val(Float64)) where T<:Real
    D, M = size(U)
    K, _ = size(V)
    jac = zeros(K*D,J)
    JwF = zeros(T,D,J,M)
    _JG = zeros(T,D,J,K)
    JG = zeros(T,K,D,J)
    jacG = zeros(T,K*D, J)
    ∇resFun(jac, U, V, JwF, _JG, JG, jacG, jacwf!)
end
function ∇resFun(prob::AbstractWENDyProblem, params::Union{WENDyParameters, Nothing}=nothing, ::Val{T}=Val(Float64)) where T<:Real 
    ∇resFun(prob.U, prob.V, prob.jacwf!, prob.J, Val(T))
end
# allocating
function (m::∇resFun)(RT::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn, allocate::Bool=false)
    if allocate 
        J = length(w)
        KD, _ = size(RT)
        jac = zeros(KD,J)
        _∇res!(jac, RT, m.U, m.V, m.JwF, m._JG, m.JG, m.jacG, m.jacwf!, w; ll=ll)
        return jac
    end
    _∇res!(m.jac, RT, m.U, m.V, m.JwF, m._JG, m.JG, m.jacG, m.jacwf!, w; ll=ll)
    return m.jac
end 
# in place
function (m::∇resFun)(jac::AbstractMatrix{<:Real}, RT::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; ll::Logging.LogLevel=Logging.Warn)scale
    _∇res!(jac, RT, m.U, m.V, m.JwF, m._JG, m.JG, m.jacG, m.jacwf!, w; ll=ll)
    nothing
end 
# 
function ∇res(RT::AbstractMatrix{<:Real}, U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, jacwf!::Function, w::AbstractVector{<:Real}, ::Val{T}=Val(Float64); ll::Logging.LogLevel=Logging.Warn) where T<:Real
    _∇res = ∇resFun(U, V, jacwf!, length(w), Val(T))
    _∇res(RT, w; ll=ll)
end
# ## Maholinobis distance 
# struct MFun
#     b0::AbstractVector{<:Real}
#     _RT::RTFun
#     _res::ResFun
#     function MFun( U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, b0::AbstractVector, sig::AbstractVector, diagReg::Real, f!::Function, jacuf!::Function)
#         _RT = RTFun(U, V, Vp, sig, jacuf!,diagReg)
#         _res = ResFun(U, V, f!)
#         new(b0, _RT, _res)
#     end
# end

# function (m::MFun)(w::AbstractVector{W};ll::Logging.LogLevel=Logging.Warn) where W<:Real
#     with_logger(ConsoleLogger(stderr, ll)) do 
#         @info "++++ Maholinobis Distance Eval ++++"
#         @info "Get RT"
#         dt = @elapsed a = @allocations RT = m._RT(w)
#         @info "  $dt s, $a allocations"
#         @info "Get b"
#         dt = @elapsed a = @allocations b = RT \ b0
#         @info "  $dt s, $a allocations"
#         @info "Get weighted residual"
#         dt = @elapsed a = @allocations r = m._res(RT,b,w)
#         @info "  $dt s, $a allocations"
#         return 1/2*norm(r)^2
#     end
# end

# struct ∇mFun!
#     U::AbstractMatrix{<:Real}
#     V::AbstractMatrix{<:Real}
#     Vp::AbstractMatrix{<:Real}
#     b0::AbstractVector{<:Real}
#     sig::AbstractVector{<:Real}
#     diagReg::Real
#     f!::Function
#     _L::LFun
#     jacwf!::Function
#     jacwjacuf!::Function
#     JwJuF::AbstractArray{<:Real,4} 
#     _Lbuf0::AbstractArray{<:Real,5} 
#     _Lbuf1::AbstractArray{<:Real,5} 
#     ∇Sw_buff::AbstractArray{<:Real,3} 
#     ∇Sw::AbstractArray{<:Real,3} 
#     function ∇mFun!(U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, b0::AbstractVector, sig::AbstractVector, diagReg::Real, f!::Function, jacuf!::Function, jacwf!::Function, jacwjacuf!::Function, J::Int, ::Val{T}=Val(Float64)) where T
#         D,M  = size(U)
#         K, _ = size(V)
#         JwJuF = zeros(T,D,D,J,M)
#         _Lbuf0 = zeros(T,K,D,D,J,M)
#         _Lbuf1 = zeros(T,K,D,M,D,J)
#         ∇Sw_buff = zeros(T,K*D,K*D,J)
#         ∇Sw = zeros(T,K*D,K*D,J)
#         _L = LFun(U,V,Vp,sig, jacuf!)
#         new(U,V,Vp,b0,sig,diagReg,f!,_L,jacwf!,jacwjacuf!,JwJuF,_Lbuf0,_Lbuf1,∇Sw_buff,∇Sw)
#     end
# end

# function (m::∇mFun!)(∇m::AbstractVector{<:Real}, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W<:Real
#     _∇m!(∇m,m.U,m.V,m.Vp,m.b0,m.sig,m.diagReg,
#         m.f!,m._L,m.jacwf!,m.jacwjacuf!,
#         m.JwJuF,m._Lbuf0,m._Lbuf1,m.∇Sw_buff,m.∇Sw,
#         w; ll=ll
#     )
#     nothing
# end
# function _∇m!(U, V, Vp, b0, sig, diagReg, f!, _L, jacwf!, jacwjacuf!, JwJuF, _Lbuf0, _Lbuf1, ∇Sw_buff, ∇Sw, w;ll=Logging.Warn)
#     with_logger(ConsoleLogger(stderr, ll)) do 
#         @info "++++ ∇w[Maholinobis Dist] ++++"
#         D, M = size(U)
#         K, _ = size(V)
#         @info "  $dt s, $a allocations"
#         @info " Compute Lw"
#         dt = @elapsed a = @allocations L = _L(w) 
#         @info "  $dt s, $a allocations"
#         @info " Get Cholesky of Sw"
#         dt = @elapsed a = @allocations R = _R(L, diagReg)
#         @info "  $dt s, $a allocations"
#         @info " Compute ∇w[∇uF] "
#         dt = @elapsed a = @allocations @inbounds for m in 1:M
#             jacwjacuf!(view(JwJuF,:,:,:,m), w, view(U,:,m))
#         end
#         @info "  $dt s, $a allocations"
#         @info " Compute Lbuff0 "
#         dt = @elapsed a = @allocations begin
#             @tullio _Lbuf0[k,d2,d1,j,m] = V[k,m] * JwJuF[d2,d1,j,m] * sig[d1]
#         end
#         @info "  $dt s, $a allocations"
#         @info " Permutedims Lw "
#         dt = @elapsed a = @allocations permutedims!(_Lbuf1, _Lbuf0, (1,2,5,3,4))
#         @info "  $dt s, $a allocations"
#         @info " Reshape Lw"
#         dt = @elapsed a = @allocations ∇L = reshape(_Lbuf1,K*D,M*D,J)
#         @info "  $dt s, $a allocations"
#         @info " Compute ∇Sw(prt1)"
#         @show typeof(∇Sw_buff)
#         @show typeof(∇L)
#         @show typeof(L)
#         dt = @elapsed a = @allocations begin 
#             @inbounds for j = 1:J 
#                 mul!(view(∇Sw_buff,:,:,j), view(∇L,:,:,j), L')
#             end
#         end 
#         @info "  $dt s, $a allocations"
#         @info " Compute ∇Sw(prt2)"
#         dt = @elapsed a = @allocations begin 
#             @tullio ∇Sw[kd1,kd2,j] = ∇Sw_buff[kd1,kd2,j] + ∇Sw_buff[kd2,kd1,j]
#         end
#         @info "  $dt s, $a allocations"
#         @info " Compute Residual"
#         dt = @elapsed a = @allocations begin 
#             b = R' \ b0
#             r = res(R', U, V, b, f!, w)
#         end
#         @info "  $dt s, $a allocations"
#         @info " Applying R^{-1}"
#         dt = @elapsed a = @allocations rr = R \ r
#         @info "  $dt s, $a allocations"
#         @info " Compute first part of ∇m"
#         dt = @elapsed a = @allocations begin
#             @inbounds for j in 1:J 
#                 ∇m[j] = -1/2*dot(rr, ∇Sw[:,:,j], rr)
#             end 
#         end
#         @info "  $dt s, $a allocations"
#         @info " Compute second part of ∇m"
#         dt = @elapsed a = @allocations ∇m .+= ∇res(R',U,V,jacwf!,w)' * r
#         @info "  $dt s, $a allocations"
#         nothing
#     end
# end