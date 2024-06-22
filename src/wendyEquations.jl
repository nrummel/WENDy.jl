using Tullio, LinearAlgebra, Logging
## Basic Functions

function Lw(U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, jacuf!::Function, w::AbstractVector{W};ll::Logging.LogLevel=Logging.Warn) where W<:Real
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Lw Eval ++++"
        # Preallocate for L 
        D, M = size(U)
        K, _ = size(V)
        @info " Allocated buffers "
        dt = @elapsed a = @allocations begin
            JuF = zeros(W,D,D,M)
            _Lbuff0 = zeros(W,K,D,D,M)
            _Lbuff1  = zeros(W,K,D,M,D)
        end        
        @info "  $dt s, $a allocations"
        # Get _Lbuff1
        K,D,M,_ = size(_Lbuff1)
        @info " Compute ∇uF "
        dt = @elapsed a = @allocations @inbounds for m in 1:M
            jacuf!(view(JuF,:,:,m), w, view(U,:,m))
        end
        @info "  $dt s, $a allocations"
        @info " Compute L1 "
        dt = @elapsed a = @allocations @tullio _Lbuff0[k,d2,d1,m] = V[k,m] * JuF[d2,d1,m] * sig[d1]
        @info "  $dt s, $a allocations"
        @info " add L0 "
        dt = @elapsed a = @allocations @tullio _Lbuff0[k,d,d,m] += Vp[k,m]*sig[d]
        @info "  $dt s, $a allocations"
        @info " permutedims Lw "
        dt = @elapsed a = @allocations permutedims!(_Lbuff1,_Lbuff0,(1,2,4,3))
        @info "  $dt s, $a allocations"
        @info " Reshape Lw"
        dt = @elapsed a = @allocations Lw = reshape(_Lbuff1,K*D,M*D)
        @info "  $dt s, $a allocations"
        return Lw
    end
end
function _Rfun(L::AbstractMatrix, diagReg::AbstractFloat; ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info " Compute Sw "
        dt = @elapsed a = @allocations S = L * L'
        @info "  $dt s, $a allocations"
        # regularize for possible ill conditioning
        @info " Regularize Sw "
        dt = @elapsed a = @allocations R = (1-diagReg)*S + diagReg*I
        @info "  $dt s, $a allocations"
        # compute cholesky for lin solv efficiency
        @info " Compute cholesky of Sw "
        dt = @elapsed a = @allocations cholesky!(Symmetric(R))
        @info "  $dt s, $a allocations"
        return UpperTriangular(R)
    end
end
function RTfun(U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, diagReg::Real, jacuf!::Function, w::AbstractVector{W};ll::Logging.LogLevel=Logging.Warn) where W<:Real
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Res Eval ++++"
        # 
        L = Lw(U, V, Vp, sig, jacuf!, w;ll=ll) 
        # compute covariance
        R = _Rfun(L, diagReg;ll=ll)
        return R'
    end
end
##
function _res!(res::AbstractVector, RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, b::AbstractVector, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W<:Real
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Res Eval ++++"
        K, M = size(V)
        D, _ = size(U)
        @info " Evaluate F "
        dt = @elapsed a = @allocations begin 
            F = zeros(W, D, M)
            for m in 1:size(F,2)
                f!(view(F,:,m), w, view(U,:,m))
            end
        end
        # @info "  Is Real? $(eltype(F)<:Real), $(eltype(F))"
        @info "  $dt s, $a allocations"
        @info " Mat Mat mult "
        dt = @elapsed a = @allocations G = V * F'
        # @info "  Is Real? $(eltype(G)<:Real), $(eltype(G))"
        @info "  $dt s, $a allocations"
        @info " Reshape "
        dt = @elapsed a = @allocations g = reshape(G, K*D)
        # @info "  Is Real? $(eltype(g)<:Real), $(eltype(g))"
        @info "  $dt s, $a allocations"
        @info " Linear Solve "
        dt = @elapsed a = @allocations ldiv!(res, RT, g)
        # @info "  Is Real? $(eltype(res)<:Real), $(eltype(res))"
        @info "  $dt s, $a allocations"
        @info " Vec Vec add "
        dt = @elapsed a = @allocations res .-= b
        # @info "  Is Real? $(eltype(res)<:Real), $(eltype(res))"
        @info "  $dt s, $a allocations"
        @info "++++++++++++++++++"
        nothing
    end
end
function res(RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    KD, _ = size(RT)
    r = zeros(KD)
    _res!(r, RT, U, V, b, f!, w; ll=ll)
    return r
end

function _∇res!(∇res::AbstractMatrix, RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W<:Real
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Jac Res Eval ++++"
        dtTotal = @elapsed aTotal = @allocations  begin
            K, M = size(V)
            D, _ = size(U)
            J = length(w)
            @info " Evaluating jacobian of F"
            dt = @elapsed a = @allocations begin
                JwF = zeros(W,D,J,M)
                for m in 1:M
                    jacwf!(view(JwF,:,:,m), w, view(U,:,m))
                end
            end
            @info "  $dt s, $a allocations"
            @info " Computing ∇G = V ∘ ∇F"
            dt = @elapsed a = @allocations @tullio _JG[d,j,k] := V[k,m] * JwF[d,j,m] 
            @info "  $dt s, $a allocations"
            @info " Permutedims for ∇G"
            dt = @elapsed a = @allocations JG = permutedims(_JG,(3,1,2))
            @info "  $dt s, $a allocations"
            @info " Reshape ∇G"
            dt = @elapsed a = @allocations jacG = reshape(JG, K*D, J)
            @info "  $dt s, $a allocations"
            @info " RT \\ ∇G"
            dt = @elapsed a = @allocations ∇res .= RT \ jacG
            @info "  $dt s, $a allocations"
        end
        @info " Totals "
        @info "  $dtTotal s, $aTotal allocations "
        @info "+++++++++++++"
        nothing
    end
end
function ∇res(RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, w::AbstractVector; ll::Logging.LogLevel=Logging.Warn)
    J = length(w)
    KD, _ = size(RT)
    jac = zeros(KD,J)
    _∇res!(jac, RT, U, V, jacwf!, w; ll=ll)
    return jac
end 
function _weighted_l2_error(RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    (1/2 * norm(res(RT, U, V, b, f!, w; ll=ll))^2)
end

function _gradient_weighted_l2_error!(gradient::AbstractVector{T}, RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, f!::Function, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    gradient .= ∇res(RT,U,V,jacwf!,w;ll=ll)' * res(RT, U, V, b, f!, w; ll=ll) 
    nothing
end
## Maholinobis distance 
function _m(U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, b0::AbstractVector, sig::AbstractVector, diagReg::Real, f!::Function, jacuf!::Function, w::AbstractVector{W};ll::Logging.LogLevel=Logging.Warn) where W<:Real
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Maholinobis Distance Eval ++++"
        @info "Get RT"
        dt = @elapsed a = @allocations RT = RTfun(U,V,Vp,sig,diagReg,jacuf!,w)
        @info "  $dt s, $a allocations"
        @info "Get b"
        dt = @elapsed a = @allocations b = RT \ b0
        @info "  $dt s, $a allocations"
        @info "Get weighted residual"
        dt = @elapsed a = @allocations r = res(RT, U, V, b, f!, w)
        @info "  $dt s, $a allocations"
        return 1/2*norm(r)^2
    end
end

struct ∇mFun!
    U::AbstractMatrix
    V::AbstractMatrix
    Vp::AbstractMatrix
    b0::AbstractVector
    sig::AbstractVector
    diagReg::Real
    f!::Function
    jacuf!::Function
    jacwf!::Function
    jacwjacuf!::Function
    JwJuF::AbstractArray{<:Any,4} 
    _Lbuf0::AbstractArray{<:Any,5} 
    _Lbuf1::AbstractArray{<:Any,5} 
    ∇Sw_buff::AbstractArray{<:Any,3} 
    ∇Sw::AbstractArray{<:Any,3} 
    function ∇mFun!(U::AbstractMatrix{W}, V::AbstractMatrix, Vp::AbstractMatrix, b0::AbstractVector, sig::AbstractVector, diagReg::Real, f!::Function, jacuf!::Function, jacwf!::Function, jacwjacuf!::Function, J::Int) where W
        D,M  = size(U)
        K, _ = size(V)
        JwJuF = zeros(W,D,D,J,M)
        _Lbuf0 = zeros(W,K,D,D,J,M)
        _Lbuf1 = zeros(W,K,D,M,D,J)
        ∇Sw_buff = zeros(W,K*D,K*D,J)
        ∇Sw = zeros(W,K*D,K*D,J)
        new(U,V,Vp,b0,sig,diagReg,f!,jacuf!,jacwf!,jacwjacuf!,JwJuF,_Lbuf0,_Lbuf1,∇Sw_buff,∇Sw)
    end
end

function (m::∇mFun!)(∇m::AbstractVector, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W<:AbstractFloat
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ ∇w[Maholinobis Dist] ++++"
        @info " Extract storage from struct" 
       
        dt = @elapsed a = @allocations begin
            U = m.U
            V = m.V
            Vp = m.Vp
            b0 = m.b0
            sig = m.sig
            diagReg = m.diagReg
            f! = m.f!
            jacuf! = m.jacuf!
            jacwf! = m.jacwf!
            jacwjacuf! = m.jacwjacuf!
            JwJuF = m.JwJuF
            _Lbuf0 = m._Lbuf0
            _Lbuf1 = m._Lbuf1
            ∇Sw_buff = m.∇Sw_buff
            ∇Sw = m.∇Sw
            D, M = size(U)
            K, _ = size(V)
        end
        @info "  $dt s, $a allocations"
        @info " Compute Lw"
        dt = @elapsed a = @allocations L = Lw(U, V, Vp, sig, jacuf!, w) 
        @info "  $dt s, $a allocations"
        @info " Get Cholesky of Sw"
        dt = @elapsed a = @allocations R = _Rfun(L, diagReg)
        @info "  $dt s, $a allocations"
        @info " Compute ∇w[∇uF] "
        dt = @elapsed a = @allocations @inbounds for m in 1:M
            jacwjacuf!(view(JwJuF,:,:,:,m), w, view(U,:,m))
        end
        @info "  $dt s, $a allocations"
        @info " Compute Lbuff0 "
        dt = @elapsed a = @allocations begin
            @tullio _Lbuf0[k,d2,d1,j,m] = V[k,m] * JwJuF[d2,d1,j,m] * sig[d1]
        end
        @info "  $dt s, $a allocations"
        @info " Permutedims Lw "
        dt = @elapsed a = @allocations permutedims!(_Lbuf1, _Lbuf0, (1,2,5,3,4))
        @info "  $dt s, $a allocations"
        @info " Reshape Lw"
        dt = @elapsed a = @allocations ∇L = reshape(_Lbuf1,K*D,M*D,J)
        @info "  $dt s, $a allocations"
        @info " Compute ∇Sw(prt1)"
        dt = @elapsed a = @allocations begin 
            @inbounds for j = 1:J 
                mul!(view(∇Sw_buff,:,:,j), view(∇L,:,:,j), L')
                # @view ∇Sw_buff[:,:,j] = ∇L[,:,:,j]*L'
            end
        end 
        @info "  $dt s, $a allocations"
        @info " Compute ∇Sw(prt2)"
        dt = @elapsed a = @allocations begin 
            @tullio ∇Sw[kd1,kd2,j] = ∇Sw_buff[kd1,kd2,j] + ∇Sw_buff[kd2,kd1,j]
        end
        @info "  $dt s, $a allocations"
        @info " Compute Residual"
        dt = @elapsed a = @allocations begin 
            b = R' \ b0
            r = res(R', U, V, b, f!, w)
        end
        @info "  $dt s, $a allocations"
        @info " Applying R^{-1}"
        dt = @elapsed a = @allocations rr = R \ r
        @info "  $dt s, $a allocations"
        @info " Compute first part of ∇m"
        dt = @elapsed a = @allocations begin
            @inbounds for j in 1:J 
                ∇m[j] = -1/2*dot(rr, ∇Sw[:,:,j], rr)
            end 
        end
        @info "  $dt s, $a allocations"
        @info " Compute second part of ∇m"
        dt = @elapsed a = @allocations ∇m .+= ∇res(R',U,V,jacwf!,w)' * r
        @info "  $dt s, $a allocations"
        nothing
    end
end