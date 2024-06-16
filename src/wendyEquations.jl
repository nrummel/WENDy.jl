using Tullio, LinearAlgebra, Logging
## Basic Functions
function RTfun(U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, diagReg::Real, jacuf!::Function, w::AbstractVector;ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Res Eval ++++"
        # Preallocate for L 
        D, M = size(U)
        K, _ = size(V)
        @info " Allocated buffers "
        dt = @elapsed a = @allocations begin
            JuF = zeros(D,D,M)
            _L0 = zeros(K,D,D,M)
            _L1  = zeros(K,D,M,D)
        end        
        @info "  $dt s, $a allocations"
        # Get _L1
        K,D,M,_ = size(_L1)
        @info " Compute ∇uF "
        dt = @elapsed a = @allocations @inbounds for m in 1:M
            jacuf!(view(JuF,:,:,m), w, view(U,:,m))
        end
        @info "  $dt s, $a allocations"
        @info " Compute L1 "
        dt = @elapsed a = @allocations @tullio _L0[k,d2,d1,m] = V[k,m] * JuF[d2,d1,m] * sig[d1]
        @info "  $dt s, $a allocations"
        @info " add L0 "
        dt = @elapsed a = @allocations @tullio _L0[k,d,d,m] += Vp[k,m]*sig[d]
        @info "  $dt s, $a allocations"
        @info " permutedims Lw "
        dt = @elapsed a = @allocations permutedims!(_L1,_L0,(1,2,4,3))
        @info "  $dt s, $a allocations"
        @info " Reshape Lw"
        dt = @elapsed a = @allocations L = reshape(_L1,K*D,M*D)
        @info "  $dt s, $a allocations"
        # compute covariance
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
        return UpperTriangular(R)'
    end
end
# 
function res(RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
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
        @info "  Is Real? $(eltype(F)<:Real), $(eltype(F))"
        @info "  $dt s, $a allocations"
        @info " Mat Mat mult "
        dt = @elapsed a = @allocations G = V * F'
        @info "  Is Real? $(eltype(G)<:Real), $(eltype(G))"
        @info "  $dt s, $a allocations"
        @info " Reshape "
        dt = @elapsed a = @allocations g = reshape(G, K*D)
        @info "  Is Real? $(eltype(g)<:Real), $(eltype(g))"
        @info "  $dt s, $a allocations"
        @info " Linear Solve "
        dt = @elapsed a = @allocations res = RT \ g
        @info "  Is Real? $(eltype(res)<:Real), $(eltype(res))"
        @info "  $dt s, $a allocations"
        @info " Vec Vec add "
        dt = @elapsed a = @allocations res .-= b
        @info "  Is Real? $(eltype(res)<:Real), $(eltype(res))"
        @info "  $dt s, $a allocations"
        @info "++++++++++++++++++"
        return res::AbstractVector{T}
    end
end

function ∇res(RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Jac Res Eval ++++"
        K, M = size(V)
        D, _ = size(U)
        J = length(w)
        @info " Evaluating jacobian of F"
        dt = @elapsed a = @allocations begin
            JwF = zeros(W,D,J,M)
            for m in 1:M
                try 
                    jacwf!(view(JwF,:,:,m), w, view(U,:,m))
                catch e
                    println("m = $m")
                    println("jwfm")
                    show(stderr, "text/plain", view(JwF,:,:,m))
                    println("w")
                    show(stderr, "text/plain", w)
                    println("um")
                    show(stderr, "text/plain", view(U,:,m)')
                    throw(e)
                    @assert false
                end
            end
        end
        @info "  Is Real? $(eltype(JwF)<:Real), $(eltype(JwF))"
        @info "  $dt s, $a allocations"
        @info " Computing V ×_3 jacF with tullio"
        dt = @elapsed a = @allocations @tullio _JG[d,j,k] := V[k,m] * JwF[d,j,m] 
        @info "  Is Real? $(eltype(_JG)<:Real), $(eltype(_JG))"
        @info "  $dt s, $a allocations"
        @info " permutedims"
        dt = @elapsed a = @allocations JG = permutedims(_JG,(3,1,2))
        @info "  Is Real? $(eltype(JG)<:Real), $(eltype(JG))"
        @info "  $dt s, $a allocations"
        @info " Reshape"
        dt = @elapsed a = @allocations jacG = reshape(JG, K*D, J)
        @info "  Is Real? $(eltype(jacG)<:Real), $(eltype(jacG))"
        @info "  $dt s, $a allocations"
        @info " Linsolve"
        dt = @elapsed a = @allocations ∇res = RT \ jacG
        @info "  Is Real? $(eltype(∇res)<:Real), $(eltype(∇res))"
        @info "  $dt s, $a allocations"
        return ∇res::AbstractMatrix
    end
end

function _weighted_l2_error(RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    (1/2 * norm(res(RT, U, V, b, f!, w; ll=ll))^2)
end

function _gradient_weighted_l2_error!(gradient::AbstractVector{T}, RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, f!::Function, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    gradient .= ∇res(RT,U,V,jacwf!,w;ll=ll)' * res(RT, U, V, b, f!, w; ll=ll) 
    nothing
end