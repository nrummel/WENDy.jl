using Tullio,LinearAlgebra
##
function F!(FF::AbstractMatrix{T}, w::AbstractVector{T}, u::AbstractMatrix{T}, _F!::Function) where T<:Real
    for m in 1:size(FF,2)
        _F!(view(FF,:,m), w,  view(u,:,m))
    end
    nothing
end

function G!(GG::AbstractMatrix{T}, FF::AbstractMatrix{T}, V::AbstractMatrix{T}, w::AbstractVector{T}, u::AbstractMatrix{T}, _F!::Function)where T<:Real
    F!(FF, w, u, _F!)
    mul!(GG, V, FF')
    nothing
end

function B!(BB::AbstractMatrix{T}, Vp::AbstractMatrix{T},u::AbstractMatrix{T})where T<:Real
    mul!(BB, Vp, u',-1,0)
    nothing
end

function _J!(JJ::AbstractArray{T,3},w::AbstractVector{T},u::AbstractMatrix{T}, _jacuF!::Function) where T<:Real
    @inbounds for m in 1:size(JJ,3)
        um = view(u,:,m)
        jm = view(JJ,:,:,m)
        _jacuF!(jm, w,um)
    end
    nothing
end

function _L!(L_mat::AbstractMatrix{T}, L::AbstractArray{T,4}, LL::AbstractArray{T,4}, JJ::AbstractArray{T,3},w::AbstractVector{T},u::AbstractMatrix{T}, V::AbstractMatrix{T}, Vp::AbstractMatrix{T}, sig::AbstractVector{T}, _jacuF!::Function) where T<:Real
    K,D,M,_ = size(L)
    _J!(JJ, w, u, _jacuF!)
    @tullio LL[k,d2,d1,m] = V[k,m] * JJ[d2,d1,m] * sig[d1]
    @tullio LL[k,d,d,m] += Vp[k,m]*sig[d]
    # LL_tmp = reshape
    permutedims!(L,LL,(1,2,4,3))
    L_mat .= reshape(L,K*D,M*D)
    nothing
end


##
abstract type LMatrix end

struct LNonlinear{T}<:LMatrix where T<:Real
    JJ::AbstractArray{T,3}
    LL::AbstractArray{T,4}
    L::AbstractArray{T,4}
    L_mat::AbstractMatrix{T}
    V::AbstractMatrix{T}
    Vp::AbstractMatrix{T}
    u::AbstractMatrix{T}
    sig::AbstractVector{T}
    _jacuF!::Function 
end

function LNonlinear(u::AbstractMatrix{T}, V::AbstractMatrix{T}, Vp::AbstractMatrix{T}, sig::AbstractVector{T}, _jacuF!::Function) where T<:Real
    D, M = size(u)
    K, _ = size(V)
    JJ = zeros(D,D,M)
    LL = zeros(K,D,D,M)
    L  = zeros(K,D,M,D)
    L_mat  = zeros(K*D,M*D)
    LNonlinear(JJ, LL, L, L_mat, V, Vp, u, sig, _jacuF!)
end

function (s::LNonlinear)(w::AbstractVector{<:Real})
    _L!(s.L_mat, s.L, s.LL, s.JJ, w, s.u, s.V, s.Vp, s.sig, s._jacuF!)
    return s.L_mat 
end

abstract type IRWLS_Iter end 

struct IRWLS_Iter_Linear{T}<:IRWLS_Iter where T<:Real
    Lgetter::LNonlinear
    S::AbstractMatrix{T}
    R0::AbstractMatrix{T}
    R::AbstractMatrix{T}
    G0::AbstractMatrix{T}
    G::AbstractMatrix{T}
    b0::AbstractVector{T}
    b::AbstractVector{T}
    diag_reg::T
end

function IRWLS_Iter_Linear(u::AbstractMatrix{T}, V::AbstractMatrix{T}, Vp::AbstractMatrix{T}, sig::AbstractVector{T}, _jacuF!::Function, G0::AbstractMatrix{T}, b0::AbstractVector{T}) where T<:Real
    D, M = size(u)
    K, _ = size(V)
    S = zeros(K*D,K*D)
    R0 = Matrix{Float64}(I,K*D,K*D)
    R = similar(R0)
    G = similar(G0)
    b = similar(b0)
    Lgetter = LNonlinear(u,V,Vp,sig,_jacuF!)
    IRWLS_Iter_Linear(Lgetter, S, R0, R, G0, G, b0, b, diag_reg)
end

function (s::IRWLS_Iter_Linear)(wim1::AbstractVector{T}) where T<:Real
    L = s.Lgetter(wim1)
    mul!(s.S, L, L')
    s.R .= s.R0
    mul!(s.R, s.S, s.R0, 1-diag_reg,diag_reg)
    cholesky!(Symmetric(s.R))
    ldiv!(s.G, UpperTriangular(s.R)', s.G0)
    ldiv!(s.b, UpperTriangular(s.R)', s.b0)
    return  s.G \ s.b
    # ldiv!(wi, s.G, s.b) ## this gives an error for some reason
    # nothing
end

function _G0!(G0::AbstractMatrix{T}, u::AbstractMatrix{T}, V::AbstractMatrix{T}, _F!::Function) where T<:Real
    _, J = size(G0)
    K, M = size(V)
    D, _ = size(u)
    FF = zeros(D,M)
    GG = zeros(K,D)
    ej = zeros(J)
    for j in 1:J 
        ej[j] = 1
        G!(GG, FF, V, ej, u, _F!)
        G0[:,j] = reshape(GG, K*D)
        ej[j] = 0
    end
end

function IRWLS_Linear(u::AbstractMatrix{T}, V::AbstractMatrix{T}, Vp::AbstractMatrix{T}, sig::AbstractVector{T}, _F!::Function, _jacuF!::Function, J::Int; maxIt::Int=100,tol::AbstractFloat=1e-10) where T<:Real
    ## Get dimensions
    D, M = size(u)
    K,_ = size(V)
    ## Preallocate
    wit = zeros(J,maxIt)
    B = zeros(K,D)
    G0 = zeros(K*D, J)
    B!(B, Vp, u)
    b0 = reshape(B, K*D)
    _G0!(G0, u, V, _F!)
    iter = IRWLS_Iter_Linear(u, V, Vp, sig, _jacuF!, G0, b0) 
    ## start algorithm
    wit[:,1] = G0 \ b0
    for i in 2:maxIt 
        wit[:,i] = iter(wit[:,i-1])
        if norm(wit[:,i] - wit[:,i-1]) / norm(wit[:,i-1]) < tol 
            wit = wit[:,1:i]
            break 
        end
        if i == maxIt 
            @warn "Did not converge"
        end
    end
    return wit[:,end], wit
end