using Tullio,LinearAlgebra
##
function F!(FF::AbstractMatrix, w::AbstractVector,  u::AbstractMatrix, _F!::Function) 
    for m in 1:size(FF,2)
        _F!(view(FF,:,m), w, view(u,:,m))
    end
    nothing
end

function G!(GG::AbstractMatrix, FF::AbstractMatrix, V::AbstractMatrix, w::AbstractVector,  u::AbstractMatrix, _F!::Function)
    F!(FF, w, u, _F!)
    mul!(GG, V, FF')
    nothing
end

function B!(BB::AbstractMatrix, Vp::AbstractMatrix,u::AbstractMatrix) 
    mul!(BB, Vp, u',-1,0)
    nothing
end

function _J!(JJ::AbstractArray{<:Any,3}, w::AbstractVector, u::AbstractMatrix, _jacF!::Function) 
    @inbounds for m in 1:size(JJ,3)
        um = view(u,:,m)
        jm = view(JJ,:,:,m)
        _jacF!(jm, w, um)
    end
    nothing
end

function _L!(L_mat::AbstractMatrix, L::AbstractArray{<:Any,4}, LL::AbstractArray{<:Any,4}, JJ::AbstractArray{<:Any,3},w::AbstractVector, u::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, _jacuF!::Function) 
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

struct LNonlinear<:LMatrix 
    JJ::AbstractArray{<:Any,3}
    LL::AbstractArray{<:Any,4}
    L::AbstractArray{<:Any,4}
    L_mat::AbstractMatrix
    V::AbstractMatrix
    Vp::AbstractMatrix
    u::AbstractMatrix
    sig::AbstractVector
    _jacuF!::Function 
end

function LNonlinear(u::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, _jacuF!::Function) 
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

struct IRWLS_Iter_Linear<:IRWLS_Iter 
    Lgetter::LNonlinear
    S::AbstractMatrix
    R0::AbstractMatrix
    R::AbstractMatrix
    G0::AbstractMatrix
    G::AbstractMatrix
    b0::AbstractVector
    b::AbstractVector
    diag_reg::Real
end

function IRWLS_Iter_Linear(u::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, _jacuF!::Function, G0::AbstractMatrix, b0::AbstractVector)
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

function (s::IRWLS_Iter_Linear)(wim1::AbstractVector) 
    L = s.Lgetter(wim1)
    mul!(s.S, L, L')
    s.R .= s.R0
    mul!(s.R, s.S, s.R0, 1-s.diag_reg, s.diag_reg)
    cholesky!(Symmetric(s.R))
    ldiv!(s.G, UpperTriangular(s.R)', s.G0)
    ldiv!(s.b, UpperTriangular(s.R)', s.b0)
    return  s.G \ s.b
    # ldiv!(wi, s.G, s.b) ## this gives an error for some reason
    # nothing
end

function _G0!(G0::AbstractMatrix, u::AbstractMatrix, V::AbstractMatrix, _F!::Function)
    _, J = size(G0)
    K, M = size(V)
    D, _ = size(u)
    FF = zeros(D,M)
    GG = zeros(K,D)
    ej = zeros(J)
    for j in 1:J 
        ej[j] = 1
        G!(GG, FF, V, ej, u, _F!)
        G0[:,j] .= reshape(GG, K*D)
        ej[j] = 0
    end
end

function IRWLS_Linear(u::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, _F!::Function, _jacuF!::Function, J::Int; maxIt::Int=100,tol::AbstractFloat=1e-10)
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
