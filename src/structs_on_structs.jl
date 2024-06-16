

function F!(F::AbstractMatrix, U::AbstractMatrix, f!::Function, w::AbstractVector) 
    @assert size(U,2) == size(F,2)

    for m in 1:size(U,2)
        f!(view(F,:,m), view(U,:,m), w)
    end
    nothing
end

function G!(G::AbstractMatrix, F::AbstractMatrix, V::AbstractMatrix, U::AbstractMatrix, f!::Function, w::AbstractVector)
    F!(F, U, f!, w)
    mul!(G, V, F')
    nothing
end

struct GFun 
    # Internal buffers that are preallocated
    G::AbstractMatrix
    F::AbstractMatrix 
    # Data specific to this problem
    U::AbstractMatrix 
    V::AbstractMatrix 
    f!::Function
    # Output we want
    g::AbstractVector
end
# constructor to preallocate
function GFun(U::AbstractMatrix, V::AbstractMatrix, f!::Function, ::Val{T}) where T
    K, M = size(V)
    D, _ = size(U)
    F = zeros(T,D,M)
    G = zeros(T,K,D)
    # reshape into vector for use in cost function
    g = reshape(G,K*D) 
    GFun(G,F,U,V,f!,g)
end 
# define a method to use internal memory 
function (s::GFun)(w::AbstractVector)
    G!(s.G, s.F, s.V, s.U, s.f!, w)
    # s.g should be view of s.G that is vectorized
    return s.g
end
## Jacobian struct method
function jacwF!(JF::AbstractArray{<:Any,3}, U::AbstractMatrix, jacf!::Function, w::AbstractVector,) 
    @inbounds for m in 1:size(U,2)
        jacf!(view(JF,:,:,m), view(U,:,m),w)
    end
    nothing
end

function jacwG!(JG::AbstractArray{<:Any,3}, _JG::AbstractArray{<:Any,3}, JF::AbstractArray{<:Any,3}, U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, w::AbstractVector) 
    jacwF!(JF, U, jacwf!, w)
    @tullio _JG[d,j,k] = V[k,m] * JF[d,j,m] 
    permutedims!(JG, _JG, (3,1,2))
    nothing
end

struct JacwGFun 
    # Internal buffers that are preallocated
    JG::AbstractArray{<:Any,3}
    _JG::AbstractArray{<:Any,3}
    JF::AbstractArray{<:Any,3}
    # Data specific to this problem 
    U::AbstractMatrix
    V::AbstractMatrix
    jacwf!::Function
    # Output we want 
    jacGmat::AbstractMatrix
end
# constructor to preallocate
function JacwGFun(U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, J::Int, ::Val{T}) where T
    K, M = size(V)
    D, _ = size(U)
    JG = zeros(T,K,D,J)
    _JG = zeros(T,D,J,K)
    JF = zeros(T,D,J,M)
    # reshape into matrix for use in grad based methods
    jacGmat = reshape(JG,K*D,J) 
    JacwGFun(JG, _JG, JF, U,V, jacwf!, jacGmat )
end
# define a method to use internal memory 
function (s::JacwGFun)(w::AbstractVector)
    jacwG!(s.JG, s._JG, s.JF, s.U, s.V, s.jacwf!, w) 
    return s.jacGmat
end 

function jacG!(jacG::AbstractArray{<:Any,3}, jacGbuf::AbstractArray{<:Any,3}, JJ::AbstractArray{<:Any,3},  w::AbstractVector, u::AbstractMatrix, _jacwF!::Function) 
    _J!(JJ, w, u, _jacwF!)
    @tullio jacGbuf[d,j,k] = V[k,m] * JJ[d,j,m] 
    permutedims!(jacG, jacGbuf, (3,1,2))
    nothing
end
##
struct GFun<:Any
    u::AbstractMatrix
    V::AbstractMatrix
    GG::AbstractMatrix
    FF::AbstractMatrix
    g::AbstractVector
    _F!::Function 
    K::Int
    D::Int
end
##
function GFun(u::AbstractMatrix{T}, V::AbstractMatrix{T}, _F!::Function) where T
    D, M = size(u)
    K, _ = size(V)
    # FF = Matrix{Union{T, AbstractJuMPScalar}}(undef,D,M)
    # GG = Matrix{Union{T, AbstractJuMPScalar}}(undef,K,D)
    FF = zeros(AffExpr,D,M)
    GG = zeros(AffExpr,K,D)
    g = reshape(GG,K*D)
    GFun(u,V,GG,FF,g,_F!,K,D)
end
##
function (s::GFun)(w::AbstractVector) 
    G!(s.GG, s.FF, s.V, w, s.u, s._F!)
    return s.g 
end

struct JacGgetter<:Any
    u::AbstractMatrix
    jacG::AbstractArray{<:Any,3}
    jacGbuf::AbstractArray{<:Any,3}
    JJ::AbstractArray{<:Any,3}
    jacGmat::AbstractMatrix
    _jacwF!::Function
end

function JacGgetter(u::AbstractMatrix, V::AbstractMatrix, _jacwF!::Function)
    D, M = size(u)
    K, _ = size(V)
    jacG = zeros(K,D,J)
    jacGbuf = zeros(D,J,K)
    JJ = zeros(D,J,M);
    jacGmat = reshape(jacG,K*D,J) 
    JacGgetter(u, jacG, jacGbuf, JJ, jacGmat, _jacwF!)
end

function (s::JacGgetter)(w::AbstractVector) 
    jacG!(s.jacG, s.jacGbuf, s.JJ, w, s.u, s._jacwF!)
    return s.jacGmat
end