using Ipopt, JuMP
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

struct IRWLS_Iter_Nonlinear<:IRWLS_Iter
    Lgetter::LNonlinear
    G::GFun
    jac::JacGgetter
    S::AbstractMatrix
    R0::AbstractMatrix
    R::AbstractMatrix
    b0::Vector
    b::Vector
    diag_reg::Real
end

function IRWLS_Iter_Nonlinear(u::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, _F!::Function, _jacuF!::Function, _jacwF!::Function, diag_reg::Real)
    D, M = size(u)
    K, _ = size(V)
    S  = zeros(K*D,K*D)
    R0 = Matrix(I,K*D,K*D)
    R = zeros(K*D,K*D)
    B = zeros(K,D)
    B!(B, Vp, u)
    b0 = reshape(B, K*D)
    b = zeros(K*D)
    Lgetter = LNonlinear(u,V,Vp,sig,_jacuF!)
    G = GFun(u,V,_F!)
    jac = JacGgetter(u,V,_jacwF!)
    IRWLS_Iter_Nonlinear(Lgetter, G, jac, S, R0, R, b0, b, diag_reg)
end

function (s::IRWLS_Iter_Nonlinear)(wim1::AbstractVector; maxIter::Int= 100, tol::AbstractFloat=1e-4)
    J = length(wim1)
    L = s.Lgetter(wim1)
    mul!(s.S, L, L')
    s.R .= s.R0
    mul!(s.R, s.S, s.R0, 1-s.diag_reg, s.diag_reg)
    cholesky!(Symmetric(s.R))
    ldiv!(s.b, UpperTriangular(s.R)', s.b0)

    # g = similar(s.b)
    # function F(w)
    #     ldiv!(g, UpperTriangular(s.R)', s.G(w)) 
    #     norm(g - s.b)^2
    # end
    # tmp1 = similar(s.b) 
    # tmp2 = similar(s.b) 
    # tmp3 = similar(s.b)
    # function gradF!(g, w) 
    #     tmp1 .= s.G(w) - s.b
    #     ldiv!(tmp2, UpperTriangular(s.R)', tmp1)
    #     ldiv!(tmp3, UpperTriangular(s.R), tmp2)
    #     mul!(g, s.jac(w)', tmp3)
    #     nothing
    # end
        
    # wi, wit, ve, fe = gradientDescent(wim1,F, gradF!;
    # jac=s.jac, R=UpperTriangular(s.R), G=s.G, b=s.b)
    # @show cond(s.jac(wim1))
    # wstar = (UpperTriangular(s.R)' \ s.jac(wim1)) \ s.b
    # @show norm(wi  - wstar) / norm(wstar)
    # p = plot(1:length(ve), ve, yaxis=:log,label="value err")
    # plot!(1:length(fe), fe, yaxis=:log, label="func err")
    # display(p)
    # @assert false 
    B = s.jac(wim1)
    wstar = (UpperTriangular(s.R)' \ B) \ s.b ## if everything was linear this would be it 
    # res(ww) = UpperTriangular(s.R)' \ (B*ww) - s.b
    # res(ww) = UpperTriangular(s.R)' \ (s.jac(ww)*ww) - s.b
    KD = length(s.b)
    buf = zeros(AffExpr, KD)
    r = zeros(AffExpr, J)
    function res(ww) 
        @time buf = s.G(ww)
        @time r = UpperTriangular(s.R)' \ buf
        @time r - s.b
    end
    # jac(ww) = UpperTriangular(s.R)' \ s.jac(ww)
    jac(ww) = UpperTriangular(s.R)' \ s.jac(ww)
    wrand = zeros(J)
    @info "here"
    res(wrand)
    @time jac(wrand)
    @info "there"
    mdl = Model(Ipopt.Optimizer)
    J = length(wim1)
    KD, _ = size(L)
    @variable(mdl, w[i = 1:J], start = wim1[i])
    # @variable(mdl, r[k = 1:KD])
    # @operator(mdl, f, J, res, jac)
    # @constraint(mdl, r == f(w))
    # @objective(mdl, Min, sum(r.^2) ) 
    @objective(mdl, Min, sum((UpperTriangular(s.R)' \ B*w-s.b).^2) ) 
    # @objective(mdl, Min, sum(f(w).^2) ) 
    set_silent(mdl)
    optimize!(mdl)
    wi = value.(w)

    relErr = norm(wi - wstar) / norm(wstar)
    # @info """ How did we do?
    #     relative Error     = $relErr
    #     termination_status = $(termination_status(mdl))
    #     primal_status      = $(primal_status(mdl))
    #     objective_value    = $(objective_value(mdl))
    # """
    # @assert false
    # @show typeof(wi)
    # @show res(wi)
    # @show typeof(res(wi))
    return wi, norm((objective_value(mdl)))
end

function IRWLS_Nonlinear(u::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, _F!::Function, _jacuF!::Function, _jacwF!::Function, J::Int; maxIt::Int=100,tol::AbstractFloat=1e-10, diag_reg::Real=1e-10)
    ## Get dimensions
    D, M = size(u)
    K,_ = size(V)
    ## Preallocate
    wit = zeros(J,maxIt) 
    iter = IRWLS_Iter_Nonlinear(u, V, Vp, sig, _F!, _jacuF!, _jacwF!, diag_reg) 
    ## start algorithm
    wit[:,1] = iter.jac(zeros(J)) \ iter.b0
    @info """Initializing 
      w1 = $(wit[:,1]')"""
    for i in 2:maxIt 
        @info "iter $i"
        @time wit[:,i], costi = iter(wit[:,i-1])
        @info "  w$i = $(round.(100 .*wit[:,i]';digits=2)/100)"
        @info "  cost = $(costi)"
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