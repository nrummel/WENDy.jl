using JuMP, Ipopt, LinearAlgebra, Tullio, BenchmarkTools
## 
# Set size of problem 
K = 500
M = 1000
J = 10 
D = 3
# Build Data Matrix
U = rand(D, M)
u = reshape(U, D*M)
# Build random matrices with orthogonal rows
V = Matrix(qr(rand(M,K)).Q')[1:K,:]
Vp = Matrix(qr(rand(M,K)).Q')[1:K,:];
## Build rhs 
b0 = reshape(Vp * U', K*D)
# Build random SPD matrix 
A1 = rand(K*D, K*D)
fact = qr(A1)
Q = fact.Q
S = Q *diagm(100*rand(K*D))*Q'
S = (S+S')/2
cholesky!(S)
RT = UpperTriangular(S)'
b = RT\b0
# define random weights for testing 
w_rand = rand(J)
nothing
## Define G function to do as little allocation as possible
function f!(fm, um, w)
    fm[1] = w[1] * um[2] + w[2] * um[1]^3 + w[3] * um[1]^2 + w[4] * um[3] 
    fm[2] = w[5] + w[6] * um[1]^2 + w[7] * um[2] 
    fm[3] = w[8] *um[1] + w[9] + w[10] * um[3]
end

# this is a slightly naive version of the G function 
function naiveG(U::AbstractMatrix, V::AbstractMatrix, f!::Function, w::AbstractVector{W}, ::Val{T}) where {W,T}
    K, M = size(V)
    D, _ = size(U)
    @assert size(U,2) == size(V,2) "We should have U ∈ R^{$D×$M} and V ∈ R^{$K×$M}, but U ∈ R^{$D×$(size(U,2))}"
    F = zeros(T, D, M)
    @inbounds for m in 1:M
        f!(view(F,:,m), view(U,:,m), w)
    end
    G = V * F'
    return reshape(G, K*D)
end
# Define the jacobian of the G function and eventually the 
function jacwf!(jfm, um, w)
    jfm .= 0
    jfm[1] = um[2]
    jfm[4] = (^)(um[1], 3)
    jfm[7] = (^)(um[1], 2)
    jfm[10] = um[3]
    jfm[14] = 1
    jfm[17] = (^)(um[1], 2)
    jfm[20] = um[2]
    jfm[24] = um[1]
    jfm[27] = 1
    jfm[30] = um[3]
    nothing
end

function naiveJacG(U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, w::AbstractVector{W}, ::Val{T}; showFlag::Bool=false) where {W,T}
    showFlag && println(stderr, "Hello")
    K, M = size(V)
    D, _ = size(U)
    J = length(w)
    @assert size(U,2) == size(V,2) "We should have U ∈ R^{$D×$M} and V ∈ R^{$K×$M}, but U ∈ R^{$D×$(size(U,2))}"
    JF = zeros(T,D,J,M)
    @inbounds for m in 1:M
        jacwf!(view(JF,:,:,m), view(U,:,m), w)
    end
    @tullio _JG[d,j,k] := V[k,m] * JF[d,j,m] 
    JG = permutedims(_JG,(3,1,2))
    return reshape(JG, K*D, J)
end
##
@info "Timing G(w)"
@btime naiveG($U,$V,$f!,$w_rand, Val(Float64));
@info "Timing jacG(w)"
@btime naiveJacG($U,$V,$jacwf!,$w_rand,Val(Float64));
nothing
##
res(w) = RT \ naiveG(U,V,f!,w, Val(Float64)) - b
gradRes(w) = RT \ naiveJacG(U,V,jacwf!,w,Val(Float64))
@info "Timing res(w)"
@btime res($w_rand)
@info "Timing gradRes(w)"
@btime gradRes($w_rand)
nothing
##
# check that G is linear... that its jacobian is constant 
##
m = 1
um = U[:,m]
jfm1 = zeros(D,J);
jfm2 = zeros(D,J);
jacwf!(jfm1, um, rand(J))
jacwf!(jfm2, um, rand(J))
norm(jfm1 - jfm2) / norm(jfm1) < eps() ? (@info "Jacobian at m=$m looks constant in w") : @warn ("Something is wrong in the jacobian computation")
##
JF1 = naiveJacG(U, V, jacwf!, rand(J), Val(Float64))
println()
JF2 = naiveJacG(U, V, jacwf!, rand(J), Val(Float64))
norm(JF1 - JF2) / norm(JF1) < eps() ? (@info "Jacobian looks constant in w") : @warn ("Something is wrong in the jacobian computation")
##
# Compute the least square solution 
@info "Computing the least squares solution because we have a linear system"
A = RT \ naiveJacG(U,V,jacwf!,w_rand, Val(Float64)) 
w_star = A \ b
nothing 
##
# try to solve with ipopt for the non-linear case
res_AffExpr(w::AbstractVector) = RT \ naiveG(U,V,f!,w, Val(AffExpr)) - b
gradRes_AffExpr(w::AbstractVector) =  RT \ naiveJacG(U,V,jacwf!,w,Val{AffExpr}; showFlag=true)
@info "Building Model..."
mdl = Model(Ipopt.Optimizer)
@variable(mdl, w[i = 1:J])
@variable(mdl, r[k = 1:K*D])
@operator(mdl, f, J, res_AffExpr, gradRes_AffExpr)
@constraint(mdl, r == f(w))
@objective(mdl, Min, sum(r.^2 ) ) 
set_silent(mdl)
@info "Running Ipopt..."
@time optimize!(mdl)
wi = value.(w)
relErr = norm(wi - w_star) / norm(w_star)
@info """ How did we do?
    relative Error     = $relErr
    termination_status = $(termination_status(mdl))
    primal_status      = $(primal_status(mdl))
    objective_value    = $(objective_value(mdl))
 """