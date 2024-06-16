@info "Loading dependencies"
# std lib
using LinearAlgebra, Logging, Random
# external dependencies
using JuMP, Ipopt,  Tullio, BenchmarkTools
## 
# Set size of problem 
@info "Generate random data..."
Random.seed!(1)
K = 100
M = 1000
J = 10 
D = 3
diag_reg = 1e-10
# Build Random data datrix
U = rand(D, M)
u = reshape(U, D*M)
# Give random noise 
sig = rand(D)
# Build random matrices with orthogonal rows
V = Matrix(qr(rand(M,K)).Q')[1:K,:]
Vp = Matrix(qr(rand(M,K)).Q')[1:K,:];
# Build lhs of ODE
b0 = reshape(Vp * U', K*D)
nothing
## Define necessary functions 
@info "Building functions..."
# Define functions from the ODE RHS
function f!(fm, w, um)
    fm[1] = w[1] * um[2] + w[2] * um[1]^3 + w[3] * um[1]^2 + w[4] * um[3] 
    fm[2] = w[5] + w[6] * um[1]^2 + w[7] * um[2] 
    fm[3] = w[8] *um[1] + w[9] + w[10] * um[3]
end
# jacobian of the RHS with respect to u
function jacuf!(jfm, w, um)
    jfm[1] = 2*w[3]*um[1] + 3*w[2]*um[1]^2
    jfm[2] = 2*w[6]*um[1]
    jfm[3] = w[8]
    jfm[4] = w[1]
    jfm[5] = w[7]
    jfm[6] = 0
    jfm[7] = w[4]
    jfm[8] = 0
    jfm[9] = w[10]
    nothing 
end 
# jacobian of the right hand side with respect to w
function jacwf!(jfm, w, um)
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
# Get the cholesky decomposition of our current approximation of the covariance
function RTfun(U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, diag_reg::Real, jacuf!::Function, w::AbstractVector)
    # Preallocate for L 
    D, M = size(U)
    K, _ = size(V)
    JuF = zeros(D,D,M)
    _L0 = zeros(K,D,D,M)
    _L1  = zeros(K,D,M,D)
    # Get _L1
    K,D,M,_ = size(_L1)
    @inbounds for m in 1:M
        jacuf!(view(JuF,:,:,m), w, view(U,:,m))
    end
    @tullio _L0[k,d2,d1,m] = V[k,m] * JuF[d2,d1,m] * sig[d1]
    @tullio _L0[k,d,d,m] += Vp[k,m]*sig[d]
    permutedims!(_L1,_L0,(1,2,4,3))
    L = reshape(_L1,K*D,M*D)
    # compute covariance
    S = L * L'
    # regularize for possible ill conditioning
    R = (1-diag_reg)*S + diag_reg*I
    # compute cholesky for lin solv efficiency
    cholesky!(Symmetric(R))
    return UpperTriangular(R)'
end
# Define residual function 
function res(RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, b::AbstractVector, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W
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
        @info "  $dt s, $a allocations"
        @info " Mat Mat mult "
        dt = @elapsed a = @allocations G = V * F'
        @info "  $dt s, $a allocations"
        @info " Reshape "
        dt = @elapsed a = @allocations g = reshape(G, K*D)
        @info "  $dt s, $a allocations"
        @info " Linear Solve "
        dt = @elapsed a = @allocations res = RT \ g
        @info "  $dt s, $a allocations"
        @info " Vec Vec add "
        dt = @elapsed a = @allocations res .-= b
        @info "  $dt s, $a allocations"
        @info "++++++++++++++++++"
        return res
    end
end
# Define Jacobian of residual with respect to the parameters w 
function ∇res(RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Jac Res Eval ++++"
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
        @info " Computing V ×_3 jacF with tullio"
        dt = @elapsed a = @allocations @tullio _JG[d,j,k] := V[k,m] * JwF[d,j,m] 
        @info "  $dt s, $a allocations"
        @info " permutedims"
        dt = @elapsed a = @allocations JG = permutedims(_JG,(3,1,2))
        @info "  $dt s, $a allocations"
        @info " Reshape"
        dt = @elapsed a = @allocations jacG = reshape(JG, K*D, J)
        @info "  $dt s, $a allocations"
        @info " Linsolve"
        dt = @elapsed a = @allocations ∇res = RT \ jacG
        @info "  $dt s, $a allocations"
        return ∇res
    end
end
function memorize_f(f::Function, KD::Int, ::Val{J}, ::Val{W}) where W
    _w, _f = NTuple{J,W}(zeros(W,J)), zeros(W,J)
    function f_kd(kd::Int, wargs::W...)
        if wargs !== _w
                _w, _f = wargs, f(wargs...)
        end
        return  _f[kd]::W 
    end
    return [(wargs...) -> f_kd(kd, wargs...) for kd in 1:KD]
end
 
function memorize_∇f!(∇f::Function, KD::Int, ::Val{J}, ::Val{W}) where W
    _w, _∇f = NTuple{J,W}(zeros(W,J)), zeros(W,KD,J)
    function ∇f_kd!(ret::AbstractVector{W}, kd::Int, wargs::W...)
        if wargs !== _w
            _w = wargs
            _∇f = ∇f(wargs...)
        end
        ret .= view(_∇f, kd, :)  
        nothing 
    end
    return [(∇f_kd, wargs...) -> ∇f_kd!(∇f_kd, kd, wargs...) for kd in 1:KD]
end
#
res(w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W = res(RT,U,V,b,f!,w;ll=ll)
∇res(w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W = ∇res(RT,U,V,jacwf!,w;ll=ll)
res(wargs::W...; ll::Logging.LogLevel=Logging.Warn) where W = res(collect(wargs);ll=ll)
∇res(wargs::W...; ll::Logging.LogLevel=Logging.Warn) where W = ∇res(collect(wargs);ll=ll)
res_kd = memorize_f(res, K*D, Val(J), Val(eltype(U)))
∇res_kd = memorize_∇f!(∇res, K*D, Val(J), Val(eltype(U)))
res_mem(wargs::W...) where W = reduce(vcat, res_kd[kd](wargs...) for kd in 1:K*D)
function ∇res_mem!(∇res::AbstractMatrix{W}, wargs::W...) where W
    for kd in 1:K*D
        ∇res_kd[kd](view(∇res,kd,:), wargs...) 
    end
    nothing
end 
## 
@info "Test Allocations with with Float64"
w_rand = rand(J)
RT = RTfun(U,V,Vp,sig,diag_reg,jacuf!,w_rand)
b = RT \ b0 
# because this problem is linear in w the jacobian is constant, 
# and we could solve this problem with backslash because 
# it is really a linear least squares problem
w_star = ∇res(zeros(J)) \ b 
##
res_w_rand = res(w_rand)
res_w_rand_splat = res(w_rand...)
res_mem_w_rand = res_mem(w_rand...)
@assert all(res_w_rand .== res_w_rand_splat)
@assert all(res_w_rand .== res_mem_w_rand)
∇res_w_rand = ∇res(w_rand)
∇res_w_rand_splat = ∇res(w_rand...) 
∇res_mem_w_rand = zeros(K*D,J)
∇res_mem!(∇res_mem_w_rand,w_rand...)
@assert all(∇res_w_rand .== ∇res_w_rand_splat)
@assert all(∇res_w_rand .== ∇res_mem_w_rand)
nothing 
##
@info " Benchmark res"
@time res(rand(J))
@info " Benchmark res splatting"
@time res(rand(J)...)
@info " Benchmark res memorized"
@time res_mem(rand(J)...)
@info " Benchmark ∇res "
@time ∇res(rand(J))
@info " Benchmark ∇res splatting"
@time ∇res(rand(J)...) 
@info " Benchmark ∇res memorized"
∇res_mem_w_rand = zeros(K*D,J)
@time ∇res_mem!(∇res_mem_w_rand,rand(J)...);
nothing
##
@info "Test the NLS in isolation"
@info "Defining model for Ipopt"
mdl = Model(Ipopt.Optimizer)
@variable(mdl, w[j = 1:J], start = w_rand[j])
@variable(mdl, r[k = 1:K*D])
for kd in 1:K*D
    fk = Symbol("f$kd")
    register(mdl, fk, J, res_kd[kd], ∇res_kd[kd];autodiff=false)
    add_nonlinear_constraint(mdl, :($(fk)($(w)...) == $(r[kd])) )
end
@objective(mdl, Min, sum(r .^2 ) ) 
set_silent(mdl)
@info "Running optimizer"
@time optimize!(mdl)
w_hat = value.(w)
abserr = norm(w_hat - w_star)
relerr = abserr / norm(w_star)
@info "Absolute coeff error = $abserr"
@info "Relative coeff error = $relerr"
obj_star = sum(res(w_star...).^2)
abserr = abs(objective_value(mdl) - obj_star)
relerr = abserr / obj_star
@info "Absolute coeff error = $abserr"
@info "Relative coeff error = $relerr"
# ## Compute the non lin least square solution 
# @info "Defining IRWLS_Nonlinear..."
# function IRWLS_Nonlinear(U, V, Vp, b0, sig, diag_reg, J, f!, jacuf!, jacwf!; ll=Logging.Info,maxIt=100, relTol=1e-4)
#     with_logger(ConsoleLogger(stderr,ll)) do 
#         @info "Initializing the linearization least squares solution  ..."
#         D, M = size(U)
#         K, _ = size(V)
#         G0 = _∇res(Matrix{Float64}(I, K*D,K*D), U, V, jacwf!,, zeros(J))
#         w0 = G0 \ b0 
#         wit = zeros(J,maxIt)
#         resit = zeros(J,maxIt)
#         wnm1 = w0 
#         wn = similar(w0)
#         for n = 1:maxIt 
#             RT = RTfun(U,V,Vp,sig,diag_reg,jacuf!,wnm1)
#             b = RT \ b0 
#             ll = (n == 1) ? Logging.Info : Logging.Warn 
#             res_AffExpr(w::AbstractVector) = _res(RT,U,V,b,f!,Val(AffExpr), w; ll=ll)
#             ∇res_AffExpr(w::AbstractVector) = _∇res(RT,U,V,jacwf!,Val(AffExpr),w;showFlag=true)
#             w_star = _∇res(RT, U, V, jacwf!,, zeros(J)) \ b 
#             ##
#             @info "Defining model for Ipopt"
#             mdl = Model(Ipopt.Optimizer)
#             @variable(mdl, w[j = 1:J], start = wnm1[j])
#             @variable(mdl, r[k = 1:K*D])
#             @show typeof(w)
#             @operator(mdl, f, J, res_AffExpr, ∇res_AffExpr)
#             @constraint(mdl, r == f(w))
#             @objective(mdl, Min, sum(r.^2 ) ) 
#             set_silent(mdl)
#             @info "Running optimizer"
#             dtn = @elapsed optimize!(mdl)
#             wn = value.(w)
#             resn = wn - w_star
#             relErr = norm(resn) / norm(w_star)
#             resit[:,n] = resn
#             wit[:,n] = wn
#             @info """ Iteration $n
#                 relative Error     = $dtn
#                 relative Error     = $relErr
#                 barrier_iterations = $(barrier_iterations(mdl))
#                 termination_status = $(termination_status(mdl))
#                 primal_status      = $(primal_status(mdl))
#                 objective_value    = $(objective_value(mdl))
#             """
#             if norm(wnm1 - wn) / norm(wnm1) < relTol
#                 resit = resit[:,1:n] 
#                 wit = wit[:,1:n] 
#                 @info "Convergence Criterion met!"
#                 return wn, wit, resit 
#             end
#             wnm1 = wn
#         end
#         @warn "Maxiteration met..."
#         return wn, wit, resit 
#     end
# end 
# ##
# @info "IRWLS (Nonlinear): "
# @info "   Runtime info: "
# @time what, wit, resit = IRWLS_Nonlinear(U, V, Vp, b0, sig, diag_reg, J, f!, jacuf!, jacwf!; ll=Logging.Warn)
# @info "   iterations    = $(size(wit,2)-1)"