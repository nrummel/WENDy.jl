## 
using MKL
using Revise
using DifferentialEquations, ModelingToolkit, BSON, MAT
using Random, Statistics, LinearAlgebra, Logging, StaticArrays # stdlib
using ModelingToolkit: t_nounits as t, D_nounits
using ModelingToolkit: @mtkmodel, @mtkbuild, ODESystem
using Symbolics: jacobian
using ImageFiltering: imfilter, Inner # equivalent to conv
using FFTW: fft, ifft
using BenchmarkTools: @btime
using Test: @test
using Tullio, BlockDiagonals
using Plots 
gr()
includet("exampleProblems.jl")
includet("testFunctions.jl")
includet("computeGradients.jl")
includet("generateNoise.jl")
includet("linearSystem.jl")
##
# mdl                 = LOGISTIC_GROWTH_MODEL
# exampleFile         = joinpath(@__DIR__, "../data/LogisticGrowth.bson")
mdl                 = HINDMARSH_ROSE_MODEL
exampleFile         = joinpath(@__DIR__, "../data/HindmarshRose.bson")
ϕ                   = ExponentialTestFun()
reg                 = 1e-10
noise_ratio         = 0.01
time_subsample_rate = 2
mt_params           = 2 .^(0:3)
seed                = Int(1)
K_min               = 10
K_max               = Int(5e3)
pruneMeth           = SingularValuePruningMethod( 
    MtminRadMethod(),
    UniformDiscritizationMethod()
);
##
function plotData(t, u, tobs, uobs, mdl)
    D,Mp1 = size(uobs)
    plot(tobs, uobs[1,:], label="u_1+ϵ",line=:scatter)
    plot!(t, u[1,:], label="u_1")
    for d = 2:D 
        plot!(tobs, uobs[d,:], label="u_$d + ϵ", line=:scatter)
        plot!(t, u[d,:], label="u_$d")
    end 
    title!("Data")
end 
Random.seed!(Int(seed))
data = BSON.load(exampleFile) 
tt = data[:t] 
u = data[:u] 
num_rad = length(mt_params)
tobs = tt[1:time_subsample_rate:end]
uobs = u[:,1:time_subsample_rate:end]
uobs, noise, noise_ratio_obs, sigma = generateNoise(uobs, noise_ratio)
##
mdl = HINDMARSH_ROSE_MODEL
data = matread(joinpath(@__DIR__, "../data/Lw_hindmarsh_test.mat"))
tobs = Vector(data["tobs"][:])
uobs = Matrix(data["xobs"]')
G_matlab = data["G_0"]
b_matlab = data["b_0"][:]
V = data["V"]
Vp = data["Vp"];
L0_matlab = data["L0"];
L1_matlab = data["L1"];
##
sig = estimate_std(uobs)
##
# V,Vp,Vfull = pruneMeth(tobs,uobs,ϕ,K_min,K_max,mt_params);

D, Mp1 = size(uobs)
J = length(parameters(mdl))
K = size(V, 1)
##
_F, _F!         = getRHS(mdl)
_jacuF, _jacuF! = getJacobian(mdl)
_jacwF, _jacwF! = getParameterJacobian(mdl)
##
r = zeros(K*D)
G = zeros(K, D)
B = zeros(K,D)
FF = zeros(Mp1,D)
B!(B, Vp, uobs)
b = reshape(B, K*D)
S = zeros(K*D,K*D)
L = zeros(K,D,Mp1,D)
LL = zeros(K,Mp1,D,D) 
JJ = zeros(Mp1,D,D)
L0 = zeros(K,Mp1,D,D)
L0!(L0,Vp,sig);
##
u1u1sqr = hcat(uobs[1,:],uobs[1,:].^2)'
custom_uobs = zeros(D^2*J, Mp1);
for m in 1:Mp1
    matView = zeros(D^2, J)
    matView[1,2] = u1u1sqr[2,m]
    matView[1,3] = u1u1sqr[1,m]
    matView[2,6] = u1u1sqr[1,m]
    matView[3,8] = 1
    matView[4,1] = 1
    matView[5,7] = 1
    matView[7,4] = 1
    matView[9,10] = 1
    custom_uobs[:,m] .= matView[:]
end
function _custom_jacF!(out, w, u)
    # out[1] = (+)((*)((*)(2, w[3]), u[1]), (*)((*)(3, w[2]), u[2]))
    # out[2] = (*)((*)(2, w[6]), u[1])
    # out[3] = w[8]
    # out[4] = w[1]
    # out[5] = w[7]
    # out[6] = 0
    # out[7] = w[4]
    # out[8] = 0
    # out[9] = w[10]
    mul!(out, reshape(u, D^2, J),w)
    # mul!(reshape(out,, reshape(u, D^2, J), w)
end
function getResidual(w)
    residual!(r,G,FF,B,w,V,Vp,_F!,uobs)
    return reshape(L, K*D, Mp1*D);
end
function getL(w,L0=L0, f=_jacuF!, u=uobs)
    L!(L, LL, JJ, w, sig, L0, f, u) 
    return reshape(L, K*D, Mp1*D);
end
function getS!(w)
    L = getL(w)
    mul!(S, L,L')
    nothing 
end
w_rand = wit[:,2]
@info "time to get residual"
@time getResidual(w_rand);
@info "time for get L"
@time L_mat = getL(w_rand);
@info "time for get S"
@time getS!(w_rand);
@info "Error for S"
S_true = L_mat*L_mat'
@show norm(S - S_true) / norm(S_true)
##
L1 = zeros(K*D, Mp1*D,J)
for j in 1:J 
    ej = zeros(J)
    ej[j] = 1
    L1[:,:,j] = getL(ej, zeros(K*D, Mp1*D))
end
##
data = matread(joinpath(@__DIR__, "../data/firstIter.mat"))
RT_matlab  = data["RT"]
Lw_matlab  = data["Lw"]
Sw_matlab  = data["Cov"]
G_matlab  = data["G"]
b_matlab  = data["b"]
G_0_matlab  = data["G_0"]
b_0_matlab  = data["b_0"][:]
wjm1_matlab  = data["wjm1"][:]
wj_matlab  = data["wj"][:]
##
Gmat = zeros(K*D,J)
for j in 1:J
    G0 = zeros(K,D)
    ej = zeros(J)
    ej[j] = 1
    G!(G0,FF,ej,V,_F!,uobs)
    Gmat[:,j] = reshape(G0,K*D) 
end
norm(Gmat - G_matlab) / norm(G_matlab)
@assert norm(Gmat - G_0_matlab) / norm(G_0_matlab) < 1e2*eps()
##
w0 = Gmat \ b
@assert norm(w0 - wjm1_matlab) / norm(wjm1_matlab) < 1e2*eps()
##
Lw = getL(w0) 
@assert norm(Lw - Lw_matlab) / norm(Lw) < 1e2*eps()
##
getS!(w0)
Sw = S 
@assert norm(Sw - Sw_matlab) / norm(Sw) < 1e2*eps()
##
R = cholesky((1-reg)*Sw + I*reg).U
@assert norm(R' - RT_matlab) / norm(RT_matlab) < 1e2*eps()
##
Giter = R' \ Gmat 
biter = R' \ b 
@assert norm(Giter - G_matlab) / norm(G_matlab) < 1e2*eps()
@assert norm(biter - b_matlab) / norm(b_matlab) < 1e2*eps()
##
wj = Giter \ biter
@assert norm(wj - wj_matlab) / norm(wj_matlab) < 1e2*eps()
##
L0 = zeros(K,Mp1,D,D)
L0!(L0,Vp,sig);
L0_mat = reshape(permutedims(L0,(1,3,2,4)), K*D, Mp1*D)
function IRWLS(L0;maxIt =100,tol = 1e-10)
    L = zeros(K,D,Mp1,D)
    LL = zeros(K,Mp1,D,D) 
    JJ = zeros(Mp1,D,D)
    Sw = zeros(K*D,K*D)
    R0 = Matrix{Float64}(I,K*D,K*D)
    R = similar(R0)
    wit = zeros(J,maxIt)
    wit[:,1] =  Gmat \ b
    for i in 2:maxIt 
        L!(L, LL, JJ, wit[:,i-1], sig, L0, _custom_jacF!, custom_uobs) 
        @assert false
        Lw = reshape(L, K*D, Mp1*D)
        mul!(Sw, Lw,Lw')
        R .= R0
        mul!(R, Sw, R0, 1-reg,reg)
        cholesky!(R)
        Giter = UpperTriangular(R)' \ Gmat
        biter = UpperTriangular(R)' \ b
        wit[:,i] = Giter \ (biter) 
        if norm(wit[:,i] - wit[:,i-1]) / norm(wit[:,i-1]) < tol 
            wit = wit[:,1:i]
            break 
        end
        if i == maxIt 
            @warn "did not converge"
        end
    end
    return wit[:,end], wit
end

function mypagetime!(Lw, L1, w)
    mul!(reshape(Lw,K*Mp1*D^2), reshape(L1,K*Mp1*D^2,J),w,1,1)
    # @tullio Lw[k,m] = L1[k,m,j] * w[j]
    nothing
end

function IRWLS(L0, L1; maxIt =100,tol = 1e-10)
    Lw = zeros(K*D, Mp1*D)
    Sw = zeros(K*D,K*D)
    wit = zeros(J,maxIt)
    R0 = Matrix{Float64}(I,K*D,K*D)
    R = similar(R0)
    wit[:,1] =  Gmat \ b
    for i in 2:maxIt 
        w = wit[:,i-1]
        

        # @time begin
            Lw .= L0
            mypagetime!(Lw, L1,w)
        # end
        # @time begin 
            mul!(Sw, Lw,Lw')
        # end
        # @time begin
            R .= R0
            mul!(R, Sw, R0, 1-reg,reg)
            cholesky!(Symmetric(R);check=false)
        # end
        # @time begin
            Giter = UpperTriangular(R)' \ Gmat
            biter = UpperTriangular(R)' \ b
        # end
        # @time begin
            wit[:,i] = Giter \ biter 
        # end
        # println()
        if norm(wit[:,i] - wit[:,i-1]) / norm(wit[:,i-1]) < tol 
            wit = wit[:,1:i]
            break 
        end
        if i == maxIt 
            @warn "did not converge"
        end
    end
    return wit[:,end], wit
end
##
@info "IRWLS (non_lin): "
@info "   Runtime info: "
@time what, wit = IRWLS(L0)
w_true = [ModelingToolkit.getdefault(p) for p in parameters(mdl)]
relErr = norm(wit[:,end] - w_true) / norm(w_true)
@info "   coeff rel err = $relErr"
@info "   iterations    = $(size(wit,2)-1)"
@info "IRWLS (linearize): "
@info "   Runtime info: "
@time what, wit = IRWLS(L0_mat,L1)
w_true = [ModelingToolkit.getdefault(p) for p in parameters(mdl)]
relErr = norm(wit[:,end] - w_true) / norm(w_true)
@info "   coeff rel err = $relErr"
@info "   iterations    = $(size(wit,2)-1)"
##
# S1 = zeros(K,D,K,D)
# S2 = zeros(K,D,K,D)
# function getS1(w)
#     LL!(LL,JJ, w, sig, L0, _jacuF!, uobs) 
#     @tullio S1[k1,d1,k2,d2] = LL[k1,m,d1,d] * LL[k2,m,d2,d];
#     reshape(S1, K*D, K*D)
# end
# function getS2(w)
#     L!(L,LL,JJ, w, sig, L0, _jacuF!, uobs) 
#     @tullio S2[k1,d1,k2,d2] = L[k1,d1,m,d] * L[k2,d2,m,d];
#     reshape(S2, K*D, K*D)
# end
##
# @time S1_mat = getS1(w_rand);
# @time S2_mat = getS2(w_rand);
# @time R = cholesky(S_true).U;

# @show norm(S1_mat - S_true) / norm(S_true)
# @show norm(S2_mat- S_true) / norm(S_true)
