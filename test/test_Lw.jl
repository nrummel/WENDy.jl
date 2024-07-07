using BenchmarkTools,  MAT, UnicodePlots
@info "Loading generateNoise..."
includet("../src/wendyNoise.jl")
@info "Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info "Loading computeGradients..."
includet("../src/wendySymbolics.jl")
@info "Loading linearSystem..."
includet("../src/wendyEquations.jl")
## Build custome functions to try to beat what the ModelingToolkit builds 
function _f!(jm::AbstractMatrix{T}, w::AbstractVector{T}, um::AbstractVector{T}) where T<:Real
    jm[1] = 2*w[3]*um[1] + 3*w[2]*um[1]^2
    jm[2] = 2*w[6]*um[1]
    jm[3] = w[8]
    jm[4] = w[1]
    jm[5] = w[7]
    jm[6] = 0
    jm[7] = w[4]
    jm[8] = 0
    jm[9] = w[10]
    nothing
end

function _fprecomp!(jm::AbstractMatrix{T}, w::AbstractVector{T}, ump::AbstractVector{T}) where T<:Real
    jm[1] = w[3]*ump[1] + w[2]*ump[2]
    jm[2] = w[6]*ump[1]
    jm[3] = w[8]
    jm[4] = w[1]
    jm[5] = w[7]
    jm[6] = 0
    jm[7] = w[4]
    jm[8] = 0
    jm[9] = w[10]
    nothing
end

function _L0!(L0::AbstractArray{T,4}, Vp::AbstractMatrix{T},sig::AbstractVector{T}) where T<:Real
    @tullio L0[k,d,d,m] = Vp[k,m]*sig[d]
    nothing
end 

function _L0(Vp::AbstractMatrix{T}, sig::AbstractVector{T}) where T<:Real
    L0 = zeros(K,D,D,M)
    _L0!(L0, Vp, sig)
    return L0 
end 

function _L1!(L1::AbstractArray{T,3}, V::AbstractMatrix{T},sig::AbstractVector{T}, _jj!::Function, J::Int) where T<:Real
    K, M = size(Vp)
    D = length(sig)
    JJ = zeros(D,D,M)
    LL = zeros(K,D,D,M)
    L = zeros(K,D,M,D)
    ZZ = zeros(K,D,D,M)
    for j in 1:J 
        ej = zeros(J)
        ej[j] = 1
        _L!(view(L1,:,:,j),L, LL, ZZ, JJ, ej, u, V, sig, _jj!)
    end
end
function _L1(V::AbstractMatrix{T}, sig::AbstractVector{T}, _jj!::Function, J::Int) where T<:Real
    L1 = zeros(K*D, M*D, J)
    _L1!(L1, V, sig, _jj!, J)
    return L1 
end
# Precompute L0 for mat mat add rather than tullio
function _L!(L_mat::AbstractMatrix{T}, L::AbstractArray{T,4}, LL::AbstractArray{T,4}, L0::AbstractArray{T,4}, JJ::AbstractArray{T,3},w::AbstractVector{T},u::AbstractMatrix{T}, V::AbstractMatrix{T},sig::AbstractVector{T}, _jacuF!::Function) where T<:Real
    K,D,M,_ = size(L)
    _J!(JJ, w, u, _jacuF!)
    @tullio LL[k,d2,d1,m] = V[k,m] * JJ[d2,d1,m] * sig[d1]
    LL .+= L0
    # LL_tmp = reshape
    permutedims!(L,LL,(1,2,4,3))
    L_mat .= reshape(L,K*D,M*D)
    nothing
end
#  Use blas like matlab implementation
function _L!(L_mat::AbstractMatrix{T}, L1::AbstractArray{T,3}, L0_mat::AbstractMatrix{T}, w::AbstractVector{T}) where T<:Real
    J = length(w)
    KD, MD = size(L0_mat)
    @assert all(size(L1) .== (KD,MD,J)) "L1 is not the appropriate size"
    L_mat .= L0_mat 
    mul!(reshape(L_mat,KD*MD), reshape(L1,KD*MD,J),w,1,1)
    nothing
end


## Load data from matlab to prepare
mdl = HINDMARSH_ROSE_MODEL
data = matread(joinpath(@__DIR__, "../data/Lw_hindmarsh_test.mat"))
u = Matrix(data["xobs"]')
V = data["V"]
Vp = data["Vp"];
L0_matlab = data["L0"];
L1_matlab = data["L1"];
Lw_matlab = data["Lw"];
Sw_matlab = data["Sw"];
RT_matlab = data["RT"];
true_vec = data["true_vec"][:];
diagReg = data["diagReg"];
Ltime_matlab = data["Ltime"];
KD,MD,J = size(L1_matlab)
K,M = size(V)
D = Int(KD/K)
_, _jacuF! = getJacu(mdl);
nothing
##
sig = estimate_std(u)
uprecomp = reduce(hcat, [2*u[1,m], 3*u[1,m]^2] for m in 1:M)
JJ = zeros(D,D,M)
LL = zeros(K,D,D,M)
L = zeros(K,D,M,D);
L_matMTK = zeros(K*D,M*D);
L_mat0 = zeros(K*D,M*D);
L_mat1 = zeros(K*D,M*D);
L_mat2 = zeros(K*D,M*D);
L_mat3 = zeros(K*D,M*D);
L_matT = zeros(K*D,M*D);
Lgetter = LNonlinear(u, V, Vp, sig, _f!, K, M, D)
## Prepare the linear case 
ZZ = zeros(K,D,D,M)
L0 = _L0(Vp, sig)
L1 = _L1(V, sig, _f!, J)
L0_mat = reshape(permutedims(L0,(1,2,4,3)), K*D, M*D)
@assert norm(L0_mat - L0_matlab) / norm(L0_matlab) < 1e2*eps()
@assert norm(L1 - L1_matlab) / norm(L1_matlab) < 1e2*eps()
##
w = true_vec
@info "Non-linear with ModelingToolKit Gen Jac"
@time _L!(L_matMTK, L, LL, L0, JJ, w, u, V, sig, _jacuF!)
@info "Non-linear"
@time _L!(L_mat0, L, LL, L0, JJ, w, u, V, sig, _f!)
@info "Remove L0 precomp"
@time _L!(L_mat1, L, LL, JJ, w, u, V, Vp, sig, _f!)
@info "Precomp non-linear"
@time _L!(L_mat2, L, LL, L0, JJ, w, uprecomp, V, sig, _fprecomp!)
@info "BLAS"
@time _L!(L_mat3, L1, L0_mat, w)
@info "BLAS (with matlab input)"
@time _L!(L_matT, L1_matlab, L0_matlab, w)
@info "Non-liner L_getter"
@time Lgettermat = Lgetter(w)
@info """ MATLAB reference 
time $(Ltime_matlab) s"""
@assert norm(L_matMTK - Lw_matlab) / norm(Lw_matlab) < 1e2*eps() "Modeling Tool Kit does not work "
@assert norm(L_matT - Lw_matlab) / norm(Lw_matlab) < 1e2*eps() "Linear Method does not work"
@assert norm(Lgettermat - Lw_matlab) / norm(Lw_matlab) < 1e2*eps() "Getter does not work"
@assert norm(L_matT - L_mat0) / norm(L_matT) < 1e2*eps() "Non-linear does not work"
@assert norm(L_matT - L_mat1) / norm(L_matT) < 1e2*eps() "Removing L0 does not work"
@assert norm(L_matT - L_mat2) / norm(L_matT) < 1e2*eps() "Custome u and corresponding function does not work"
@assert norm(L_matT - L_mat3) / norm(L_matT) < 1e2*eps() "Linearization does not work"
##
@info "Non-linear"
@btime _L!($L_mat0, $L, $LL, $L0, $JJ, $w, $u, $V, $sig, $_f!)
@info "Remove L0 precomp"
@btime _L!($L_mat1, $L, $LL, $JJ, $w, $u, $V, $Vp, $sig, $_f!)
@info "Precomp non-linear"
@btime _L!($L_mat2, $L, $LL, $L0, $JJ, $w, $uprecomp, $V, $sig, $_fprecomp!)
@info "BLAS"
@btime _L!($L_mat3, $L1, $L0_mat, $w)
@info "Non-liner L_getter"
@btime Lgettermat = Lgetter($w)
@info """ MATLAB reference 
    time $(Ltime_matlab) s"""
##
JJ1 = similar(JJ)
JJ2 = similar(JJ)
@time _J!(JJ1, w, u, _f!)
@time _J!(JJ2, w, uprecomp, _fprecomp!)
@assert norm(JJ2 - JJ1)/norm(JJ1) < 1e2*eps()
##
@btime _J!($JJ, $w, $u, $_F!)
@btime _jacuF!($JJ, $w, $uprecomp, $_fprecomp!)