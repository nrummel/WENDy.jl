using BenchmarkTools,  MAT, UnicodePlots
@info "Loading generateNoise..."
includet("../src/wendyNoise.jl")
@info "Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info "Loading computeGradients..."
includet("../src/wendySymbolics.jl")
@info "Loading linearSystem..."
includet("../src/wendyEquations.jl")
##
mdl = HINDMARSH_ROSE_MODEL
data = matread(joinpath(@__DIR__, "../data/Lw_hindmarsh_test.mat"))
u = Matrix(data["xobs"]')
V = data["V"]
Vp = data["Vp"];
G_0_matlab = data["G_0"];
b_0_matlab = data["b_0"][:];
L0_matlab = data["L0"];
L1_matlab = data["L1"];
Lw_matlab = data["Lw"];
Sw_matlab = data["Sw"];
RT_matlab = data["RT"];
true_vec = data["true_vec"][:];
diagReg = data["diagReg"];
w0_matlab = data["w0"];
KD,MD,J = size(L1_matlab)
K,M = size(V)
D = Int(KD/K)
_, _F!         = getRHS(mdl)
_, _jacuF! = getJacu(mdl);
Lgetter = LNonlinear(u, V, Vp, sig, _jacuF!, K, M, D)
S = zeros(KD,KD)
R = zeros(KD,KD)
R0 = Matrix{Float64}(I,KD,KD)
FF = zeros(D, M);
##
B = zeros(K,D)
B!(B, Vp, u)
b₀ = reshape(B, KD)
G0 = zeros(K*D,J)
for j in 1:J
    tmp = zeros(K,D)
    ej = zeros(J)
    ej[j] = 1
    G!(tmp,FF,V,ej,u,_F!)
    G0[:,j] = reshape(tmp,K*D) 
end
## do one step of IRWLS
@time begin
w0 = G0 \ b₀

L = Lgetter(w)
mul!(S, L, L')
R .= R0
mul!(R, S, R0, 1-diagReg,diagReg)
cholesky!(Symmetric(R))

Giter = UpperTriangular(R)' \ G0
biter = UpperTriangular(R)' \ b₀;
end
nothing
##
@assert norm(b₀ - b_0_matlab) / norm(b_0_matlab) < 1e2*eps()
@assert norm(G0 - G_0_matlab) / norm(G_0_matlab) < 1e2*eps()
@assert norm(S - Sw_matlab) / norm(Sw_matlab) < 1e2*eps()
@assert norm(L - Lw_matlab) / norm(Lw_matlab) < 1e2*eps()
@assert norm(UpperTriangular(R)' - LowerTriangular(RT_matlab)) / norm(RT_matlab) < 1e2*eps()
