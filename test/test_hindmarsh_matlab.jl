using BenchmarkTools,  MAT
@info "Loading generateNoise..."
includet("generateNoise.jl")
@info "Loading exampleProblems..."
includet("exampleProblems.jl")
@info "Loading computeGradients..."
includet("computeGradients.jl")
@info "Loading linearSystem..."
includet("linearSystem.jl")
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
diag_reg = data["diag_reg"];
w0_matlab = data["w0"];
KD,MD,J = size(L1_matlab)
K,M = size(V)
D = Int(KD/K)
_, _F!         = getRHS(mdl)
_, _jacuF! = getJacobian(mdl)
sig = estimate_std(u);

##
@info "IRWLS (Linear): "
@info "   Runtime info: "
@time what, wit = IRWLS_Linear(u,V,Vp, sig, _F!, _jacuF!, J)
relErr = norm(wit[:,end] - true_vec) / norm(true_vec)
@info "   coeff rel err = $relErr"
@info "   iterations    = $(size(wit,2)-1)"