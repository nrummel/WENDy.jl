using BenchmarkTools,  MAT
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
what_matlab = data["w_hat"];
wits_matlab = data["w_hat_its"];
dt_matlab = data["dt"];
KD,MD,J = size(L1_matlab)
K,M = size(V)
D = Int(KD/K)
_, _F!         = getRHS(mdl)
_, _jacuF! = getJacobian(mdl)
sig = estimate_std(u);

##
@info "IRWLS (Linear): "

dt = @elapsed what, wit = IRWLS_Linear(u,V,Vp, sig, _F!, _jacuF!, J)
relErr = norm(what - true_vec) / norm(true_vec)
relErr_matlab  = norm(what_matlab - true_vec) / norm(true_vec)
@info """Julia
   run time   = $dt s
   rel err    = $(relErr*100)%
   iterations = $(size(wit,2)-1)
MATLAB comparison:
    run time   = $dt_matlab s
    rel err    = $(relErr_matlab*100)%
    iterations = $(size(wits_matlab,2)-1)
"""