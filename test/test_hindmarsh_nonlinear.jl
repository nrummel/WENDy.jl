## 
using BenchmarkTools, Random
@info "Loading generateNoise..."
includet("../src/generateNoise.jl")
@info "Loading exampleProblems..."
includet("../src/exampleProblems.jl")
@info "Loading computeGradients..."
includet("../src/computeGradients.jl")
@info "Loading linearSystem..."
includet("../src/linearSystem.jl")
@info "Loading nonlinearSystem..."
# includet("../src/gradDescent.jl")
includet("../src/nonlinearSystem.jl")
@info "Loading testFunctions..."
includet("../src/testFunctions.jl")
##
# mdl                 = LOGISTIC_GROWTH_MODEL
# exampleFile         = joinpath(@__DIR__, "../data/LogisticGrowth.bson")
mdl                 = HINDMARSH_ROSE_MODEL
exampleFile         = joinpath(@__DIR__, "../data/HindmarshRose.bson")
ϕ                   = ExponentialTestFun()
reg                 = 1e-10
noise_ratio         = 0.05
time_subsample_rate = 2
mt_params           = 2 .^(0:3)
seed                = Int(1)
K_min               = 10
K_max               = Int(5e3)
pruneMeth           = SingularValuePruningMethod( 
    MtminRadMethod(),
    UniformDiscritizationMethod()
);
w_true = Float64[ModelingToolkit.getdefault(p) for p in parameters(mdl)]
_, _F!         = getRHS(mdl)
_, _jacuF! = getJacobian(mdl);
_, _jacwF! = getParameterJacobian(mdl);
Random.seed!(seed)
data = BSON.load(exampleFile) 
tt = data[:t] 
u = data[:u] 
num_rad = length(mt_params)
tobs = tt[1:time_subsample_rate:end]
uobs = u[:,1:time_subsample_rate:end]
uobs, noise, noise_ratio_obs, sigma = generateNoise(uobs, noise_ratio)
#
sig = estimate_std(uobs)
#
V,Vp,Vfull = pruneMeth(tobs,uobs,ϕ,K_min,K_max,mt_params);
##
K,M = size(V)
D, _ = size(uobs)
J = length(w_true)
G0 = zeros(K*D, J)
_G0!(G0, uobs, V, _F!)
B = zeros(K,D)
G0 = zeros(K*D, J)
_G0!(G0, uobs, V, _F!)
B!(B, Vp, uobs)
b0 = reshape(B, K*D);
##
jac = JacGgetter(uobs,V,_jacwF!)
jacG = jac(zeros(J))
@assert norm(reshape(jacG,K*D,J) - G0) / norm(G0) < 1e2*eps()
##
G = GFun(uobs,V,_F!)
res = G(w_true) - b0
@assert norm(res - (G0*w_true -b0)) / norm(res) < 1e2*eps()
##
diag_reg = 1e-10
Lgetter = LNonlinear(uobs,V,Vp,sig,_jacuF!);
##
@info "IRWLS (Nonlinear): "
@info "   Runtime info: "
@time what, wit = IRWLS_Nonlinear(uobs, V, Vp, sig, _F!, _jacuF!, _jacwF!, J)
relErr = norm(wit[:,end] - w_true) / norm(w_true)
@info "   coeff rel err = $relErr"
@info "   iterations    = $(size(wit,2)-1)"
