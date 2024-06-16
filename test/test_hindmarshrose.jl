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
_, _F!         = getRHS(mdl)
_, _jacuF! = getJacobian(mdl);
##
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
@info "IRWLS (Linear): "
@info "   Runtime info: "
w_true = [ModelingToolkit.getdefault(p) for p in parameters(mdl)]
diag_reg = 1e-10
J = length(w_true)
@time what, wit = IRWLS_Linear(uobs, V, Vp, sig, _F!, _jacuF!, J)
relErr = norm(wit[:,end] - w_true) / norm(w_true)
@info "   coeff rel err = $relErr"
@info "   iterations    = $(size(wit,2)-1)"