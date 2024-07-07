## 
using BenchmarkTools, Random
@info "Loading generateNoise..."
includet("../src/wendyNoise.jl")
@info "Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info "Loading computeGradients..."
includet("../src/wendySymbolics.jl")
@info "Loading linearSystem..."
includet("../src/wendyEquations.jl")
@info "Loading testFunctions..."
includet("../src/wendyTestFunctions.jl")
##
# mdl                 = LOGISTIC_GROWTH_MODEL
# exampleFile         = joinpath(@__DIR__, "../data/LogisticGrowth.bson")
mdl                 = HINDMARSH_ROSE_MODEL
exampleFile         = joinpath(@__DIR__, "../data/HindmarshRose.bson")
ϕ                   = ExponentialTestFun()
reg                 = 1e-10
noiseRatio         = 0.05
timeSubsampleRate = 2
mtParams           = 2 .^(0:3)
seed                = Int(1)
Kmin               = 10
Kmax               = Int(5e3)
pruneMeth           = SingularValuePruningMethod( 
    MtminRadMethod(),
    UniformDiscritizationMethod()
);
_, _F!         = getRHS(mdl)
_, _jacuF! = getJacu(mdl);
##
Random.seed!(seed)
data = BSON.load(exampleFile) 
tt = data[:t] 
u = data[:u] 
numRad = length(mtParams)
tobs = tt[1:timeSubsampleRate:end]
uobs = u[:,1:timeSubsampleRate:end]
uobs, noise, noise_ratio_obs, sigma = generateNoise(uobs, noiseRatio)
#
sig = estimate_std(uobs)
#
V,Vp,Vfull = pruneMeth(tobs,uobs,ϕ,Kmin,Kmax,mtParams);
##
@info "IRWLS (Linear): "
@info "   Runtime info: "
wTrue = [ModelingToolkit.getdefault(p) for p in parameters(mdl)]
diagReg = 1e-10
J = length(wTrue)
@time what, wit = IRWLS_Linear(uobs, V, Vp, sig, _F!, _jacuF!, J)
relErr = norm(wit[:,end] - wTrue) / norm(wTrue)
@info "   coeff rel err = $relErr"
@info "   iterations    = $(size(wit,2)-1)"