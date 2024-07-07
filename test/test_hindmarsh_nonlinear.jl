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
@info "Loading nonlinearSystem..."
# includet("../src/gradDescent.jl")
includet("../src/nonlinearSystem.jl")
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
wTrue = Float64[ModelingToolkit.getdefault(p) for p in parameters(mdl)]
_, _F!         = getRHS(mdl)
_, _jacuF! = getJacu(mdl);
_, _jacwF! = getJacw(mdl);
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
K,M = size(V)
D, _ = size(uobs)
J = length(wTrue)
G0 = zeros(K*D, J)
_G0!(G0, uobs, V, _F!)
B = zeros(K,D)
G0 = zeros(K*D, J)
_G0!(G0, uobs, V, _F!)
B!(B, Vp, uobs)
b₀ = reshape(B, K*D);
##
jac = JacGgetter(uobs,V,_jacwF!)
jacG = jac(zeros(J))
@assert norm(reshape(jacG,K*D,J) - G0) / norm(G0) < 1e2*eps()
##
G = GFun(uobs,V,_F!)
res = G(wTrue) - b₀
@assert norm(res - (G0*wTrue -b₀)) / norm(res) < 1e2*eps()
##
diagReg = 1e-10
Lgetter = LNonlinear(uobs,V,Vp,sig,_jacuF!);
##
@info "IRWLS (Nonlinear): "
@info "   Runtime info: "
@time what, wit = IRWLS_Nonlinear(uobs, V, Vp, sig, _F!, _jacuF!, _jacwF!, J)
relErr = norm(wit[:,end] - wTrue) / norm(wTrue)
@info "   coeff rel err = $relErr"
@info "   iterations    = $(size(wit,2)-1)"
