@info "Loading dependencies"
@info " Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info " Loading symbolic calcutations..."
includet("../src/wendySymbolics.jl")
@info " Loading wendy equations..."
includet("../src/wendyEquations.jl")
@info " Loading IRWLS..."
includet("../src/wendyIRWLS.jl")
@info " Loading other dependencies"
using BenchmarkTools, OptimizationOptimJL, Plots
gr()
nothing 
##
# ex = LOGISTIC_GROWTH
# ex = HINDMARSH_ROSE
# ex = FITZHUG_NAGUMO
# ex = LOOP
ex = MENDES_EXAMPLES[1]
params = WENDyParameters(;noiseRatio=0.05)
# wendyProb = _MATLAB_WENDyProblem(ex, params;ll=Logging.Info)
wendyProb = WENDyProblem(ex, params; ll=Logging.Info)
iter = NLS_iter(wendyProb, params)
# iterLin = NLopt_iter(wendyProb, params)
iterLin = Linear_IRWLS_Iter(wendyProb, params)
wTrue = wendyProb.wTrue
J = length(wTrue)
# w0 = wendyProb.data["w0"][:]
μ = 0.1
w0 = wTrue + μ * abs.(wTrue) .* randn(J)
# w0 = iterLin(zeros(J));
nothing
##
@info "IRWLS"
dt = @elapsed a = @allocations what, wit, resit = IRWLS(
    wendyProb, params, iter, w0; 
    ll=Logging.Info, 
    # trueIter=iterLin, 
    iterll=Logging.Info
)
@info """   
    iterations  = $(size(wit,2)-1)
    time        = $(dt)
    allocations = $(a)
"""
if typeof(what) <:AbstractVector 
    relErr = norm(wit[:,end] - wTrue) / norm(wTrue)
    @info "   coeff rel err = $relErr"
end

## solve with Maximum Likelihood Estimate
m(
    w::AbstractVector{<:Real},
    ::Any=nothing; 
    ll::Logging.LogLevel=Logging.Warn
) = _m(wendyProb.U, wendyProb.V, wendyProb.Vp, wendyProb.b0, wendyProb.sig, params.diagReg, wendyProb.f!, wendyProb.jacuf!, w; ll=ll)
_∇m! = ∇mFun!(wendyProb.U, wendyProb.V, wendyProb.Vp, wendyProb.b0, wendyProb.sig, params.diagReg, wendyProb.f!, wendyProb.jacuf!, wendyProb.jacwf!, wendyProb.jacwjacuf!, J)
∇m!(
    ∇m::AbstractVector{<:Real}, 
    w::AbstractVector{<:Real}, 
    ::Any=nothing;
    ll::Logging.LogLevel=Logging.Warn
) = _∇m!(∇m, w; ll=ll)
##
optFun = OptimizationFunction(m; grad=∇m!)#, hess=fake_hess!)
problem = OptimizationProblem(
    optFun, w0; 
    x_tol=1e-8, 
    # show_trace=true,show_every=1
)
@info "MLE"
a = @allocations sol = solve(
    problem,  
    Optim.LBFGS()
    # Optim.NewtonTrustRegion()
)
relErr = norm(sol.u - wTrue) / norm(wTrue)
@info """   
    iterations      = $(sol.stats.iterations)
    time            = $(sol.stats.time)
    allocations     = $(a)
    objective_value = $(sol.objective)
    ret code        = $(sol.retcode)
    coeff rel err   = $relErr
"""
