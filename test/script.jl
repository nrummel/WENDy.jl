@info "Loading dependencies"
@info " Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info " Loading symbolic calcutations..."
includet("../src/wendySymbolics.jl")
@info " Loading wendyProblems"
includet("../src/wendyProblems.jl")
@info " Loading wendy equations..."
includet("../src/wendyEquations.jl")
@info " Loading wendy methods..."
includet("../src/wendyMethods.jl")
@info " Loading IRWLS..."
includet("../src/wendyIRWLS.jl")
@info " Loading other dependencies"
using BenchmarkTools, OptimizationOptimJL, Plots, ForwardDiff, FiniteDiff
gr()
includet("../src/wendyCSE.jl")
nothing 
##
# ex = LOGISTIC_GROWTH
ex = HINDMARSH_ROSE
# ex = FITZHUG_NAGUMO
# ex = LOOP
# ex = MENDES_EXAMPLES[1]
params = WENDyParameters(;noiseRatio=0.05)
wendyProb = WENDyProblem(ex, params; ll=Logging.Info)
iter = NLS_iter(wendyProb, params)
iterLin = Linear_IRWLS_Iter(wendyProb, params)
wTrue = wendyProb.wTrue
J = length(wTrue)
μ = 1
w0 = wTrue + μ * abs.(wTrue) .* randn(J)
K,M,D = wendyProb.K, wendyProb.M, wendyProb.D
w0 = iterLin(zeros(J));
nothing
## solve with Maximum Likelihood Estimate
_m_ = mw(wendyProb, params);
_∇m!_ = ∇mw(wendyProb, params);
_Hm!_ = Hmw(wendyProb, params);
##
@info "Cost function call "
@time _m_(w0)
## Gradient computation 
∇m0 = zeros(J)
@info "  Finite Diff for gradient"
@time ∇m_fd = FiniteDiff.finite_difference_gradient(_m_, w0)
@info "  Explict Gradient Computation"
@time _∇m!_(∇m0, w0); 
relErr = norm(∇m0 - ∇m_fd) / norm(∇m_fd)
@info "  relErr = $relErr"
## Hessian computation 
# @info "Hessian from autodiff"
# @time H_auto = ForwardDiff.hessian(m, w0)
##
# testHessian 
H0 = zeros(J,J)
@info "Explicit Hessian "
@time _Hm!_(H0, w0)
@info "Finite Differences Hessian "
Hfd = zeros(J,J)
cache = FiniteDiff.HessianCache(w0,Val{:hcentral},Val{true})
@time FiniteDiff.finite_difference_hessian!(Hfd, _m_, w0, cache)
@info "Rel Error (finite diff) $(norm(H0 - Hfd) / norm(Hfd))"

##
f(w,p=nothing) = _m_(w)
∇m(grad, w,p=nothing) = _∇m!_(grad,w)
cache = FiniteDiff.HessianCache(w0)
h(H,w,p=nothing) = FiniteDiff.finite_difference_hessian!(H, _m, w, cache)
funfd = OptimizationFunction(f; grad=∇m, hess=h)
probfd = OptimizationProblem(funfd, w0)
solFd = solve(probfd, NewtonTrustRegion(), show_trace=true)
@info "Trust Region Solve with Finite Difference Hessian "
relErr = norm(solFd.u - wTrue) / norm(wTrue)
@info "  relErr =  $(relErr*100)%"
@info "  iter   =  $(solFd.stats.iterations)"
##
f(w,p=nothing) = _m_(w)
∇m(grad, w,p=nothing) = _∇m!_(grad,w)
h(H,w,p=nothing) = _Hm!_(H, w)
fun = OptimizationFunction(f; grad=∇m, hess=h)
prob = OptimizationProblem(fun, w0)
sol = solve(prob, NewtonTrustRegion(),show_trace=true)
@info "Trust Region Solve with Explicit Hessian "
relErr = norm(sol.u - wTrue) / norm(wTrue)
@info "  relErr =  $(relErr*100)%"
@info "  iter   =  $(sol.stats.iterations)"
##
# v0 = w0
# norm(Hfd * v0 - FiniteDiff.finite_difference_gradient(w->dot(_∇m(w),v0), w0)) / norm(Hfd * v0)
# ##
# function fg!(g, w)
#     _∇m(g, w)
#     _m(w)
# end
# function hv!(Hv, w, v)
#     FiniteDiff.finite_difference_gradient!(
#         Hv,
#         w -> dot(_∇m(w),v),
#         w
#     )
#     nothing
# end
# d = Optim.TwiceDifferentiableHV(m, fg!, hv!, zeros(length(w0)))
# result = Optim.optimize(d, w0, Optim.KrylovTrustRegion())
## 
# includet("/Users/user/Documents/School/becker-misc/GradientTests/julia/gradientCheck.jl")
# function ∇m(
#     w::AbstractVector{<:Real}, 
#     ::Any=nothing;
#     ll::Logging.LogLevel=Logging.Warn
# )
#     grad = zeros(J)
#     ∇m!(grad, w; ll=ll)
#     return grad 
# end
# ## 
# gradientCheck(m, ∇m, w0; ll=Logging.Info, scaling=1e-6, makePlot=true)

##
# optFun = OptimizationFunction(m; grad=∇m!)#, hess=fake_hess!)
# problem = OptimizationProblem(
#     optFun, w0; 
#     x_tol=1e-8, 
#     show_trace=true,show_every=1
# )
# @info "MLE"
# a = @allocations sol = solve(
#     problem,  
#     Optim.LBFGS()
#     # Optim.NewtonTrustRegion()
# )
# relErr = norm(sol.u - wTrue) / norm(wTrue)
# @info """   
#     iterations      = $(sol.stats.iterations)
#     time            = $(sol.stats.time)
#     allocations     = $(a)
#     objective_value = $(sol.objective)
#     ret code        = $(sol.retcode)
#     coeff rel err   = $relErr
# """
##
# SFN 
# @info "Loading SFN..."
# # using SFN: SFNOptimizer, iterate!, hvp_power, Optimizer,RHvpOperator
# import SFN.minimize!
# includet("../../SFN/src/SFN.jl")
# includet("../../SFN/src/optimizers.jl")
# includet("../../SFN/src/stats.jl")
# includet("../../SFN/src/hvp.jl")
# includet("../../SFN/src/solvers.jl")
# includet("../../SFN/src/minimize.jl")
# includet("../../SFN/src/linesearch.jl")
# @info "Extending minimize to only take gradient but not hessian..."
# function minimize!(opt::O, x::S, f::F1, fg!::F2; itmax::I=1000, time_limit::T=Inf) where {O<:Optimizer, T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, I}
#     @info "build the RHv operator"
#     Hv = RHvpOperator(f, x, power=hvp_power(opt.solver))

#     #iterate
#     @info "iterate!"
#     stats = iterate!(opt, x, f, fg!, Hv, itmax, time_limit)

#     return stats
# end
# ##
# @info "Build optimizer"
# opt = SFNOptimizer(length(w0))
# @info "minimize!"
# minimize!(opt, w0, m, ∇m!; itmax=10)





##
# @info "IRWLS"
# dt = @elapsed a = @allocations what, wit, resit = IRWLS(
#     wendyProb, params, iter, w0; 
#     ll=Logging.Info, 
#     # trueIter=iterLin, 
#     iterll=Logging.Info
# )
# @info """   
#     iterations  = $(size(wit,2)-1)
#     time        = $(dt)
#     allocations = $(a)
# """
# if typeof(what) <:AbstractVector 
#     relErr = norm(wit[:,end] - wTrue) / norm(wTrue)
#     @info "   coeff rel err = $relErr"
# end