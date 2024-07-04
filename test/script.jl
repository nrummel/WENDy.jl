@info "Loading dependencies"
@info " Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info " Loading symbolic calcutations..."
includet("../src/wendySymbolics.jl")
@info " Loading wendyProblems"
includet("../src/wendyProblems.jl")
@info " Loading wendy equations..."
includet("../src/wendyEquations.jl")
@info " Loading IRWLS..."
includet("../src/wendyIRWLS.jl")
@info " Loading other dependencies"
using BenchmarkTools, OptimizationOptimJL, Plots, ForwardDiff
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
wendyProb = _MATLAB_WENDyProblem(ex, params;ll=Logging.Info)
# wendyProb = WENDyProblem(ex, params; ll=Logging.Info)
iter = NLS_iter(wendyProb, params)
# iterLin = NLopt_iter(wendyProb, params)
iterLin = Linear_IRWLS_Iter(wendyProb, params)
wTrue = wendyProb.wTrue
J = length(wTrue)
w0 = wendyProb.data["w0"][:]
# μ = 0.1
# w0 = wTrue + μ * abs.(wTrue) .* randn(J)
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
)::Real = _m(wendyProb.U, wendyProb.V, wendyProb.Vp, wendyProb.b0, wendyProb.sig, params.diagReg, wendyProb.f!, wendyProb.jacuf!, w; ll=ll)
_∇m! = ∇mFun!(wendyProb.U, wendyProb.V, wendyProb.Vp, wendyProb.b0, wendyProb.sig, params.diagReg, wendyProb.f!, wendyProb.jacuf!, wendyProb.jacwf!, wendyProb.jacwjacuf!, J)
∇m!(
    ∇m::AbstractVector{<:AbstractFloat}, 
    w::AbstractVector{<:AbstractFloat}, 
    ::Any=nothing;
    ll::Logging.LogLevel=Logging.Warn
) = _∇m!(∇m, w; ll=ll)
_∇m!_dual = ∇mFun!(wendyProb.U, wendyProb.V, wendyProb.Vp, wendyProb.b0, wendyProb.sig, params.diagReg, wendyProb.f!, wendyProb.jacuf!, wendyProb.jacwf!, wendyProb.jacwjacuf!, J, Val(ForwardDiff.Dual{ForwardDiff.Tag{Function, Float64}, Float64, 10}))
∇m!(
    ∇m::AbstractVector{T1}, 
    w::AbstractVector{T2}, 
    ::Any=nothing;
    ll::Logging.LogLevel=Logging.Warn
) where {T1<:ForwardDiff.Dual, T2<:ForwardDiff.Dual}= _∇m!_dual(∇m, w; ll=ll)
function grad(w::AbstractVector{T}) where T<:Real
    ∇m = zeros(T, length(w))
    ∇m!(∇m, w)
    return ∇m
end
# run once for JIT
m(w0)
∇m_test = zeros(J)
##
∇m!(∇m_test, w0)
##
grad(w0)
##
@info "Cost function call "
@time m(w0)
## Gradient computation 
@info "Explict Gradient Computation"
@time ∇m!(∇m_test, w0); 
@info "Auto Diff for gradient "
@time ∇m_auto = ForwardDiff.gradient(m, w0)::Vector{Float64}
relErr = norm(∇m_test- ∇m_auto) / norm(∇m_test)
@info "  relErr = $relErr"
## Hessian computation 
# @info "Hessian from gradient"
# @time ForwardDiff.jacobian(grad, w0)
@info "Hessian from obj"
@time ForwardDiff.hessian(m, w0)
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
@info "Loading SFN..."
# # using SFN: SFNOptimizer, iterate!, hvp_power, Optimizer,RHvpOperator
# import SFN.minimize!
# includet("../../SFN/src/SFN.jl")
includet("../../SFN/src/optimizers.jl")
includet("../../SFN/src/stats.jl")
includet("../../SFN/src/hvp.jl")
includet("../../SFN/src/solvers.jl")
includet("../../SFN/src/minimize.jl")
includet("../../SFN/src/linesearch.jl")
@info "Extending minimize to only take gradient but not hessian..."
function minimize!(opt::O, x::S, f::F1, fg!::F2; itmax::I=1000, time_limit::T=Inf) where {O<:Optimizer, T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, I}
    @info "build the RHv operator"
    Hv = RHvpOperator(f, x, power=hvp_power(opt.solver))

    #iterate
    @info "iterate!"
    stats = iterate!(opt, x, f, fg!, Hv, itmax, time_limit)

    return stats
end
##
@info "Build optimizer"
opt = SFNOptimizer(length(w0))
@info "minimize!"
minimize!(opt, w0, m, ∇m!; itmax=10)

