using Revise
@info "Loading dependencies"
@info " Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
# includet("../examples/brusselator.jl")
includet("../examples/robertson.jl")
@info " Loading symbolic calcutations..."
includet("../src/wendySymbolics.jl")
@info " Loading wendySymbolics"
includet("../src/wendySymbolics.jl")
@info " Loading wendyNoise"
includet("../src/wendyNoise.jl")
@info " Loading wendyTestFunctions"
includet("../src/wendyTestFunctions.jl")
@info " Loading wendyProblems"
includet("../src/wendyProblems.jl")
@info " Loading wendy data..."
includet("../src/wendyData.jl")
@info " Loading wendy equations..."
includet("../src/wendyEquations.jl")
@info " Loading wendy methods..."
includet("../src/wendyMethods.jl")
@info " Loading IRWLS..."
includet("../src/wendyIRWLS.jl")
@info " Loading FSNLS..."
includet("../src/wendyFSNLS.jl")
push!(LOAD_PATH,joinpath(@__DIR__,"../../SFN.jl/src"))
@info " Loading SFN/ARC..."
using SFN
@info " Loading external dependencies"
using FiniteDiff, Optim
##
ex = ROBERTSON
# ex = EXPONENTIAL
# ex = LORENZ
# ex = LOGISTIC_GROWTH
# ex = HINDMARSH_ROSE
# ex = FITZHUG_NAGUMO
# ex = LOOP
# ex = MENDES_EXAMPLES[1]
# ex = BRUSSELATOR
params = WENDyParameters(;
    noiseRatio=0.01, 
    seed=Int(1), 
    timeSubsampleRate=1
)
wendyProb = WENDyProblem(ex, params; ll=Logging.Info)
wTrue = wendyProb.wTrue
J = length(wTrue)
μ = .1
w0 = wTrue + μ * abs.(wTrue) .* randn(J);
## solve with Maximum Likelihood Estimate
_m_ = mw(wendyProb, params);
_∇m!_ = ∇mw(wendyProb, params);
_Hm!_ = Hmw(wendyProb, params);
##
@info "Cost function call "
@time _m_(w0)
## Gradient computation 
∇m0 = zeros(J)
@info "  Finite difference gradient"
@time ∇m_fd = FiniteDiff.finite_difference_gradient(_m_, w0)
@info "  Analytic gradient"
@time _∇m!_(∇m0, w0); 
relErr = norm(∇m0 - ∇m_fd) / norm(∇m_fd)
@info "  relErr = $relErr"
## Hessian computation 
function Hm_fd!(H,w,p=nothing) 
    FiniteDiff.finite_difference_jacobian!(H, _∇m!_, w)
    @views H .= 1/2*(H + H')
    @views H .= Symmetric(H)
    nothing 
end 
##
H0 = zeros(J,J)
@info "==============================================="
@info "==== Comparing Hess ====="
@info "  Analytic Hessian "
@time _Hm!_(H0, w0)
@info "  Finite Differences Hessian from _m_"
Hfd = zeros(J,J)
@time FiniteDiff.finite_difference_hessian!(Hfd, _m_, w0)
@info "  Finite Differences Hessian from _∇m_"
Hfd2 = zeros(J,J)
@time Hm_fd!(Hfd2, w0)
@info "   Rel Error (analytic vs finite diff obj) $(norm(H0 - Hfd) / norm(Hfd))"
@info "   Rel Error (analytic vs finite diff res) $(norm(H0 - Hfd2) / norm(Hfd2))"
@info "   Rel Error (finite diff obj vs finite diff res) $(norm(Hfd - Hfd2) / norm(Hfd))"
@info "==============================================="
##
@info "Trust Region Solve with Analytic Hessian "
@time res = Optim.optimize(
    _m_, _∇m!_, _Hm!_, w0, Optim.NewtonTrustRegion(),
    Optim.Options(
        show_trace=true, show_every=10,
        extended_trace=true, 
        x_reltol=1e-8
    )
)
wts = res.minimizer
relErr = norm(wts - wTrue) / norm(wTrue)
fsRelErr = forwardSolveRelErr(wendyProb, ex, wts)
@info """  
    coef relErr =  $(relErr*100)%
    fs relErr   =  $(fsRelErr*100)%
    iter        =  $(res.iterations)
"""
##
function fg!(grads, w)
    _∇m!_(grads,w)
    _m_(w)
end
function _Hm_(w)
    _Hm!_(_Hm!_.H, w)
    _Hm!_.H
end
##
opt = SFN.ARCOptimizer(
    length(w0);
    atol=1e-8 ,rtol=1e-8)
@info "Calling ARC"
@time stats, warc = minimize!(
    opt, copy(w0), _m_, fg!, _Hm_;
    show_trace=true, show_every=10,
    # extended_trace=true
)
relErr = norm(warc - wTrue) / norm(wTrue)
fsRelErr = forwardSolveRelErr(wendyProb, ex, warc)
@info """  
    coef relErr =  $(relErr*100)%
    fs relErr   =  $(fsRelErr*100)%
    iter        =  $(res.iterations)
"""
##
@info "Forward Solve Nonlinear Least Squares"
try 
    @time  begin
    wfsnls, solfsnls = FSNLS(
        wendyProb, ex.ode, ones(J);
        OptAlg=NewtonTrustRegion(), 
        reltol=1e-8, maxiters=1000
        # show_trace=true, show_every=10
    )
    end
    relErr = norm(wfsnls - wTrue) / norm(wTrue)
    fsRelErr = solfsnls.objective
    @info """  
        coef relErr =  $(relErr*100)%
        fs relErr   =  $(fsRelErr*100)%
        iter        =  $(res.iterations)
    """
catch e
    @warn " FSNLS failed"
end
## 
@info "IRWLS"
iter = NLS_iter(wendyProb, params, maxiters=1000, reltol=1e-8); # this tolerence is for nonlinear solve 
@time wirwls, wit, resit = IRWLS(
    wendyProb, params, iter, w0; 
    relTol=1e-8,
    maxIt=1000
    # ll=Logging.Info, 
    # trueIter=iterLin, 
    # iterll=Logging.Info
)
relErr = norm(wirwls - wTrue) / norm(wTrue)
fsRelErr = try
    forwardSolveRelErr(wendyProb, ex, wirwls)
catch 
    1e6 
end
@info """  
    coef relErr =  $(relErr*100)%
    fs relErr   =  $(fsRelErr*100)%
    iter        =  $(res.iterations)
"""

# ##
# @info "SFN with analytic hessian"
# # opt = SFNOptimizer(length(w0),Symbol("EigenSolver"), linesearch=true)
# opt = SFNOptimizer(length(w0),Symbol("GLKSolver"), linesearch=true)
# @time stats,wsfn = minimize!(
#     opt, copy(w0), _m_, fg!, _Hm_;
#     show_trace=true, show_every=10, extended_trace=true
# )
# relErr = norm(wsfn - wTrue) / norm(wTrue)
# @info "  relErr =  $(relErr*100)%"
# @info "  iter   =  $(stats.iterations)"
# ##
# @info "Trust Region Solve with Finite Difference Hessian "
# @time begin
# resfd = Optim.optimize(
#     _m_, _∇m!_, Hm_fd!, w0, 
#     Optim.NewtonTrustRegion(;
#         initial_delta = 1.0, # matches matlab
#         delta_hat = 100.0,
#         eta = 0.1,
#         rho_lower = 0.25, # matches matlab
#         rho_upper = 0.75 # matches matlab
#     ),
#     Optim.Options(
#         # show_trace=true, extended_trace=true, store_trace=true, show_every=10, 
#         x_reltol=1e-8, x_abstol=1e-8
#     )
# )
# end
# relErr = norm(resfd.minimizer - wTrue) / norm(wTrue)
# @info "  relErr =  $(relErr*100)%"
# @info "  iter   =  $(resfd.iterations)"