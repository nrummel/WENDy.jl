using Revise
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using WENDy
##
for ex in WENDy.EXAMPLES 
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
@info "Trust Region Solve with Analytic Hessian "
@time 
relErr = norm(wts - wTrue) / norm(wTrue)
fsRelErr = forwardSolveRelErr(wendyProb, ex, wts)
@info """  
    coef relErr =  $(relErr*100)%
    fs relErr   =  $(fsRelErr*100)%
    iter        =  $(res.iterations)
"""
##

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