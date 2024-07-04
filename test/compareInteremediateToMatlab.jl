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
using BenchmarkTools
nothing 
##
ex = HINDMARSH_ROSE
params = WENDyParameters(;noiseRatio=0.05)
wendyProb = _MATLAB_WENDyProblem(ex, params;ll=Logging.Info)
iter = NLS_iter(wendyProb, params)
iterLin = Linear_IRWLS_Iter(wendyProb, params)
wTrue = wendyProb.wTrue
J = length(wTrue)
w0 = wendyProb.data["w0"][:]
nothing
## Lw
Lw = LFun(wendyProb,params)
@time Lw0 = Lw(w0)
relErr = norm(Lw0 - wendyProb.data["Lw"]) /norm( wendyProb.data["Lw"])
relErr < 1e-14 ? (@info "Lw in spec") : (@warn "Lw out of spec")
## Sw/RT
_RT = RTFun(wendyProb,params)
@time RT0  = _RT(w0)
relErr = norm(_RT.S - wendyProb.data["Sw"]) /norm( wendyProb.data["Sw"])
relErr < 1e-14 ? (@info "Sw in spec") : (@warn "Sw out of spec")
relErr = norm(RT0 - wendyProb.data["RT"]) /norm( wendyProb.data["RT"])
relErr < 1e-14 ? (@info "RT in spec") : (@warn "RT out of spec")
## Residual
RT_matlab = wendyProb.data["RT"]
b_matlab = RT_matlab \ wendyProb.data["b_0"][:]
_res = ResFun(wendyProb, params)
@time r0 = _res(RT_matlab, b_matlab, w0; ll=Logging.Info)
r0_matlab = RT_matlab \ (wendyProb.data["G_0"] * w0) - b_matlab
relErr = norm(r0 - r0_matlab) / norm( r0_matlab)
relErr < eps()*1e2 ? (@info "r0 in spec") : (@warn "r0 out of spec")
## jacobian of residual 
# because the this ode is linear in paramets ∇_wr = G 
G_matlab = RT_matlab \ wendyProb.data["G_0"]
jacFun = ∇resFun(wendyProb,params)
@time G = jacFun(RT_matlab, w0)
Gtilde = jacFun(RT_matlab, rand(J))
@assert norm(G-Gtilde) /norm(G) < eps()*1e2 
relErr = norm(G - G_matlab) / norm(G_matlab)
relErr < eps()*1e2 ? (@info "∇r in spec") : (@warn "∇r out of spec")
## IRWLS
@info "IRWLS linear"
dt = @elapsed a = @allocations what, wit, resit = IRWLS(
    wendyProb, params, iterLin, w0
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
relErr = norm(wendyProb.data["w_hat_its"] - wit)/norm(wit)
relErr < eps()*1e2 ? (@info "wits in spec with matlab") : (@warn "wits out of spec with matlab")
##
@info "IRWLS non-linear"
dt = @elapsed a = @allocations what, wit, resit = IRWLS(
    wendyProb, params, iter, w0;
    # ll=Logging.Info, iterll=Logging.Info, trueIter=iterLin
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