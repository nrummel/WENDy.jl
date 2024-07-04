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
wTrue = wendyProb.wTrue
J = length(wTrue)
w0 = wendyProb.data["w0"][:]
nothing
## Lw
Lw = LFun(wendyProb,params)
@time Lw0 = Lw(w0)
relErr = norm(Lw0 - wendyProb.data["Lw"]) /norm( wendyProb.data["Lw"])
relErr < eps()*1e2 ? (@info "Lw in spec (relerr = $relErr)") : (@warn "Lw out of spec (relerr = $relErr)")
## Sw/RT
_RT = RTFun(wendyProb,params)
@time RT0  = _RT(w0)
relErr = norm(_RT.S - wendyProb.data["Sw"]) /norm( wendyProb.data["Sw"])
relErr < eps()*1e2 ? (@info "Sw in spec (relerr = $relErr)") : (@warn "Sw out of spec (relerr = $relErr)")
relErr = norm(RT0 - wendyProb.data["RT"]) /norm( wendyProb.data["RT"])
relErr < eps()*1e2 ? (@info "RT in spec (relerr = $relErr)") : (@warn "RT out of spec (relerr = $relErr)")
## Residual
RT_matlab = wendyProb.data["RT"]
b_matlab = RT_matlab \ wendyProb.data["b_0"][:]
_res = ResFun(wendyProb, params)
@time r0 = _res(RT_matlab, b_matlab, w0; ll=Logging.Info)
r0_matlab = RT_matlab \ (wendyProb.data["G_0"] * w0) - b_matlab
relErr = norm(r0 - r0_matlab) / norm( r0_matlab)
relErr < eps()*1e2 ? (@info "r0 in spec (relerr = $relErr)") : (@warn "r0 out of spec (relerr = $relErr)")
## jacobian of residual 
# because the this ode is linear in paramets ∇_wr = G 
G_matlab = RT_matlab \ wendyProb.data["G_0"]
jacFun = ∇resFun(wendyProb,params)
@time G = jacFun(RT_matlab, w0)
Gtilde = jacFun(RT_matlab, rand(J))
@assert norm(G-Gtilde) /norm(G) < eps()*1e2 
relErr = norm(G - G_matlab) / norm(G_matlab)
relErr < eps()*1e2 ? (@info "∇r in spec (relerr = $relErr)") : (@warn "∇r out of spec (relerr = $relErr)")
## IRWLS
@info "IRWLS linear"
iterLin = Linear_IRWLS_Iter(wendyProb, params)
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
relErr < eps()*1e2 ? (@info "wits in spec (relerr = $relErr) with matlab") : (@warn "wits out of spec (relerr = $relErr) with matlab")
##
@info "IRWLS non-linear"
iter = NLS_iter(wendyProb, params; maxiters=10)
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
## check Maholinobis distance 
_m = MFun(wendyProb, params)
@time m = _m(w0)
m_matlab = wendyProb.data["m"][1]
relErr = abs(2*m - m_matlab) / abs(m_matlab)
relErr < eps()*1e2 ? (@info "m in spec (relerr = $relErr) with matlab") : (@warn "m out of spec (relerr = $relErr) with matlab")
## check gradient of Maholinobis distance
_∇m = ∇mFun(wendyProb, params)
@time ∇m = _∇m(w0;ll=Logging.Info)
∇m_matlab = wendyProb.data["gradm"][:]
relErr = norm(2*∇m - ∇m_matlab) / norm(∇m_matlab)
relErr < eps()*1e2 ? (@info "∇m in spec (relerr = $relErr) with matlab") : (@warn "∇m out of spec (relerr = $relErr) with matlab")
