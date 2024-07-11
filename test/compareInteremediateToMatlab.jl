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
wendyProb = _MATLAB_WENDyProblem(ex, params;ll=Info)
wTrue = wendyProb.wTrue
J = length(wTrue)
w0 = wendyProb._matdata["w0"][:]
K,M,D = wendyProb.K, wendyProb.M, wendyProb.D
nothing
## Lw
_L! = Lw(wendyProb,params)
Lw0 = zeros(K*D,M*D)
@time _L!(Lw0,w0)
relErr = norm(Lw0 - wendyProb._matdata["Lw"]) /norm( wendyProb._matdata["Lw"])
relErr < eps()*1e2 ? (@info "Lw in spec (relerr = $relErr)") : (@warn "Lw out of spec (relerr = $relErr)")
## Sw/RT
S0_matlab = wendyProb._matdata["Sw"]
_R!_ = Rw(wendyProb,params)
S₀ = zeros(K*D,K*D)
@time _R!_(S₀,w0,transpose=true, doChol=false) # this regularizes the output... so inspect the non-regularized
relErr = norm(_R!_.S - S0_matlab) /norm(S0_matlab)
relErr < eps()*1e2 ? (@info "Sw in spec (relerr = $relErr)") : (@warn "Sw out of spec (relerr = $relErr)")
Rᵀ₀ = zeros(K*D,K*D)
@time _R!_(Rᵀ₀,w0,transpose=true)
relErr = norm(Rᵀ₀ - wendyProb._matdata["RT"]) /norm( wendyProb._matdata["RT"])
relErr < eps()*1e2 ? (@info "RT in spec (relerr = $relErr)") : (@warn "RT out of spec (relerr = $relErr)")
## Residual
RT_matlab = wendyProb._matdata["RT"]
G_matlab = wendyProb._matdata["G_0"]
b_matlab = wendyProb._matdata["b_0"][:]
_res! = rw(wendyProb, params)
r0 = zeros(K*D)
@time _res!(r0, b_matlab, w0)
r0_matlab = G_matlab * w0 - b_matlab
relErr = norm(r0 - r0_matlab) / norm( r0_matlab)
relErr < eps()*1e2 ? (@info "r0 in spec (relerr = $relErr)") : (@warn "r0 out of spec (relerr = $relErr)")
##
RTinvr0 = zeros(K*D)
@time _res!(RTinvr0, RT_matlab\b_matlab, w0; Rᵀ=RT_matlab)
RTinvr0_matlab = RT_matlab \ (G_matlab*w0) - RT_matlab \b_matlab
relErr = norm(RTinvr0 - RTinvr0_matlab) / norm(RTinvr0_matlab)
relErr < eps()*1e2 ? (@info "RT\\r0 in spec (relerr = $relErr)") : (@warn "RT\\r0 out of spec (relerr = $relErr)")
## jacobian of residual 
# because the this ode is linear in paramets ∇_wr = G 
G_matlab = RT_matlab \ wendyProb._matdata["G_0"]
_∇r!_ = ∇rw(wendyProb,params)
G = similar(G_matlab)
@time _∇r!_(G, w0; Rᵀ=RT_matlab)
relErr = norm(G - G_matlab) / norm(G_matlab)
relErr < eps()*1e2 ? (@info "∇r in spec (relerr = $relErr)") : (@warn "∇r out of spec (relerr = $relErr)")
## check Maholinobis distance 
_m_ = mw(wendyProb, params)
@time m0 = _m_(w0)
m_matlab = wendyProb._matdata["m"][1]
relErr = abs(2*m0 - m_matlab) / abs(m_matlab)
relErr < eps()*1e2 ? (@info "m in spec (relerr = $relErr) with matlab") : (@warn "m out of spec (relerr = $relErr) with matlab")
## check gradient of Maholinobis distance
_∇m!_ = ∇mw(wendyProb, params)
∇m0 = zeros(J)
@time _∇m!_(∇m0, w0)
∇m0_fd = FiniteDiff.finite_difference_gradient(_m_,w0)
∇m_matlab = wendyProb._matdata["gradm"][:]
relErr = norm(2*∇m0 - ∇m_matlab) / norm(∇m_matlab)
relErr < eps()*1e2 ? (@info "∇m in spec (relerr = $relErr) with matlab") : (@warn "∇m out of spec (relerr = $relErr) with matlab")
relErr = norm(∇m0_fd - ∇m0) / norm(∇m0_fd)
relErr < eps()*1e2 ? (@info "∇m in spec (relerr = $relErr) with finite differences") : (@warn "∇m out of spec (relerr = $relErr) with finite differences")
relErr = norm(∇m0_fd - ∇m_matlab) / norm(∇m_matlab)
relErr < eps()*1e2 ? (@info "∇m_fd in spec (relerr = $relErr) with matlab") : (@warn "∇m_fd out of spec (relerr = $relErr) with matlab")
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
relErr = norm(wendyProb._matdata["w_hat_its"] - wit)/norm(wit)
relErr < eps()*1e2 ? (@info "wits in spec (relerr = $relErr) with matlab") : (@warn "wits out of spec (relerr = $relErr) with matlab")
##
@info "IRWLS non-linear"
iter = NLS_iter(wendyProb, params; maxiters=10)
dt = @elapsed a = @allocations what, wit, resit = IRWLS(
    wendyProb, params, iter, w0;
    # ll=Info, iterll=Info, trueIter=iterLin
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