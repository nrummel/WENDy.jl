@info "Loading external dependencies"
using OrdinaryDiffEq, ModelingToolkit
using ModelingToolkit: D_nounits, t_nounits as t
using DiffEqParamEstim, Optimization, OptimizationOptimJL
using Statistics, Random 
@info "Build test problem"
## Call from ODEProblem Directly
function f(du, u, w, t)
   du[1] = w[1] * u[2] + w[2] * u[1]^3 + w[3] * u[1]^2 + w[4] * u[3] 
   du[2] = w[5] + w[6] * u[1]^2 + w[7] * u[2] 
   du[3] = w[8] * u[1] + w[9] + w[10] * u[3]
end
u0= [-1.31; -7.6; -0.2]
tspan = (0.0, 10.0)
wTrue = [10,-10,30,-10,10,-50,-10,0.04,0.0319,-0.01]
D = length(u0)
J = length(wTrue)
M = 1024
σ = 0.1 # snr for noise to data
μ = 0.1 # snr for initCond 
Random.seed!(1)
w0 = wTrue + μ .* abs.(wTrue) .* rand(J)
opt = OptimizationOptimJL.NewtonTrustRegion();
##
prob = ODEProblem(f, u0, tspan, wTrue)
sol = solve(prob, Rosenbrock23())
tt = collect(range(tspan[1], stop = tspan[end], length = M))
U_exact = reduce(hcat, sol(tt[i]) for i in 1:M)
U = U_exact + σ*sqrt(mean(U_exact.^2))*rand(D,M);
##
obj = build_loss_objective(prob, Rosenbrock23(), L2Loss(tt, U), Optimization.AutoForwardDiff())
optprob = Optimization.OptimizationProblem(obj, w0);
##
@info "Solving ODE param estimation problem with default ODEProb construction"
res = solve(optprob, opt, show_trace=true, show_every=100);
## try to use ModelingToolkit
## See Wendy paper
@mtkmodel HindmarshRoseModel begin
    @variables begin
        u1(t) = -1.31
        u2(t) = -7.6
        u3(t) = -0.2
    end
    @parameters begin
        w1 = 10
        w2 = -10
        w3 = 30
        w4 = -10
        w5 = 10
        w6 = -50
        w7 = -10
        w8 = 0.04
        w9 = 0.0319
        w10= -0.01
    end
    @equations begin
        D_nounits(u1) ~ w1 * u2 + w2 * u1^3 + w3 * u1^2 + w4 * u3 
        D_nounits(u2) ~ w5 + w6 * u1^2 + w7 * u2 
        D_nounits(u3) ~ w8 *u1 + w9 + w10 * u3
    end
end
@mtkbuild HINDMARSH_ROSE_SYSTEM = HindmarshRoseModel()
mtk_prob = ODEProblem(
    HINDMARSH_ROSE_SYSTEM, 
    ModelingToolkit.getdefault.(unknowns(HINDMARSH_ROSE_SYSTEM)), 
    tspan, 
    ModelingToolkit.getdefault.(parameters(HINDMARSH_ROSE_SYSTEM))
);
##
mtk_obj = build_loss_objective(mtk_prob, Rosenbrock23(), L2Loss(tt, U), Optimization.AutoForwardDiff())
mtk_optprob = Optimization.OptimizationProblem(mtk_obj, w0);
##
@info "Solving ODE param estimation problem with default MTK construction"
res = solve(mtk_optprob, opt, show_trace=true, show_every=100);