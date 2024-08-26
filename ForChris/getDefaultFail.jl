using ModelingToolkit, OrdinaryDiffEq
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
##
@show ModelingToolkit.getdefault.(parameters(HINDMARSH_ROSE_SYSTEM))
@show ModelingToolkit.getdefault.(unknowns(HINDMARSH_ROSE_SYSTEM))
##
function f(du, u, w, t)
    du[1] = w[1] * u[2] + w[2] * u[1]^3 + w[3] * u[1]^2 + w[4] * u[3] 
    du[2] = w[5] + w[6] * u[1]^2 + w[7] * u[2] 
    du[3] = w[8] * u[1] + w[9] + w[10] * u[3]
end
u0= [-1.31; -7.6; -0.2]
tspan = (0.0, 10.0)
wTrue = [10,-10,30,-10,10,-50,-10,0.04,0.0319,-0.01]
HINDMARSH_ROSE_ODE = ODEProblem(f, u0, tspan, wTrue)

@mtkbuild HINDMARSH_ROSE_SYSTEM_2 = modelingtoolkitize(HINDMARSH_ROSE_ODE)
##
@show ModelingToolkit.getdefault.(parameters(HINDMARSH_ROSE_SYSTEM_2))
@show ModelingToolkit.getdefault.(unknowns(HINDMARSH_ROSE_SYSTEM_2))