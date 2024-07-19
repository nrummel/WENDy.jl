# Found in https://en.wikipedia.org/wiki/Stiff_equation
# which references Robertson, H. H. (1966). "The solution of a set of reaction rate equations". Numerical analysis: an introduction. Academic Press. pp. 178–182.
# @mtkmodel Robertson begin
#     @variables begin 
#         x(t) = 1
#         y(t) = 0
#         z(t) = 0
#     end
#     @parameters begin
#         p₁ = 0.04
#         p₂ = 1e4
#         p₃ = 1e7
#         p₄ = 3e7
#         γ = 2
#         β = 2
#     end
#     @equations begin
#         D_nounits(x) ~ -p₁ * x + p₂* y * z 
#         D_nounits(y) ~ p₁ * x - p₂ * y * z - p₄ * y^β 
#         D_nounits(z) ~ 3 * p₃ * y^γ
#     end
# end
function ROBERTSON_f!(du, u, w, t)
    du[1] = -w[1] * u[1] + w[2]* u[2] * u[3] 
    du[2] = w[1] * u[1] - w[2] * u[2] * u[3] - w[3] * u[2]^w[4] 
    du[3] = w[3] * u[2]^w[4]
    nothing
end

function ROBERTSON_logf!(du, u, w, t)
    ROBERTSON_f!(du, u, w, t)
    du ./= u
    nothing
end

ROBERTSON_TRUE_PARAMS = [0.04, 1e4, 3e7, 2]
ROBERTSON_INIT_COND = [1,0,0]
ROBERTSON_TRNG = (0.0, 1e2)

ROBERTSON_ODE = ODEProblem(
    ROBERTSON_f!, 
    ROBERTSON_INIT_COND, 
    ROBERTSON_TRNG, 
    ROBERTSON_TRUE_PARAMS
)
@mtkbuild ROBERTSON_SYSTEM = modelingtoolkitize(ROBERTSON_ODE)

ROBERTSON = SimulatedWENDyData(
    "robertson", 
    ROBERTSON_SYSTEM,
    ROBERTSON_ODE,
    ROBERTSON_f!,
    ROBERTSON_INIT_COND,
    ROBERTSON_TRNG,
    ROBERTSON_TRUE_PARAMS;
    linearInParameters=Val(false),
    noiseDist=Val(LogNormal),
    forceOdeSolve=true
);