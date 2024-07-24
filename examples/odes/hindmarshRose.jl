## See Wendy paper
function HINDMARSH_f!(du, u, w, t)
    du[1] = w[1] * u[2] + w[2] * u[1]^3 + w[3] * u[1]^2 + w[4] * u[3] 
    du[2] = w[5] + w[6] * u[1]^2 + w[7] * u[2] 
    du[3] = w[8] * u[1] + w[9] + w[10] * u[3]
    nothing
end
HINDMARSH_INIT_COND = [-1.31; -7.6; -0.2]
HINDMARSH_TRNG = (0.0, 10.0)
HINDMARSH_TRUE_PARAMS = [10,-10,30,-10,10,-50,-10,0.04,0.0319,-0.01]

HINDMARSH_ROSE_ODE = ODEProblem(
    HINDMARSH_f!, 
    HINDMARSH_INIT_COND, 
    HINDMARSH_TRNG, 
    HINDMARSH_TRUE_PARAMS
)
@mtkbuild HINDMARSH_ROSE_SYSTEM = modelingtoolkitize(HINDMARSH_ROSE_ODE)

HINDMARSH_ROSE = SimulatedWENDyData(
    "hindmarshRose", 
    HINDMARSH_ROSE_SYSTEM,
    HINDMARSH_ROSE_ODE,
    HINDMARSH_f!,
    HINDMARSH_INIT_COND,
    HINDMARSH_TRNG,
    HINDMARSH_TRUE_PARAMS;
    linearInParameters=Val(true),
);