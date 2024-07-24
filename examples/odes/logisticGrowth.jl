## See Wendy paper

function LOGISTIC_f!(du, u, w, t)
    du[1] = w[1] * u[1] + w[2] * u[1]^2
    nothing
end
LOGISTIC_TRNG = (0.0, 10.0)
LOGISTIC_INIT_COND = [0.01]
LOGISTIC_TRUE_PARAMS = [1,-1]

LOGISTIC_ODE = ODEProblem(
    LOGISTIC_f!, 
    LOGISTIC_INIT_COND, 
    LOGISTIC_TRNG, 
    LOGISTIC_TRUE_PARAMS
)

@mtkbuild LOGISTIC_SYSTEM = modelingtoolkitize(LOGISTIC_ODE)

LOGISTIC_GROWTH = SimulatedWENDyData(
    "LogisticGrowth", 
    LOGISTIC_SYSTEM,
    LOGISTIC_ODE,
    LOGISTIC_f!,
    LOGISTIC_INIT_COND,
    LOGISTIC_TRNG,
    LOGISTIC_TRUE_PARAMS;
    linearInParameters=Val(true),
);
