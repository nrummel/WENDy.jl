## See https://tspace.libraru.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=48
function GOODWIN_f!(du,u,w,t)
    du[1] = w[1] / (w[3] + u[3]^w[4]) - w[2] *u[1]
    du[2] = w[5]*u[1]- w[6]*u[2]
    du[3] = w[7]*u[2]-w[8]*u[3]
    nothing
end
GOODWIN_TRNG = (0.0, 80.0)            
GOODWIN_INIT_COND = [0.3617, 0.9137, 1.3934]
GOODWIN_TRUE_PARAMS = [3.4884, 0.0969, 2.15, 10, 0.0969, 0.0581, 0.0969, 0.0775]

GOODWIN_ODE = ODEProblem(
    GOODWIN_f!, 
    GOODWIN_INIT_COND, 
    GOODWIN_TRNG, 
    GOODWIN_TRUE_PARAMS
)

@mtkbuild GOODWIN_SYSTEM = modelingtoolkitize(GOODWIN_ODE)

GOODWIN = SimulatedWENDyData(
    "goodwin", 
    GOODWIN_SYSTEM,
    GOODWIN_ODE,
    GOODWIN_f!,
    GOODWIN_INIT_COND,
    GOODWIN_TRNG,
    GOODWIN_TRUE_PARAMS;
    linearInParameters=Val(false), # NONLINEAR!
);
