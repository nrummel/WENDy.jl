## See https://tspace.libraru.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=48
function MULTIMODAL_f!(du,u,w,t)
    du[1] = w[1] / (36 + w[2]*u[2]) - w[3] 
    du[2] = w[4]*u[1] - w[5]
    nothing
end
MULTIMODAL_TRNG = (0.0, 60.0)            
MULTIMODAL_INIT_COND = [7.0, -10.0]
MULTIMODAL_TRUE_PARAMS = [72.0, 1.0, 2.0, 1.0, 1.0]

MULTIMODAL_ODE = ODEProblem(
    MULTIMODAL_f!, 
    MULTIMODAL_INIT_COND, 
    MULTIMODAL_TRNG, 
    MULTIMODAL_TRUE_PARAMS
)

@mtkbuild MULTIMODAL_SYSTEM = modelingtoolkitize(MULTIMODAL_ODE)

MULTIMODAL = SimulatedWENDyData(
    "MULTIMODAL", 
    MULTIMODAL_SYSTEM,
    MULTIMODAL_ODE,
    MULTIMODAL_f!,
    MULTIMODAL_INIT_COND,
    MULTIMODAL_TRNG,
    MULTIMODAL_TRUE_PARAMS;
    linearInParameters=Val(false), # NONLINEAR!
    forceOdeSolve=true
);
