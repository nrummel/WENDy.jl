# From Mathematical Biology: An Introduction with Maple and Matlab on page 318
function SIR_f!(du, u, w, t)
    du[1] = -w[1] * u[1] + w[2] * u[2] + w[3] * u[3] 
    du[2] = w[1] * u[1] - w[2] * u[2] - w[4] * (1 - exp(-w[5] * t^2)) * u[2] 
    du[3] = w[4] * (1 - exp(-w[5] * t^2)) * u[2] - w[3] * u[3]
end

function _beta(h, tau=1.5)
    (h * exp(-h * tau)) / (1 - exp(-h * tau))
end

h = 1.99
ρ = 0.074
β = _beta(h)
R = 0.113
v = 0.0024 
SIR_TRUE_PARAMS = [h,ρ,β,R,v]
SIR_INIT_COND = [1,0,0]
SIR_TRNG = (0.0, 50.0)

SIR_ODE = ODEProblem(
    SIR_f!, 
    SIR_INIT_COND, 
    SIR_TRNG, 
    SIR_TRUE_PARAMS
)
@mtkbuild SIR_SYSTEM = modelingtoolkitize(SIR_ODE)

SIR = SimulatedWENDyData(
    "sir", 
    SIR_SYSTEM,
    SIR_ODE,
    SIR_f!,
    SIR_INIT_COND,
    SIR_TRNG,
    SIR_TRUE_PARAMS;
    linearInParameters=Val(false),
    noiseDist=Val(LogNormal),
    # forceOdeSolve=true
);