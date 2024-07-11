## Following from https://docs.sciml.ai/ModelingToolkit/stable/examples/sparse_jacobians/
BRUSSELATOR_N = 5
xyd_brusselator = range(0, stop = 1, length = BRUSSELATOR_N)
BRUSSELATOR_PARAMS = [3.4, 1.0, 10.0]
BRUSSELATOR_T_RNG = (0.0, 11.5)
BRUSSELATOR_FILE = joinpath(@__DIR__, "../data/brusselator.bson")

brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
limit(a, N) = a == BRUSSELATOR_N + 1 ? 1 : a == 0 ? BRUSSELATOR_N : a
function brusselator_2d_loop(du, u, p, t)
    dx = step(xyd_brusselator)
    A, B, alpha= p
    alpha = alpha / dx^2
    @inbounds for I in CartesianIndices((BRUSSELATOR_N, BRUSSELATOR_N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = limit(i + 1, BRUSSELATOR_N), limit(i - 1, BRUSSELATOR_N), limit(j + 1, BRUSSELATOR_N),
        limit(j - 1, BRUSSELATOR_N)
        du[i, j, 1] = alpha * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                       4u[i, j, 1]) +
                      B + u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] +
                      brusselator_f(x, y, t)
        du[i, j, 2] = alpha * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                       4u[i, j, 2]) +
                      A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    u
end
BRUSSELATOR_INIT_COND = init_brusselator_2d(xyd_brusselator)

BRUSSELATOR_ODE = ODEProblem(
    brusselator_2d_loop, 
    BRUSSELATOR_INIT_COND, 
    BRUSSELATOR_T_RNG, 
    BRUSSELATOR_PARAMS
)
@mtkbuild BRUSSELATOR_SYS = modelingtoolkitize(BRUSSELATOR_ODE)
BRUSSELATOR = SimulatedWENDyData(
    "Brusselator", 
    BRUSSELATOR_SYS,
    BRUSSELATOR_T_RNG,
    1024;
    initCond=BRUSSELATOR_INIT_COND[:],
    trueParameters=BRUSSELATOR_PARAMS,
    linearInParameters=Val(true)
);