## The odes of the compile latency benchmark. lotkaVolterra is only used to warm up (compile everything that does not
## depend on f!); the targets are timed after it.
function lotkaVolterra!(du, u, p, t)
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = p[3]*u[1]*u[2] - p[4]*u[2]
    nothing
end
function logistic!(du, u, p, t)
    du[1] = p[1]*u[1] - p[2]*u[1]^2
    nothing
end
function hindmarshRose!(du, u, p, t)
    du[1] = p[1]*u[2] - p[2]*u[1]^3 + p[3]*u[1]^2 - p[4]*u[3]
    du[2] = p[5] - p[6]*u[1]^2 - p[7]*u[2]
    du[3] = p[8]*u[1] + p[9] - p[10]*u[3]
    nothing
end
# name => (f!, u₀, tRng, p*, params); the same settings as wendyScripts' examples
const ODES = Dict(
    "lotkaVolterra" => (lotkaVolterra!, [1.0, 1.0], (0.0, 10.0), [1.5, 1.0, 1.0, 3.0], WENDyParameters()),
    "logistic" => (logistic!, [0.01], (0.0, 10.0), [1.0, 1.0], WENDyParameters(; Kmax=1000)),
    "hindmarshRose" => (hindmarshRose!, [-1.31, -7.6, -0.2], (0.0, 10.0), [10, 10, 30, 10, 10, 50, 10, 0.04, 0.0319, 0.01],
        WENDyParameters(; radiiParams=[4, 8, 16], optimTimelimit=500.0)),
)
