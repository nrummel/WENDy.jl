## Compile latency of WENDy and output error least squares for a new rhs f!, after a warm-up on another rhs.
##   julia --project=benchmark benchmark/compileLatency.jl <target> <config> <rep> <out.tsv> [--trace <trace.txt>]
## target: logistic | hindmarshRose (data from benchmark/makeData.jl). config: base, or switches joined by "+"
## (cse, wrap, oejac; e.g. cse+wrap+oejac). One process per (target, config, rep), so nothing compiled for the target
## leaks between configurations. With --trace, every method compiled for the target is written to trace.txt
## (Base.@trace_compile); keep those runs out of the timings, the tracing slows compilation down.
## Appends one line per step to out.tsv:
##   target config rep step time compile_time recompile_time retcode estimate
const t_start = time()
using WENDy, OrdinaryDiffEq, Serialization, LinearAlgebra
using WENDy: WLS, IRLS, TrustRegion, HybridTrustRegionOELS
const t_load = time() - t_start
include(joinpath(@__DIR__, "odes.jl"))

target, config, rep, out = ARGS[1], ARGS[2], parse(Int, ARGS[3]), ARGS[4]
trace = (i = findfirst(==("--trace"), ARGS); isnothing(i) ? nothing : ARGS[i+1])
switches = config == "base" ? String[] : split(config, "+")
for s in switches
    s == "cse" ? (WENDy._CSE[] = true) : s == "wrap" ? (WENDy._WRAP[] = true) :
        s == "oejac" ? (WENDy._OE_SYMBOLIC[] = true) : error("unknown switch $s")
end
LinearAlgebra.BLAS.set_num_threads(1) # several processes run at once

## The steps, in this order (later methods reuse what earlier ones compiled): build both problems, then each method
## twice (first call = compile + run, second = run only).
const METHODS = [
    "wls" => (prob, oe, p₀, params) -> WENDy.solve(prob, p₀, params; alg=WLS()),
    "irls" => (prob, oe, p₀, params) -> WENDy.solve(prob, p₀, params; alg=IRLS()),
    "mle" => (prob, oe, p₀, params) -> WENDy.solve(prob, p₀, params; alg=TrustRegion()),
    "oels" => (prob, oe, p₀, params) -> WENDy.solve(oe, p₀, params),
    "hybrid" => (prob, oe, p₀, params) -> WENDy.solve(prob, p₀, params; alg=HybridTrustRegionOELS(oe)),
]
function steps(name; record)
    f!, _, _, pstar, params = ODES[name]
    (; tt, U, p₀) = deserialize(joinpath(@__DIR__, "data", "$name.jls"))
    J = length(pstar)
    t = @timed WENDyProblem(tt, U, f!, J; linearInParameters=true, params, ll=WENDy.Warn)
    record("build_wendy", t, "", Float64[])
    prob = t.value
    t = @timed OutputErrorProblem(tt, U, f!, J; params)
    record("build_oe", t, "", Float64[])
    oe = t.value
    for (m, run) in METHODS, call in ("first", "second")
        t = @timed run(prob, oe, p₀, params)
        record("$(m)_$call", t, string(t.value.retcode), t.value.p̂)
    end
end

# warm-up on a different rhs: compiles everything that does not depend on f!
t_warm = @elapsed steps("lotkaVolterra"; record=(args...) -> nothing)
lines = String[]
push!(lines, join((target, config, rep, "load", t_load, NaN, NaN, "", ""), '\t'))
push!(lines, join((target, config, rep, "warmup", t_warm, NaN, NaN, "", ""), '\t'))
record(step, t, retcode, p̂) = push!(lines, join((target, config, rep, step, t.time, t.compile_time, t.recompile_time,
    retcode, join(p̂, ',')), '\t'))
if isnothing(trace)
    steps(target; record)
else
    open(trace, "w") do io
        redirect_stderr(io) do
            Base.@trace_compile steps(target; record)
        end
    end
end
open(out, "a") do io
    foreach(l -> println(io, l), lines)
end
