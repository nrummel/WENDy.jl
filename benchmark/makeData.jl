## Simulate the benchmark data once, in its own process, so the timed processes never solve a target ode before they
## are timed. Writes benchmark/data/<name>.jls with (tt, U, p₀).
##   julia --project=benchmark benchmark/makeData.jl
using WENDy, OrdinaryDiffEq, Random, Statistics, Serialization
include(joinpath(@__DIR__, "odes.jl"))
mkpath(joinpath(@__DIR__, "data"))
for (name, (f!, u₀, tRng, pstar, _)) in ODES
    rng = Xoshiro(1)
    tt = collect(range(tRng..., length=1025))[1:2:end] # Mp1 = 1025, subsampled by 2, as in wendyScripts
    sol = OrdinaryDiffEq.solve(ODEProblem(f!, u₀, tRng, Float64.(pstar)), Rodas5P(); saveat=tt, reltol=1e-12, abstol=1e-12)
    Uclean = reduce(vcat, u' for u in sol.u)
    U = Uclean .+ 0.05 .* std(Uclean, dims=1) .* randn(rng, size(Uclean)) # noise ratio 5%, Normal
    p₀ = pstar .* (1 .+ 0.1 .* randn(rng, length(pstar)))
    serialize(joinpath(@__DIR__, "data", "$name.jls"), (; tt, U, p₀=Float64.(p₀)))
    println(name, ": Mp1 = ", length(tt), ", D = ", size(U, 2), ", J = ", length(pstar))
end
