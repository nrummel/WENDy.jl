## Exponential growth (mod )
## See [ModelingToolkit.jl/examples](https://docs.sciml.ai/ModelingToolkit/stable/examples/higher_order/)
_EXPONENTIAL_T_RNG = (0.0, 2.0)
@mtkmodel ExponentialGrowth begin
    @variables begin 
        u(t) = 2.0
    end
    @parameters begin
        λ = 1.1
    end
    @equations begin
        D_nounits(u) ~ λ^2 * u
    end
end
@mtkbuild EXPONENTIAL_SYSTEM = ExponentialGrowth()
EXPONENTIAL_FILE = joinpath(@__DIR__, "../data/Exponential.bson")
EXPONENTIAL = (
    name="ExponentialGrowth", 
    ode=EXPONENTIAL_SYSTEM,
    tRng=_EXPONENTIAL_T_RNG,
    M=1024,
    file=EXPONENTIAL_FILE
);