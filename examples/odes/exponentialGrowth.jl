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
EXPONENTIAL = SimulatedWENDyData(
    "ExponentialGrowth", 
    EXPONENTIAL_SYSTEM,
    _EXPONENTIAL_T_RNG,
    1024;
    linearInParameters=Val(true)
);