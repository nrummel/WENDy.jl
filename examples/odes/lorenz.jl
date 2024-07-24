## See [ModelingToolkit.jl/examples](https://docs.sciml.ai/ModelingToolkit/stable/examples/higher_order/)
_LORENZ_T_RNG = (0.0, 100.0)
@mtkmodel Lorenz begin
    @variables begin 
        xₜ(t) = 2.0
        y(t) = 1.0
        z(t) = 0.0
        x(t) = 0.0
    end
    @parameters begin
        σ = 28.0
        ρ = 10.0
        β = 8.0 / 3.0
    end
    @equations begin
        D_nounits(xₜ) ~ (-x + y)*σ
        D_nounits(y) ~ -y + x*(-z + ρ)
        D_nounits(z) ~ x*y-z*β
        D_nounits(x) ~ xₜ
    end
end
@mtkbuild LORENZ_SYSTEM = Lorenz()
LORENZ = SimulatedWENDyData(
    "Lorenz", 
    LORENZ_SYSTEM,
    _LORENZ_T_RNG,
    1024;
    linearInParameters=Val(true)
);