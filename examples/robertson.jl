# Found in https://en.wikipedia.org/wiki/Stiff_equation
# which references Robertson, H. H. (1966). "The solution of a set of reaction rate equations". Numerical analysis: an introduction. Academic Press. pp. 178–182.
@mtkmodel Robertson begin
    @variables begin 
        x(t) = 1
        y(t) = 0
        z(t) = 0
    end
    @parameters begin
        p₁ = 0.04
        p₂ = 1e4
        p₃ = 1e7
        p₄ = 3e7
        γ = 2
        β = 2
    end
    @equations begin
        D_nounits(x) ~ -p₁ * x + p₂* y * z 
        D_nounits(y) ~ p₁ * x - p₂ * y * z - p₄ * y^β 
        D_nounits(z) ~ 3 * p₃ * y^γ
    end
end
@mtkbuild ROBERTSON_SYSTEM = Robertson()
_ROBERTSON_T_RNG = (0, 1e2)
ROBERTSON_FILE = joinpath(@__DIR__, "../data/Robertson.bson")
ROBERTSON = (
    name="Robertson", 
    ode=ROBERTSON_SYSTEM,
    tRng=_ROBERTSON_T_RNG,
    M=1025,
    file=ROBERTSON_FILE, 
    noise_dist=LogNormal
);