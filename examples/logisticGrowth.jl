## See Wendy paper
_LOGISTIC_T_RNG = (0, 10)
@mtkmodel LogisticGrowthModel begin
    @variables begin 
        u(t) = 0.01
    end
    @parameters begin
        w1 = 1
        w2 = -1
    end
    @equations begin
        D_nounits(u) ~ w1 * u + w2 * u^2
    end
end
@mtkbuild LOGISTIC_GROWTH_SYSTEM = LogisticGrowthModel()
LOGISTIC_GROWTH_FILE = joinpath(@__DIR__, "../data/LogisticGrowth.bson")
LOGISTIC_GROWTH = (
    name="logisticGrowth", 
    ode=LOGISTIC_GROWTH_SYSTEM,
    tRng=_LOGISTIC_T_RNG,M=1024,
    file=LOGISTIC_GROWTH_FILE
)