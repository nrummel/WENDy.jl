## See Wendy paper
_LOGISTIC_T_RNG = (0.0, 10.0)
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
LOGISTIC_GROWTH = SimulatedWENDyData(
    "LogisticGrowth", 
    LOGISTIC_GROWTH_SYSTEM,
    _LOGISTIC_T_RNG,
    1024;
    linearInParameters=Val(true)
);