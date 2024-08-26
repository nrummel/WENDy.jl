## Given in RAMSAY 2007
_FITZ_T_RNG = (0.0, 20.0)         # Time in ms 20
_FITZ_NUM_PTS = 400               # from the paper
_FTZ_PARAM_RNG = NaN              # Not mentioned in paper
@mtkmodel FitzHugNagumo begin
    @variables begin
        u1(t) = -1.0
        u2(t) = 1
    end
    @parameters begin
        a = 0.2
        b = 0.2
        c = 3
    end
    # define equations
    @equations begin
        D_nounits(u1) ~ c * (u1 - u1^3 / 3 + u2);
        D_nounits(u2) ~ -1 / c * (u1 - a + b * u2);
    end
end
@mtkbuild FITZHUG_NAGUMO_SYSTEM = FitzHugNagumo()
FITZHUG_NAGUMO = SimulatedWENDyData(
    "fitzHugNagumo",
    FITZHUG_NAGUMO_SYSTEM,
    _FITZ_T_RNG,
    1024,
);