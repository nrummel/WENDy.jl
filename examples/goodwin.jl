## See https://tspace.libraru.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=48
_GOODWIN_T_RNG = (0.0, 80.0)                  #
_GOODWIN_NLPARAM_RNG = [1e-2, 1e2]            # This is for σ and A?
_GOODWIN_NUM_PTS = 800                        # Not mentioned in paper
_GOODWIN_LPARAM_RNG = NaN                     # Not mentioned in paper
@mtkmodel GoodwinModel begin
    @variables begin
        u1(t) = 0.3617
        u2(t) = 0.9137
        u3(t) = 1.3934
    end
    @parameters begin
        a = 3.4884
        b = 0.0969
        A = 2.15
        σ = 10
        α = 0.0969
        β = 0.0581
        γ = 0.0969
        δ = 0.0775
    end
    # define equations
    @equations begin
        D_nounits(u1) ~ (a) / (A + u3^σ) - b *u1
        D_nounits(u2) ~ α*u1- β*u2
        D_nounits(u3) ~ γ*u2-δ*u3
    end
end
@mtkbuild GOODWIN_SYSTEM = GoodwinModel()
GOODWIN_FILE = joinpath(@__DIR__, "../data/goodwin.bson")
GOODWIN = SimulatedWENDyData(
    "Goodwin", 
    GOODWIN_SYSTEM, 
    _GOODWIN_T_RNG,
    1024, 
);