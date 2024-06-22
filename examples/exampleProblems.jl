include("../src/wendyData.jl")
using Distributions
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
    file=LOGISTIC_GROWTH_FILE)
## See Wendy paper
_HINDMARSH_T_RNG = (0, 10)
@mtkmodel HindmarshRoseModel begin
    @variables begin
        u1(t) = -1.31
        u2(t) = -7.6
        u3(t) = -0.2
    end
    @parameters begin
        w1  = 10
        w2  = -10
        w3  = 30
        w4  = -10
        w5  = 10
        w6  = -50
        w7  = -10
        w8  = 0.04
        w9  = 0.0319
        w10 = -0.01
    end
    @equations begin
        D_nounits(u1) ~ w1 * u2 + w2 * u1^3 + w3 * u1^2 + w4 * u3 
        D_nounits(u2) ~ w5 + w6 * u1^2 + w7 * u2 
        D_nounits(u3) ~ w8 *u1 + w9 + w10 * u3
    end
end
@mtkbuild HINDMARSH_ROSE_SYSTEM = HindmarshRoseModel()
HINDMARSH_ROSE_FILE = joinpath(@__DIR__, "../data/HindmarshRose.bson")
MATLAB_HINDMARSH_ROSE_FILE = joinpath(@__DIR__, "../data/Hindmarsh_Rose_MATLAB.mat")
HINDMARSH_ROSE = (name="hindmarshRose", ode=HINDMARSH_ROSE_SYSTEM, tRng=_HINDMARSH_T_RNG,M=1024,file=HINDMARSH_ROSE_FILE,matlab_file=MATLAB_HINDMARSH_ROSE_FILE)
## Given in RAMSAY 2007
_FITZ_T_RNG = (0.0, 20.0)         # Time in ms 20
_FITZ_NUM_PTS = 400               # from the paper
_FTZ_PARAM_RNG = NaN              # Not mentioned in paper
@mtkmodel FitzHugNagumoModel begin
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
@mtkbuild FITZHUG_NAGUMO_SYSTEM = FitzHugNagumoModel()
FITZHUG_NAGUMO_FILE = joinpath(@__DIR__, "../data/FitzHug_Nagumo.bson")
FITZHUG_NAGUMO = (name="fitzHugNagumo", ode=FITZHUG_NAGUMO_SYSTEM, tRng=_FITZ_T_RNG,M=1024, file=FITZHUG_NAGUMO_FILE)
## See https://tspace.libraru.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=48
_LOOP_T_RNG = (0.0, 80.0)                  #
_LOOP_NLPARAM_RNG = [1e-2, 1e2]            # This is for σ and A?
_LOOP_NUM_PTS = 800                        # Not mentioned in paper
_LOOP_LPARAM_RNG = NaN                     # Not mentioned in paper
@mtkmodel LoopModel begin
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
@mtkbuild LOOP_SYSTEM = LoopModel()
LOOP_FILE = joinpath(@__DIR__, "../data/loop.bson")
LOOP = (name="loopModel", ode=LOOP_SYSTEM, tRng=_LOOP_T_RNG,M=1024, file=LOOP_FILE)
## See https://tspace.libraru.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=51
_MENDES_S_VALS = [0.1,0.46416,2.1544,10]
_MENDES_P_VALS = [0.05,0.13572,0.3684,1]
_MENDES_T_RNG = (0.0,120.0)
_MENDES_NUM_PTS = 120               # This was 8 to be small for the forward solves
_MENDES_HILL_PARAM_RNG = [1e-1,1e2] # q2,4,6,8,10,12 
_MENDES_PARAM_RNG = [1e−12, 1e6]    # everuthing else k, and q

@mtkmodel MendesModel begin
    @variables begin
        u1(t) = 0.66667
        u2(t) = 0.57254
        u3(t) = 0.41758
        u4(t) = 0.4
        u5(t) = 0.36409
        u6(t) = 0.29457
        u7(t) = 1.419
        u8(t) = 0.93464
    end
    # # TODO: Do something smarter here... 
    # @constants begin 
    #     S = 0.1
    #     P = 0.05 
    # end
    @parameters begin
        k1=1
        k2=1
        k3=1
        k4=1
        k5=1
        k6=1
        k7=0.1
        k8=0.1
        k9=0.1
        k10=0.1
        k11=0.1
        k12=0.1
        k13=1
        k14=1
        k15=1
        q1=1
        q3=1
        q5=1
        q7=1
        q9=1
        q11=1
        q2=2
        q4=2
        q6=2
        q8=2
        q10=2
        q12=2
        q13=1
        q14=1
        q15=1
        q16=1
        q17=1
        q18=1
        q19=1
        q20=1
        q21=1                                             
    end
    # define equations
    @equations begin
        D_nounits(u1) ~ k1 / (1 + q1*0.05^(q2) + q3 * (0.1)^(-q4))-k2 * u1 
        D_nounits(u2) ~ k3 / (1 + q5*0.05^(q6) + q7 * u7^(-q8))-k4 * u2 
        D_nounits(u3) ~ k5 / (1 + q9*0.05^(q10) + q11 * u8^(-q12))-k6 * u3 
        D_nounits(u4) ~ (k7 * u1) / (u1+q13)-k8 * u4 
        D_nounits(u5) ~ (k9 * u2) / (u2+q14)-k10 * u5 
        D_nounits(u6) ~ (k11 * u3) / (u3+q15)-k12 * u6 
        D_nounits(u7) ~ ((k13 * u4*((1) / (q16))*((0.1)-u7)) / (1+(((0.1)) / (q16))+((u7) / (q17)))
              - (k14 * u5*((1) / (q18))*(u7-u8)) / (1+((u7) / (q18))+((u8) / (q19))) )
        D_nounits(u8) ~ ((k14 * u5 * ((1) / (q18))*(u7-u8)) / (1+((u7) / (q18))+((u8) / (q19)))
              - (k15 * u6 * ((1) / (q20))*(u8-0.05)) / (1+((u8) / (q20))+((0.05) / (q21))))
    end
end
MENDES_EXAMPLES = [
    (
        name="mendes_S=$(S)_P=$P",
        ode = begin 
            @mtkbuild ode = MendesModel() # usually could set S, P here but now that they are constants we cannot 
            ode 
        end,
        tRng=_MENDES_T_RNG,
        M=1024,
        file=joinpath(@__DIR__, "../data/Mendes_S=$(S)_P=$P.bson"),
        noise_dist=LogNormal
    )
for S in _MENDES_S_VALS, P in _MENDES_P_VALS][:]

## 
EXAMPLES = vcat([LOGISTIC_GROWTH, HINDMARSH_ROSE, LOOP], MENDES_EXAMPLES)
