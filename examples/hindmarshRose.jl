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