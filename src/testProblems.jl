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
        D(u1) ~ c * (u1 - u1^3 / 3 + u2);
        D(u2) ~ -1 / c * (u1 - a + b * u2);
    end
end
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
        D(u1) ~ (a) / (A + u3^σ) - b *u1
        D(u2) ~ α*u1- β*u2
        D(u3) ~ γ*u2-δ*u3
    end
end

## See https://tspace.libraru.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=51
_MENDES_S_VALS = [0.1,0.46416,2.1544,10]
_MENDES_P_VALS = [0.05,0.13572,0.3684,1]
_MENDES_TRUE_PARAMS = Dict(
   
)
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
    @parameters begin
        S
        P
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
        D(u1) ~ k1 / (1+((P) / (q1))^(q2)+(q3 / S)^(q4))-k2 * u1 
        D(u2) ~ k3 / (1+((P) / (q5))^(q6)+(q7 / u7)^(q8))-k4 * u2 
        D(u3) ~ k5 / (1+((P) / (q9))^(q10)+(q11 / u8)^(q12))-k6 * u3 
        D(u4) ~ (k7 * u1) / (u1+q13)-k8 * u4 
        D(u5) ~ (k9 * u2) / (u2+q14)-k10 * u5 
        D(u6) ~ (k11 * u3) / (u3+q15)-k12 * u6 
        D(u7) ~ ((k13 * u4*((1) / (q16))*(S-u7)) / (1+((S) / (q16))+((u7) / (q17)))
              - (k14 * u5*((1) / (q18))*(u7-u8)) / (1+((u7) / (q18))+((u8) / (q19))) )
        D(u8) ~ ((k14 * u5 * ((1) / (q18))*(u7-u8)) / (1+((u7) / (q18))+((u8) / (q19)))
              - (k15 * u6 * ((1) / (q20))*(u8-P)) / (1+((u8) / (q20))+((P) / (q21))))
    end
end
## Function used to odes problems in the format used here
function _solve_ode(mdl::ODESystem, t_rng::Tuple, num_pts::Int;
    alg=FBDF(), reltol::Real= 1e-8,abstol::Real=1e-8)
    # Build parameter dictionary from the "true params" 
    params = [p => ModelingToolkit.getdefault(p) for p in parameters(mdl)]
    # Build the initial condition from the true initial condition
    init_cond = [ic=>ModelingToolkit.getdefault(ic) for ic in unknowns(mdl)]
    p = ODEProblem(mdl,init_cond , t_rng, params)

    t_step = (t_rng[end]-t_rng[1])/num_pts
    return solve(p, alg,reltol=reltol, abstol = abstol, saveat=t_step)
end
##
function create_test_data(;ll::Logging.LogLevel=Logging.Warn, saveFlag::Bool=false, NumberOfPoints::Int=1024)
    with_logger(ConsoleLogger(stderr,ll)) do
        @info "Creating Fitz Data..."
        @mtkbuild fitzModel = FitzHugNagumoModel()
        fitz_sol = _solve_ode(fitzModel, _FITZ_T_RNG, NumberOfPoints)
        @info "Creating Loop Data..."
        @mtkbuild loopMdl = LoopModel()
        loop_sol = _solve_ode(loopMdl, _LOOP_T_RNG, NumberOfPoints)
        @info "Creating Mendes Data..."
        mendes_sol = Matrix(undef, length(_MENDES_S_VALS), length(_MENDES_P_VALS))
        for (i,S) in enumerate(_MENDES_S_VALS), (j,P) in enumerate(_MENDES_P_VALS)
            @info "\tS=$S, P=$P"
            @mtkbuild mendesMdl = MendesModel(S=S,P=P)
            mendes_sol[i,j] = _solve_ode(mendesMdl, _MENDES_T_RNG, NumberOfPoints)
        end
        if saveFlag
            BSON.@save joinpath(@__DIR__, "..", "FitzHug_Nagumo.bson") fitz_sol
            BSON.@save joinpath(@__DIR__, "..", "Loop_Model.bson") loop_sol
            BSON.@save joinpath(@__DIR__, "..", "Mendes_Problem.bson") mendes_sol
        end
        return fitz_sol, loop_sol, mendes_sol
    end
end
