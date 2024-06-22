@info " Loading wendySymbolics"
include("wendySymbolics.jl")
@info " Loading wendyNoise"
include("wendyNoise.jl")
@info " Loading wendyTestFunctions"
include("wendyTestFunctions.jl")
using Random, Logging, MAT, NaNMath
## Defaults values for params
!(@isdefined  DEFAULT_TIME_SUBSAMPLE_RATE )&& const DEFAULT_TIME_SUBSAMPLE_RATE = Ref{Int}(2)
!(@isdefined  DEFAULT_SEED )&& const DEFAULT_SEED = Ref{Int}(Int(1))
!(@isdefined  DEFAULT_K_MIN )&& const DEFAULT_K_MIN = Ref{Int}(10)
!(@isdefined  DEFAULT_K_MAX )&& const DEFAULT_K_MAX = Ref{Int}(Int(5.0e3))
!(@isdefined  DEFAULT_DIAG_REG )&& const DEFAULT_DIAG_REG = Ref{AbstractFloat}(1.0e-10)
!(@isdefined  DEFAULT_NOISE_RATIO )&& const DEFAULT_NOISE_RATIO = Ref{AbstractFloat}(0.01)
!(@isdefined  DEFAULT_MT_PARAMS )&& const DEFAULT_MT_PARAMS = Ref{AbstractVector{<:Int}}( 2 .^(0:3))
!(@isdefined  DEFAULT_TEST_FUNCTION )&& const DEFAULT_TEST_FUNCTION = Ref{TestFunction}(ExponentialTestFun())
!(@isdefined  DEFAULT_PRUNE_METHOD )&& const DEFAULT_PRUNE_METHOD = Ref{TestFunctionPruningMethod}(SingularValuePruningMethod(MtminRadMethod(),UniformDiscritizationMethod()))
##
struct WENDyParameters
    timeSubsampleRate::Int 
    seed::Int   
    Kmin::Int               
    Kmax::Int   
    diagReg::AbstractFloat    
    noiseRatio::AbstractFloat        
    mtParams::AbstractVector{<:Real}           
    ϕ::TestFunction                  
    pruneMeth::TestFunctionPruningMethod   
    function WENDyParameters(;
        timeSubsampleRate=DEFAULT_TIME_SUBSAMPLE_RATE[],
        seed=DEFAULT_SEED[],
        Kmin=DEFAULT_K_MIN[],
        Kmax=DEFAULT_K_MAX[],
        diagReg=DEFAULT_DIAG_REG[],
        noiseRatio=DEFAULT_NOISE_RATIO[],
        mtParams=DEFAULT_MT_PARAMS[],
        ϕ=DEFAULT_TEST_FUNCTION[],
        pruneMeth=DEFAULT_PRUNE_METHOD[]
    )
        @assert timeSubsampleRate >= 1
        @assert Kmin > 0
        @assert Kmax > Kmin 
        @assert length(mtParams) >0 && all(mtParams .>= 1)
        new(timeSubsampleRate,seed,Kmin,Kmax,diagReg,noiseRatio,mtParams,ϕ,pruneMeth)
    end
end
##
abstract type AbstractWENDyProblem end 
struct WENDyProblem <: AbstractWENDyProblem
    D::Int
    J::Int
    M::Int
    K::Int
    numRad::Int
    sigTrue::AbstractFloat
    wTrue::AbstractVector{<:AbstractFloat}
    b0::AbstractVector{<:AbstractFloat}
    sig::AbstractVector{<:AbstractFloat}
    tt::AbstractVector{<:AbstractFloat}
    U::AbstractMatrix{<:AbstractFloat} 
    noise::AbstractMatrix{<:AbstractFloat}
    V::AbstractMatrix{<:AbstractFloat}
    Vp::AbstractMatrix{<:AbstractFloat}
    f!::Function 
    jacuf!::Function
    jacwf!::Function
    jacwjacuf!::Function
end 

function WENDyProblem(ex::NamedTuple,params::WENDyParameters;ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        wTrue = Float64[ModelingToolkit.getdefault(p) for p in parameters(ex.ode)]
        J = length(wTrue)
        @info "Build julia functions from symbolic expressions of ODE..."
        _,f!     = getRHS(ex.ode)
        _,jacuf! = getJacobian(ex.ode);
        _,jacwf! = getParameterJacobian(ex.ode);
        # f! = mendesf!
        # jacuf! = mendesJacuf!
        # jacwf! = mendesJacwf!
        _,jacwjacuf! = getDoubleJacobian(ex.ode);

        Random.seed!(params.seed)
        @info "Load data from file..."
        tt_full, U_full = getData(ex)
        numRad = length(params.mtParams)
        @info "Subsample data and add noise..."
        tt = tt_full[1:params.timeSubsampleRate:end]
        U = U_full[:,1:params.timeSubsampleRate:end]
        U, noise, noise_ratio_obs, sigTrue = :noise_dist in keys(ex) ? generateNoise(U, params.noiseRatio, Val(ex.noise_dist)) : generateNoise(U, params.noiseRatio)
        D, M = size(U)
        @info "============================="
        @info "Start of Algo..."
        @info "Estimate the noise in each dimension..."
        sig = estimate_std(U)
        @info "Build test function matrices..."
        V, Vp, Vfull = params.pruneMeth(tt,U,params.ϕ,params.Kmin,params.Kmax,params.mtParams);
        K,_ = size(V)
        @info "Build right hand side to NLS..."
        b0 = reshape(-Vp * U', K*D);

        return WENDyProblem(D, J, M, K, numRad, sigTrue, wTrue, b0, sig, tt, U, noise, V, Vp, f!, jacuf!, jacwf!,jacwjacuf!)
    end
end

function mendesf!(dudt, w, um)
    @inbounds begin
        dudt[1] = (+)((/)(w[3], (+)((+)(1, (*)(w[18], (^)(w[2], w[24]))), (*)(w[19], (^)(w[1], (*)(-1, w[25]))))), (*)((*)(-1, w[4]), um[1]))
        dudt[2] = (+)((/)(w[5], (+)((+)(1, (*)(w[20], (^)(w[2], w[26]))), (*)(w[21], (^)(um[7], (*)(-1, w[27]))))), (*)((*)(-1, w[6]), um[2]))
        dudt[3] = (+)((/)(w[7], (+)((+)(1, (*)(w[23], (^)(um[8], (*)(-1, w[29])))), (*)(w[22], (^)(w[2], w[28])))), (*)((*)(-1, w[8]), um[3]))
        dudt[4] = (+)((/)((*)(w[9], um[1]), (+)(w[30], um[1])), (*)((*)(-1, w[10]), um[4]))
        dudt[5] = (+)((/)((*)(w[11], um[2]), (+)(w[31], um[2])), (*)((*)(-1, w[12]), um[5]))
        dudt[6] = (+)((/)((*)(w[13], um[3]), (+)(w[32], um[3])), (*)((*)(-1, w[14]), um[6]))
        dudt[7] = (+)((/)((*)((*)((*)(-1, w[16]), um[5]), (+)((*)(-1, um[8]), um[7])), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])))), (/)((*)((*)((+)(w[1], (*)(-1, um[7])), w[15]), um[4]), (*)(w[33], (+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])))))
        dudt[8] = (+)((/)((*)((*)((+)(w[2], (*)(-1, um[8])), w[17]), um[6]), (*)(w[37], (+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])))), (/)((*)((*)(w[16], um[5]), (+)((*)(-1, um[8]), um[7])), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])))))
        nothing
    end
end

function mendesJacwf!(∇gm, w, um)
    ∇gm .= 0
    @inbounds begin 
        ∇gm[1] = (*)((*)((*)(w[19], w[25]), (/)(w[3], (^)((+)((+)(1, (*)(w[18], (^)(w[2], w[24]))), (*)(w[19], (^)(w[1], (*)(-1, w[25])))), 2))), (^)(w[1], (+)(-1, (*)(-1, w[25]))))
        ∇gm[7] = (+)((/)((*)(w[15], um[4]), (*)(w[33], (+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])))), (/)((*)((*)((*)(-1, (+)(w[1], (*)(-1, um[7]))), w[15]), um[4]), (*)((^)(w[33], 2), (^)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), 2))))
        ∇gm[9] = (*)((*)((*)((*)(-1, w[18]), w[24]), (/)(w[3], (^)((+)((+)(1, (*)(w[18], (^)(w[2], w[24]))), (*)(w[19], (^)(w[1], (*)(-1, w[25])))), 2))), (^)(w[2], (+)(-1, w[24])))
        ∇gm[10] = (*)((*)((*)((*)(-1, w[20]), w[26]), (^)(w[2], (+)(-1, w[26]))), (/)(w[5], (^)((+)((+)(1, (*)(w[20], (^)(w[2], w[26]))), (*)(w[21], (^)(um[7], (*)(-1, w[27])))), 2)))
        ∇gm[11] = (*)((*)((*)((*)(-1, w[28]), w[22]), (/)(w[7], (^)((+)((+)(1, (*)(w[23], (^)(um[8], (*)(-1, w[29])))), (*)(w[22], (^)(w[2], w[28]))), 2))), (^)(w[2], (+)(-1, w[28])))
        ∇gm[16] = (+)((/)((*)((*)(-1, w[37]), (/)((*)((*)((+)(w[2], (*)(-1, um[8])), w[17]), um[6]), (*)((^)(w[37], 2), (^)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), 2)))), w[38]), (/)((*)(w[17], um[6]), (*)(w[37], (+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])))))
        ∇gm[17] = (/)(1, (+)((+)(1, (*)(w[18], (^)(w[2], w[24]))), (*)(w[19], (^)(w[1], (*)(-1, w[25])))))
        ∇gm[25] = (*)(-1, um[1])
        ∇gm[34] = (/)(1, (+)((+)(1, (*)(w[20], (^)(w[2], w[26]))), (*)(w[21], (^)(um[7], (*)(-1, w[27])))))
        ∇gm[42] = (*)(-1, um[2])
        ∇gm[51] = (/)(1, (+)((+)(1, (*)(w[23], (^)(um[8], (*)(-1, w[29])))), (*)(w[22], (^)(w[2], w[28]))))
        ∇gm[59] = (*)(-1, um[3])
        ∇gm[68] = (/)(um[1], (+)(w[30], um[1]))
        ∇gm[76] = (*)(-1, um[4])
        ∇gm[85] = (/)(um[2], (+)(w[31], um[2]))
        ∇gm[93] = (*)(-1, um[5])
        ∇gm[102] = (/)(um[3], (+)(w[32], um[3]))
        ∇gm[110] = (*)(-1, um[6])
        ∇gm[119] = (/)((*)((+)(w[1], (*)(-1, um[7])), um[4]), (*)(w[33], (+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33]))))
        ∇gm[127] = (/)((*)((*)(-1, um[5]), (+)((*)(-1, um[8]), um[7])), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36]))))
        ∇gm[128] = (/)((*)(um[5], (+)((*)(-1, um[8]), um[7])), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36]))))
        ∇gm[136] = (/)((*)((+)(w[2], (*)(-1, um[8])), um[6]), (*)(w[37], (+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38]))))
        ∇gm[137] = (*)((*)(-1, (/)(w[3], (^)((+)((+)(1, (*)(w[18], (^)(w[2], w[24]))), (*)(w[19], (^)(w[1], (*)(-1, w[25])))), 2))), (^)(w[2], w[24]))
        ∇gm[145] = (*)((*)(-1, (/)(w[3], (^)((+)((+)(1, (*)(w[18], (^)(w[2], w[24]))), (*)(w[19], (^)(w[1], (*)(-1, w[25])))), 2))), (^)(w[1], (*)(-1, w[25])))
        ∇gm[154] = (*)((*)(-1, (/)(w[5], (^)((+)((+)(1, (*)(w[20], (^)(w[2], w[26]))), (*)(w[21], (^)(um[7], (*)(-1, w[27])))), 2))), (^)(w[2], w[26]))
        ∇gm[162] = (*)((*)(-1, (/)(w[5], (^)((+)((+)(1, (*)(w[20], (^)(w[2], w[26]))), (*)(w[21], (^)(um[7], (*)(-1, w[27])))), 2))), (^)(um[7], (*)(-1, w[27])))
        ∇gm[171] = (*)((*)(-1, (/)(w[7], (^)((+)((+)(1, (*)(w[23], (^)(um[8], (*)(-1, w[29])))), (*)(w[22], (^)(w[2], w[28]))), 2))), (^)(w[2], w[28]))
        ∇gm[179] = (*)((*)(-1, (^)(um[8], (*)(-1, w[29]))), (/)(w[7], (^)((+)((+)(1, (*)(w[23], (^)(um[8], (*)(-1, w[29])))), (*)(w[22], (^)(w[2], w[28]))), 2)))
        ∇gm[185] = (*)((*)((*)((*)(-1, w[18]), (/)(w[3], (^)((+)((+)(1, (*)(w[18], (^)(w[2], w[24]))), (*)(w[19], (^)(w[1], (*)(-1, w[25])))), 2))), NaNMath.log(w[2])), (^)(w[2], w[24]))
        ∇gm[193] = (*)((*)((*)(w[19], (/)(w[3], (^)((+)((+)(1, (*)(w[18], (^)(w[2], w[24]))), (*)(w[19], (^)(w[1], (*)(-1, w[25])))), 2))), (^)(w[1], (*)(-1, w[25]))), NaNMath.log(w[1]))
        ∇gm[202] = (*)((*)((*)((*)(-1, w[20]), NaNMath.log(w[2])), (/)(w[5], (^)((+)((+)(1, (*)(w[20], (^)(w[2], w[26]))), (*)(w[21], (^)(um[7], (*)(-1, w[27])))), 2))), (^)(w[2], w[26]))
        ∇gm[210] = (*)((*)((*)(w[21], (/)(w[5], (^)((+)((+)(1, (*)(w[20], (^)(w[2], w[26]))), (*)(w[21], (^)(um[7], (*)(-1, w[27])))), 2))), (^)(um[7], (*)(-1, w[27]))), NaNMath.log(um[7]))
        ∇gm[219] = (*)((*)((*)((*)(-1, w[22]), NaNMath.log(w[2])), (/)(w[7], (^)((+)((+)(1, (*)(w[23], (^)(um[8], (*)(-1, w[29])))), (*)(w[22], (^)(w[2], w[28]))), 2))), (^)(w[2], w[28]))
        ∇gm[227] = (*)((*)((*)(w[23], NaNMath.log(um[8])), (^)(um[8], (*)(-1, w[29]))), (/)(w[7], (^)((+)((+)(1, (*)(w[23], (^)(um[8], (*)(-1, w[29])))), (*)(w[22], (^)(w[2], w[28]))), 2)))
        ∇gm[236] = (*)(-1, (/)((*)(w[9], um[1]), (^)((+)(w[30], um[1]), 2)))
        ∇gm[245] = (*)(-1, (/)((*)(w[11], um[2]), (^)((+)(w[31], um[2]), 2)))
        ∇gm[254] = (*)(-1, (/)((*)(w[13], um[3]), (^)((+)(w[32], um[3]), 2)))
        ∇gm[263] = (*)((*)(-1, (+)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), (*)((*)(-1, w[33]), (/)(w[1], (^)(w[33], 2))))), (/)((*)((*)((+)(w[1], (*)(-1, um[7])), w[15]), um[4]), (*)((^)(w[33], 2), (^)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), 2))))
        ∇gm[271] = (*)((*)(w[33], (/)((*)((*)((+)(w[1], (*)(-1, um[7])), w[15]), um[4]), (*)((^)(w[33], 2), (^)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), 2)))), (/)(um[7], (^)(w[34], 2)))
        ∇gm[279] = (*)((*)(-1, (+)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), (*)((*)(-1, w[35]), (/)(um[7], (^)(w[35], 2))))), (/)((*)((*)((*)(-1, w[16]), um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
        ∇gm[280] = (*)((*)(-1, (+)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), (*)((*)(-1, w[35]), (/)(um[7], (^)(w[35], 2))))), (/)((*)((*)(w[16], um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
        ∇gm[287] = (*)((*)(w[35], (/)(um[8], (^)(w[36], 2))), (/)((*)((*)((*)(-1, w[16]), um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
        ∇gm[288] = (*)((*)(w[35], (/)(um[8], (^)(w[36], 2))), (/)((*)((*)(w[16], um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
        ∇gm[296] = (*)((*)(-1, (+)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), (*)((*)(-1, w[37]), (/)(um[8], (^)(w[37], 2))))), (/)((*)((*)((+)(w[2], (*)(-1, um[8])), w[17]), um[6]), (*)((^)(w[37], 2), (^)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), 2))))
        ∇gm[304] = (*)((*)(w[37], (/)((*)((*)((+)(w[2], (*)(-1, um[8])), w[17]), um[6]), (*)((^)(w[37], 2), (^)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), 2)))), (/)(w[2], (^)(w[38], 2)))
    end
    nothing
end


function mendesJacuf!(∇gm, w, um)
    ∇gm .= 0
    @inbounds begin
        ∇gm[1] = (*)(-1, w[4])
        ∇gm[4] = (+)((/)((*)((*)(-1, w[9]), um[1]), (^)((+)(w[30], um[1]), 2)), (/)(w[9], (+)(w[30], um[1])))
        ∇gm[10] = (*)(-1, w[6])
        ∇gm[13] = (+)((/)(w[11], (+)(w[31], um[2])), (/)((*)((*)(-1, w[11]), um[2]), (^)((+)(w[31], um[2]), 2)))
        ∇gm[19] = (*)(-1, w[8])
        ∇gm[22] = (+)((/)((*)((*)(-1, w[13]), um[3]), (^)((+)(w[32], um[3]), 2)), (/)(w[13], (+)(w[32], um[3])))
        ∇gm[28] = (*)(-1, w[10])
        ∇gm[31] = (/)((*)((+)(w[1], (*)(-1, um[7])), w[15]), (*)(w[33], (+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33]))))
        ∇gm[37] = (*)(-1, w[12])
        ∇gm[39] = (/)((*)((*)(-1, w[16]), (+)((*)(-1, um[8]), um[7])), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36]))))
        ∇gm[40] = (/)((*)(w[16], (+)((*)(-1, um[8]), um[7])), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36]))))
        ∇gm[46] = (*)(-1, w[14])
        ∇gm[48] = (/)((*)((+)(w[2], (*)(-1, um[8])), w[17]), (*)(w[37], (+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38]))))
        ∇gm[50] = (*)((*)((*)(w[21], w[27]), (/)(w[5], (^)((+)((+)(1, (*)(w[20], (^)(w[2], w[26]))), (*)(w[21], (^)(um[7], (*)(-1, w[27])))), 2))), (^)(um[7], (+)(-1, (*)(-1, w[27]))))
        ∇gm[55] = (+)((+)((+)((/)((*)((*)(-1, w[16]), um[5]), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])))), (/)((*)((*)(-1, w[33]), (/)((*)((*)((+)(w[1], (*)(-1, um[7])), w[15]), um[4]), (*)((^)(w[33], 2), (^)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), 2)))), w[34])), (/)((*)((*)(w[16], um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2)))), (/)((*)((*)(-1, w[15]), um[4]), (*)(w[33], (+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])))))
        ∇gm[56] = (+)((/)((*)(w[16], um[5]), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])))), (/)((*)((*)((*)(-1, w[16]), um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
        ∇gm[59] = (*)((*)((*)(w[23], w[29]), (/)(w[7], (^)((+)((+)(1, (*)(w[23], (^)(um[8], (*)(-1, w[29])))), (*)(w[22], (^)(w[2], w[28]))), 2))), (^)(um[8], (+)(-1, (*)(-1, w[29]))))
        ∇gm[63] = (+)((/)((*)((*)(-1, w[35]), (/)((*)((*)((*)(-1, w[16]), um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2)))), w[36]), (/)((*)(w[16], um[5]), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])))))
        ∇gm[64] = (+)((+)((+)((/)((*)((*)(-1, w[16]), um[5]), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])))), (/)((*)((*)((*)(-1, (+)(w[2], (*)(-1, um[8]))), w[17]), um[6]), (*)((^)(w[37], 2), (^)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), 2)))), (/)((*)((*)(-1, w[35]), (/)((*)((*)(w[16], um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2)))), w[36])), (/)((*)((*)(-1, w[17]), um[6]), (*)(w[37], (+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])))))
    end
    nothing
end

struct _MATLAB_WENDyProblem <: AbstractWENDyProblem
    D::Int
    J::Int
    M::Int
    K::Int
    wTrue::AbstractVector{<:AbstractFloat}
    b0::AbstractVector{<:AbstractFloat}
    sig::AbstractVector{<:AbstractFloat}
    tt::AbstractVector{<:AbstractFloat}
    U::AbstractMatrix{<:AbstractFloat} 
    V::AbstractMatrix{<:AbstractFloat}
    Vp::AbstractMatrix{<:AbstractFloat}
    f!::Function 
    jacuf!::Function
    jacwf!::Function
    jacwjacuf!::Function
    data::Dict
    function _MATLAB_WENDyProblem(ex::NamedTuple, ::Any=nothing; ll::Logging.LogLevel=Logging.Warn)
        with_logger(ConsoleLogger(stderr, ll)) do
            @info "Loading from MatFile "
            data =  matread(ex.matlab_file)
            U = Matrix(data["xobs"]')
            tt = data["tobs"][:]
            V = data["V"]
            Vp = data["Vp"]
            true_vec = data["true_vec"][:]
            sig_ests = data["sig_ests"][:]
            ##
            wTrue = true_vec[:]
            J = length(wTrue)
            @info "Build julia functions from symbolic expressions of ODE..."
            _,f!     = getRHS(ex.ode)
            _,jacuf! = getJacobian(ex.ode);
            _,jacwf! = getParameterJacobian(ex.ode);
            _,jacwjacuf! = getDoubleJacobian(ex.ode);
            D, M = size(U)
            @info "============================="
            @info "Start of Algo..."
            @info "Estimate the noise in each dimension..."
            sig = estimate_std(U)
            @assert norm(sig -sig_ests) / norm(sig_ests) < 1e2*eps() "Out estimation of noise is wrong"
            @info "Build test function matrices..."
            ## TODO: check that our V/Vp is the same up to a rotation
            # V, Vp, Vfull = params.pruneMeth(tt,U,params.ϕ,params.Kmin,params.Kmax,params.mtParams);
            K,_ = size(V)
            @info "Build right hand side to NLS..."
            b0 = reshape(-Vp * U', K*D);

            return new(D, J, M, K, wTrue, b0, sig, tt, U, V, Vp, f!, jacuf!, jacwf!, jacwjacuf!, data)
        end
    end 
end 
##
import Plots: plot
function plot(prob::WENDyProblem)
    D = prob.D 
    plot(
        prob.tt,
        [prob.U[d,:] for d in 1:D],
        label=["u_$d" for d in 1:D],
        title="WENDy Problem"
    )
    xlabel!("time")


end