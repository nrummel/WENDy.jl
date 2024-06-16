@info " Loading wendySymbolics"
include("wendySymbolics.jl")
@info " Loading wendyNoise"
include("wendyNoise.jl")
@info " Loading wendyTestFunctions"
include("wendyTestFunctions.jl")
using Random, Logging
##
!(@isdefined  DEFAULT_TIME_SUBSAMPLE_RATE )&& const DEFAULT_TIME_SUBSAMPLE_RATE = Ref{Int}(2)
!(@isdefined  DEFAULT_SEED )&& const DEFAULT_SEED = Ref{Int}(Int(1))
!(@isdefined  DEFAULT_K_MIN )&& const DEFAULT_K_MIN = Ref{Int}(10)
!(@isdefined  DEFAULT_K_MAX )&& const DEFAULT_K_MAX = Ref{Int}(Int(5.0e3))
!(@isdefined  DEFAULT_DIAG_REG )&& const DEFAULT_DIAG_REG = Ref{AbstractFloat}(1.0e-10)
!(@isdefined  DEFAULT_NOISE_RATIO )&& const DEFAULT_NOISE_RATIO = Ref{AbstractFloat}(0.05)
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
        @assert diagReg >= 0
        @assert noiseRatio >= 0 
        @assert length(mtParams) >0 && all(mtParams .>= 1)
        new(timeSubsampleRate,seed,Kmin,Kmax,diagReg,noiseRatio,mtParams,ϕ,pruneMeth)
    end
end
##
struct WENDyProblem 
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
end 

function WENDyProblem(ex::NamedTuple,params::WENDyParameters;ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        wTrue = Float64[ModelingToolkit.getdefault(p) for p in parameters(ex.ode)]
        J = length(wTrue)
        @info "Build julia functions from symbolic expressions of ODE..."
        _,f!     = getRHS(ex.ode)
        _,jacuf! = getJacobian(ex.ode);
        _,jacwf! = getParameterJacobian(ex.ode);

        Random.seed!(params.seed)
        @info "Load data from file..."
        tt_full, U_full = getData(ex)
        numRad = length(params.mtParams)
        @info "Subsample data and add noise..."
        tt = tt_full[1:params.timeSubsampleRate:end]
        U = U_full[:,1:params.timeSubsampleRate:end]
        U, noise, noise_ratio_obs, sigTrue = generateNoise(U, params.noiseRatio)
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

        return WENDyProblem(D, J, M, K, numRad, sigTrue, wTrue, b0, sig, tt, U, noise, V, Vp, f!, jacuf!, jacwf!)
    end
end