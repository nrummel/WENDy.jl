## Defaults values for params
!(@isdefined DEFAULT_TIME_SUBSAMPLE_RATE )&& const DEFAULT_TIME_SUBSAMPLE_RATE = Ref{Int}(2)
!(@isdefined DEFAULT_SEED )&& const DEFAULT_SEED = Ref{Int}(Int(1))
!(@isdefined DEFAULT_K_MAX )&& const DEFAULT_K_MAX = Ref{Int}(Int(5.0e3))
!(@isdefined DEFAULT_DIAG_REG )&& const DEFAULT_DIAG_REG = Ref{AbstractFloat}(1.0e-10)
!(@isdefined DEFAULT_NOISE_RATIO )&& const DEFAULT_NOISE_RATIO = Ref{AbstractFloat}(0.01)
!(@isdefined DEFAULT_TEST_FUN_RADII )&& const DEFAULT_TEST_FUN_RADII = Ref{AbstractVector{<:Int}}( 2 .^(0:3))
!(@isdefined DEFAULT_TEST_FUNCTION )&& const DEFAULT_TEST_FUNCTION = Ref{TestFunction}(ExponentialTestFun())
!(@isdefined DEFAULT_PRUNE_METHOD )&& const DEFAULT_PRUNE_METHOD = Ref{TestFunctionPruningMethod}(SingularValuePruningMethod(MtminRadMethod(),UniformDiscritizationMethod()))
!(@isdefined DEFAULT_FORCE_NONLINEAR )&& const DEFAULT_FORCE_NONLINEAR = Ref{Bool}(false)
!(@isdefined DEFAULT_NLS_ABSTOL )&& const DEFAULT_NLS_ABSTOL = Ref{Float64}
(1e-8)
!(@isdefined DEFAULT_NLS_RELTOL )&& const DEFAULT_NLS_RELTOL = Ref{Float64}(1e-8)
!(@isdefined DEFAULT_NLS_MAXITERS )&& const DEFAULT_NLS_MAXITERS = Ref{Int}(1000)
!(@isdefined DEFAULT_OPTIM_ABSTOL )&& const DEFAULT_OPTIM_ABSTOL = Ref{Float64}(1e-8)
!(@isdefined DEFAULT_OPTIM_RELTOL )&& const DEFAULT_OPTIM_RELTOL = Ref{Float64}(1e-8)
!(@isdefined DEFAULT_OPTIM_MAXITERS )&& const DEFAULT_OPTIM_MAXITERS = Ref{Int}(1000)
##
struct WENDyParameters
    timeSubsampleRate::Int 
    seed::Int   
    Kmax::Int   
    diagReg::AbstractFloat    
    noiseRatio::AbstractFloat        
    testFuctionRadii::AbstractVector{<:Real}           
    ϕ::TestFunction                  
    pruneMeth::TestFunctionPruningMethod   
    forceNonlinear::Bool  
    nlsAbstol::AbstractFloat  
    nlsReltol::AbstractFloat  
    nlsMaxiters::Int
    optimAbstol::AbstractFloat  
    optimReltol::AbstractFloat  
    optimMaxiters::Int
    function WENDyParameters(;
        timeSubsampleRate=DEFAULT_TIME_SUBSAMPLE_RATE[],
        seed=DEFAULT_SEED[],
        Kmax=DEFAULT_K_MAX[],
        diagReg=DEFAULT_DIAG_REG[],
        noiseRatio=DEFAULT_NOISE_RATIO[],
        testFuctionRadii=DEFAULT_TEST_FUN_RADII[],
        ϕ=DEFAULT_TEST_FUNCTION[],
        pruneMeth=DEFAULT_PRUNE_METHOD[],
        forceNonlinear=DEFAULT_FORCE_NONLINEAR[],
        nlsAbstol=DEFAULT_NLS_ABSTOL[],
        nlsReltol=DEFAULT_NLS_RELTOL[],
        nlsMaxiters=DEFAULT_NLS_MAXITERS[],
        optimAbstol=DEFAULT_OPTIM_ABSTOL[],
        optimReltol=DEFAULT_OPTIM_RELTOL[],
        optimMaxiters=DEFAULT_OPTIM_MAXITERS[],
    )
        @assert timeSubsampleRate >= 1
        @assert length(testFuctionRadii) >0 && all(testFuctionRadii .>= 1)
        new(timeSubsampleRate,seed,Kmax,diagReg,noiseRatio,testFuctionRadii,ϕ,pruneMeth,forceNonlinear)
    end
end
##
abstract type WENDyData{LinearInParameters,DistType<:Distribution} end 
## For observed data
struct EmpricalWENDyData{LinearInParameters,DistType}<:WENDyData{LinearInParameters,DistType}
    name::String
    ode::ODESystem
    tt_full::AbstractVector{<:Real}
    U_full::AbstractMatrix{<:Real}
end

function EmpricalWENDyData(name,ode,tt_full, U_full, ::Val{LinearInParameters}=Val(false), ::Val{DistType}=Val(Normal)) where {LinearInParameters,DistType<:Distribution}
    EmpricalWENDyData{LinearInParameters, DistType}(name, ode, tt_full, U_full)
end

struct SimulatedWENDyData{LinearInParameters,DistType}<:WENDyData{LinearInParameters,DistType}
    name::String
    ode::ODESystem
    tRng::NTuple{2,<:Real}
    M::Int
    file::String 
    trueParameters::AbstractVector{<:Real}
    initCond::AbstractVector{<:Real}
    tt_full::AbstractVector{<:Real}
    U_full::AbstractMatrix{<:Real}
end

function SimulatedWENDyData( 
    name::String,
    ode::ODESystem,
    tRng::NTuple{2,<:Real},
    M::Int;
    linearInParameters::Val{LinearInParameters}=Val(false),
    file::Union{String, Nothing}=nothing,
    trueParameters::Union{AbstractVector{<:Real}, Nothing}=nothing,
    initCond::Union{AbstractVector{<:Real}, Nothing}=nothing,
    noiseDist::Val{DistType}=Val(Normal)
) where {LinearInParameters, DistType<:Distribution}
    isnothing(file) && (file = joinpath(@__DIR__, "../data/$name.bson"))
    isnothing(trueParameters) && (trueParameters = [ModelingToolkit.getdefault(p) for p in parameters(ode)])
    isnothing(initCond) && (initCond = [ModelingToolkit.getdefault(p) for p in unknowns(ode)])
    tt_full, U_full = _getData(ode, tRng, M, trueParameters, initCond, file)
    if DistType == LogNormal && any(U_full .<= 0)
        ix = findall( all(U_full[:,m] .> 0) for m in 1:size(U_full,2))
        @warn " Removing data that is zero so that logrithms are well defined: $(length(tt_full) - length(ix)) data point(s) aree invalid"
        tt_full = tt_full[ix]
        U_full = U_full[:,ix]
    end
    SimulatedWENDyData{LinearInParameters,DistType}(name, ode, tRng, M,file,trueParameters,initCond,tt_full,U_full)
end