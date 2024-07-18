##
struct WENDyParameters
    timeSubsampleRate::Int 
    seed::Union{Int,Nothing}   
    Kmax::Int   
    diagReg::AbstractFloat    
    noiseRatio::AbstractFloat        
    testFuctionRadii::AbstractVector{<:Real}           
    ϕ::TestFunction                  
    pruneMeth::TestFunctionPruningMethod   
    nlsAbstol::AbstractFloat  
    nlsReltol::AbstractFloat  
    nlsMaxiters::Int
    optimAbstol::AbstractFloat  
    optimReltol::AbstractFloat  
    optimMaxiters::Int
    optimTimelimit::AbstractFloat
end
## Defaults values for params are set in the kwargs
function WENDyParameters(;
    timeSubsampleRate::Int=2,
    seed::Union{Int,Nothing}=nothing,
    Kmax::Int=Int(5.0e3),
    diagReg::AbstractFloat=1.0e-10,
    noiseRatio::AbstractFloat=0.01,
    testFuctionRadii::AbstractVector{Int}=Int.(2 .^(0:3)),
    ϕ::TestFunction=ExponentialTestFun(),
    pruneMeth::TestFunctionPruningMethod=SingularValuePruningMethod(MtminRadMethod(),UniformDiscritizationMethod()),
    nlsAbstol::AbstractFloat=1e-8,
    nlsReltol::AbstractFloat=1e-8,
    nlsMaxiters::Int=1000,
    optimAbstol::AbstractFloat=1e-8,
    optimReltol::AbstractFloat=1e-8,
    optimMaxiters::Int=200,
    optimTimelimit::Real=200.0,
)
    @assert timeSubsampleRate >= 1
    @assert length(testFuctionRadii) >0 && all(testFuctionRadii .>= 1)
    WENDyParameters(timeSubsampleRate,seed,Kmax,diagReg,noiseRatio,testFuctionRadii,ϕ,pruneMeth,nlsAbstol, nlsReltol, nlsMaxiters, optimAbstol, optimReltol, optimMaxiters, float(optimTimelimit))
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
##
using Plots
function _getData(f!::Function, tRng::NTuple{2,<:AbstractFloat}, M::Int, trueParameters::AbstractVector{<:Real}, initCond::AbstractVector{<:Real}, file::String; forceOdeSolve::Bool=false, ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stdout, ll)) do 
        if forceOdeSolve || !isfile(file)
            @info "  Generating data for $file by solving ode"
            sol = _solve_ode(f!, tRng, M, trueParameters, initCond)
            # plot(sol,title="Hindmarsh Rose Solution")
            # Plots.savefig(joinpath("/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig", "hindmarshRose_sol.png"))
            u = reduce(hcat, sol.u)
            t = sol.t
            BSON.@save file t u
            return t,u, sol
        end
        @info "Loading from file"
        data = BSON.load(file) 
        tt_full = data[:t] 
        U_full = data[:u] 
        return tt_full, U_full
    end
end
##
struct SimulatedWENDyData{LinearInParameters,DistType}<:WENDyData{LinearInParameters,DistType}
    name::String
    ode::ODESystem
    odeprob::ODEProblem
    f!::Function
    initCond::AbstractVector{<:Real}
    tRng::NTuple{2,<:Real}
    trueParameters::AbstractVector{<:Real}
    # defaul values provided
    M::Int
    file::String 
    tt_full::AbstractVector{<:Real}
    U_exact::AbstractMatrix{<:Real}
end

function SimulatedWENDyData( 
    name::String,
    ode::ODESystem,
    odeprob::ODEProblem,
    f!::Function,
    initCond::AbstractVector{<:Real},
    tRng::NTuple{2,<:Real},
    trueParameters::AbstractVector{<:Real};
    M::Int=1024,
    linearInParameters::Val{LinearInParameters}=Val(false),
    file::Union{String, Nothing}=nothing,
    noiseDist::Val{DistType}=Val(Normal),
    forceOdeSolve::Bool=false
) where {LinearInParameters, DistType<:Distribution}
    isnothing(file) && (file = joinpath(@__DIR__, "../data/$name.bson"))
    tt_full, U_exact = _getData(
        f!, tRng, M, trueParameters, initCond, file;
        forceOdeSolve=forceOdeSolve, ll=Info
    )
    @assert DistType == Normal || DistType == LogNormal "Only LogNormal and Normal Noise distributions are supported"

    if DistType == LogNormal && any(U_exact .<= 0)
        ix = findall( all(U_exact[:,m] .> 0) for m in 1:size(U_exact,2))
        @warn " Removing data that is zero so that logrithms are well defined: $(length(tt_full) - length(ix)) data point(s) aree invalid"
        tt_full = tt_full[ix]
        U_exact = U_exact[:,ix]
        initCond = U_exact[:,1]
        tRng = (tt_full[1], tt_full[end])
    end

    SimulatedWENDyData{LinearInParameters,DistType}(
        name, ode, odeprob, f!, initCond, tRng, trueParameters, M, file, tt_full, U_exact
    )
end

function SimulatedWENDyData(data::SimulatedWENDyData{<:Any, DistType}, ::Val{LinearInParameters}) where {DistType<:Distribution, LinearInParameters}
    SimulatedWENDyData{LinearInParameters,DistType}(
        data.name, data.ode, data.odeprob, data.f!, data.initCond, data.tRng, data.trueParameters, data.M, data.file, data.tt_full, data.U_exact
    )
end