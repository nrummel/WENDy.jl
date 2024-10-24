##
struct WENDyParameters
    Kmax::Int   
    diagReg::AbstractFloat    
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
    Kmax::Int=Int(5.0e3),
    diagReg::AbstractFloat=1.0e-10,
    testFuctionRadii::AbstractVector{<:Real}=2 .^(0:3),
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
    @assert length(testFuctionRadii) >0 && all(testFuctionRadii .>= 1)
    WENDyParameters(Kmax,diagReg,testFuctionRadii,ϕ,pruneMeth,nlsAbstol, nlsReltol, nlsMaxiters, optimAbstol, optimReltol, optimMaxiters, float(optimTimelimit))
end

##
abstract type WENDyData{lip,DistType<:Distribution} end 
## For observed data
struct EmpricalWENDyData{lip,DistType}<:WENDyData{lip,DistType}
    name::String
    odeprob::ODEProblem
    f!::Function
    initCond::AbstractVector{<:Real}
    tRng::NTuple{2,<:Real}
    tt_full::AbstractVector{<:Real}
    U_full::AbstractMatrix{<:Real}
    paramRng::AbstractVector{<:Tuple}
end
##
function _getData(f!::Function, tRng::NTuple{2,<:AbstractFloat}, Mp1::Int, wTrue::AbstractVector{<:Real}, initCond::AbstractVector{<:Real}, file::String; forceOdeSolve::Bool=false, ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stdout, ll)) do 
        if forceOdeSolve || !isfile(file)
            @info "  Generating data for $file by solving ode"
            sol = _solve_ode(f!, tRng, Mp1, wTrue, initCond)
            t = sol.t
            u = reduce(hcat, sol.u)
            mkpath(dirname(file))
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
## Simulated data for testing
@kwdef struct SimulationParameters
    timeSubsampleRate::Int = 2
    seed::Union{Int,Nothing} = nothing
    noiseRatio::AbstractFloat = 0.01
end

struct SimulatedWENDyData{lip,DistType}<:WENDyData{lip,DistType}
    name::String
    odeprob::ODEProblem
    f!::Function
    initCond::AbstractVector{<:Real}
    tRng::NTuple{2,<:Real}
    wTrue::AbstractVector{<:Real}
    paramRng::AbstractVector{<:Tuple}
    # defaul values provided
    Mp1::Int
    file::String 
    tt_full::AbstractVector{<:Real}
    U_full::AbstractMatrix{<:Real}
    ## subsampled
    tt::Ref{Union{Nothing,AbstractVector{<:Real}}}
    U::Ref{Union{Nothing,AbstractMatrix{<:Real}}}
    U_exact::Ref{Union{Nothing,AbstractMatrix{<:Real}}}
    sigTrue::Ref{Union{Nothing,AbstractVector{<:Real}}}
    noise::Ref{Union{Nothing,AbstractMatrix{<:Real}}}
end

function SimulatedWENDyData( 
    name::String,
    odeprob::ODEProblem,
    f!::Function,
    initCond::AbstractVector{<:Real},
    tRng::NTuple{2,<:Real},
    wTrue::AbstractVector{<:Real},
    paramRng::AbstractVector{<:Tuple};
    Mp1::Int=1025,
    linearInParameters::Val{lip}=Val(false), #linearInParameters
    file::Union{String, Nothing}=nothing,
    noiseDist::Val{DistType}=Val(Normal), #distributionType
    forceOdeSolve::Bool=false,
    ll::LogLevel=Warn
) where {lip, DistType<:Distribution}
    isnothing(file) && (file = joinpath(@__DIR__, "../data/$name.bson"))
    tt_full, U_exact = _getData(
        f!, tRng, Mp1, wTrue, initCond, file;
        forceOdeSolve=forceOdeSolve, ll=ll
    )
    @assert DistType == Normal || DistType == LogNormal "Only LogNormal and Normal Noise distributions are supported"

    if DistType == LogNormal && any(U_exact .<= 0)
        ix = findall( all(U_exact[:,m] .> 0) for m in 1:size(U_exact,2))
        @warn " Removing data that is zero so that logrithms are well defined: $(length(tt_full) - length(ix)) data point(s) are invalid"
        tt_full = tt_full[ix]
        U_exact = U_exact[:,ix]
        initCond = U_exact[:,1]
        tRng = (tt_full[1], tt_full[end])
    end

    SimulatedWENDyData{lip,DistType}(
        name, odeprob, f!, initCond, tRng, wTrue, paramRng, 
        Mp1, file, tt_full, U_exact,
        nothing, nothing, nothing, nothing, nothing # subsampled and noisey data needs to be simulated with the simulate! function
    )
end
## Change data's lip or dist type
function SimulatedWENDyData(data::SimulatedWENDyData{old_lip, old_DistType}, ::Val{new_lip}=Val(nothing), ::Val{new_DistType}=Val(nothing)) where {old_lip, old_DistType, new_DistType, new_lip}
    lip = isnothing(new_lip) ? old_lip : new_lip 
    DistType = isnothing(new_DistType) ? old_DistType : new_DistType 
    SimulatedWENDyData(
        data.name, data.odeprob, data.f!, data.initCond, data.tRng, data.wTrue; 
        Mp1=data.Mp1, file=data.file, linearInParameters=Val(lip), noiseDist=Val(DistType)
    )
end

## add noise and subsample data
function simulate!(data::SimulatedWENDyData{lip,DistType}, params::SimulationParameters; ll::LogLevel=Warn) where {lip, DistType<:Distribution}
    with_logger(ConsoleLogger(stdout,ll)) do 
        @info "Simulated Data"
        @info "  Subsample data "
        tt = data.tt_full[1:params.timeSubsampleRate:end]
        U_exact = data.U_full[:,1:params.timeSubsampleRate:end]
        D, Mp1 = size(U_exact)
        @info "Adding noise from Distribution $DistType"
        U, noise, _, sigTrue = generateNoise(U_exact, params, Val(DistType))
        data.tt[] = tt
        data.U[] = U
        data.U_exact[] = U_exact
        data.sigTrue[] = sigTrue
        data.noise[] = noise
        nothing
    end
end 