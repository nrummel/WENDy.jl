##
@kwdef struct WENDyParameters   
    diagReg::Real                       = 1.0e-10
    radiusMinTime::Real                 = 0.01
    radiusMaxTime::Real                 = 2.0
    numRadii::Int                       = 100
    radiiParams::AbstractVector{<:Real} = 2 .^(0:3)
    testFunSubRate::Real                = 2.0
    maxTestFunCondNum::Real             = 1e2
    Kmax::Int                           = 200
    Kᵣ::Union{Nothing,Int}              = 100
    nlsAbstol::Real                     = 1e-8
    nlsReltol::Real                     = 1e-8
    nlsMaxiters::Int                    = 1000
    optimAbstol::Real                   = 1e-8
    optimReltol::Real                   = 1e-8
    optimMaxiters::Int                  = 200
    optimTimelimit::Real                = 200.0
end 

##
abstract type WENDyData{lip,DistType<:Distribution} end 

## 
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
## Constructor
function SimulatedWENDyData( 
    name::String,
    odeprob::ODEProblem,
    f!::Function,
    initCond::AbstractVector{<:Real},
    tRng::NTuple{2,<:Real},
    wTrue::AbstractVector{<:Real},
    paramRng::AbstractVector{<:Tuple};
    Mp1::Union{Nothing, Int}=1025,
    dt::Union{Nothing, Real}=nothing,
    linearInParameters::Val{lip}=Val(false), #linearInParameters
    file::Union{String, Nothing}=nothing,
    noiseDist::Val{DistType}=Val(Normal), #distributionType
    forceOdeSolve::Bool=false,
    ll::LogLevel=Warn
) where {lip, DistType<:Distribution}
    with_logger(ConsoleLogger(stderr, ll)) do 
        @assert DistType == Normal || DistType == LogNormal "Only LogNormal and Normal Noise distributions are supported"
        @assert !(isnothing(Mp1) && isnothing(dt)) "One must either set the dt or number of time points"
        isnothing(file) && (file = joinpath(@__DIR__, "../data/$name.bson"))
        if !isnothing(dt) 
            Mp1 = Int(floor((tRng[end]- tRng[1]) / dt)) + 1
        end

        tt_full, U_full = if forceOdeSolve || !isfile(file)
            @info "  Generating data for $file by solving ode"
            sol = _solve_ode(f!, tRng, Mp1, wTrue, initCond)
            t = sol.t
            u = reduce(hcat, sol.u)
            mkpath(dirname(file))
            BSON.@save file t u
            t, u
        else
            @info "Loading from file"
            data = BSON.load(file) 
            tt_full= data[:t] 
            U_full = data[:u] 
            tt_full, U_full 
        end
        ## handle the case for LogNormal Noise by removing any data where the value is zero or neg
        if DistType == LogNormal && any(U_full .<= 0)
            ix = findall( all(U_full[:,m] .> 0) for m in 1:size(U_full,2))
            @info " Removing data that is zero so that logrithms are well defined: $(length(tt_full) - length(ix)) data point(s) are invalid"
            tt_full = tt_full[ix]
            U_full = U_full[:,ix]
            initCond = U_full[:,1]
        end
        # this may be modified by the log normal handeling or from dt being used instead of Mp1
        tRng = (tt_full[1], tt_full[end])
        return SimulatedWENDyData{lip,DistType}(
            name, odeprob, f!, initCond, tRng, wTrue, paramRng, 
            Mp1, file, tt_full, U_full,
            # subsampled and noisey data needs to be simulated with the simulate! function
            nothing, nothing, nothing, nothing, nothing 
        )
    end
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
## Simulated data for testing
@kwdef struct SimulationParameters
    timeSubsampleRate::Int = 2
    seed::Union{Int,Nothing} = nothing
    noiseRatio::AbstractFloat = 0.01
    isotropic::Bool = true
end
## add noise and subsample data
function simulate!(data::SimulatedWENDyData{lip,DistType}, params::SimulationParameters; ll::LogLevel=Debug) where {lip, DistType<:Distribution}
    with_logger(ConsoleLogger(stdout,ll)) do 
        @info "Simulating ..."
        @info "  Subsample data with rate $(params.timeSubsampleRate)"
        tt = data.tt_full[1:params.timeSubsampleRate:end]
        U_exact = data.U_full[:,1:params.timeSubsampleRate:end]
        @info "  D,Mp1_full = $(size(data.U_full))"
        D, Mp1 = size(U_exact)
        @info "  D,Mp1 = $(size(U_exact))"
        @info "  Adding noise from Distribution $DistType and noise ratio $(params.noiseRatio)"
        U, noise, sigTrue = generateNoise(U_exact, params, Val(DistType))
        data.tt[] = tt
        data.U[] = U
        data.U_exact[] = U_exact
        data.sigTrue[] = sigTrue
        data.noise[] = noise
        nothing
    end
end 