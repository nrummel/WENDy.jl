using Random, Logging, MAT
## Defaults values for params
!(@isdefined  DEFAULT_TIME_SUBSAMPLE_RATE )&& const DEFAULT_TIME_SUBSAMPLE_RATE = Ref{Int}(2)
!(@isdefined  DEFAULT_SEED )&& const DEFAULT_SEED = Ref{Int}(Int(1))
!(@isdefined  DEFAULT_K_MIN )&& const DEFAULT_K_MIN = Ref{Int}(10)
!(@isdefined  DEFAULT_K_MAX )&& const DEFAULT_K_MAX = Ref{Int}(Int(5.0e3))
!(@isdefined  DEFAULT_DIAG_REG )&& const DEFAULT_DIAG_REG = Ref{AbstractFloat}(1.0e-10)
!(@isdefined  DEFAULT_NOISE_RATIO )&& const DEFAULT_NOISE_RATIO = Ref{AbstractFloat}(0.01)
!(@isdefined  DEFAULT_TEST_FUN_RADII )&& const DEFAULT_TEST_FUN_RADII = Ref{AbstractVector{<:Int}}( 2 .^(0:3))
!(@isdefined  DEFAULT_TEST_FUNCTION )&& const DEFAULT_TEST_FUNCTION = Ref{TestFunction}(ExponentialTestFun())
!(@isdefined  DEFAULT_PRUNE_METHOD )&& const DEFAULT_PRUNE_METHOD = Ref{TestFunctionPruningMethod}(SingularValuePruningMethod(MtminRadMethod(),UniformDiscritizationMethod()))
!(@isdefined  DEFAULT_FORCE_NONLINEAR )&& const FORCE_NONLINEAR = Ref{Bool}(false)
##
struct WENDyParameters
    timeSubsampleRate::Int 
    seed::Int   
    Kmin::Int               
    Kmax::Int   
    diagReg::AbstractFloat    
    noiseRatio::AbstractFloat        
    testFuctionRadii::AbstractVector{<:Real}           
    ϕ::TestFunction                  
    pruneMeth::TestFunctionPruningMethod   
    forceNonlinear::Bool  
    function WENDyParameters(;
        timeSubsampleRate=DEFAULT_TIME_SUBSAMPLE_RATE[],
        seed=DEFAULT_SEED[],
        Kmin=DEFAULT_K_MIN[],
        Kmax=DEFAULT_K_MAX[],
        diagReg=DEFAULT_DIAG_REG[],
        noiseRatio=DEFAULT_NOISE_RATIO[],
        testFuctionRadii=DEFAULT_TEST_FUN_RADII[],
        ϕ=DEFAULT_TEST_FUNCTION[],
        pruneMeth=DEFAULT_PRUNE_METHOD[]
        forceNonlinear=DEFAULT_FORCE_NONLINEAR[]
    )
        @assert timeSubsampleRate >= 1
        @assert Kmin > 0
        @assert Kmax > Kmin 
        @assert length(testFuctionRadii) >0 && all(testFuctionRadii .>= 1)
        new(timeSubsampleRate,seed,Kmin,Kmax,diagReg,noiseRatio,testFuctionRadii,ϕ,pruneMeth,forceNonlinear)
    end
end
##
abstract type WENDyData end 
## For observed data
struct EmpricalWENDyData
    name::String
    ode::ODESystem 
    noiseDist::T 
    tt_full::AbstractVector{<:Real}
    U_full::AbstractMatrix{<:Real}
end

struct SimulatedWENDyData{T}<:WENDyData where T<:Distribution
    name::String
    ode::ODESystem 
    tRng::NTuple{2,<:Real}
    M::Int
    file::String 
    trueParameters::AbstractVector{<:Real}
    initCond::AbstractVector{<:Real}
    noiseDist::T 
    tt_full::AbstractVector{<:Real}
    U_full::AbstractMatrix{<:Real}
end

function SimulatedWENDyData( 
    name::String,
    ode::ODESystem,
    tRng::NTuple{2,<:Real},
    M::Int;
    file::Union{String, Nothing}=nothing,
    trueParameters::Union{AbstractVector{<:Real}, Nothing}=nothing,
    initCond::Union{AbstractVector{<:Real}, Nothing}=nothing,
    noiseDist::Distribution=Normal
)
    isnothing(file) && file = joinpath(@__DIR__, "../data/$name.bson")
    isnothing(trueParameters) && trueParameters = [ModelingToolkit.getdefault(p) for p in parameters(ode)]
    isnothing(initCond) && initCond = [ModelingToolkit.getdefault(p) for p in unknowns(ode)]
    tt_full, U_full = _getData(ode, file)
    if noiseDist == LogNormal && any(U_full .<= 0)
        ix = findall( all(U_full[:,m] .> 0) for m in 1:size(U_full,2))
        @warn " Removing data that is zero so that logrithms are well defined: $(lenth(tt_full) - length(ix)) data points aree invalid"
        tt_full = tt_full[ix]
        U_full = U_full[:,ix]
    end
    WENDyData(name, ode,tRng,M,file,params,initCond,noiseDist,tt_full,U_full)
end
##
abstract type WENDyProblem end 

struct LinearWENDyProblem <: WENDyProblem
    D::Int
    J::Int
    M::Int
    K::Int
    testFuctionRadii::Int
    b₀::AbstractVector{<:AbstractFloat}
    sig::AbstractVector{<:AbstractFloat}
    tt::AbstractVector{<:AbstractFloat}
    U::AbstractMatrix{<:AbstractFloat} 
    V::AbstractMatrix{<:AbstractFloat}
    Vp::AbstractMatrix{<:AbstractFloat}
    f!::Function 
    jacuf!::Function
    data::WENDyData
    # truth information
    wTrue::AbstractVector{<:AbstractFloat}
    sigTrue::AbstractVector{<:AbstractFloat}
    noise::AbstractMatrix{<:AbstractFloat}
end

function LinearWENDyProblem(data::T, params::WENDyParameters; ll:::Logging.LogLevel=Logging.Warn) where T<:WENDyData
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "Build julia functions from symbolic expressions of ODE..."
        noiseDist = data.noiseDist
        _,f!     = getRHS(data.ode, Val(noiseDist))
        _,jacuf! = getJacu(data.ode, Val(noiseDist));
        J = length(parameters(data.ode))
       
        testFuctionRadii = length(params.testFuctionRadii)
        @info "Subsample data and add noise..."
        tt = tt_full[1:params.timeSubsampleRate:end]
        U = U_full[:,1:params.timeSubsampleRate:end]
        D, M = size(U)
        U, noise, noise_ratio_obs, sigTrue = generateNoise(U, params.noiseRatio, Val(noiseDist)) 
        @info "============================="
        @info "Start of Algo..."
        @info "Estimate the noise in each dimension..."
        sig = estimate_std(U, Val(noiseDist))
        noiseEstRelErr = norm(sigTrue - sig) / norm(sigTrue)
        @info "  Relative Error in noise estimate $noiseEstRelErr"
        @info "Build test function matrices..."
        V, Vp, Vfull = params.pruneMeth(tt,U,params.ϕ,params.Kmin,params.Kmax,params.testFuctionRadii);
        K,_ = size(V)
        @info "Build right hand side to NLS..."
        b₀ = reshape(-Vp * U', K*D);

        wTrue = if T == SimulatedWENDyData 
            data.trueParameters 
        else
            NaN*ones(J)
        end

        return LinearWENDyProblem(
            D,J,M,K,testFuctionRadii,
            b₀,sig,tt,U,V,Vp,
            f!,jacuf!,
            data,
            wTrue,sigTrue,noise
        )
    end
end



struct NonlinearWENDyProblem <: WENDyProblem
    D::Int
    J::Int
    M::Int
    K::Int
    testFuctionRadii::Int
    b₀::AbstractVector{<:AbstractFloat}
    sig::AbstractVector{<:AbstractFloat}
    tt::AbstractVector{<:AbstractFloat}
    U::AbstractMatrix{<:AbstractFloat} 
    V::AbstractMatrix{<:AbstractFloat}
    Vp::AbstractMatrix{<:AbstractFloat}
    f!::Function 
    jacuf!::Function
    jacwf!::Function
    jacwjacuf!::Function
    heswf!::Function
    heswjacuf!::Function
    data::WENDyData
    ## Truth info
    sigTrue::AbstractVector{AbstractFloat}
    wTrue::AbstractVector{<:AbstractFloat}
    noise::AbstractMatrix{<:AbstractFloat}
end 

function NonlinearWENDyProblem(data::WENDyData, params::WENDyParameters; ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "Build julia functions from symbolic expressions of ODE..."
        noiseDist = data.noiseDist
        _,f!     = getRHS(data.ode, Val(noiseDist))
        _,jacuf! = getJacu(data.ode, Val(noiseDist));
        _,jacwf! = getJacw(data.ode, Val(noiseDist));
        _,jacwjacuf! = getJacwJacu(data.ode, Val(noiseDist));
        _,heswf! = getHesw(data.ode, Val(noiseDist));
        _,heswjacuf! = getHeswJacu(data.ode, Val(noiseDist));
        wTrue = data.trueParameters
        J = length(parameters(data.ode))
        testFuctionRadii = length(params.testFuctionRadii)
        @info "Subsample data and add noise..."
        tt = tt_full[1:params.timeSubsampleRate:end]
        U = U_full[:,1:params.timeSubsampleRate:end]
        D, M = size(U)
        U, noise, noise_ratio_obs, sigTrue = generateNoise(U, params.noiseRatio, Val(noiseDist)) 
        @info "============================="
        @info "Start of Algo..."
        @info "Estimate the noise in each dimension..."
        sig = estimate_std(U, Val(noiseDist))
        noiseEstRelErr = norm(sigTrue - sig) / norm(sigTrue)
        @info "  Relative Error in noise estimate $noiseEstRelErr"
        @info "Build test function matrices..."
        V, Vp, Vfull = params.pruneMeth(tt,U,params.ϕ,params.Kmin,params.Kmax,params.testFuctionRadii);
        K,_ = size(V)
        @info "Build right hand side to NLS..."
        b₀ = reshape(-Vp * U', K*D);

        return WENDyProblem(D, J, M, K, testFuctionRadii, sigTrue, wTrue, b₀, sig, tt, U, noise, V, Vp, f!, jacuf!, jacwf!,jacwjacuf!,heswf!, heswjacuf!)
    end
end
## For comparing against the matlab implementation
struct _MATLAB_WENDyProblem <: WENDyProblem
    D::Int
    J::Int
    M::Int
    K::Int
    wTrue::AbstractVector{<:AbstractFloat}
    b₀::AbstractVector{<:AbstractFloat}
    sig::AbstractVector{<:AbstractFloat}
    tt::AbstractVector{<:AbstractFloat}
    U::AbstractMatrix{<:AbstractFloat} 
    V::AbstractMatrix{<:AbstractFloat}
    Vp::AbstractMatrix{<:AbstractFloat}
    f!::Function 
    jacuf!::Function
    jacwf!::Function
    jacwjacuf!::Function
    heswf!::Function
    heswjacuf!::Function
    data::WENDyData
    _matdata::Dict
    function _MATLAB_WENDyProblem(data::WENDyData, ::Any=nothing; ll::Logging.LogLevel=Logging.Warn)
        with_logger(ConsoleLogger(stderr, ll)) do
            @info "Loading from MatFile"
            _matdata =  matread(data.matlab_file)
            U = Matrix(data["xobs"]')
            tt = data["tobs"][:]
            V = data["V"]
            Vp = data["Vp"]
            true_vec = data["true_vec"][:]
            sig_ests = data["sig_ests"][:]
            ##
            wTrue = true_vec[:]
            J = length(parameters(data.ode))
            @info "Build julia functions from symbolic expressions of ODE..."
            _,f!     = getRHS(data.ode)
            _,jacuf! = getJacu(data.ode);
            _,jacwf! = getJacw(data.ode);
            _,jacwjacuf! = getJacwJacu(data.ode);
            _,heswf! = getHesw(data.ode);
            _,heswjacuf! = getHeswJacu(data.ode);
            D, M = size(U)
            @info "============================="
            @info "Start of Algo..."
            @info "Estimate the noise in each dimension..."
            sig = estimate_std(U)
            @assert norm(sig -sig_ests) / norm(sig_ests) < 1e2*eps() "Out estimation of noise is wrong"
            @info "Build test function matrices..."
            ## TODO: check that our V/Vp is the same up to a rotation
            # V, Vp, Vfull = params.pruneMeth(tt,U,params.ϕ,params.Kmin,params.Kmax,params.testFuctionRadii);
            K,_ = size(V)
            @info "Build right hand side to NLS..."
            b₀ = reshape(-Vp * U', K*D);

            return new(D, J, M, K, wTrue, b₀, sig, tt, U, V, Vp, f!, jacuf!, jacwf!, jacwjacuf!,heswf!, heswjacuf!, data,_matdata)
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