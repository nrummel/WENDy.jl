using Random, Logging, MAT
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
    sigTrue::AbstractVector{AbstractFloat}
    wTrue::AbstractVector{<:AbstractFloat}
    b₀::AbstractVector{<:AbstractFloat}
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
    heswf!::Function
    heswjacuf!::Function
end 

function WENDyProblem(ex::NamedTuple,params::WENDyParameters;ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        wTrue = if :params in keys(ex) 
            Float64.(ex.params)
        else 
            Float64[ModelingToolkit.getdefault(p) for p in parameters(ex.ode)]
        end
        J = length(wTrue)
        @info "Build julia functions from symbolic expressions of ODE..."
        noise_dist = :noise_dist in keys(ex) ? ex.noise_dist : Normal
        _,f!     = getRHS(ex.ode, Val(noise_dist))
        _,jacuf! = getJacu(ex.ode, Val(noise_dist));
        _,jacwf! = getJacw(ex.ode, Val(noise_dist));
        _,jacwjacuf! = getJacwJacu(ex.ode, Val(noise_dist));
        _,heswf! = getHesw(ex.ode, Val(noise_dist));
        _,heswjacuf! = getHeswJacu(ex.ode, Val(noise_dist));
        
        Random.seed!(params.seed)
        @info "Load data from file..."
        tt_full, U_full = getData(ex)
        if noise_dist == LogNormal && any(U_full .<= 0)
            @info " Removing data that is zero so that logrithms are well defined"
            ix = findall( all(U_full[:,m] .> 0) for m in 1:size(U_full,2))
            tt_full = tt_full[ix]
            U_full = U_full[:,ix]
        end
        numRad = length(params.mtParams)
        @info "Subsample data and add noise..."
        tt = tt_full[1:params.timeSubsampleRate:end]
        U = U_full[:,1:params.timeSubsampleRate:end]
        D, M = size(U)
        U, noise, noise_ratio_obs, sigTrue = generateNoise(U, params.noiseRatio, Val(noise_dist)) 
        @info "============================="
        @info "Start of Algo..."
        @info "Estimate the noise in each dimension..."
        sig = estimate_std(U, Val(noise_dist))
        noiseEstRelErr = norm(sigTrue - sig) / norm(sigTrue)
        @info "  Relative Error in noise estimate $noiseEstRelErr"
        @info "Build test function matrices..."
        V, Vp, Vfull = params.pruneMeth(tt,U,params.ϕ,params.Kmin,params.Kmax,params.mtParams);
        K,_ = size(V)
        @info "Build right hand side to NLS..."
        b₀ = reshape(-Vp * U', K*D);

        return WENDyProblem(D, J, M, K, numRad, sigTrue, wTrue, b₀, sig, tt, U, noise, V, Vp, f!, jacuf!, jacwf!,jacwjacuf!,heswf!, heswjacuf!)
    end
end

struct _MATLAB_WENDyProblem <: AbstractWENDyProblem
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
            _,jacuf! = getJacu(ex.ode);
            _,jacwf! = getJacw(ex.ode);
            _,jacwjacuf! = getJacwJacu(ex.ode);
            _,heswf! = getHesw(ex.ode);
            _,heswjacuf! = getHeswJacu(ex.ode);
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
            b₀ = reshape(-Vp * U', K*D);

            return new(D, J, M, K, wTrue, b₀, sig, tt, U, V, Vp, f!, jacuf!, jacwf!, jacwjacuf!,heswf!, heswjacuf!, data)
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