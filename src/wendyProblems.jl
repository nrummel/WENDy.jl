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

function _jacuf!(out, w, u)
@inbounds begin
        out[1] = (+)((*)((*)(2, w[3]), u[1]), (*)((*)(3, w[2]), (^)(u[1], 2)))
        out[2] = (*)((*)(2, w[6]), u[1])
        out[3] = w[8]
        out[4] = w[1]
        out[5] = w[7]
        out[6] = 0
        out[7] = w[4]
        out[8] = 0
        out[9] = w[10]
        #= /Users/user/.julia/packages/SymbolicUtils/dtCid/src/code.jl:420 =#
        nothing
    end
end
function WENDyProblem(ex::NamedTuple,params::WENDyParameters;ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        wTrue = Float64[ModelingToolkit.getdefault(p) for p in parameters(ex.ode)]
        J = length(wTrue)
        @info "Build julia functions from symbolic expressions of ODE..."
        _,f!     = getRHS(ex.ode)
        jacuf! = _jacuf!
        # _,jacuf! = getJacu(ex.ode);
        _,jacwf! = getJacw(ex.ode);
        _,jacwjacuf! = getJacwJacu(ex.ode);
        _,heswf! = getHesw(ex.ode);
        _,heswjacuf! = getHeswJacu(ex.ode);

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