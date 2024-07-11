##
abstract type WENDyProblem end 

function WENDyProblem(data::WENDyData{LinearInParameters, DistType}, params::WENDyParameters; ll::LogLevel=Warn) where {LinearInParameters, DistType}
    if LinearInParameters == true && !params.forceNonlinear
        return LinearWENDyProblem(data, params; ll=ll)
    end
    NonlinearWENDyProblem(data, params; ll=ll)
end
struct LinearWENDyProblem <: WENDyProblem
    D::Int
    J::Int
    M::Int
    K::Int
    testFuctionRadii::AbstractVector{<:Int}
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

function LinearWENDyProblem(data::WENDyData{LinearInParameters,DistType}, params::WENDyParameters; ll::LogLevel=Warn) where {LinearInParameters, DistType}
    with_logger(ConsoleLogger(stderr, ll)) do
        @assert LinearInParameters "The ODE must be linear in order to use this method"
        @info "Build julia functions from symbolic expressions of ODE..."
        _,f!     = getRHS(data.ode, Val(DistType))
        _,jacuf! = getJacu(data.ode, Val(DistType));
        J = length(parameters(data.ode))
        @show testFuctionRadii = params.testFuctionRadii
        U, noise, noise_ratio_obs, sigTrue = generateNoise(data, params) 
        @info "Subsample data and add noise..."
        tt = data.tt_full[1:params.timeSubsampleRate:end]
        U = U[:,1:params.timeSubsampleRate:end]
        @show D, M = size(U)
        @info "============================="
        @info "Start of Algo..."
        @info "Estimate the noise in each dimension..."
        sig = estimate_std(U, Val(DistType))
        noiseEstRelErr = norm(sigTrue - sig) / norm(sigTrue)
        @info "  Relative Error in noise estimate $noiseEstRelErr"
        @info "Build test function matrices..."
        @show params.testFuctionRadii
        @show typeof(params.pruneMeth)
        V, Vp, Vfull = params.pruneMeth(
            tt,U,params.ϕ,J,params.Kmax,
            params.testFuctionRadii
        );
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
    testFuctionRadii::AbstractVector{<:Int}
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

function NonlinearWENDyProblem(data::WENDyData, params::WENDyParameters; ll::LogLevel=Warn)
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
        testFuctionRadii = params.testFuctionRadii
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
        V, Vp, Vfull = params.pruneMeth(tt,U,params.ϕ,J,params.Kmax,params.testFuctionRadii);
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
    function _MATLAB_WENDyProblem(data::WENDyData, ::Any=nothing; ll::LogLevel=Warn)
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
            # V, Vp, Vfull = params.pruneMeth(tt,U,params.ϕ,J,params.Kmax,params.testFuctionRadii);
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