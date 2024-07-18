##
struct WENDyProblem{LinearInParameters, DistType} 
    D::Int
    J::Int
    M::Int
    K::Int
    b₀::AbstractVector{<:AbstractFloat}
    sig::AbstractVector{<:AbstractFloat}
    tt::AbstractVector{<:AbstractFloat}
    U_exact::AbstractMatrix{<:AbstractFloat} 
    U::AbstractMatrix{<:AbstractFloat} 
    _Y::AbstractMatrix{<:AbstractFloat} 
    V::AbstractMatrix{<:AbstractFloat}
    Vp::AbstractMatrix{<:AbstractFloat}
    f!::Function 
    jacuf!::Function
    # Only valid when the problem is linear 
    G::AbstractMatrix{<:Real}
    # Only necessary when the problem is non linear
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
## convience constructor if we wish to arbitrarily say its lineear or not
function WENDyProblem(data::SimulatedWENDyData{LinearInParameters, DistType}, params::WENDyParameters, forceLinear::Val{LinearInParameters}; kwargs...) where {LinearInParameters, DistType<:Distribution}
    if !isnothing(forceLinear)
        data = SimulatedWENDyData(data, Val(LinearInParameters))
    end
    WENDyProblem(data, params; kwargs...)
end
## Helper function to unpack data and then simulate noise
function _unpackData(data::SimulatedWENDyData{LinearInParameters, DistType}, params::WENDyParameters) where {LinearInParameters, DistType<:Distribution}
    @info "Simulated Data"
    @info "  Subsample data "
    tt = data.tt_full[1:params.timeSubsampleRate:end]
    U_exact = data.U_exact[:,1:params.timeSubsampleRate:end]
    D, M = size(U_exact)
    @info "Adding noise from Distribution $DistType"
    U, noise, noise_ratio_obs, sigTrue = generateNoise(U_exact, params, Val(DistType))
    return (
        tt, U, sigTrue, noise, Float64.(data.trueParameters), U_exact
    )
end
## helper function that unpacks data and fill in NaN for truth
function _unpackData(data::EmpricalWENDyData, params::WENDyParameters)
    @info "Using EmpricalWENDyData"
    data.tt_full, data.U_full, NaN*ones(size(data.U_full,1)), NaN*ones(size(data.U_full)), NaN*ones(J), NaN*size(data.U_full)
end
## helper function to build G matrix
function _buildGmat(f!::Function, U::AbstractMatrix, V::AbstractMatrix, J::Int)
    D, M = size(U)
    K, _ = size(V)
    @info " Build G mat for linear problem"
    eⱼ = zeros(J)
    F = zeros(D,M)
    G = zeros(K*D,J)
    for j in 1:J 
        eⱼ .= 0
        eⱼ[j] = 1
        for m in 1:M 
            @views f!(F[:,m], eⱼ, U[:,m])
        end 
        gⱼ = V * F'
        @views G[:,j] .= reshape(gⱼ,K*D)
    end
    G
end
## Helper function to throw error when linear problem tries to call nonlinear functions that dont exist
function _foo!(::Any, ::Any, ::Any) 
    @assert false "This function is not implemented of linear problems"
end
## linear Wendy problem 
function WENDyProblem(data::WENDyData{true, DistType}, params::WENDyParameters; ll::LogLevel=Warn, matlab_data::Union{Dict,Nothing}=nothing) where DistType<:Distribution
    with_logger(ConsoleLogger(stderr, ll)) do
        J = length(parameters(data.ode))
        tt, U, sigTrue, noise, wTrue, U_exact = _unpackData(data, params)
        D, M = size(U)
        @info "============================="
        @info "Start of Algo"
        @info " Estimate the noise in each dimension"
        _Y = DistType == Normal ? U : log.(U)
        sig = estimate_std(_Y)
        noiseEstRelErr = norm(sigTrue - sig) / norm(sigTrue)
        @info "  Relative Error in noise estimate $noiseEstRelErr"
        V,Vp,_ = isnothing(matlab_data) ? params.pruneMeth(tt,_Y,params.ϕ,J,params.Kmax,params.testFuctionRadii) : (matlab_data["V"], matlab_data["Vp"], nothing)
        K, _ = size(V)
        @info " Building the LHS to the residual"
        b₀ = reshape(-Vp * _Y', K*D);
        @info " Build julia functions from symbolic expressions of ODE"
        _,f!     = getRHS(data) # the derivatives wrt u are only affected by noise dist
        _,jacuf! = getJacu(data);
        G = _buildGmat(f!, U, V, J)
        return WENDyProblem{true, DistType}(
            D,J,M,K,
            b₀,sig,tt,U_exact,U,_Y,V,Vp,
            f!,jacuf!,
            G, # Linear Only
            _foo!,_foo!,_foo!,_foo!, # Nonlinear only
            data,
            sigTrue,wTrue,noise
        )
    end
end
## nonlinear Wendy problem 
function WENDyProblem(data::WENDyData{false, DistType}, params::WENDyParameters; ll::LogLevel=Warn, matlab_data::Union{Dict,Nothing}=nothing) where DistType<:Distribution
    with_logger(ConsoleLogger(stderr, ll)) do
        J = length(parameters(data.ode))
        J = length(parameters(data.ode))
        tt, U, sigTrue, noise, wTrue, U_exact = _unpackData(data, params)
        D, M = size(U)
        @info "============================="
        @info "Start of Algo"
        @info " Estimate the noise in each dimension"
        _Y = DistType == Normal ? U : log.(U)
        sig = estimate_std(_Y)
        noiseEstRelErr = norm(sigTrue - sig) / norm(sigTrue)
        @debug "  Relative Error in noise estimate $noiseEstRelErr"
        V,Vp,_ = isnothing(matlab_data) ? params.pruneMeth(tt,_Y,params.ϕ,J,params.Kmax,params.testFuctionRadii) : (matlab_data["V"], matlab_data["Vp"], nothing)
        K, _ = size(V)
        @info " Building the LHS to the residual"
        b₀ = reshape(-Vp * _Y', K*D);
        @info " Build julia functions from symbolic expressions of ODE"
        _,f!     = getRHS(data) # the derivatives wrt u are only affected by noise dist
        _,jacuf! = getJacu(data);
        @info " Computing additional symbolic functions for nonlinear problem"
        G = NaN.*ones(K*D, J)
        _,jacwf! = getJacw(data); # the derivatives wrt u are only affected by noise dist
        _,jacwjacuf! = getJacwJacu(data);
        _,heswf! = getHesw(data); # the derivatives wrt u are only affected by noise dist
        _,heswjacuf! = getHeswJacu(data);
        return WENDyProblem{false, DistType}(
            D,J,M,K,
            b₀,sig,tt,U_exact,U,_Y,V,Vp,
            f!,jacuf!,
            G,
            jacwf!,jacwjacuf!,heswf!,heswjacuf!,
            data,
            sigTrue,wTrue,noise
        )
    end
end
##
import Plots: plot
function Plots.plot(prob::WENDyProblem)
    D = prob.D 
    plot(
        prob.tt,
        [prob.U[d,:] for d in 1:D],
        label=["u_$d" for d in 1:D],
        title="WENDy Problem"
    )
    xlabel!("time")
end