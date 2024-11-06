##
struct WENDyProblem{lip, DistType} 
    D::Int # number of state variables
    J::Int # number of parameters (to be estimated)
    Mp1::Int # (Mp1+1) number of data points in time 
    K::Int # number of test functions 
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
    # Truth info
    sigTrue::AbstractVector{AbstractFloat}
    wTrue::AbstractVector{<:AbstractFloat}
    paramRng::AbstractVector{<:Tuple}
    noise::AbstractMatrix{<:AbstractFloat}
end 
## convience constructor if we wish to arbitrarily say its lineear or not
function WENDyProblem(data::SimulatedWENDyData{lip, DistType}, params::WENDyParameters, ::Val{new_lip}; kwargs...) where {lip, new_lip, DistType<:Distribution}
    if lip !== new_lip
        new_data = SimulatedWENDyData(data, Val(lip))
        return WENDyProblem(new_data, params; kwargs...)
    end
    data
end
## Helper function to unpack data and then simulate noise
function _unpackData(data::SimulatedWENDyData{lip, DistType}, params::WENDyParameters) where {lip, DistType<:Distribution}
    @assert !isnothing(data.tt) "tt is nothing, please call the simulate! function to generate noise"
    @assert !isnothing(data.U) "U is nothing, please call the simulate! function to generate noise"
    @assert !isnothing(data.sigTrue) "sigTrue is nothing, please call the simulate! function to generate noise"
    @assert !isnothing(data.noise) "noise is nothing, please call the simulate! function to generate noise"
    return (
        data.tt[], data.U[], data.sigTrue[], data.noise[], Float64.(data.wTrue), data.paramRng, data.U_exact[]
    )
end
## helper function to build G matrix
function _buildGmat(f!::Function, tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, J::Int)
    D, Mp1 = size(U)
    K, _ = size(V)
    eⱼ = zeros(J)
    F = zeros(D,Mp1)
    G = zeros(K*D,J)
    for j in 1:J 
        eⱼ .= 0
        eⱼ[j] = 1
        for m in 1:Mp1 
            @views f!(F[:,m], U[:,m],eⱼ,tt[m])
        end 
        gⱼ = V * F'
        @views G[:,j] .= reshape(gⱼ,K*D)
    end
    G
end
## Helper function to throw error when function should not exist
function _foo!(::Any, ::Any, ::Any) 
    @assert false "This function is not implemented of linear problems"
end

function _getRhsAndDerivatives(data, tt, U, V, D, J, K, ::Val{true})
    _,f!     = getRHS(data) # the derivatives wrt u are only affected by noise dist
    _,jacuf! = getJacu(data);
    @info " Computing G matrix for linear problem"
    G = _buildGmat(f!, tt, U, V, J)
    return (
        f!, jacuf!, 
        G, 
        _foo!,_foo!,_foo!,_foo!
    )
end

function _getRhsAndDerivatives(data, tt, U, V, D, J, K, ::Val{false})
    _,f!     = getRHS(data) # the derivatives wrt u are only affected by noise dist
    _,jacuf! = getJacu(data);
    G = NaN.*ones(K*D, J)
    @info " Computing additional symbolic functions for nonlinear problem"
    _,jacwf! = getJacw(data); # the derivatives wrt u are only affected by noise dist
    _,jacwjacuf! = getJacwJacu(data);
    _,heswf! = getHesw(data); # the derivatives wrt u are only affected by noise dist
    _,heswjacuf! = getHeswJacu(data);
    return (
        f!,jacuf!,
        G,
        jacwf!,jacwjacuf!,heswf!,heswjacuf!
    )
end

## constructor
function WENDyProblem(data::WENDyData{lip, DistType}, params::WENDyParameters; ll::LogLevel=Warn, matlab_data::Union{Dict,Nothing}=nothing) where {lip,DistType<:Distribution}
    with_logger(ConsoleLogger(stderr, ll)) do
        J = length(data.odeprob.p)
        tt, U, sigTrue, noise, wTrue, paramRng, U_exact = _unpackData(data, params)
        D, Mp1 = size(U)
        @info "============================="
        @info "Start of Algo"
        @info " Estimate the noise in each dimension"
        _Y = DistType == Normal ? U : log.(U)
        sig = estimate_std(_Y)
        noiseEstRelErr = norm(sigTrue - sig) / norm(sigTrue)
        @debug "  Relative Error in noise estimate $noiseEstRelErr"
        V, Vp = getTestFunctionMatrices(tt, U, params)
        K, _ = size(V)
        @info " Building the LHS to the residual"
        b₀ = reshape(-Vp * _Y', K*D);
        @info " Build julia functions from symbolic expressions of ODE"
        (
            f!,jacuf!,
            G,
            jacwf!,jacwjacuf!,heswf!,heswjacuf!
        ) = _getRhsAndDerivatives(data, tt, U, V, D, J, K, Val(lip))
        return WENDyProblem{lip, DistType}(
            D,J,Mp1,K,
            b₀,sig,tt,U_exact,U,_Y,V,Vp,
            f!,jacuf!,
            G,
            jacwf!,jacwjacuf!,heswf!,heswjacuf!,
            data,
            sigTrue, wTrue, paramRng, noise
        )
    end
end
