##
struct WENDyProblem{lip, DistType} 
    D::Int # number of state variables
    J::Int # number of parameters (to be estimated)
    Mp1::Int # (Mp1+1) number of data points in time 
    K::Int # number of test functions 
    b₀::AbstractVector{<:AbstractFloat}
    sig::AbstractVector{<:AbstractFloat}
    tt::AbstractVector{<:AbstractFloat}
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
end 
## convience constructor if we wish to arbitrarily say its lineear or not
function WENDyProblem(data::SimulatedWENDyData{lip, DistType}, params::WENDyParameters, ::Val{new_lip}; kwargs...) where {lip, new_lip, DistType<:Distribution}
    if lip !== new_lip
        new_data = SimulatedWENDyData(data, Val(lip))
        return WENDyProblem(new_data, params; kwargs...)
    end
    data
end
## helper function to build G matrix
function _buildGmat(f!::Function, tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, J::Int)
    Mp1, D = size(U)
    K, _ = size(V)
    eⱼ = zeros(J)
    F = similar(U)
    G = zeros(K*D,J)
    for j in 1:J 
        eⱼ .= 0
        eⱼ[j] = 1
        for m in 1:Mp1 
            @views f!(F[m,:], U[m,:], eⱼ, tt[m])
        end 
        gⱼ = V * F
        @views G[:,j] .= reshape(gⱼ,K*D)
    end
    G
end
## Helper function to throw error when function should not exist
function _foo!(::Any, ::Any, ::Any) 
    @assert false "This function is not implemented of linear problems"
end

function _getRhsAndDerivatives_linear(f!::Function, tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, D::Int, J::Int, ::Val{DistType}) where DistType<:Distribution
    _,f!     = getRHS(f!, D, J, Val(DistType)) # the derivatives wrt u are only affected by noise dist
    _,jacuf! = getJacu(f!, D, J, Val(DistType))
    @info " Computing G matrix for linear problem"
    G = _buildGmat(f!, tt, U, V, J)
    return (
        f!, jacuf!, 
        G, 
        _foo!,_foo!,_foo!,_foo!
    )
end

function _getRhsAndDerivatives_nonlinear(f!::Function, D::Int, J::Int, K::Int, ::Val{DistType}) where DistType<:Distribution
    _,f!     = getRHS(f!, D, J, Val(DistType)) # the derivatives wrt u are only affected by noise dist
    _,jacuf! = getJacu(f!, D, J, Val(DistType))
    G = NaN.*ones(K*D, J)
    @info " Computing additional symbolic functions for nonlinear problem"
    _,jacwf! = getJacw(f!, D, J, Val(DistType)) # the derivatives wrt u are only affected by noise dist
    _,jacwjacuf! = getJacwJacu(f!, D, J, Val(DistType))
    _,heswf! = getHesw(f!, D, J, Val(DistType)) # the derivatives wrt u are only affected by noise dist
    _,heswjacuf! = getHeswJacu(f!, D, J, Val(DistType))
    return (
        f!,jacuf!,
        G,
        jacwf!,jacwjacuf!,heswf!,heswjacuf!
    )
end
## constructor
function WENDyProblem(tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, f!::Function, J::Int, ::Val{lip}, ::Val{DistType},
    params::WENDyParameters=WENDyParameters(); ll::LogLevel=Warn) where {lip,DistType<:Distribution}
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "Building WENDyProblem"
        Mp1, D = size(U)
        @info "  Estimate the noise in each dimension"
        _Y = DistType == Normal ? U : log.(U)
        sig = estimate_std(_Y)
        V, Vp = getTestFunctionMatrices(tt, U, params; ll=ll)
        K, _ = size(V)
        @info "  Building the LHS to the residual"
        b₀ = reshape(-Vp * _Y, K*D)
        @info "  Build julia functions from symbolic expressions of ODE"
        (
            f!,jacuf!,
            G,
            jacwf!,jacwjacuf!,heswf!,heswjacuf!
        ) = lip ? _getRhsAndDerivatives_linear(f!, tt, U, V, D, J, Val(DistType)) : _getRhsAndDerivatives_nonlinear(f!, D, J, K, Val(DistType))
        return WENDyProblem{lip, DistType}(
            D,J,Mp1,K,
            b₀,sig,tt,U,_Y,V,Vp,
            f!,jacuf!,
            G, # only used in the linear case
            jacwf!,jacwjacuf!,heswf!,heswjacuf! # only used in the nonlinear case
        )
    end
end
## Convience Wrapper
function WENDyProblem(data::WENDyData{lip, DistType}, params::WENDyParameters=WENDyParameters(); ll::LogLevel=Warn) where {lip,DistType<:Distribution}
    @assert !isnothing(data.tt[]) "tt is nothing, please call the simulate! function to generate noise"
    @assert !isnothing(data.U[]) "U is nothing, please call the simulate! function to generate noise"
    WENDyProblem(data.tt[], data.U[], data.f!, length(data.wTrue), Val(lip), Val(DistType), params; ll=ll)
end
