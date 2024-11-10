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
    fsAbstol::Real                      = 1e-8
    fsReltol::Real                      = 1e-8
    nlsAbstol::Real                     = 1e-8
    nlsReltol::Real                     = 1e-8
    nlsMaxiters::Int                    = 1000
    optimAbstol::Real                   = 1e-8
    optimReltol::Real                   = 1e-8
    optimMaxiters::Int                  = 200
    optimTimelimit::Real                = 200.0
    fsAlg::OrdinaryDiffEqAlgorithm      = Rosenbrock23()
end 
##
struct WENDyInternals{lip, DistType}
    J::Int # number of parameters (to be estimated)
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
##
struct WENDyProblem{lip, DistType}
    D::Int # number of state variables
    J::Int # number of parameters (to be estimated)
    Mp1::Int # (Mp1+1) number of data points in time 
    K::Int # number of test functions 
    data::WENDyInternals{lip, DistType}
    # Cost functions 
    fslsq::LeastSquaresCostFunction
    wlsq::LeastSquaresCostFunction
    wnll::SecondOrderCostFunction 
end 
## helper function to build G matrix
function _buildGmat(f!::Function, tt::AbstractVector{<:Real}, Y::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, J::Int)
    Mp1, D = size(Y)
    K, _ = size(V)
    eⱼ = zeros(J)
    F = similar(Y)
    G = zeros(K*D,J)
    for j in 1:J 
        eⱼ .= 0
        eⱼ[j] = 1
        for m in 1:Mp1 
            @views f!(F[m,:], Y[m,:], eⱼ, tt[m])
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
##
function _getRhsAndDerivatives_linear(f!::Function, tt::AbstractVector{<:Real}, Y::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, D::Int, J::Int, ::Val{DistType}) where DistType<:Distribution
    _,f!     = getRHS(f!, D, J, Val(DistType)) # the derivatives wrt u are only affected by noise dist
    _,jacuf! = getJacu(f!, D, J, Val(DistType))
    @info " Computing G matrix for linear problem"
    G = _buildGmat(f!, tt, Y, V, J)
    return (
        f!, jacuf!, 
        G, 
        _foo!,_foo!,_foo!,_foo!
    )
end
##
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
##
function _fsres(w, tt, U, alg)
    Mp1, D = size(U)
    reltol, abstol = params.fsReltol, params.fsAbstol
    try 
        u0 = U[1,:]
        tRng = (tt[0], tt[end])
        dt = (tRng[end]-tRng[1]) / (Mp1-1)
        odeprob = ODEProblem{true, SciMLBase.FullSpecialize}(ogf!, u0, tRng, w)
        sol = solve(odeprob, alg; 
            reltol=reltol, abstol=abstol, 
            saveat=dt
        )
        Uhat = reduce(vcat, um' for um in sol.u)
        r = (Uhat - U) 
        return r[:]
    catch 
        NaN*ones(Mp1*D)
    end
end

function _buildCostFunctions(data::WENDyInternals, params::WENDyParameters)
    Mp1, D = size(data.U)
    K, _ = size(data.V)

    fslsq = LeastSquaresCostFunction(
        (r, w) -> r .= _fsres(w, data.tt, data.U, params.fsAlg), 
        (jac,w) -> ForwardDiff.jacobian!(jac, _fsres, w),
        Mp1*D
    )

    wlsq = LeastSquaresCostFunction(
        WENDy.Residual(data, params), 
        WENDy.JacobianResidual(data, params),
        K*D
    )

    wnll = SecondOrderCostFunction(
        WeakNLL(data, params),
        GradientWeakNLL(data, params),
        HesianWeakNLL(data, params)
    )
    return fslsq, wlsq, wnll
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
        (f!,jacuf!,G,jacwf!,jacwjacuf!,heswf!,heswjacuf!) = if lip  
            _getRhsAndDerivatives_linear(f!, tt, _Y, V, D, J, Val(DistType)) 
        else 
            _getRhsAndDerivatives_nonlinear(f!, D, J, K, Val(DistType))
        end
        @info "  Building Internal Data structures"
        data = WENDyInternals{lip,DistType}(
            J,b₀,sig,tt,U,_Y,V,Vp,
            f!,jacuf!,
            G, # only used in the linear case
            jacwf!,jacwjacuf!,heswf!,heswjacuf! # only used in the nonlinear case
        )
        ## Build Cost Functions 
        @info "  Building Cost Functions"
        fslsq, wlsq, wnll = _buildCostFunction(data, params)

        return WENDyProblem{lip, DistType}(
            D,J,Mp1,K,
            data, fslsq, wlsq, wnll
        )
    end
end