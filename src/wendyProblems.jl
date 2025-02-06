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
##
function _getRhsAndDerivatives_linear(_f!::Function, tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, D::Int, J::Int, ::Val{DistType}) where DistType<:Distribution
    @info " Computing G matrix for linear problem"
    f!   = _getf(_f!, D, J, Val(DistType))
    ∇ₓf! = _get∇ₓf(f!, D, J)
    G    = _buildGmat(f!, tt, X, V, J)
    function _foo!(::Any, ::Any, ::Any) 
        @assert false "This function is not implemented of linear problems"
    end
    return f!,∇ₓf!,G,_foo!,_foo!,_foo!,_foo!
end
##
function _getRhsAndDerivatives_nonlinear(_f!::Function, D::Int, J::Int, K::Int, ::Val{DistType}) where DistType<:Distribution
    @info " Computing additional symbolic functions for nonlinear problem"
    f!     = _getf(_f!, D, J, Val(DistType))
    ∇ₓf!   = _get∇ₓf(f!, D, J)
    G      = NaN .* ones(K*D, J)
    ∇ₚf!   = _get∇ₚf(f!, D, J)
    ∇ₚ∇ₓf! = _get∇ₚ∇ₓf(f!, D, J)
    Hₚf!   = _getHₚf(f!, D, J)
    Hₚ∇ₓf! = _getHₚ∇ₓf(f!, D, J)
    return f!,∇ₓf!,G,∇ₚf!,∇ₚ∇ₓf!,Hₚf!,Hₚ∇ₓf!
end
##
function _forwardSolveResidual(wu0, J, _tt, U, _f!, alg, reltol, abstol, u0Free)
    _Mp1, D = size(U)
    try 
        p,u₀ = if u0Free
            wu0[1:J], wu0[J+1:end]
        else 
            wu0[1:J], U[1,:]
        end
        tRng = (_tt[1], _tt[end])
        dt = (_tt[end] - _tt[1]) / (length(_tt) - 1)
        odeprob = ODEProblem(_f!, u₀, tRng, p)
        sol = solve_ode(odeprob, alg; 
            reltol=reltol, abstol=abstol,
            saveat=dt, verbose=false
        )
        Uhat = reduce(vcat, um' for um in sol.u)
        r = (Uhat - U) 
        return r[:]
    catch 
        NaN*ones(_Mp1*D)
    end
end

function _buildCostFunctions(J::Int, _tt::AbstractVector{<:Real}, _f!::Function, U::AbstractMatrix{<:Real}, data::WENDyInternals, params::WENDyParameters)
    _Mp1, D = size(U)
    K, _ = size(data.V)
    f(wu0) = _forwardSolveResidual(wu0, J, _tt, U, _f!, params.fsAlg, params.fsReltol, params.fsAbstol, params.fsU0Free)
    oels = LeastSquaresCostFunction(
        (r, wu0) -> r .= f(wu0), 
        (jac,wu0) -> ForwardDiff.jacobian!(jac, f, wu0),
        _Mp1*D
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
    return oels, wlsq, wnll
end

## constructor
function WENDyProblem(
    _tt::AbstractVector{<:Real}, U::AbstractVecOrMat{<:Real}, _f!::Function, J::Int, 
    ::Val{lip}=Val(false), ::Val{DistType}=Val(Normal), params::WENDyParameters=WENDyParameters(); 
    constraints::Union{Nothing,AbstractVector{Tuple{T1,T2}}}=nothing,
    ll::LogLevel=Warn) where {lip, DistType<:Distribution,T1<:Real, T2<:Real}
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "Building WENDyProblem"
        if typeof(U) <: AbstractVector 
            @info "  Reshaping U to be an (M+1)x1 matrix"
            U = reshape(U, (length(U),1))
        end
        _Mp1, D = size(U)
        @info "  Estimate the noise in each dimension"
        Mp1, tt, X = if DistType == Normal 
            _Mp1, _tt, U
        else
            ix = findall( all(um .> 0) for um in eachrow(U))
            if length(ix) < _Mp1
                @info " Removing data that is zero so that logrithms are well defined: $(_Mp1 - length(ix)) data point(s) are invalid"
            end
            length(ix), _tt[ix], log.(U[ix,:])
        end
        sig = estimate_std(X)
        @info "  sig = $sig"
        V, Vp = getTestFunctionMatrices(tt, X, params; ll=ll)
        K, _ = size(V)
        @info "  Building the LHS to the residual"
        b₀ = reshape(-Vp * X, K*D)
        @info "  Build julia functions from symbolic expressions of ODE"
        (f!,∇ₓf!,G,∇ₚf!,∇ₚ∇ₓf!,Hₚf!,Hₚ∇ₓf!) = if lip  
            _getRhsAndDerivatives_linear(_f!, tt, X, V, D, J, Val(DistType)) 
        else 
            _getRhsAndDerivatives_nonlinear(_f!, D, J, K, Val(DistType))
        end
        @info "  Building Internal Data structures"
        data = WENDyInternals{lip,DistType}(
            J,b₀,sig,tt,X,V,Vp,
            f!,∇ₓf!,
            G, # only used in the linear case
            ∇ₚf!,∇ₚ∇ₓf!,Hₚf!,Hₚ∇ₓf! # only used in the nonlinear case
        )
        ## Build Cost Functions 
        @info "  Building Cost Functions"
        oels, wlsq, wnll = _buildCostFunctions(J, _tt, _f!, U, data, params)

        return WENDyProblem{lip, DistType}(
            D,J,Mp1,K,U[1,:],constraints,
            data, oels, wlsq, wnll
        )
    end
end