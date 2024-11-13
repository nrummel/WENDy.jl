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
function _getRhsAndDerivatives_linear(_f!::Function, tt::AbstractVector{<:Real}, Y::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}, D::Int, J::Int, ::Val{DistType}) where DistType<:Distribution
    @info " Computing G matrix for linear problem"
    f!   = _getf(_f!, D, J, Val(DistType))
    ∇ₓf! = _get∇ₓf(f!, D, J)
    G    = _buildGmat(f!, tt, Y, V, J)
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
    ∇ₚ∇ₓf! = _get∇ₚ∇ₓf(_f!, D, J)
    Hₚf!   = _getHₚf(_f!, D, J)
    Hₚ∇ₓf! = _getHₚ∇ₓf(_f!, D, J)
    return f!,∇ₓf!,G,∇ₚf!,∇ₚ∇ₓf!,Hₚf!,Hₚ∇ₓf!
end
##
function _forwardSolveResidual(w, tt, U, _f!, alg, reltol, abstol)
    Mp1, D = size(U)
    try 
        u0 = U[1,:]
        tRng = (tt[1], tt[end])
        dt = mean(diff(tt))
        odeprob = ODEProblem(_f!, u0, tRng, w)
        sol = solve_ode(odeprob, alg; 
            reltol=reltol, abstol=abstol,
            saveat=dt, force_dtmin=true
        )
        Uhat = reduce(vcat, um' for um in sol.u)
        r = (Uhat - U) 
        return r[:]
    catch 
        NaN*ones(Mp1*D)
    end
end

function _buildCostFunctions(_f!::Function, U::AbstractMatrix{<:Real}, data::WENDyInternals, params::WENDyParameters)
    Mp1, D = size(data.X)
    K, _ = size(data.V)
    f(w) = _forwardSolveResidual(w, data.tt, U, _f!, params.fsAlg, params.fsReltol, params.fsAbstol)
    fslsq = LeastSquaresCostFunction(
        (r, w) -> r .= f(w), 
        (jac,w) -> ForwardDiff.jacobian!(jac, f, w),
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
function WENDyProblem(
    tt::AbstractVector{<:Real}, U::AbstractVecOrMat{<:Real}, _f!::Function, J::Int, 
    ::Val{lip}=Val(false), ::Val{DistType}=Val(Normal),params::WENDyParameters=WENDyParameters(); 
    ll::LogLevel=Warn) where {lip, DistType<:Distribution}
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "Building WENDyProblem"
        if typeof(U) <: AbstractVector 
            @info "  Reshaping U to be an (M+1)x1 matrix"
            U = reshape(U, (length(U),1))
        end
        Mp1, D = size(U)
        @info "  Estimate the noise in each dimension"
        _Y = DistType == Normal ? U : log.(U)
        sig = estimate_std(_Y)
        V, Vp = getTestFunctionMatrices(tt, U, params; ll=ll)
        K, _ = size(V)
        @info "  Building the LHS to the residual"
        b₀ = reshape(-Vp * _Y, K*D)
        @info "  Build julia functions from symbolic expressions of ODE"
        (f!,∇ₓf!,G,∇ₚf!,∇ₚ∇ₓf!,Hₚf!,Hₚ∇ₓf!) = if lip  
            _getRhsAndDerivatives_linear(_f!, tt, _Y, V, D, J, Val(DistType)) 
        else 
            _getRhsAndDerivatives_nonlinear(_f!, D, J, K, Val(DistType))
        end
        @info "  Building Internal Data structures"
        data = WENDyInternals{lip,DistType}(
            J,b₀,sig,tt,_Y,V,Vp,
            f!,∇ₓf!,
            G, # only used in the linear case
            ∇ₚf!,∇ₚ∇ₓf!,Hₚf!,Hₚ∇ₓf! # only used in the nonlinear case
        )
        ## Build Cost Functions 
        @info "  Building Cost Functions"
        fslsq, wlsq, wnll = _buildCostFunctions(_f!, U, data, params)

        return WENDyProblem{lip, DistType}(
            D,J,Mp1,K,
            data, fslsq, wlsq, wnll
        )
    end
end