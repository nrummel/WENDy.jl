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
        G[:,j] = vec(gⱼ)
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
"""
    WENDyProblem{lip, DistType}(...)

A WENDyProblem holds the weak form of the parameter inverse problem: the data, test functions, the symbolic 
right hand side and its derivatives, and a workspace for evaluating the weak form negative log-likelihood 
(see wnll, ∇wnll!, Hwnll!) and the weak residual (see weakResidual!, weakJacobian!, weakCovariance).
For the output error (forward solve) least squares problem see OutputErrorProblem.

# Constructor 
    WENDyProblem(
        _tt::AbstractVector{<:Real}, 
        U::AbstractVecOrMat{<:Real}, 
        _f!::Function, 
        J::Int; 
        linearInParameters::Val{lip}=Val(false), 
        noiseDist::Val{DistType}=Val(Normal), 
        params::WENDyParameters=WENDyParameters(), 
        constraints::Union{Nothing,AbstractVector{Tuple{<:Real,<:Real}}}=nothing, 
        ll::LogLevel=Warn
    )
## Arguments 
- _tt::AbstractVector{<:Real} : vector of times (equispaced)
- U::AbstractVecOrMat{<:Real} : Corrupted state variable data 
- _f!::Function : Right hand-side of the differential equation
    Must be of the form f!(du, u, p, t)
- J::Int : number of parameters (to be estimated)
- linearInParameters::Val{lip}=Val(false) : (optional) specify whether the right hand side is `linear in parameters' for improved computational efficiency
- noiseDist::Val{DistType}=Val(Normal) : (optional) specify the distribution of the measurement noise. Choose either Val(Normal) for additive Gaussian noise of Val(LogNormal) for multiplicative LogNormal noise.
- params::WENDyParameters : (optional) struct of hyper-parameters for the WENDy Algorithm (see the doc for WENDyParameters)
- constraints=nothing : (optional) Linear box constraints for each parameter, ∀j ∈ [1, ⋯,J], ℓⱼ ≤ pⱼ ≤ uⱼ. Accepts constraints as a list of tuples, [(ℓ₁,u₁), ⋯]. Note: this only is compatible with the TrustRegion solver.
- ll::LogLevel=Warn : (optional) see additional algorithm information by setting ll=Info

It can also be rebuilt from its internal data (e.g. data that was saved) with 
    WENDyProblem(data::WENDyInternals, params::WENDyParameters; constraints=nothing, ll=Warn)

# Fields
- D::Int : number of state variables
- J::Int : number of parameters (to be estimated)
- Mp1::Int : (Mp1+1) number of data points in time 
- K::Int : number of test functions 
- constraints : vector of tuples containing linear constraints for each parameter
- data::WENDyInternals : data, test functions and the symbolic right hand side and its derivatives
- ws::WENDyWorkspace : shared state and buffers for m(p), ∇m(p) and Hm(p)
"""
struct WENDyProblem{lip, DistType}
    D::Int # number of state variables
    J::Int # number of parameters (to be estimated)
    Mp1::Int # (Mp1+1) number of data points in time 
    K::Int # number of test functions 
    constraints::Union{Nothing,AbstractVector{Tuple{<:Real,<:Real}}}
    data::WENDyInternals{lip, DistType}
    ws::WENDyWorkspace
end
## constructor from the internal data
function WENDyProblem(
    data::WENDyInternals{lip, DistType}, params::WENDyParameters; 
    constraints::Union{Nothing,AbstractVector{<:Tuple{<:Real,<:Real}}}=nothing, ll::LogLevel=Warn
) where {lip, DistType}
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "  Building workspace"
        Mp1, D = size(data.X)
        K, _ = size(data.V)
        WENDyProblem{lip, DistType}(D, data.J, Mp1, K, constraints, data, WENDyWorkspace(data, params))
    end
end
## constructor
function WENDyProblem(
    _tt::AbstractVector{<:Real}, 
    U::AbstractVecOrMat{<:Real}, 
    _f!::Function, 
    J::Int;
    linearInParameters::Val{lip}=Val(false), 
    noiseDist::Val{DistType}=Val(Normal), params::WENDyParameters=WENDyParameters(),
    constraints::Union{Nothing,AbstractVector{Tuple{T1,T2}}}=nothing,
    ll::LogLevel=Warn
) where {lip, DistType<:Distribution,T1<:Real, T2<:Real}
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
        return WENDyProblem(data, params; constraints=constraints, ll=ll)
    end
end
## weak form negative log-likelihood, its gradient and hessian
"""
    wnll(prob::WENDyProblem, p)

Weak form negative log-likelihood m(p) = ½rᵀS⁻¹r + ½log|S| + (KD/2)log(2π), where r(p) is the weak residual and S(p) its covariance
"""
wnll(prob::WENDyProblem, p::AbstractVector{<:Real}) = wnll(prob.ws, p)
"""
    ∇wnll(prob::WENDyProblem, p)
    ∇wnll!(g, prob::WENDyProblem, p)

Gradient of the weak form negative log-likelihood (∇wnll! writes it into g)
"""
∇wnll(prob::WENDyProblem, p::AbstractVector{<:Real}) = ∇wnll(prob.ws, p)
∇wnll!(g::AbstractVector{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}) = g .= ∇wnll(prob, p)
"""
    Hwnll(prob::WENDyProblem, p; approx=false)
    Hwnll!(H, prob::WENDyProblem, p; approx=false)

Hessian of the weak form negative log-likelihood (Hwnll! writes it into H). With approx = true the curvature 
of the log-determinant term is left out (see approxHessian in WENDyParameters)
"""
Hwnll(prob::WENDyProblem, p::AbstractVector{<:Real}; approx::Bool=false) = Hwnll(prob.ws, p; approx=approx)
Hwnll!(H::AbstractMatrix{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}; approx::Bool=false) = H .= Hwnll(prob, p; approx=approx)
## weak residual, its jacobian and covariance
"""
    weakResidual(prob::WENDyProblem, p)
    weakResidual!(r, prob::WENDyProblem, p)

Weak residual r(p) = G(p) - b₀ (weakResidual! writes it into r)
"""
weakResidual(prob::WENDyProblem, p::AbstractVector{<:Real}) = _r(prob.ws, p)
weakResidual!(r::AbstractVector{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}) = r .= weakResidual(prob, p)
"""
    weakJacobian(prob::WENDyProblem, p)
    weakJacobian!(∇r, prob::WENDyProblem, p)

Jacobian of the weak residual wrt p (weakJacobian! writes it into ∇r)
"""
weakJacobian(prob::WENDyProblem, p::AbstractVector{<:Real}) = copy(_∇r(prob.ws, p))
weakJacobian!(∇r::AbstractMatrix{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}) = ∇r .= _∇r(prob.ws, p)
"""
    weakCovariance(prob::WENDyProblem, p)

Covariance of the weak residual S(p) = (1-δ)L(p)L(p)ᵀ + δI
"""
weakCovariance(prob::WENDyProblem, p::AbstractVector{<:Real}) = copy(_update!(prob.ws, p).S)
