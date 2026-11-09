## helper function to build G matrix
function _buildGmat(f!::Function, tt::AbstractVector{<:Real}, Y::AbstractMatrix{<:Real}, Φ::AbstractMatrix{<:Real}, J::Int)
    Mp1, D = size(Y)
    K, _ = size(Φ)
    eⱼ = zeros(J)
    F = similar(Y)
    G = zeros(K*D,J)
    for j in 1:J 
        eⱼ .= 0
        eⱼ[j] = 1
        for m in 1:Mp1 
            @views f!(F[m,:], Y[m,:], eⱼ, tt[m])
        end 
        gⱼ = Φ * F
        G[:,j] = vec(gⱼ)
    end
    G
end
"""
    WENDyProblem

A WENDyProblem holds the weak form of the parameter inverse problem: the (transformed) data, the test functions, 
the weak form (see LinearWeakForm, NonlinearWeakForm) and a workspace for evaluating the weak form negative 
log-likelihood (see ℓ, ∇ℓ!, ∇²ℓ!) and the weak residual (see weakResidual!, weakJacobian!, weakCovariance).
For the output error (forward solve) least squares problem see OutputErrorProblem.

# Constructor 
    WENDyProblem(
        _tt::AbstractVector{<:Real}, 
        U::AbstractVecOrMat{<:Real}, 
        _f!::Function, 
        J::Int; 
        linearInParameters::Bool=false, 
        noiseDist::Type{<:Distribution}=Normal, 
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
- linearInParameters::Bool=false : (optional) specify whether the right hand side is `linear in parameters' for improved computational efficiency
- noiseDist::Type{<:Distribution}=Normal : (optional) specify the distribution of the measurement noise. Choose either Normal for additive Gaussian noise of LogNormal for multiplicative LogNormal noise.
- params::WENDyParameters : (optional) struct of hyper-parameters for the WENDy Algorithm (see the doc for WENDyParameters)
- constraints=nothing : (optional) Linear box constraints for each parameter, ∀j ∈ [1, ⋯,J], ℓⱼ ≤ pⱼ ≤ uⱼ. Accepts constraints as a list of tuples, [(ℓ₁,u₁), ⋯]. Note: this only is compatible with the TrustRegion solver.
- ll::LogLevel=Warn : (optional) see additional algorithm information by setting ll=Info
(linearInParameters=Val(true) and noiseDist=Val(LogNormal), as in earlier versions, are also accepted.)

It can also be built from data that was already transformed and its weak form (e.g. to rebuild a saved problem) with 
    WENDyProblem(tt, X, Φ, Φ̇, σ, b, wform::WeakForm, params::WENDyParameters; noiseDist=Normal, constraints=nothing, ll=Warn)
where wform = weakForm(f!, tt, X, Φ, Φ̇, σ, b, J; linearInParameters, noiseDist).

# Fields
- D::Int : number of state variables
- J::Int : number of parameters (to be estimated)
- Mp1::Int : (Mp1+1) number of data points in time 
- K::Int : number of test functions 
- tt, X : times and data (X = log(U) for LogNormal noise, with nonpositive data removed)
- Φ, Φ̇ : test function matrices Φ and Φ̇
- σ : estimated noise level in each state variable
- b : -vec(V′X)
- noiseDist : the noise model the problem was built for, Normal or LogNormal
- constraints : vector of tuples containing linear constraints for each parameter
- ws::WENDyWorkspace : the weak form and shared state for ℓ(p), ∇ℓ(p) and ∇²ℓ(p)
"""
struct WENDyProblem
    D::Int # number of state variables
    J::Int # number of parameters (to be estimated)
    Mp1::Int # (Mp1+1) number of data points in time 
    K::Int # number of test functions 
    tt::Vector{Float64}
    X::Matrix{Float64}
    Φ::Matrix{Float64}
    Φ̇::Matrix{Float64}
    σ::Vector{Float64}
    b::Vector{Float64}
    noiseDist::Type{<:Distribution}
    constraints::Union{Nothing,AbstractVector{Tuple{<:Real,<:Real}}}
    ws::WENDyWorkspace
end
## constructor from transformed data and its weak form
function WENDyProblem(
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, Φ::AbstractMatrix{<:Real}, Φ̇::AbstractMatrix{<:Real}, 
    σ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, wform::WeakForm, params::WENDyParameters; 
    noiseDist::Type{<:Distribution}=Normal, constraints::Union{Nothing,AbstractVector{<:Tuple{<:Real,<:Real}}}=nothing, ll::LogLevel=Warn
)
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "  Building workspace"
        Mp1, D = size(X)
        K, _ = size(Φ)
        WENDyProblem(D, _J(wform), Mp1, K, tt, X, Φ, Φ̇, σ, b, noiseDist, constraints, WENDyWorkspace(wform, params))
    end
end
# noise models: LogNormal noise is Normal noise on log(U), so only the data and rhs are transformed
_transformData(::Type{Normal}, tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}) = tt, U
function _transformData(::Type{LogNormal}, tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real})
    ix = findall(all(um .> 0) for um in eachrow(U))
    if length(ix) < size(U, 1)
        @info " Removing data that is zero so that logrithms are well defined: $(size(U, 1) - length(ix)) data point(s) are invalid"
    end
    tt[ix], log.(U[ix,:])
end
# earlier versions took Val(true) / Val(LogNormal)
_unwrap(::Val{x}) where x = x
_unwrap(x) = x
## constructor
function WENDyProblem(
    _tt::AbstractVector{<:Real}, 
    U::AbstractVecOrMat{<:Real}, 
    _f!::Function, 
    J::Int;
    linearInParameters::Union{Bool,Val}=false, 
    noiseDist::Union{Type{<:Distribution},Val}=Normal, 
    params::WENDyParameters=WENDyParameters(),
    constraints::Union{Nothing,AbstractVector{<:Tuple{<:Real,<:Real}}}=nothing,
    ll::LogLevel=Warn
)
    linearInParameters, noiseDist = _unwrap(linearInParameters), _unwrap(noiseDist)
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "Building WENDyProblem"
        U = U isa AbstractVector ? reshape(U, length(U), 1) : U
        @info "  Estimate the noise in each dimension"
        tt, X = _transformData(noiseDist, _tt, U)
        σ = estimate_std(X)
        @info "  σ = $σ"
        Φ, Φ̇ = getTestFunctionMatrices(tt, X, params; ll=ll)
        @info "  Building the LHS to the residual"
        b = vec(-Φ̇ * X)
        @info "  Build the weak form from symbolic expressions of the ODE"
        wform = weakForm(_f!, tt, X, Φ, Φ̇, σ, b, J; linearInParameters=linearInParameters, noiseDist=noiseDist)
        return WENDyProblem(tt, X, Φ, Φ̇, σ, b, wform, params; noiseDist=noiseDist, constraints=constraints, ll=ll)
    end
end
## weak form negative log-likelihood, its gradient and hessian
"""
    ℓ(prob::WENDyProblem, p)

Weak form negative log-likelihood ℓ(p) = ½rᵀS⁻¹r + ½log|S| + (KD/2)log(2π), where r(p) is the weak residual and S(p) its covariance.
This is half the paper's ℓ(p) = logdet S + rᵀS⁻¹r plus a constant (same minimizer); kept for comparison with earlier results.
"""
ℓ(prob::WENDyProblem, p::AbstractVector{<:Real}) = ℓ(prob.ws, p)
"""
    ∇ℓ(prob::WENDyProblem, p)
    ∇ℓ!(gradℓ, prob::WENDyProblem, p)

Gradient of the weak form negative log-likelihood (∇ℓ! writes it into gradℓ)
"""
∇ℓ(prob::WENDyProblem, p::AbstractVector{<:Real}) = ∇ℓ(prob.ws, p)
∇ℓ!(gradℓ::AbstractVector{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}) = gradℓ .= ∇ℓ(prob, p)
"""
    ∇²ℓ(prob::WENDyProblem, p; approx=false)
    ∇²ℓ!(hessℓ, prob::WENDyProblem, p; approx=false)

Hessian of the weak form negative log-likelihood (∇²ℓ! writes it into hessℓ). With approx = true the curvature 
of the log-determinant term is left out (see approxHessian in WENDyParameters)
"""
∇²ℓ(prob::WENDyProblem, p::AbstractVector{<:Real}; approx::Bool=false) = ∇²ℓ(prob.ws, p; approx=approx)
∇²ℓ!(hessℓ::AbstractMatrix{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}; approx::Bool=false) = hessℓ .= ∇²ℓ(prob, p; approx=approx)
## weak residual, its jacobian and covariance
"""
    weakResidual(prob::WENDyProblem, p)
    weakResidual!(r, prob::WENDyProblem, p)

Weak residual r(p) = g(p) - b (weakResidual! writes it into r)
"""
weakResidual(prob::WENDyProblem, p::AbstractVector{<:Real}) = _r(prob.ws.wform, p)
weakResidual!(r::AbstractVector{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}) = r .= weakResidual(prob, p)
"""
    weakJacobian(prob::WENDyProblem, p)
    weakJacobian!(∇r, prob::WENDyProblem, p)

Jacobian of the weak residual wrt p (weakJacobian! writes it into ∇r)
"""
weakJacobian(prob::WENDyProblem, p::AbstractVector{<:Real}) = copy(_∇r(prob.ws.wform, p))
weakJacobian!(∇r::AbstractMatrix{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}) = ∇r .= _∇r(prob.ws.wform, p)
"""
    weakCovariance(prob::WENDyProblem, p)

Covariance of the weak residual S(p) = (1-δ)L(p)L(p)ᵀ + δI
"""
weakCovariance(prob::WENDyProblem, p::AbstractVector{<:Real}) = copy(_update!(prob.ws, p).S)
## uncertainty of the estimate
"""
    Ĉ(prob::WENDyProblem, p̂)
    Ĉ(prob::WENDyProblem, sol::WENDySolution)

Covariance of the estimate from linearizing the weak form at p̂: Ĉ = (∇ₚgᵀ S⁻¹ ∇ₚg)⁻¹, where ∇ₚg = ∇ₚr is the
jacobian of the weak residual and S(p̂) its covariance
"""
function Ĉ(prob::WENDyProblem, p::AbstractVector{<:Real})
    ∇g = weakJacobian(prob, p)
    Matrix(inv(Symmetric(∇g' * (_update!(prob.ws, p).Sfact \ ∇g))))
end
"""
    C_cons(prob::WENDyProblem, p̂)
    C_cons(prob::WENDyProblem, sol::WENDySolution)

Conservative covariance of the estimate C_cons = ∇ₚg⁺ S ∇ₚg⁺ᵀ ⪰ Ĉ at p̂, the one used for the intervals reported
in the paper
"""
function C_cons(prob::WENDyProblem, p::AbstractVector{<:Real})
    ∇g⁺ = pinv(weakJacobian(prob, p))
    Matrix(Symmetric(∇g⁺ * _update!(prob.ws, p).S * ∇g⁺'))
end
"""
    C_∇²ℓ(prob::WENDyProblem, p̂)
    C_∇²ℓ(prob::WENDyProblem, sol::WENDySolution)

Covariance of the estimate from the curvature of the weak negative log-likelihood at p̂: (∇²ₚℓ)⁻¹
"""
# ℓ here is the negative log-likelihood itself (½rᵀS⁻¹r + ½log|S| + const), so its inverse hessian is the
# asymptotic covariance; with the paper's ℓ = logdet S + rᵀS⁻¹r (twice this) it would be 2(∇²ₚℓ)⁻¹
C_∇²ℓ(prob::WENDyProblem, p::AbstractVector{<:Real}) = Matrix(inv(Symmetric(∇²ℓ(prob, p))))
for C in (:Ĉ, :C_cons, :C_∇²ℓ)
    @eval $C(prob::WENDyProblem, sol::WENDySolution) = $C(prob, sol.p̂)
end
