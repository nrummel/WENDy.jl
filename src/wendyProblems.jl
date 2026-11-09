## helper function to build G matrix: column j is r(eⱼ) + b₀, with row block d equal to V_d F(eⱼ)[:,d]
function _buildGmat(f!::Function, tt::AbstractVector{<:Real}, Y::AbstractMatrix{<:Real}, V::AbstractVector{<:AbstractMatrix{<:Real}}, J::Int)
    Mp1, D = size(Y)
    eⱼ = zeros(J)
    F = similar(Y)
    G = zeros(sum(size.(V, 1)), J)
    for j in 1:J 
        eⱼ .= 0
        eⱼ[j] = 1
        for m in 1:Mp1 
            @views f!(F[m,:], Y[m,:], eⱼ, tt[m])
        end 
        G[:,j] = reduce(vcat, V[d] * F[:,d] for d in 1:D)
    end
    G
end
"""
    WENDyProblem

A WENDyProblem holds the weak form of the parameter inverse problem: the (transformed) data, the test functions, 
the weak form (see LinearWeakForm, NonlinearWeakForm) and a workspace for evaluating the weak form negative 
log-likelihood (see wnll, ∇wnll!, Hwnll!) and the weak residual (see weakResidual!, weakJacobian!, weakCovariance).
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
    WENDyProblem(tt, X, V, Vp, sig, b₀, form::WeakForm, params::WENDyParameters; noiseDist=Normal, constraints=nothing, ll=Warn)
where form = weakForm(f!, tt, X, V, Vp, sig, b₀, J; linearInParameters, noiseDist). V and Vp are vectors with the 
test function matrices of each state (a single matrix is used for every state).

# Fields
- D::Int : number of state variables
- J::Int : number of parameters (to be estimated)
- Mp1::Int : (Mp1+1) number of data points in time 
- K::Vector{Int} : number of test functions of each state variable (see snrThreshold in WENDyParameters)
- tt, X : times and data (X = log(U) for LogNormal noise, with nonpositive data removed)
- V, Vp : test function matrices V_d and V′_d of each state variable
- sig : estimated noise level in each state variable
- b₀ : the stacked -V′_d X[:,d]
- noiseDist : the noise model the problem was built for, Normal or LogNormal
- constraints : vector of tuples containing linear constraints for each parameter
- ws::WENDyWorkspace : the weak form and shared state for m(p), ∇m(p) and Hm(p)
"""
struct WENDyProblem
    D::Int # number of state variables
    J::Int # number of parameters (to be estimated)
    Mp1::Int # (Mp1+1) number of data points in time 
    K::Vector{Int} # number of test functions of each state variable
    tt::Vector{Float64}
    X::Matrix{Float64}
    V::Vector{Matrix{Float64}}
    Vp::Vector{Matrix{Float64}}
    sig::Vector{Float64}
    b₀::Vector{Float64}
    noiseDist::Type{<:Distribution}
    constraints::Union{Nothing,AbstractVector{Tuple{<:Real,<:Real}}}
    ws::WENDyWorkspace
end
## constructor from transformed data and its weak form
function WENDyProblem(
    tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, V, Vp, 
    sig::AbstractVector{<:Real}, b₀::AbstractVector{<:Real}, form::WeakForm, params::WENDyParameters; 
    noiseDist::Type{<:Distribution}=Normal, constraints::Union{Nothing,AbstractVector{<:Tuple{<:Real,<:Real}}}=nothing, ll::LogLevel=Warn
)
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "  Building workspace"
        Mp1, D = size(X)
        V, Vp = _perState(V, D), _perState(Vp, D)
        WENDyProblem(D, _J(form), Mp1, size.(V, 1), tt, X, V, Vp, sig, b₀, noiseDist, constraints, WENDyWorkspace(form, params))
    end
end
# test function matrices of each state variable; a single matrix is shared by all of them
_perState(V::AbstractVector{<:AbstractMatrix}, D::Int) = V
_perState(V::AbstractMatrix, D::Int) = fill(V, D)
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
        sig = estimate_std(X)
        @info "  sig = $sig"
        V, Vp = testFunctionMatrices(tt, X, params; ll=ll)
        @info "  Building the LHS to the residual"
        b₀ = reduce(vcat, -Vp[d] * X[:,d] for d in 1:size(X, 2))
        @info "  Build the weak form from symbolic expressions of the ODE"
        form = weakForm(_f!, tt, X, V, Vp, sig, b₀, J; linearInParameters=linearInParameters, noiseDist=noiseDist)
        return WENDyProblem(tt, X, V, Vp, sig, b₀, form, params; noiseDist=noiseDist, constraints=constraints, ll=ll)
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
weakResidual(prob::WENDyProblem, p::AbstractVector{<:Real}) = _r(prob.ws.form, p)
weakResidual!(r::AbstractVector{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}) = r .= weakResidual(prob, p)
"""
    weakJacobian(prob::WENDyProblem, p)
    weakJacobian!(∇r, prob::WENDyProblem, p)

Jacobian of the weak residual wrt p (weakJacobian! writes it into ∇r)
"""
weakJacobian(prob::WENDyProblem, p::AbstractVector{<:Real}) = copy(_∇r(prob.ws.form, p))
weakJacobian!(∇r::AbstractMatrix{<:Real}, prob::WENDyProblem, p::AbstractVector{<:Real}) = ∇r .= _∇r(prob.ws.form, p)
"""
    weakCovariance(prob::WENDyProblem, p)

Covariance of the weak residual S(p) = (1-δ)L(p)L(p)ᵀ + δI
"""
weakCovariance(prob::WENDyProblem, p::AbstractVector{<:Real}) = copy(_update!(prob.ws, p).S)
