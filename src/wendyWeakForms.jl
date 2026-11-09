## Weak forms: what the weak residual r(p) and the covariance factor L(p) are made of
abstract type WeakForm end
"""
Weak form of an ODE that is linear in its parameters: r(p) = Gp - b and L(p) = L₀ + ∑ⱼpⱼL₁[:,:,j] are
precomputed from the rhs f! and its state jacobian ∇ₓf!, which are kept only for reference.
"""
struct LinearWeakForm <: WeakForm
    dims::WENDyDims
    b::Vector{Float64}
    G::Matrix{Float64}
    L₀::Matrix{Float64}
    L₁::Array{Float64,3}
    f!::Function
    ∇ₓf!::Function
end
"""
Weak form of an ODE that is not linear in its parameters: r(p) = vec(Φ F(p)) - b with F(p)[m,:] = f(xₘ, p, tₘ)
and L(p) = L₀ + Φ ∘ ∇ₓf ∘ diag(σ), evaluated from the symbolic right hand side and its derivatives.
"""
struct NonlinearWeakForm <: WeakForm
    dims::WENDyDims
    tt::Vector{Float64}
    X::Matrix{Float64}
    Φ::Matrix{Float64}
    σ::Vector{Float64}
    b::Vector{Float64}
    L₀::Matrix{Float64}
    # symbolic rhs and derivatives (kept ::Function so the large kernels are not recompiled for every new problem)
    f!::Function
    ∇ₓf!::Function
    ∇ₚf!::Function
    ∇ₚ∇ₓf!::Function
    Hₚf!::Function
    Hₚ∇ₓf!::Function
end
## L(p), ∇L(p), r(p), ∇r(p)
_L(wform::LinearWeakForm, p::AbstractVector{<:Real}) = _L(p, wform.L₀, wform.L₁, wform.dims)
_L(wform::NonlinearWeakForm, p::AbstractVector{<:Real}) = wform.L₀ + _Lₚ(p, wform.tt, wform.X, wform.Φ, wform.σ, wform.∇ₓf!, wform.dims)
_∇L(wform::LinearWeakForm, p::AbstractVector{<:Real}) = wform.L₁
_∇L(wform::NonlinearWeakForm, p::AbstractVector{<:Real}) = _∇L(p, wform.tt, wform.X, wform.Φ, wform.σ, wform.∇ₚ∇ₓf!, wform.dims)
_r(wform::LinearWeakForm, p::AbstractVector{<:Real}) = wform.G*p - wform.b
_r(wform::NonlinearWeakForm, p::AbstractVector{<:Real}) = _r(p, wform.tt, wform.X, wform.Φ, wform.b, wform.f!, wform.dims)
_∇r(wform::LinearWeakForm, p::AbstractVector{<:Real}) = wform.G
_∇r(wform::NonlinearWeakForm, p::AbstractVector{<:Real}) = _∇r(p, wform.tt, wform.X, wform.Φ, wform.∇ₚf!, wform.dims)
## second order terms of the hessian, ∂ᵢⱼrᵀy + (1-δ)⟨∂ᵢⱼL, ∇Lℓ⟩ with y = S⁻¹r and ∇Lℓ = S⁻¹L - y(Lᵀy)ᵀ
# (the workspace ws holds p and y, and computes ∇Lℓ on request)
_Hsecond(wform::LinearWeakForm, ws) = 0I # ∂ᵢⱼr = ∂ᵢⱼL = 0
_Hsecond(wform::NonlinearWeakForm, ws) = _Hsecond(ws.p, wform.tt, wform.X, wform.Φ, wform.σ, wform.Hₚf!, wform.Hₚ∇ₓf!, ws.y, _∇Lℓ!(ws), ws.diagReg, wform.dims)
## build the weak form from the rhs f!(du, u, p, t) and the data
"""
    weakForm(f!, tt, X, Φ, Φ̇, σ, b, J; linearInParameters=false, noiseDist=Normal)

Build the weak form of du/dt = f(u, p, t) from the (transformed) data X at times tt, the test function matrices Φ and Φ̇,
the noise estimate σ and b = -vec(Φ̇X). With noiseDist = LogNormal the rhs is transformed to that of log(u).
The problem's sizes (WENDyDims) are read here, once, from X and Φ.
"""
function weakForm(f!::Function, tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, Φ::AbstractMatrix{<:Real}, Φ̇::AbstractMatrix{<:Real},
    σ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, J::Int;
    linearInParameters::Bool=false, noiseDist::Type{<:Distribution}=Normal
)
    dims = WENDyDims(; D=size(X, 2), J, Mp1=size(X, 1), K=size(Φ, 1))
    (; D) = dims
    fₙ! = _getf(f!, D, J, noiseDist)
    ∇ₓf! = _get∇ₓf(fₙ!, D, J)
    L₀ = _L₀(Φ̇, σ, dims)
    if linearInParameters
        @info " Computing G and L₁ for linear problem"
        return LinearWeakForm(dims, b, _buildGmat(fₙ!, tt, X, Φ, dims), L₀, _L₁(tt, X, Φ, σ, ∇ₓf!, dims), fₙ!, ∇ₓf!)
    end
    @info " Computing additional symbolic functions for nonlinear problem"
    NonlinearWeakForm(
        dims, tt, X, Φ, σ, b, L₀,
        fₙ!, ∇ₓf!, _get∇ₚf(fₙ!, D, J), _get∇ₚ∇ₓf(fₙ!, D, J), _getHₚf(fₙ!, D, J), _getHₚ∇ₓf(fₙ!, D, J)
    )
end
