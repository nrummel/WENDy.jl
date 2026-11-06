## Weak forms: what the weak residual r(p) and the covariance factor L(p) are made of
abstract type WeakForm end
"""
Weak form of an ODE that is linear in its parameters: r(p) = Gp - b₀ and L(p) = L₀ + ∑ⱼpⱼL₁[:,:,j] are
precomputed from the rhs f! and its state jacobian ∇ₓf!, which are kept only for reference.
"""
struct LinearWeakForm <: WeakForm
    b₀::Vector{Float64}
    G::Matrix{Float64}
    L₀::Matrix{Float64}
    L₁::Array{Float64,3}
    f!::Function
    ∇ₓf!::Function
end
"""
Weak form of an ODE that is not linear in its parameters: row block d of r(p) is V_d F(p)[:,d] - b₀_d with 
F(p)[m,:] = f(xₘ, p, tₘ), and L(p) = L₀ + Lₚ(p) (see _Lₚ), evaluated from the symbolic right hand side and its derivatives.
"""
struct NonlinearWeakForm <: WeakForm
    J::Int
    tt::Vector{Float64}
    X::Matrix{Float64}
    V::Vector{Matrix{Float64}} # test functions of each state variable
    sig::Vector{Float64}
    b₀::Vector{Float64}
    L₀::Matrix{Float64}
    # symbolic rhs and derivatives (kept ::Function so the large kernels are not recompiled for every new problem)
    f!::Function
    ∇ₓf!::Function
    ∇ₚf!::Function
    ∇ₚ∇ₓf!::Function
    Hₚf!::Function
    Hₚ∇ₓf!::Function
end
_J(w::LinearWeakForm) = size(w.G, 2)
_J(w::NonlinearWeakForm) = w.J
## L(p), ∇L(p), r(p), ∇r(p)
_L(w::LinearWeakForm, p::AbstractVector{<:Real}) = _L(p, w.L₀, w.L₁)
_L(w::NonlinearWeakForm, p::AbstractVector{<:Real}) = w.L₀ + _Lₚ(p, w.tt, w.X, w.V, w.sig, w.∇ₓf!)
_∇L(w::LinearWeakForm, p::AbstractVector{<:Real}) = w.L₁
_∇L(w::NonlinearWeakForm, p::AbstractVector{<:Real}) = _∇L(p, w.tt, w.X, w.V, w.sig, w.∇ₚ∇ₓf!)
_r(w::LinearWeakForm, p::AbstractVector{<:Real}) = w.G*p - w.b₀
_r(w::NonlinearWeakForm, p::AbstractVector{<:Real}) = _r(p, w.tt, w.X, w.V, w.b₀, w.f!)
_∇r(w::LinearWeakForm, p::AbstractVector{<:Real}) = w.G
_∇r(w::NonlinearWeakForm, p::AbstractVector{<:Real}) = _∇r(p, w.tt, w.X, w.V, w.∇ₚf!)
## second order terms of the hessian, ∂ᵢⱼrᵀy + (1-δ)⟨∂ᵢⱼL, M⟩ with y = S⁻¹r and M = S⁻¹L - y(Lᵀy)ᵀ
# (the workspace ws holds p and y, and computes M on request)
_Hsecond(w::LinearWeakForm, ws) = 0I # ∂ᵢⱼr = ∂ᵢⱼL = 0
_Hsecond(w::NonlinearWeakForm, ws) = _Hsecond(ws.p, w.tt, w.X, w.V, w.sig, w.Hₚf!, w.Hₚ∇ₓf!, ws.y, _M!(ws), ws.diagReg)
## build the weak form from the rhs f!(du, u, p, t) and the data
"""
    weakForm(f!, tt, X, V, Vp, sig, b₀, J; linearInParameters=false, noiseDist=Normal)

Build the weak form of du/dt = f(u, p, t) from the (transformed) data X at times tt, the test function matrices V_d and
V′_d of each state (vectors; a single matrix is used for every state), the noise estimate σ and b₀ (the stacked 
-V′_d X[:,d]). With noiseDist = LogNormal the rhs is transformed to that of log(u).
"""
function weakForm(f!::Function, tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, V, Vp,
    sig::AbstractVector{<:Real}, b₀::AbstractVector{<:Real}, J::Int;
    linearInParameters::Bool=false, noiseDist::Type{<:Distribution}=Normal
)
    D = size(X, 2)
    V, Vp = _perState(V, D), _perState(Vp, D)
    fₙ! = _getf(f!, D, J, noiseDist)
    ∇ₓf! = _get∇ₓf(fₙ!, D, J)
    L₀ = _L₀(Vp, sig)
    if linearInParameters
        @info " Computing G and L₁ for linear problem"
        return LinearWeakForm(b₀, _buildGmat(fₙ!, tt, X, V, J), L₀, _L₁(tt, X, V, sig, ∇ₓf!, J), fₙ!, ∇ₓf!)
    end
    @info " Computing additional symbolic functions for nonlinear problem"
    NonlinearWeakForm(
        J, tt, X, V, sig, b₀, L₀,
        fₙ!, ∇ₓf!, _get∇ₚf(fₙ!, D, J), _get∇ₚ∇ₓf(fₙ!, D, J), _getHₚf(fₙ!, D, J), _getHₚ∇ₓf(fₙ!, D, J)
    )
end
