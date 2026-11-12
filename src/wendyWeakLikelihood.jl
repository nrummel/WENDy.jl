## The weak form negative log-likelihood ℓ(p) and its derivatives
##   1. Weak forms: what the weak residual r(p) and the covariance factor L(p) are made of (LinearWeakForm,
##      NonlinearWeakForm), built by weakForm
##   2. r(p), ∇r(p), L(p), the factor B(p) of ∇L(p), products with ∇L and the second order terms
##   3. Covariance strategies: how S(p) is formed and what each keeps at the current p (FactorCovariance,
##      BlockCovariance)
##   4. The workspace: what ℓ, ∇ℓ and ∇²ℓ share at the current p
##   5. ℓ, ∇ℓ, ∇²m and ∇²ℓ = ∇²m + ∇²logdet, one method per covariance strategy
##
## Notation (as in the paper). The data X (Mp1 × D, log U for LogNormal noise) at times tt, the test functions Φ and
## Φ̇ (K × Mp1) and the noise levels σ give the weak residual r(p) = g(p) - b, with g(p) = vec(Φ F(p)),
## F(p)[m,:] = f(xₘ, p, tₘ) and b = -vec(Φ̇X), and its covariance S(p) = (1-δ)L(p)L(p)ᵀ + δI, with L(p) = L₀ + Lₚ(p),
## L₀ = Φ̇ ∘ diag(σ) and Lₚ(p) = Φ ∘ ∇ₓf(x, p) ∘ diag(σ). Rows of r, L and S are indexed by (k,d) and columns of L by
## (m,d′), each flattened (KD and Mp1·D long). With y = S⁻¹r,
##     ℓ(p) = m(p) + logdet(p) + (KD/2)log(2π),    m(p) = ½rᵀS⁻¹r,    logdet(p) = ½log|S(p)|
## (half the paper's ℓ = logdet S + rᵀS⁻¹r plus a constant, with the same minimizer).
##
## How the derivatives are computed (adjoint form). ∂ⱼS = (1-δ)(∂ⱼL Lᵀ + L ∂ⱼLᵀ) is KD × KD, and forming it takes a
## KD × Mp1D × KD matrix product for each j. Wherever only its action on a vector, or its trace against S⁻¹, is needed,
## we use matrix-vector products through L (KD × Mp1D) and solves with the one factorization of S instead:
##     ∂ⱼS y      = (1-δ)(∂ⱼL(Lᵀy) + L(∂ⱼLᵀy))     two matvecs                          (W[:,j] in ∇²m)
##     yᵀ∂ⱼS y    = 2(1-δ)(∂ⱼLᵀy)ᵀ(Lᵀy)           one matvec per j                     (∇m)
##     tr(S⁻¹∂ⱼS) = 2(1-δ)⟨S⁻¹L, ∂ⱼL⟩             one solve S⁻¹L shared by every j      (∇logdet)
## Neither is ∇L (KD × Mp1D × J) formed: ∂ⱼL = Φ ∘ B[:,:,j,:] with B = ∇ₚ∇ₓf σ (D × D × J × Mp1, K times smaller), and
## every product with ∂ⱼL above is taken through Φ and B (section 2), which also takes fewer operations.
## So ∇ℓ costs one factorization of S, one solve with Mp1·D right hand sides and J inner products, and ∇²m needs only
## solves with J right hand sides (S⁻¹∇r and S⁻¹W). Only the curvature of logdet, ½tr(S⁻¹∂ᵢS S⁻¹∂ⱼS), needs the J
## matrices S⁻¹∂ⱼS. The second derivatives of r and L only enter through their contractions with y and with a
## KD × Mp1D matrix, which are taken through Φ without forming ∇²r or ∇²L. When the ode is linear in p,
## L(p) = L₀ + ∑ⱼpⱼL₁ⱼ with L₁ⱼ = ∂ⱼL = Φ ∘ B[:,:,j,:], so S is a quadratic form in q = [1; p] of the blocks L̃ₐL̃ᵦᵀ, computed once
## (BlockCovariance): after construction nothing touches the Mp1·D dimension.

## 1. Weak forms
abstract type WeakForm end
"""
Weak form of an ode that is linear in its parameters: r(p) = Gp - b, and ∇ₓf is linear in p too, so the factor B of ∇L
does not depend on p (see section 2): ∂ⱼL = Φ ∘ B[:,:,j,:], with B[:,:,j,m] = ∇ₓf(xₘ, eⱼ, tₘ) diag(σ), and
L(p) = L₀ + Φ ∘ (∑ⱼpⱼB[:,:,j,:]). G[:,j] = g(eⱼ) and B (D × D × J × Mp1) are computed once, by _linearWeakForm.
The rhs f! and its state jacobian ∇ₓf! are kept for reference.
"""
struct LinearWeakForm <: WeakForm
    dims::WENDyDims
    Φ::Matrix{Float64}
    b::Vector{Float64}
    G::Matrix{Float64}
    L₀::Matrix{Float64}
    B::Array{Float64,4}
    f!::Function
    ∇ₓf!::Function
end
"""
Weak form of an ode that is not linear in its parameters: r(p), L(p) and their derivatives are evaluated from the data
and the symbolic rhs f!(du, x, p, t) and its derivatives ∇ₓf!, ∇ₚf! and ∇ₚ∇ₓf!. The second derivatives only enter through
the p-hessian of the scalar ψ(x, p, t; a, Ω) = ⟨a, f⟩ + ⟨Ω, ∇ₓf⟩ (see _∂²ψ): ∇²ₚψ!(out, x, p, t, a, Ω) writes its upper
triangle entries that are not structurally zero, at the linear indices ∇²ₚψidx of a J × J matrix.
"""
struct NonlinearWeakForm <: WeakForm
    dims::WENDyDims
    tt::Vector{Float64}
    X::Matrix{Float64}
    Φ::Matrix{Float64}
    σ::Vector{Float64}
    b::Vector{Float64}
    L₀::Matrix{Float64}
    # the symbolic rhs and its derivatives, kept ::Function so the kernels are not recompiled for every new problem
    # (see _atEachTime!)
    f!::Function
    ∇ₓf!::Function
    ∇ₚf!::Function
    ∇ₚ∇ₓf!::Function
    ∇²ₚψ!::Function
    ∇²ₚψidx::Vector{Int}
end
"""L₀ = Φ̇ ∘ diag(σ₁,⋯,σ_D) (KD × Mp1D), the part of L(p) that does not depend on p"""
function _L₀(Φ̇::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}, dims::WENDyDims)
    (; K, D, Mp1) = dims
    L₀ = zeros(K, D, Mp1, D)
    for d in 1:D
        L₀[:,d,:,d] = Φ̇ * σ[d]
    end
    reshape(L₀, K*D, Mp1*D)
end
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
    if linearInParameters
        @info " Computing G and B for linear problem"
        return _linearWeakForm(dims, tt, X, Φ, Φ̇, σ, b, fₙ!, ∇ₓf!)
    end
    @info " Computing additional symbolic functions for nonlinear problem"
    NonlinearWeakForm(
        dims, tt, X, Φ, σ, b, _L₀(Φ̇, σ, dims),
        fₙ!, ∇ₓf!, _get∇ₚf(fₙ!, D, J), _get∇ₚ∇ₓf(fₙ!, D, J), _get∇²ₚψ(fₙ!, D, J)...
    )
end

"""
The LinearWeakForm of the (transformed) rhs f! and its state jacobian ∇ₓf!: g and ∇ₓf σ at the unit vectors, from the
same kernels a nonlinear weak form uses.
"""
function _linearWeakForm(dims::WENDyDims, tt::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, Φ::AbstractMatrix{<:Real},
    Φ̇::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}, b::AbstractVector{<:Real}, f!::Function, ∇ₓf!::Function)
    (; J) = dims
    wf = (; dims, tt, X, Φ, σ, f!, ∇ₓf!)
    eⱼ(j) = [i == j ? 1.0 : 0.0 for i in 1:J]
    G = stack(_g(wf, eⱼ(j)) for j in 1:J)
    B = stack((_∇ₓFσ(wf, eⱼ(j)) for j in 1:J); dims=3) # D × D × J × Mp1
    LinearWeakForm(dims, Φ, b, G, _L₀(Φ̇, σ, dims), B, f!, ∇ₓf!)
end

## 2. r(p), ∇r(p), L(p), the factor B(p) of ∇L(p) and the second order terms
## The rhs, or one of its derivatives, at every time point: f!(out[⋯, m], xₘ, p, tₘ) for m = 1, ⋯, Mp1 (time is the
## last index of out). The weak forms keep the symbolic functions as ::Function, so this loop (and the call of ∇²ₚψ! in
## _∂²ψ) is the only code that is compiled for each problem's functions; the kernels that call them are compiled once.
function _atEachTime!(out::AbstractArray, f!::Function, X::AbstractMatrix, p::AbstractVector, tt::AbstractVector, Mp1::Int)
    for m in 1:Mp1
        @views f!(selectdim(out, ndims(out), m), X[m,:], p, tt[m])
    end
    out
end
"""
g(p) = vec(Φ F(p)) with F(p)[m,:] = f(xₘ, p, tₘ) (KD). wf is a NonlinearWeakForm, or its fields (dims, tt, X, Φ, f!)
while a linear weak form is built.
"""
function _g(wf, p::AbstractVector{T}) where T<:Real
    (; dims, tt, X, Φ, f!) = wf
    (; D, Mp1) = dims
    Fᵀ = _atEachTime!(zeros(T, D, Mp1), f!, X, p, tt, Mp1)
    vec(Φ * permutedims(Fᵀ))
end
## Every p-dependent part of L is Φ times a D × D matrix at each time point:
##     Lₚ(p)[(k,d₂),(m,d₁)] = Φ[k,m] A[d₂,d₁,m],      A = ∇ₓF σ,       A[d₂,d₁,m] = ∂_{x_{d₁}} f_{d₂}(xₘ, p, tₘ) σ[d₁]
##     ∂ⱼL[(k,d₂),(m,d₁)]   = Φ[k,m] B[d₂,d₁,j,m], B = ∇ₚ∇ₓF σ,    B[d₂,d₁,j,m] = ∂ⱼ∂_{x_{d₁}} f_{d₂}(xₘ, p, tₘ) σ[d₁]
## so ∇L (KD × Mp1D × J, K times the size of B) is never formed: every product with it is taken through Φ and B.
"""
A = ∇ₓF σ (D × D × Mp1), the factor of Lₚ(p). wf is a NonlinearWeakForm, or its fields (dims, tt, X, σ, ∇ₓf!) while a
linear weak form is built.
"""
function _∇ₓFσ(wf, p::AbstractVector{T}) where T<:Real
    (; dims, tt, X, σ, ∇ₓf!) = wf
    (; D, Mp1) = dims
    ∇ₓF = _atEachTime!(zeros(T, D, D, Mp1), ∇ₓf!, X, p, tt, Mp1) # ∇ₓF[:,:,m] = ∇ₓf(xₘ, p, tₘ)
    ∇ₓF .*= reshape(σ, 1, D, 1)
end
"""Φ ∘ A (KD × Mp1D): out[(k,d₂),(m,d₁)] = Φ[k,m] A[d₂,d₁,m]"""
function _expand(Φ::AbstractMatrix, A::AbstractArray{<:Real,3}, dims::WENDyDims)
    (; K, D, Mp1) = dims
    @tullio out[k,d2,m,d1] := Φ[k,m] * A[d2,d1,m]
    reshape(out, K*D, Mp1*D)
end
"""
Lₚ(p) = Φ ∘ ∇ₓf(x, p) ∘ diag(σ) (KD × Mp1D), the part of L(p) that depends on p. wf is a NonlinearWeakForm, or its
fields (dims, tt, X, Φ, σ, ∇ₓf!). A linear weak form sums its B instead, since ∇ₓf(x, p) = ∑ⱼpⱼ∇ₓf(x, eⱼ).
"""
_Lₚ(wf, p::AbstractVector{<:Real}) = _expand(wf.Φ, _∇ₓFσ(wf, p), wf.dims)
function _Lₚ(wf::LinearWeakForm, p::AbstractVector{<:Real})
    (; B) = wf
    @tullio A[d2,d1,m] := B[d2,d1,j,m] * p[j]
    _expand(wf.Φ, A, wf.dims)
end
## r(p) = g(p) - b (KD) and its jacobian ∇r(p) = ∇ₚg(p) (KD × J)
_r(wf::LinearWeakForm, p::AbstractVector{<:Real}) = wf.G*p - wf.b
_r(wf::NonlinearWeakForm, p::AbstractVector{<:Real}) = _g(wf, p) - wf.b
_∇r(wf::LinearWeakForm, p::AbstractVector{<:Real}) = wf.G
function _∇r(wf::NonlinearWeakForm, p::AbstractVector{<:Real})
    (; dims, tt, X, Φ, ∇ₚf!) = wf
    (; K, D, Mp1, J) = dims
    ∇ₚF = _atEachTime!(zeros(D, J, Mp1), ∇ₚf!, X, p, tt, Mp1) # ∇ₚF[:,:,m] = ∇ₚf(xₘ, p, tₘ)
    @tullio ∇r[k,d,j] := Φ[k,m] * ∇ₚF[d,j,m]
    reshape(∇r, K*D, J)
end
## L(p) (KD × Mp1D) and the factor B(p) of ∇L(p) (D × D × J × Mp1, constant for a linear weak form)
_L(wf::WeakForm, p::AbstractVector{<:Real}) = wf.L₀ + _Lₚ(wf, p)
_B(wf::LinearWeakForm, p::AbstractVector{<:Real}) = wf.B
function _B(wf::NonlinearWeakForm, p::AbstractVector{<:Real})
    (; dims, tt, X, σ, ∇ₚ∇ₓf!) = wf
    (; D, Mp1, J) = dims
    B = _atEachTime!(zeros(D, D, J, Mp1), ∇ₚ∇ₓf!, X, p, tt, Mp1) # ∇ₚ∇ₓF[:,:,:,m] = ∇ₚ∇ₓf(xₘ, p, tₘ)
    B .*= reshape(σ, 1, D, 1, 1)
end
## Products with ∇L through Φ and B. The contractions over B, or that keep the time index m on both sides, read each
## entry about once (memory bound), so they are loops (@tullio); the sums over k, KD or Mp1D are BLAS products.
"""
T[d₂,m,d₁] = ∑ₖ Φ[k,m] M[(k,d₂),(m,d₁)] (D × Mp1 × D) for a KD × Mp1D matrix M: what any ⟨Φ ∘ A, M⟩ needs of M, as
⟨Φ ∘ A, M⟩ = ∑ A[d₂,d₁,m] T[d₂,m,d₁].
"""
function _ΦᵀM(Φ::AbstractMatrix, M::AbstractMatrix, dims::WENDyDims)
    (; K, D, Mp1) = dims
    M₄ = reshape(M, K, D, Mp1, D)
    @tullio T[d2,m,d1] := Φ[k,m] * M₄[k,d2,m,d1]
end
"""T = _ΦᵀM(Φ, y uᵀ) for vectors y (KD) and u (Mp1D), without forming y uᵀ: T[d₂,m,d₁] = (ΦᵀY)[m,d₂] U[m,d₁]"""
function _ΦᵀM(Φ::AbstractMatrix, y::AbstractVector, u::AbstractVector, dims::WENDyDims)
    (; K, D, Mp1) = dims
    ΦᵀY, U = Φ' * reshape(y, K, D), reshape(u, Mp1, D)
    @tullio T[d2,m,d1] := ΦᵀY[m,d2] * U[m,d1]
end
"""⟨∂ⱼL, M⟩ for every j (J), from T = _ΦᵀM(Φ, M)"""
_∇L_dot(B::AbstractArray{<:Real,4}, T::AbstractArray{<:Real,3}) = @tullio g[j] := B[d2,d1,j,m] * T[d2,m,d1]
"""vⱼ = ∂ⱼLᵀy for every j (Mp1D × J): V[(m,d₁),j] = ∑_{d₂} B[d₂,d₁,j,m] (ΦᵀY)[m,d₂]"""
function _∇Lᵀy(Φ::AbstractMatrix, B::AbstractArray{<:Real,4}, y::AbstractVector, dims::WENDyDims)
    (; K, D, Mp1, J) = dims
    ΦᵀY = Φ' * reshape(y, K, D)
    @tullio V[m,d1,j] := B[d2,d1,j,m] * ΦᵀY[m,d2]
    reshape(V, Mp1*D, J)
end
"""∂ⱼL u for every j (KD × J), u of length Mp1D: (∂ⱼLu)[(k,d₂)] = ∑ₘ Φ[k,m] ∑_{d₁} B[d₂,d₁,j,m] u[(m,d₁)]"""
function _∇L_u(Φ::AbstractMatrix, B::AbstractArray{<:Real,4}, u::AbstractVector, dims::WENDyDims)
    (; K, D, Mp1, J) = dims
    U = reshape(u, Mp1, D)
    @tullio Z[m,d2,j] := B[d2,d1,j,m] * U[m,d1]
    reshape(Φ * reshape(Z, Mp1, D*J), K*D, J)
end
"""
∂ⱼL Mᵀ (KD × KD) for one j and a KD × Mp1D matrix M, from Bⱼ = B[:,:,j,:]:
(∂ⱼL Mᵀ)[(k,d₂),r] = ∑ₘ Φ[k,m] Q[m,d₂,r],    Q[m,d₂,r] = ∑_{d₁} Bⱼ[d₂,d₁,m] M[r,(m,d₁)]
(K·Mp1·D·KD for Q, then one BLAS product of Φ with Q, instead of the KD × Mp1D × KD product with a formed ∂ⱼL).
"""
function _∂L_Mᵀ(Φ::AbstractMatrix, Bⱼ::AbstractArray{<:Real,3}, M::AbstractMatrix, dims::WENDyDims)
    (; K, D, Mp1, KD) = dims
    M₃ = reshape(M, KD, Mp1, D)
    @tullio Q[m,d2,r] := Bⱼ[d2,d1,m] * M₃[r,m,d1]
    reshape(Φ * reshape(Q, Mp1, D*KD), KD, KD)
end
"""E = Φ ⊗ I_D as a KD × Mp1D matrix, E[(k,d₂),(m,d₂′)] = Φ[k,m]δ(d₂,d₂′), so that ∂ⱼL = E ∘ blockdiag(B[:,:,j,m])"""
function _ΦI(Φ::AbstractMatrix, dims::WENDyDims)
    (; K, D, Mp1) = dims
    E = zeros(K, D, Mp1, D)
    for d in 1:D
        E[:,d,:,d] = Φ
    end
    reshape(E, K*D, Mp1*D)
end
"""
Second order terms, without forming ∇²r or ∇²L (both are zero for a linear weak form). For a KD × Mp1D matrix M and a
scalar c, with Y = reshape(y, K, D) and T = _ΦᵀM(Φ, M),
    ∂ᵢⱼrᵀy = ∑ ∂ᵢⱼf_d(xₘ) (ΦᵀY)[m,d]    and    ⟨∂ᵢⱼL, M⟩ = ∑ ∂ᵢⱼ∂_{x_{d₁}}f_{d₂}(xₘ) σ[d₁] T[d₂,m,d₁],
so both are p-hessians of one scalar per time point,
    ∂ᵢⱼrᵀy + c⟨∂ᵢⱼL, M⟩ = ∑ₘ ∂ᵢⱼψ(xₘ, p, tₘ; aₘ, Ωₘ),    ψ(x, p, t; a, Ω) = ⟨a, f(x, p, t)⟩ + ⟨Ω, ∇ₓf(x, p, t)⟩,
with aₘ = (ΦᵀY)[m,:] and Ωₘ[d₂,d₁] = c σ[d₁] T[d₂,m,d₁]. The generated ∇²ₚψ! (_get∇²ₚψ) writes only the upper
triangle entries that are not structurally zero, so a time point costs about their number, instead of the D × J × J
and D × D × J × J tensors ∇²ₚf and ∇²ₚ∇ₓf (written in full, zeros included) and a product with each.
"""
function _∂²ψ(wf::NonlinearWeakForm, p::AbstractVector{<:Real}, y::AbstractVector{<:Real}, c::Real, T::AbstractArray{<:Real,3})
    (; dims, tt, X, Φ, σ, ∇²ₚψ!, ∇²ₚψidx) = wf
    (; K, D, Mp1, J) = dims
    ΦᵀY = Φ' * reshape(y, K, D)
    vals, H, a, Ω = zeros(length(∇²ₚψidx)), zeros(J, J), zeros(D), zeros(D, D)
    for m in 1:Mp1
        a .= view(ΦᵀY, m, :)
        Ω .= c .* view(T, :, m, :) .* σ'
        @views ∇²ₚψ!(vals, X[m,:], p, tt[m], a, Ω)
        for (k, l) in enumerate(∇²ₚψidx)
            H[l] += vals[k]
        end
    end
    H + triu(H, 1)'
end

## 3. Covariance strategies: how S(p) = (1-δ)L(p)L(p)ᵀ + δI is formed, and what is kept at the current p
abstract type CovarianceStrategy end
"""
S(p) from L(p), for any weak form. Keeps L at the current p, and computes the factor B of ∇L, S⁻¹L, T = _ΦᵀM(Φ, S⁻¹L),
Lᵀy and the vⱼ = ∂ⱼLᵀy the first time something asks for them there (∇ℓ, ∇²m and ∇²ℓ at the same p share them).
"""
@kwdef mutable struct FactorCovariance <: CovarianceStrategy
    L::Matrix{Float64} = zeros(0, 0)
    B::Union{Nothing,Array{Float64,4}} = nothing
    S⁻¹L::Union{Nothing,Matrix{Float64}} = nothing
    ΦᵀS⁻¹L::Union{Nothing,Array{Float64,3}} = nothing
    Lᵀy::Union{Nothing,Vector{Float64}} = nothing
    ∇Lᵀy::Union{Nothing,Matrix{Float64}} = nothing # [v₁ ⋯ v_J]
end
"""
S(p) = (1-δ)∑ₐᵦ qₐqᵦ L̃ₐL̃ᵦᵀ + δI, with q = [1; p] and L̃ = [L₀, L₁₁, ⋯, L₁_J], L₁ⱼ = ∂ⱼL, from the blocks
P[:,a,b] = vec(L̃ₐL̃ᵦᵀ) of a linear weak form, computed once, so nothing per p touches the Mp1·D dimension. Keeps q at
the current p, and computes S⁻¹ (see _S⁻¹!) and the contractions C(S⁻¹ - yyᵀ) and C(yyᵀ) (see _C) the first time
something asks for them there: ∇ℓ needs the first, ∇²m the second, and ∇²ℓ C(S⁻¹), which is their sum.
"""
@kwdef mutable struct BlockCovariance <: CovarianceStrategy
    P::Array{Float64,3}
    q::Vector{Float64} = Float64[]
    S⁻¹::Union{Nothing,Matrix{Float64}} = nothing
    C_S⁻¹_yy::Union{Nothing,Matrix{Float64}} = nothing # C(S⁻¹ - yyᵀ)
    C_yy::Union{Nothing,Matrix{Float64}} = nothing     # C(yyᵀ)
end
BlockCovariance(P::Array{Float64,3}) = BlockCovariance(; P)
# Linear weak forms use the blocks unless they would take more than this much memory
const MAX_LLᵀ_BLOCK_BYTES = 2^30
covarianceStrategy(wf::NonlinearWeakForm) = FactorCovariance()
function covarianceStrategy(wf::LinearWeakForm)
    (; J, KD) = wf.dims
    (J+1)^2*KD^2*sizeof(Float64) <= MAX_LLᵀ_BLOCK_BYTES ? BlockCovariance(_LLᵀBlocks(wf)) : FactorCovariance()
end
# P[:,a,b] = vec(L̃ₐL̃ᵦᵀ) for every pair of blocks, a, b = 1, ⋯, J+1, with L̃ₐ₊₁ = Φ ∘ B[:,:,a,:]: one L̃ₐ is formed at a
# time, and L̃ᵦL̃ₐᵀ for b > 1 is taken through Φ and B (_∂L_Mᵀ)
function _LLᵀBlocks(wf::LinearWeakForm)
    (; dims, Φ, L₀, B) = wf
    (; J, KD) = dims
    P = zeros(KD*KD, J+1, J+1)
    for a in 1:J+1
        L̃ₐ = a == 1 ? L₀ : _expand(Φ, view(B,:,:,a-1,:), dims)
        for b in a:J+1
            LᵦLₐᵀ = b == 1 ? L₀*L₀' : _∂L_Mᵀ(Φ, view(B,:,:,b-1,:), L̃ₐ, dims)
            P[:,a,b] = vec(LᵦLₐᵀ')
            P[:,b,a] = vec(LᵦLₐᵀ)
        end
    end
    P
end
## S(p), keeping what the strategy needs at p (and dropping what it kept for the previous p)
function _covariance!(c::FactorCovariance, wf::WeakForm, p::AbstractVector{<:Real}, diagReg::Real)
    c.L, c.B, c.S⁻¹L, c.ΦᵀS⁻¹L, c.Lᵀy, c.∇Lᵀy = _L(wf, p), nothing, nothing, nothing, nothing, nothing
    S = c.L * c.L'
    S .*= 1-diagReg
    S[diagind(S)] .+= diagReg
    S
end
function _covariance!(c::BlockCovariance, wf::LinearWeakForm, p::AbstractVector{<:Real}, diagReg::Real)
    (; KD) = wf.dims
    c.q, c.S⁻¹, c.C_S⁻¹_yy, c.C_yy = [1; p], nothing, nothing, nothing
    S = reshape(reshape(c.P, KD*KD, :) * vec(c.q*c.q'), KD, KD)
    S .*= 1-diagReg
    S[diagind(S)] .+= diagReg
    S
end
## pieces computed the first time something asks for them at the current p
_B!(c::FactorCovariance, ws) = isnothing(c.B) ? (c.B = _B(ws.wform, ws.p)) : c.B
_S⁻¹L!(c::FactorCovariance, ws) = isnothing(c.S⁻¹L) ? (c.S⁻¹L = ws.Sfact \ c.L) : c.S⁻¹L
_ΦᵀS⁻¹L!(c::FactorCovariance, ws) =
    isnothing(c.ΦᵀS⁻¹L) ? (c.ΦᵀS⁻¹L = _ΦᵀM(ws.wform.Φ, _S⁻¹L!(c, ws), ws.wform.dims)) : c.ΦᵀS⁻¹L
# Lᵀy and vⱼ = ∂ⱼLᵀy depend on y, which _update! sets after _covariance! has dropped them, so they are always taken
# from the current y
_Lᵀy!(c::FactorCovariance, ws) = isnothing(c.Lᵀy) ? (c.Lᵀy = c.L'ws.y) : c.Lᵀy
_∇Lᵀy!(c::FactorCovariance, ws) =
    isnothing(c.∇Lᵀy) ? (c.∇Lᵀy = _∇Lᵀy(ws.wform.Φ, _B!(c, ws), ws.y, ws.wform.dims)) : c.∇Lᵀy
# S⁻¹ itself, only for the log-det terms of the block strategy: they are traces ⟨S⁻¹, L̃ₐL̃ᵦᵀ⟩ against every block,
# which would otherwise take a KD × KD solve per block. It comes from the Cholesky factor (LAPACK potri), the usual way
# to get an SPD inverse for trace terms; every linear system is solved with ws.Sfact instead.
_S⁻¹!(c::BlockCovariance, ws) = isnothing(c.S⁻¹) ? (c.S⁻¹ = inv(ws.Sfact)) : c.S⁻¹
# Qⱼ = ∑ₐqₐL̃ₐL₁ⱼᵀ = L L₁ⱼᵀ (KD × KD), so ∂ⱼS = (1-δ)(Qⱼ + Qⱼᵀ)
_Q(c::BlockCovariance, j::Int, dims::WENDyDims) = reshape(view(c.P,:,:,j+1) * c.q, dims.KD, dims.KD)
# Cₐᵦ = ⟨B, L̃ₐL̃ᵦᵀ⟩ for every pair of blocks, a, b = 1, ⋯, J+1
_C(c::BlockCovariance, B::AbstractMatrix{<:Real}, dims::WENDyDims) = reshape(reshape(c.P, dims.KD^2, :)' * vec(B), dims.J+1, dims.J+1)
_C_S⁻¹_yy!(c::BlockCovariance, ws) =
    isnothing(c.C_S⁻¹_yy) ? (c.C_S⁻¹_yy = _C(c, _S⁻¹!(c, ws) - ws.y*ws.y', ws.wform.dims)) : c.C_S⁻¹_yy
_C_yy!(c::BlockCovariance, ws) = isnothing(c.C_yy) ? (c.C_yy = _C(c, ws.y*ws.y', ws.wform.dims)) : c.C_yy
# C is linear in B, so C(S⁻¹) = C(S⁻¹ - yyᵀ) + C(yyᵀ) reuses both cached contractions
_C_S⁻¹(c::BlockCovariance, ws) = _C_S⁻¹_yy!(c, ws) + _C_yy!(c, ws)

## 4. The workspace
## ℓ, ∇ℓ and ∇²ℓ are usually evaluated at the same p (by an optimizer), so they share this state. Whenever p changes
## (compared by value), S(p), its factorization, log|S|, r(p) and y = S⁻¹r are recomputed and the covariance strategy
## drops what it kept for the old p; everything else is computed the first time something asks for it.
@kwdef mutable struct WENDyWorkspace
    wform::WeakForm
    cov::CovarianceStrategy
    diagReg::Float64
    p::Vector{Float64} = Float64[] # the parameters everything below was computed for
    S::Matrix{Float64} = zeros(0, 0)
    Sfact::Factorization = cholesky(ones(1, 1))
    logDet::Float64 = 0.0
    r::Vector{Float64} = Float64[]
    y::Vector{Float64} = Float64[] # S⁻¹r
    ∇r::Union{Nothing,Matrix{Float64}} = nothing # on request
end
WENDyWorkspace(wform::WeakForm, params::WENDyParameters; cov::CovarianceStrategy=covarianceStrategy(wform)) =
    WENDyWorkspace(; wform, cov, diagReg=params.diagReg)
## Bring the workspace up to date with p
function _update!(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    ws.p == p && return ws
    ws.p = Float64[] # marks the state as invalid until everything below is done
    ws.∇r = nothing
    ws.S = _covariance!(ws.cov, ws.wform, p, ws.diagReg)
    ws.r = _r(ws.wform, p)
    ws.Sfact, ws.logDet = try
        Sfact = cholesky(Symmetric(ws.S))
        Sfact, 2*sum(log, diag(Sfact.U))
    catch
        Sfact = lu(ws.S)
        Sfact, sum(log.(filter(x-> x >0, diag(Sfact.U))))
    end
    ws.y = ws.Sfact \ ws.r
    ws.p = copy(p)
    ws
end
_∇r!(ws::WENDyWorkspace) = isnothing(ws.∇r) ? (ws.∇r = _∇r(ws.wform, ws.p)) : ws.∇r

## 5. ℓ, ∇ℓ, ∇²m and ∇²ℓ on the workspace (see wendyProblems.jl for the versions on a WENDyProblem)
function ℓ(ws::WENDyWorkspace, p::AbstractVector{<:Real})
    _update!(ws, p)
    (dot(ws.r, ws.y) + ws.logDet)/2 + ws.wform.dims.KD/2*log(2*pi)
end
∇ℓ(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = (_update!(ws, p); ∇ℓ(ws.cov, ws))
∇²m(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = (_update!(ws, p); ∇²m(ws.cov, ws))
∇²ℓ(ws::WENDyWorkspace, p::AbstractVector{<:Real}) = (_update!(ws, p); ∇²ℓ(ws.cov, ws))
"""
∇ℓ = ∇m + ∇logdet, with vⱼ = ∂ⱼLᵀy:
    ∂ⱼm      = ∂ⱼrᵀy - ½yᵀ∂ⱼSy = ∂ⱼrᵀy - (1-δ)vⱼᵀ(Lᵀy)
    ∂ⱼlogdet = ½tr(S⁻¹∂ⱼS)   = (1-δ)⟨S⁻¹L, ∂ⱼL⟩
both through Φ and B (section 2): vⱼ from ΦᵀY, and ⟨S⁻¹L, ∂ⱼL⟩ from T = _ΦᵀM(Φ, S⁻¹L) (D × Mp1 × D).
"""
function ∇ℓ(c::FactorCovariance, ws::WENDyWorkspace)
    (; y, diagReg) = ws
    ∇m = _∇r!(ws)'y - (1-diagReg) * (_∇Lᵀy!(c, ws)' * _Lᵀy!(c, ws))
    ∇logdet = (1-diagReg) * _∇L_dot(_B!(c, ws), _ΦᵀS⁻¹L!(c, ws))
    ∇m + ∇logdet
end
"""
With the blocks, ½yᵀ∂ⱼSy = (1-δ)yᵀQⱼy and ½tr(S⁻¹∂ⱼS) = (1-δ)⟨S⁻¹, Qⱼ⟩, and since Qⱼ = ∑ₐqₐL̃ₐL̃ⱼ₊₁ᵀ both are
∑ₐqₐCₐ,ⱼ₊₁ for C = ⟨B, L̃ₐL̃ᵦᵀ⟩ with B = yyᵀ and S⁻¹. So ∇ℓ = ∇m + ∇logdet takes one contraction with B = S⁻¹ - yyᵀ:
    ∂ⱼℓ = Gⱼᵀy + (1-δ)∑ₐqₐCₐ,ⱼ₊₁
"""
function ∇ℓ(c::BlockCovariance, ws::WENDyWorkspace)
    (; wform, y, diagReg) = ws
    wform.G'y + (1-diagReg) * _C_S⁻¹_yy!(c, ws)[:,2:end]'c.q
end
"""
The hessian ∇²ℓ = ∇²m + ∇²logdet. With y = S⁻¹r, vⱼ = ∂ⱼLᵀy and W[:,j] = ∂ⱼSy = (1-δ)(∂ⱼL(Lᵀy) + Lvⱼ) (matvecs only):
    ∂ᵢⱼm      = ∂ᵢⱼrᵀy + ∂ᵢrᵀS⁻¹∂ⱼr - (S⁻¹∂ᵢr)ᵀWⱼ - (S⁻¹∂ⱼr)ᵀWᵢ + WᵢᵀS⁻¹Wⱼ - (1-δ)[vᵢᵀvⱼ + ⟨∂ᵢⱼL, y(Lᵀy)ᵀ⟩]
    ∂ᵢⱼlogdet = (1-δ)[⟨S⁻¹∂ᵢL, ∂ⱼL⟩ + ⟨S⁻¹L, ∂ᵢⱼL⟩] - ½tr(AᵢAⱼ),   Aⱼ = S⁻¹∂ⱼS
∇²m only needs solves with J right hand sides (S⁻¹∇r, S⁻¹W), and is the approximate hessian (approxHessian = true).
For ⟨S⁻¹∂ᵢL, ∂ⱼL⟩, ∂ᵢL = E ∘ blockdiag(B[:,:,i,m]) with E = Φ ⊗ I_D, so one solve Z = S⁻¹E (Mp1D right hand sides)
replaces the J solves S⁻¹∂ᵢL: ⟨S⁻¹∂ᵢL, ∂ⱼL⟩ = ∑ Ẑ[e,m,d₂] B[d₂,d₁,i,m] B[e,d₁,j,m] with Ẑ = _ΦᵀM(Φ, Z). The J
matrices Aⱼ (KD × KD) take one product through Φ (_∂L_Mᵀ) and one solve each. The terms with second derivatives
(∂ᵢⱼrᵀy and ⟨∂ᵢⱼL, ·⟩, see _∂²ψ) are zero for a linear weak form; they are linear in their second argument, so
∇²ℓ takes ⟨∂ᵢⱼL, S⁻¹L - y(Lᵀy)ᵀ⟩ in one contraction instead of one for m and one for logdet.
"""
∇²m(c::CovarianceStrategy, ws::WENDyWorkspace) = _symmetric(_∇²m₁(c, ws) + _∇²m₂(ws.wform, c, ws))
∇²ℓ(c::CovarianceStrategy, ws::WENDyWorkspace) = _symmetric(_∇²m₁(c, ws) + _∇²logdet₁(c, ws) + _∇²ℓ₂(ws.wform, c, ws))
_symmetric(H) = (H + H') / 2
## the terms with first derivatives of r and L only (subscript 1) ...
function _∇²m₁(c::FactorCovariance, ws::WENDyWorkspace)
    (; wform, Sfact, diagReg) = ws
    δ̃ = 1-diagReg
    ∇r, L = _∇r!(ws), c.L
    S⁻¹∇r = Sfact \ ∇r
    Lᵀy, ∇Lᵀy = _Lᵀy!(c, ws), _∇Lᵀy!(c, ws) # Lᵀy, vⱼ
    W = δ̃ .* (_∇L_u(wform.Φ, _B!(c, ws), Lᵀy, wform.dims) .+ L*∇Lᵀy) # W[:,j] = ∂ⱼSy
    ∇r'S⁻¹∇r - δ̃ * ∇Lᵀy'∇Lᵀy - S⁻¹∇r'W - W'S⁻¹∇r + W' * (Sfact \ W)
end
function _∇²logdet₁(c::FactorCovariance, ws::WENDyWorkspace)
    (; wform, Sfact, diagReg) = ws
    (; dims, Φ) = wform
    (; J, KD) = dims
    δ̃ = 1-diagReg
    B, L = _B!(c, ws), c.L
    Ẑ = _ΦᵀM(Φ, Sfact \ _ΦI(Φ, dims), dims) # Ẑ[e,m,d₂] = ∑ₖ Φ[k,m] (S⁻¹E)[(k,e),(m,d₂)]
    @tullio ẐB[e,m,d1,i] := Ẑ[e,m,d2] * B[d2,d1,i,m]
    @tullio H[i,j] := ẐB[e,m,d1,i] * B[e,d1,j,m] # ⟨S⁻¹∂ᵢL, ∂ⱼL⟩
    A = Array{Float64}(undef, KD, KD, J) # A[:,:,j] = S⁻¹∂ⱼS
    for j in 1:J
        ∂ⱼLLᵀ = _∂L_Mᵀ(Φ, view(B,:,:,j,:), L, dims)
        A[:,:,j] = Sfact \ (δ̃ .* (∂ⱼLLᵀ .+ ∂ⱼLLᵀ'))
    end
    δ̃ * H - [dot(transpose(view(A,:,:,i)), view(A,:,:,j)) for i in 1:J, j in 1:J] / 2
end
# With the blocks ∂ⱼS = (1-δ)(Qⱼ + Qⱼᵀ), vᵢᵀvⱼ = ⟨yyᵀ, L₁ᵢL₁ⱼᵀ⟩ and ⟨S⁻¹∂ᵢL, ∂ⱼL⟩ = ⟨S⁻¹, L₁ᵢL₁ⱼᵀ⟩ (blocks i+1, j+1)
function _∇²m₁(c::BlockCovariance, ws::WENDyWorkspace)
    (; wform, y, Sfact, diagReg) = ws
    (; dims, G) = wform
    (; J) = dims
    δ̃ = 1-diagReg
    S⁻¹∇r = Sfact \ G # ∇r = G
    W = δ̃ .* reduce(hcat, (Qⱼ = _Q(c, j, dims); Qⱼ*y + Qⱼ'y) for j in 1:J) # W[:,j] = ∂ⱼSy
    G'S⁻¹∇r - δ̃ * _C_yy!(c, ws)[2:end,2:end] - S⁻¹∇r'W - W'S⁻¹∇r + W' * (Sfact \ W)
end
function _∇²logdet₁(c::BlockCovariance, ws::WENDyWorkspace)
    (; wform, Sfact, diagReg) = ws
    (; dims) = wform
    (; J) = dims
    δ̃ = 1-diagReg
    A = stack((Qⱼ = _Q(c, j, dims); Sfact \ (δ̃ .* (Qⱼ .+ Qⱼ'))) for j in 1:J) # A[:,:,j] = S⁻¹∂ⱼS
    δ̃ * _C_S⁻¹(c, ws)[2:end,2:end] - [dot(transpose(view(A,:,:,i)), view(A,:,:,j)) for i in 1:J, j in 1:J] / 2
end
## ... and the terms with second derivatives (subscript 2), for m and for ℓ (zero for a linear weak form)
_∇²m₂(wf::LinearWeakForm, c::CovarianceStrategy, ws::WENDyWorkspace) = zeros(wf.dims.J, wf.dims.J)
_∇²ℓ₂(wf::LinearWeakForm, c::CovarianceStrategy, ws::WENDyWorkspace) = zeros(wf.dims.J, wf.dims.J)
function _∇²m₂(wf::NonlinearWeakForm, c::FactorCovariance, ws::WENDyWorkspace)
    (; p, y, diagReg) = ws
    _∂²ψ(wf, p, y, -(1-diagReg), _ΦᵀM(wf.Φ, y, _Lᵀy!(c, ws), wf.dims))
end
function _∇²ℓ₂(wf::NonlinearWeakForm, c::FactorCovariance, ws::WENDyWorkspace)
    (; p, y, diagReg) = ws
    _∂²ψ(wf, p, y, 1-diagReg, _ΦᵀS⁻¹L!(c, ws) - _ΦᵀM(wf.Φ, y, _Lᵀy!(c, ws), wf.dims))
end
