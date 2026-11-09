## The weak form negative log-likelihood ℓ(p) and its derivatives
##   1. Weak forms: what the weak residual r(p) and the covariance factor L(p) are made of (LinearWeakForm,
##      NonlinearWeakForm), built by weakForm
##   2. r(p), ∇r(p), L(p), ∇L(p) and the second order terms, one method per weak form
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
## So ∇ℓ costs one factorization of S, one solve with Mp1·D right hand sides and J inner products, and ∇²m needs only
## solves with J right hand sides (S⁻¹∇r and S⁻¹W). Only the curvature of logdet, ½tr(S⁻¹∂ᵢS S⁻¹∂ⱼS), needs the J
## matrices S⁻¹∂ⱼS. The second derivatives of r and L only enter through their contractions with y and with a
## KD × Mp1D matrix, which are taken through Φ without forming ∇²r or ∇²L. When the ode is linear in p,
## L(p) = L₀ + ∑ⱼpⱼL₁[:,:,j], so S is a quadratic form in q = [1; p] of the blocks L̃ₐL̃ᵦᵀ, computed once
## (BlockCovariance): after construction nothing touches the Mp1·D dimension.

## 1. Weak forms
abstract type WeakForm end
"""
Weak form of an ode that is linear in its parameters: r(p) = Gp - b and L(p) = L₀ + ∑ⱼpⱼL₁[:,:,j], precomputed as
G[:,j] = g(eⱼ) and L₁[:,:,j] = Lₚ(eⱼ). The rhs f! and its state jacobian ∇ₓf! are kept for reference.
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
Weak form of an ode that is not linear in its parameters: r(p), L(p) and their derivatives are evaluated from the data
and the symbolic rhs f!(du, x, p, t) and its derivatives ∇ₓf!, ∇ₚf!, ∇ₚ∇ₓf!, ∇²ₚf! and ∇²ₚ∇ₓf!.
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
    ∇²ₚf!::Function
    ∇²ₚ∇ₓf!::Function
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
    L₀ = _L₀(Φ̇, σ, dims)
    if linearInParameters
        @info " Computing G and L₁ for linear problem"
        # g and Lₚ at the unit vectors, from the same kernels a nonlinear weak form uses
        wf = (; dims, tt, X, Φ, σ, f! = fₙ!, ∇ₓf!)
        eⱼ(j) = [i == j ? 1.0 : 0.0 for i in 1:J]
        G = stack(_g(wf, eⱼ(j)) for j in 1:J)
        L₁ = stack(_Lₚ(wf, eⱼ(j)) for j in 1:J)
        return LinearWeakForm(dims, b, G, L₀, L₁, fₙ!, ∇ₓf!)
    end
    @info " Computing additional symbolic functions for nonlinear problem"
    NonlinearWeakForm(
        dims, tt, X, Φ, σ, b, L₀,
        fₙ!, ∇ₓf!, _get∇ₚf(fₙ!, D, J), _get∇ₚ∇ₓf(fₙ!, D, J), _get∇²ₚf(fₙ!, D, J), _get∇²ₚ∇ₓf(fₙ!, D, J)
    )
end

## 2. r(p), ∇r(p), L(p), ∇L(p) and the second order terms
## The rhs, or one of its derivatives, at every time point: f!(out[⋯, m], xₘ, p, tₘ) for m = 1, ⋯, Mp1 (time is the
## last index of out). The weak forms keep the symbolic functions as ::Function, so this loop is the only code that
## is compiled for each problem's functions; the kernels that call it are compiled once.
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
"""
Lₚ(p) = Φ ∘ ∇ₓf(x, p) ∘ diag(σ) (KD × Mp1D), the part of L(p) that depends on p. wf is a NonlinearWeakForm, or its
fields (dims, tt, X, Φ, σ, ∇ₓf!) while a linear weak form is built.
"""
function _Lₚ(wf, p::AbstractVector{T}) where T<:Real
    (; dims, tt, X, Φ, σ, ∇ₓf!) = wf
    (; K, D, Mp1) = dims
    ∇ₓF = _atEachTime!(zeros(T, D, D, Mp1), ∇ₓf!, X, p, tt, Mp1) # ∇ₓF[:,:,m] = ∇ₓf(xₘ, p, tₘ)
    @tullio Lₚ[k,d2,m,d1] := ∇ₓF[d2,d1,m] * Φ[k,m] * σ[d1]
    reshape(Lₚ, K*D, Mp1*D)
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
## L(p) (KD × Mp1D) and ∇L(p), with ∇L[:,:,j] = ∂ⱼL (KD × Mp1D × J)
function _L(wf::LinearWeakForm, p::AbstractVector{<:Real})
    (; dims, L₀, L₁) = wf
    (; J, KD, Mp1, D) = dims
    L₀ + reshape(reshape(L₁, :, J) * p, KD, Mp1*D)
end
_L(wf::NonlinearWeakForm, p::AbstractVector{<:Real}) = wf.L₀ + _Lₚ(wf, p)
_∇L(wf::LinearWeakForm, p::AbstractVector{<:Real}) = wf.L₁
function _∇L(wf::NonlinearWeakForm, p::AbstractVector{<:Real})
    (; dims, tt, X, Φ, σ, ∇ₚ∇ₓf!) = wf
    (; K, D, Mp1, J) = dims
    ∇ₚ∇ₓF = _atEachTime!(zeros(D, D, J, Mp1), ∇ₚ∇ₓf!, X, p, tt, Mp1) # ∇ₚ∇ₓF[:,:,:,m] = ∇ₚ∇ₓf(xₘ, p, tₘ)
    @tullio ∇L[k,d2,m,d1,j] := ∇ₚ∇ₓF[d2,d1,j,m] * Φ[k,m] * σ[d1]
    reshape(∇L, K*D, Mp1*D, J)
end
"""
Second order terms, contracted through Φ without forming ∇²r or ∇²L (both are zero for a linear weak form):
    ∂ᵢⱼrᵀy     = ∑ ∇²ₚF[d,i,j,m] (ΦᵀY)[m,d],                   Y = reshape(y, K, D)
    ⟨∂ᵢⱼL, M⟩ = ∑ ∇²ₚ∇ₓF[d₂,d₁,i,j,m] σ[d₁] (ΦᵀM)[d₂,m,d₁],    M (KD × Mp1D) reshaped to K × D × Mp1 × D
"""
function _∂²rᵀy(wf::NonlinearWeakForm, p::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    (; dims, tt, X, Φ, ∇²ₚf!) = wf
    (; K, D, Mp1, J) = dims
    ∇²ₚF = _atEachTime!(zeros(D, J, J, Mp1), ∇²ₚf!, X, p, tt, Mp1) # ∇²ₚF[:,:,:,m] = ∇²ₚf(xₘ, p, tₘ)
    ΦᵀY = Φ' * reshape(y, K, D)
    @tullio H[i,j] := ∇²ₚF[d,i,j,m] * ΦᵀY[m,d]
end
function _∂²L_M(wf::NonlinearWeakForm, p::AbstractVector{<:Real}, M::AbstractMatrix{<:Real})
    (; dims, tt, X, Φ, σ, ∇²ₚ∇ₓf!) = wf
    (; K, D, Mp1, J) = dims
    ∇²ₚ∇ₓF = _atEachTime!(zeros(D, D, J, J, Mp1), ∇²ₚ∇ₓf!, X, p, tt, Mp1) # ∇²ₚ∇ₓF[:,:,:,:,m] = ∇²ₚ∇ₓf(xₘ, p, tₘ)
    M₄ = reshape(M, K, D, Mp1, D)
    @tullio ΦᵀM[d2,m,d1] := Φ[k,m] * M₄[k,d2,m,d1]
    @tullio H[i,j] := ∇²ₚ∇ₓF[d2,d1,i,j,m] * σ[d1] * ΦᵀM[d2,m,d1]
end

## 3. Covariance strategies: how S(p) = (1-δ)L(p)L(p)ᵀ + δI is formed, and what is kept at the current p
abstract type CovarianceStrategy end
"""
S(p) from L(p), for any weak form. Keeps L at the current p, and computes ∇L and S⁻¹L the first time something asks
for them there.
"""
@kwdef mutable struct FactorCovariance <: CovarianceStrategy
    L::Matrix{Float64} = zeros(0, 0)
    ∇L::Union{Nothing,Array{Float64,3}} = nothing
    S⁻¹L::Union{Nothing,Matrix{Float64}} = nothing
end
"""
S(p) = (1-δ)∑ₐᵦ qₐqᵦ L̃ₐL̃ᵦᵀ + δI, with q = [1; p] and L̃ = [L₀, L₁[:,:,1], ⋯, L₁[:,:,J]], from the blocks
P[:,a,b] = vec(L̃ₐL̃ᵦᵀ) of a linear weak form, computed once, so nothing per p touches the Mp1·D dimension. Keeps q at
the current p, and computes S⁻¹ the first time something asks for it there (see _S⁻¹!).
"""
@kwdef mutable struct BlockCovariance <: CovarianceStrategy
    P::Array{Float64,3}
    q::Vector{Float64} = Float64[]
    S⁻¹::Union{Nothing,Matrix{Float64}} = nothing
end
BlockCovariance(P::Array{Float64,3}) = BlockCovariance(; P)
# Linear weak forms use the blocks unless they would take more than this much memory
const MAX_LLᵀ_BLOCK_BYTES = 2^30
covarianceStrategy(wf::NonlinearWeakForm) = FactorCovariance()
function covarianceStrategy(wf::LinearWeakForm)
    (; J, KD) = wf.dims
    (J+1)^2*KD^2*sizeof(Float64) <= MAX_LLᵀ_BLOCK_BYTES ? BlockCovariance(_LLᵀBlocks(wf)) : FactorCovariance()
end
# P[:,a,b] = vec(L̃ₐL̃ᵦᵀ) for every pair of blocks, a, b = 1, ⋯, J+1
function _LLᵀBlocks(wf::LinearWeakForm)
    (; dims, L₀, L₁) = wf
    (; J, KD) = dims
    L̃(a) = a == 1 ? L₀ : view(L₁,:,:,a-1)
    P = zeros(KD*KD, J+1, J+1)
    for a in 1:J+1, b in a:J+1
        LₐLᵦᵀ = L̃(a) * L̃(b)'
        P[:,a,b] = vec(LₐLᵦᵀ)
        P[:,b,a] = vec(LₐLᵦᵀ')
    end
    P
end
## S(p), keeping what the strategy needs at p (and dropping what it kept for the previous p)
function _covariance!(c::FactorCovariance, wf::WeakForm, p::AbstractVector{<:Real}, diagReg::Real)
    c.L, c.∇L, c.S⁻¹L = _L(wf, p), nothing, nothing
    S = c.L * c.L'
    S .*= 1-diagReg
    S[diagind(S)] .+= diagReg
    S
end
function _covariance!(c::BlockCovariance, wf::LinearWeakForm, p::AbstractVector{<:Real}, diagReg::Real)
    (; KD) = wf.dims
    c.q, c.S⁻¹ = [1; p], nothing
    S = reshape(reshape(c.P, KD*KD, :) * vec(c.q*c.q'), KD, KD)
    S .*= 1-diagReg
    S[diagind(S)] .+= diagReg
    S
end
## pieces computed the first time something asks for them at the current p
_∇L!(c::FactorCovariance, ws) = isnothing(c.∇L) ? (c.∇L = _∇L(ws.wform, ws.p)) : c.∇L
_S⁻¹L!(c::FactorCovariance, ws) = isnothing(c.S⁻¹L) ? (c.S⁻¹L = ws.Sfact \ c.L) : c.S⁻¹L
# S⁻¹ itself, only for the log-det terms of the block strategy: they are traces ⟨S⁻¹, L̃ₐL̃ᵦᵀ⟩ against every block,
# which would otherwise take a KD × KD solve per block. It comes from the Cholesky factor (LAPACK potri), the usual way
# to get an SPD inverse for trace terms; every linear system is solved with ws.Sfact instead.
_S⁻¹!(c::BlockCovariance, ws) = isnothing(c.S⁻¹) ? (c.S⁻¹ = inv(ws.Sfact)) : c.S⁻¹
# Qⱼ = ∑ₐqₐL̃ₐL₁ⱼᵀ = L L₁ⱼᵀ (KD × KD), so ∂ⱼS = (1-δ)(Qⱼ + Qⱼᵀ)
_Q(c::BlockCovariance, j::Int, dims::WENDyDims) = reshape(view(c.P,:,:,j+1) * c.q, dims.KD, dims.KD)
# Cₐᵦ = ⟨B, L̃ₐL̃ᵦᵀ⟩ for every pair of blocks, a, b = 1, ⋯, J+1
_C(c::BlockCovariance, B::AbstractMatrix{<:Real}, dims::WENDyDims) = reshape(reshape(c.P, dims.KD^2, :)' * vec(B), dims.J+1, dims.J+1)

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
"""
function ∇ℓ(c::FactorCovariance, ws::WENDyWorkspace)
    (; wform, y, diagReg) = ws
    (; J) = wform.dims
    ∇L = _∇L!(c, ws)
    ∇Lᵀy = reduce(hcat, view(∇L,:,:,j)'y for j in 1:J) # vⱼ
    ∇m = _∇r!(ws)'y - (1-diagReg) * (∇Lᵀy' * (c.L'y))
    ∇logdet = (1-diagReg) * (reshape(∇L, :, J)' * vec(_S⁻¹L!(c, ws)))
    ∇m + ∇logdet
end
"""
With the blocks, ½yᵀ∂ⱼSy = (1-δ)yᵀQⱼy and ½tr(S⁻¹∂ⱼS) = (1-δ)⟨S⁻¹, Qⱼ⟩, and since Qⱼ = ∑ₐqₐL̃ₐL̃ⱼ₊₁ᵀ both are
∑ₐqₐCₐ,ⱼ₊₁ for C = ⟨B, L̃ₐL̃ᵦᵀ⟩ with B = yyᵀ and S⁻¹. So ∇ℓ = ∇m + ∇logdet takes one contraction with B = S⁻¹ - yyᵀ:
    ∂ⱼℓ = Gⱼᵀy + (1-δ)∑ₐqₐCₐ,ⱼ₊₁
"""
function ∇ℓ(c::BlockCovariance, ws::WENDyWorkspace)
    (; wform, y, diagReg) = ws
    (; dims, G) = wform
    C = _C(c, _S⁻¹!(c, ws) - y*y', dims)
    G'y + (1-diagReg) * C[:,2:end]'c.q
end
"""
The hessian ∇²ℓ = ∇²m + ∇²logdet. With y = S⁻¹r, vⱼ = ∂ⱼLᵀy and W[:,j] = ∂ⱼSy = (1-δ)(∂ⱼL(Lᵀy) + Lvⱼ) (matvecs only):
    ∂ᵢⱼm      = ∂ᵢⱼrᵀy + ∂ᵢrᵀS⁻¹∂ⱼr - (S⁻¹∂ᵢr)ᵀWⱼ - (S⁻¹∂ⱼr)ᵀWᵢ + WᵢᵀS⁻¹Wⱼ - (1-δ)[vᵢᵀvⱼ + ⟨∂ᵢⱼL, y(Lᵀy)ᵀ⟩]
    ∂ᵢⱼlogdet = (1-δ)[⟨S⁻¹∂ᵢL, ∂ⱼL⟩ + ⟨S⁻¹L, ∂ᵢⱼL⟩] - ½tr(AᵢAⱼ),   Aⱼ = S⁻¹∂ⱼS
∇²m only needs solves with J right hand sides (S⁻¹∇r, S⁻¹W), and is the approximate hessian (approxHessian = true);
∇²logdet needs the J matrices Aⱼ, one KD × Mp1D × KD product and solve each. The terms with second derivatives
(∂ᵢⱼrᵀy and ⟨∂ᵢⱼL, ·⟩, see _∂²rᵀy) are zero for a linear weak form; they are linear in their second argument, so
∇²ℓ takes ⟨∂ᵢⱼL, S⁻¹L - y(Lᵀy)ᵀ⟩ in one contraction instead of one for m and one for logdet.
"""
∇²m(c::CovarianceStrategy, ws::WENDyWorkspace) = _symmetric(_∇²m₁(c, ws) + _∇²m₂(ws.wform, c, ws))
∇²ℓ(c::CovarianceStrategy, ws::WENDyWorkspace) = _symmetric(_∇²m₁(c, ws) + _∇²logdet₁(c, ws) + _∇²ℓ₂(ws.wform, c, ws))
_symmetric(H) = (H + H') / 2
## the terms with first derivatives of r and L only (subscript 1) ...
function _∇²m₁(c::FactorCovariance, ws::WENDyWorkspace)
    (; wform, y, Sfact, diagReg) = ws
    (; J) = wform.dims
    δ̃ = 1-diagReg
    ∇r, ∇L, L = _∇r!(ws), _∇L!(c, ws), c.L
    S⁻¹∇r = Sfact \ ∇r
    Lᵀy = L'y
    ∇Lᵀy = reduce(hcat, view(∇L,:,:,j)'y for j in 1:J) # vⱼ
    W = δ̃ .* (reduce(hcat, view(∇L,:,:,j)*Lᵀy for j in 1:J) .+ L*∇Lᵀy) # W[:,j] = ∂ⱼSy
    ∇r'S⁻¹∇r - δ̃ * ∇Lᵀy'∇Lᵀy - S⁻¹∇r'W - W'S⁻¹∇r + W' * (Sfact \ W)
end
function _∇²logdet₁(c::FactorCovariance, ws::WENDyWorkspace)
    (; wform, Sfact, diagReg) = ws
    (; J, KD) = wform.dims
    δ̃ = 1-diagReg
    ∇L, L = _∇L!(c, ws), c.L
    H = zeros(J, J)
    A = Array{Float64}(undef, KD, KD, J) # A[:,:,j] = S⁻¹∂ⱼS
    for j in 1:J
        ∂ⱼL = @view ∇L[:,:,j]
        ∂ⱼLLᵀ = ∂ⱼL * L'
        A[:,:,j] = Sfact \ (δ̃ .* (∂ⱼLLᵀ .+ ∂ⱼLLᵀ'))
        H[:,j] = δ̃ * (reshape(∇L, :, J)' * vec(Sfact \ ∂ⱼL)) # (1-δ)⟨S⁻¹∂ⱼL, ∂ᵢL⟩ for every i
    end
    H - [dot(transpose(view(A,:,:,i)), view(A,:,:,j)) for i in 1:J, j in 1:J] / 2
end
# With the blocks ∂ⱼS = (1-δ)(Qⱼ + Qⱼᵀ), vᵢᵀvⱼ = ⟨yyᵀ, L₁ᵢL₁ⱼᵀ⟩ and ⟨S⁻¹∂ᵢL, ∂ⱼL⟩ = ⟨S⁻¹, L₁ᵢL₁ⱼᵀ⟩ (blocks i+1, j+1)
function _∇²m₁(c::BlockCovariance, ws::WENDyWorkspace)
    (; wform, y, Sfact, diagReg) = ws
    (; dims, G) = wform
    (; J) = dims
    δ̃ = 1-diagReg
    S⁻¹∇r = Sfact \ G # ∇r = G
    W = δ̃ .* reduce(hcat, (Qⱼ = _Q(c, j, dims); Qⱼ*y + Qⱼ'y) for j in 1:J) # W[:,j] = ∂ⱼSy
    G'S⁻¹∇r - δ̃ * _C(c, y*y', dims)[2:end,2:end] - S⁻¹∇r'W - W'S⁻¹∇r + W' * (Sfact \ W)
end
function _∇²logdet₁(c::BlockCovariance, ws::WENDyWorkspace)
    (; wform, Sfact, diagReg) = ws
    (; dims) = wform
    (; J) = dims
    δ̃ = 1-diagReg
    A = stack((Qⱼ = _Q(c, j, dims); Sfact \ (δ̃ .* (Qⱼ .+ Qⱼ'))) for j in 1:J) # A[:,:,j] = S⁻¹∂ⱼS
    δ̃ * _C(c, _S⁻¹!(c, ws), dims)[2:end,2:end] - [dot(transpose(view(A,:,:,i)), view(A,:,:,j)) for i in 1:J, j in 1:J] / 2
end
## ... and the terms with second derivatives (subscript 2), for m and for ℓ (zero for a linear weak form)
_∇²m₂(wf::LinearWeakForm, c::CovarianceStrategy, ws::WENDyWorkspace) = zeros(wf.dims.J, wf.dims.J)
_∇²ℓ₂(wf::LinearWeakForm, c::CovarianceStrategy, ws::WENDyWorkspace) = zeros(wf.dims.J, wf.dims.J)
function _∇²m₂(wf::NonlinearWeakForm, c::FactorCovariance, ws::WENDyWorkspace)
    (; p, y, diagReg) = ws
    _∂²rᵀy(wf, p, y) - (1-diagReg) * _∂²L_M(wf, p, y * (c.L'y)')
end
function _∇²ℓ₂(wf::NonlinearWeakForm, c::FactorCovariance, ws::WENDyWorkspace)
    (; p, y, diagReg) = ws
    _∂²rᵀy(wf, p, y) + (1-diagReg) * _∂²L_M(wf, p, _S⁻¹L!(c, ws) - y * (c.L'y)')
end
