using Test, WENDy, LinearAlgebra, Random, Logging, Statistics
using WENDy: Normal, LogNormal
using ForwardDiff
using OrdinaryDiffEq: ODEProblem, Rodas4P, solve as solve_ode

## Reference weak negative log-likelihood written directly from its definition, generic in the element type so
## ForwardDiff can differentiate it. It shares only the L(p) and r(p) kernels with the package.
function Lr_ref(prob::WENDyProblem{lip}, p::AbstractVector{T}) where {lip, T}
    d = prob.data
    Mp1, D = size(d.X)
    K, _ = size(d.V)
    KD = K*D
    L₀ = zeros(KD, Mp1*D)
    WENDy._L₀!(L₀, d.Vp, d.sig, zeros(K,D,D,Mp1), zeros(K,D,Mp1,D))
    L = zeros(T, KD, Mp1*D)
    r = zeros(T, KD)
    WENDy._L!(L, p, d.tt, d.X, d.V, L₀, d.sig, d.∇ₓf!, zeros(T,D,D,Mp1), zeros(T,K,D,Mp1,D))
    if lip
        r .= d.G*p .- d.b₀
    else
        WENDy._r!(r, p, d.tt, d.X, d.V, d.b₀, d.f!, zeros(T,Mp1,D), zeros(T,K,D))
    end
    L, r
end
S_ref(params, L) = (1-params.diagReg)*(L*L') + params.diagReg*I
function wnll_ref(prob::WENDyProblem, params::WENDyParameters, p::AbstractVector)
    L, r = Lr_ref(prob, p)
    C = cholesky(Symmetric(S_ref(params, L)))
    z = C.L \ r
    (dot(z,z) + 2sum(log, diag(C.U)))/2 + length(r)/2*log(2pi)
end
## The approximate hessian is the hessian of the Maholinobis distance ½rᵀS⁻¹r plus (1-δ)⟨S⁻¹L, ∂ᵢⱼL⟩ (zero when linear)
function mahal_ref(prob::WENDyProblem, params::WENDyParameters, p::AbstractVector)
    L, r = Lr_ref(prob, p)
    dot(r, S_ref(params, L) \ r)/2
end
function approxH_ref(prob::WENDyProblem, params::WENDyParameters, p::AbstractVector)
    L, _ = Lr_ref(prob, p)
    Z = S_ref(params, L) \ L
    ForwardDiff.hessian(q -> mahal_ref(prob, params, q), p) +
        (1-params.diagReg)*ForwardDiff.hessian(q -> dot(Z, Lr_ref(prob, q)[1]), p)
end

function simulate(f!, u₀, tRng, dt, pstar, dist, nr; seed=1)
    Random.seed!(seed)
    tt = collect(tRng[1]:dt:tRng[end])
    sol = solve_ode(ODEProblem(f!, u₀, tRng, pstar), Rodas4P(); saveat=dt, reltol=1e-10, abstol=1e-10)
    U = reduce(vcat, um' for um in sol.u)
    U = dist == LogNormal ? U .* exp.(nr*randn(size(U))) : U + nr*std(U)*randn(size(U))
    tt, U
end

function logistic!(du, u, p, t)
    du[1] = p[1]*u[1] - p[2]*u[1]^2
    nothing
end
function lotkaVolterra!(du, u, p, t)
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = p[3]*u[1]*u[2] - p[4]*u[2]
    nothing
end
function goodwin!(du, u, p, t)
    du[1] = p[1]/(2.15 + p[3]*u[3]^p[4]) - p[2]*u[1]
    du[2] = p[5]*u[1] - p[6]*u[2]
    du[3] = p[7]*u[2] - p[8]*u[3]
    nothing
end

const CASES = [
    (name="Logistic (linear)", f! =logistic!, u₀=[0.01], tRng=(0.0,10.0), dt=0.01, pstar=[1.0,1.0],
        lip=true, dist=Normal, nr=0.1, params=WENDyParameters(Kmax=25)),
    (name="Lotka-Volterra (linear)", f! =lotkaVolterra!, u₀=[1.0,1.0], tRng=(0.0,10.0), dt=0.02, pstar=[1.5,1.0,1.0,3.0],
        lip=true, dist=Normal, nr=0.05, params=WENDyParameters(Kmax=25)),
    (name="Goodwin (nonlinear, LogNormal)", f! =goodwin!, u₀=[0.3617,0.9137,1.3934], tRng=(0.0,80.0), dt=0.5,
        pstar=[3.4884,0.0969,1,10,0.0969,0.0581,0.0969,0.0775], lip=false, dist=LogNormal, nr=0.05,
        params=WENDyParameters(Kmax=25, radiusMinTime=0.5, radiusMaxTime=16.0)),
]

@testset "WENDy.jl" begin
    for c in CASES
        tt, U = simulate(c.f!, c.u₀, c.tRng, c.dt, c.pstar, c.dist, c.nr)
        J = length(c.pstar)
        prob = with_logger(NullLogger()) do
            WENDyProblem(tt, U, c.f!, J; linearInParameters=Val(c.lip), noiseDist=Val(c.dist), params=c.params)
        end
        p = c.pstar .* (1 .+ 0.1 .* randn(MersenneTwister(2), J))
        f(q) = wnll_ref(prob, c.params, q)
        gref = ForwardDiff.gradient(f, p)
        Href = ForwardDiff.hessian(f, p)
        @testset "$(c.name)" begin
            @test prob.wnll.f(p) ≈ f(p) rtol=1e-10
            # gradient: the implementation chosen for this problem and the adjoint form (always valid)
            ∇f! = prob.wnll.∇f!
            @test ∇f! isa (c.lip ? WENDy.LinearGradientWeakNLL : WENDy.AdjointGradientWeakNLL)
            g = zeros(J); ∇f!(g, p)
            @test g ≈ gref rtol=1e-8
            @test ∇f!(p) ≈ g
            adjointCache = WENDy.WeakNLLCache(prob.data, c.params; P=nothing)
            gₐ = zeros(J); WENDy.GradientWeakNLL(prob.data, c.params; cache=adjointCache)(gₐ, p)
            @test gₐ ≈ gref rtol=1e-8
            # hessian: same two implementations
            Hf! = prob.wnll.Hf!
            @test Hf! isa (c.lip ? WENDy.LinearHesianWeakNLL : WENDy.AdjointHesianWeakNLL)
            H = zeros(J,J); Hf!(H, p)
            @test H ≈ Href rtol=1e-8
            @test issymmetric(H)
            @test Hf!(p) ≈ H
            Hₐ = zeros(J,J); WENDy.HesianWeakNLL(prob.data, c.params; cache=adjointCache)(Hₐ, p)
            @test Hₐ ≈ Href rtol=1e-8
            @test WENDy.WeakNLL(prob.data, c.params; cache=adjointCache)(p) ≈ f(p) rtol=1e-10
            # approximate hessian (no log-det curvature), both implementations
            Happroxref = approxH_ref(prob, c.params, p)
            Hf!(H, p; approx=true); @test H ≈ Happroxref rtol=1e-8
            @test issymmetric(H)
            WENDy.HesianWeakNLL(prob.data, c.params; cache=adjointCache)(Hₐ, p; approx=true)
            @test Hₐ ≈ Happroxref rtol=1e-8
            # shared cache: one cache, never stale
            cache = prob.wnll.f.cache
            @test cache === ∇f!.cache === Hf!.cache
            p₂ = c.pstar .* (1 .+ 0.1 .* randn(MersenneTwister(3), J))
            g₂ref = ForwardDiff.gradient(f, p₂)
            H₂ref = ForwardDiff.hessian(f, p₂)
            for _ in 1:2 # alternate p and p₂ so every call follows a call at the other point
                @test prob.wnll.f(p₂) ≈ f(p₂) rtol=1e-10
                ∇f!(g, p);  @test g ≈ gref rtol=1e-8
                Hf!(H, p₂); @test H ≈ H₂ref rtol=1e-8
                @test prob.wnll.f(p) ≈ f(p) rtol=1e-10
                ∇f!(g, p₂); @test g ≈ g₂ref rtol=1e-8
                Hf!(H, p);  @test H ≈ Href rtol=1e-8
            end
            # optimizers often mutate the same array in place
            q = copy(p)
            ∇f!(g, q)
            q .= p₂
            ∇f!(g, q); @test g ≈ g₂ref rtol=1e-8
            # after f and ∇f at p, Hf at p reuses the factorization rather than recomputing it
            prob.wnll.f(p); ∇f!(g, p)
            F = cache.F
            Hf!(H, p)
            @test cache.F === F
            @test H ≈ Href rtol=1e-8
        end
    end
    @testset "Logistic solve recovers parameters" begin
        c = CASES[1]
        tt, U = simulate(c.f!, c.u₀, c.tRng, c.dt, c.pstar, c.dist, c.nr)
        prob = with_logger(NullLogger()) do
            WENDyProblem(tt, U, c.f!, 2; linearInParameters=Val(true), params=c.params)
        end
        p̂ = with_logger(NullLogger()) do
            solve(prob, [0.5, 0.5], c.params)
        end
        @test norm(p̂ - c.pstar)/norm(c.pstar) < 0.05
        p̂ₐ = with_logger(NullLogger()) do
            solve(prob, [0.5, 0.5], WENDyParameters(Kmax=25, approxHessian=true))
        end
        @test norm(p̂ₐ - p̂)/norm(p̂) < 1e-6
    end
end
