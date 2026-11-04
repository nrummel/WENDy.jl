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
            # linear problems use the blocks of LLᵀ, nonlinear ones build S from L(p)
            @test isnothing(prob.ws.P) == !c.lip
            # a workspace that builds S from L(p) is always valid, so compare both
            wsL = WENDy.WENDyWorkspace(prob.data, c.params; useBlocks=false)
            @test wnll(prob, p) ≈ f(p) rtol=1e-10
            @test WENDy.wnll(wsL, p) ≈ f(p) rtol=1e-10
            g = zeros(J); gₗ = zeros(J)
            @test ∇wnll!(g, prob, p) ≈ gref rtol=1e-8
            @test WENDy.∇wnll!(gₗ, wsL, p) ≈ gref rtol=1e-8
            H = zeros(J,J); Hₗ = zeros(J,J)
            @test Hwnll!(H, prob, p) ≈ Href rtol=1e-8
            @test issymmetric(H)
            @test WENDy.Hwnll!(Hₗ, wsL, p) ≈ Href rtol=1e-8
            # approximate hessian (no log-det curvature)
            Happroxref = approxH_ref(prob, c.params, p)
            @test Hwnll!(H, prob, p; approx=true) ≈ Happroxref rtol=1e-8
            @test issymmetric(H)
            @test WENDy.Hwnll!(Hₗ, wsL, p; approx=true) ≈ Happroxref rtol=1e-8
            # weak residual, its jacobian and covariance
            L, r = Lr_ref(prob, p)
            KD = prob.K*prob.D
            @test weakResidual!(zeros(KD), prob, p) ≈ r rtol=1e-12
            @test weakJacobian!(zeros(KD, J), prob, p) ≈ ForwardDiff.jacobian(q -> Lr_ref(prob, q)[2], p) rtol=1e-8
            @test weakCovariance(prob, p) ≈ S_ref(c.params, L) rtol=1e-12
            # the workspace is never stale
            p₂ = c.pstar .* (1 .+ 0.1 .* randn(MersenneTwister(3), J))
            g₂ref = ForwardDiff.gradient(f, p₂)
            H₂ref = ForwardDiff.hessian(f, p₂)
            for _ in 1:2 # alternate p and p₂ so every call follows a call at the other point
                @test wnll(prob, p₂) ≈ f(p₂) rtol=1e-10
                @test ∇wnll!(g, prob, p) ≈ gref rtol=1e-8
                @test Hwnll!(H, prob, p₂) ≈ H₂ref rtol=1e-8
                weakResidual!(zeros(KD), prob, p₂) # must not disturb the cached state at p
                @test wnll(prob, p) ≈ f(p) rtol=1e-10
                @test ∇wnll!(g, prob, p₂) ≈ g₂ref rtol=1e-8
                @test Hwnll!(H, prob, p) ≈ Href rtol=1e-8
            end
            # optimizers often mutate the same array in place
            q = copy(p)
            ∇wnll!(g, prob, q)
            q .= p₂
            @test ∇wnll!(g, prob, q) ≈ g₂ref rtol=1e-8
            # after m and ∇m at p, Hm at p reuses the factorization rather than recomputing it
            wnll(prob, p); ∇wnll!(g, prob, p)
            F = prob.ws.F
            Hwnll!(H, prob, p)
            @test prob.ws.F === F
            @test H ≈ Href rtol=1e-8
        end
    end
    @testset "Logistic solvers" begin
        c = CASES[1]
        tt, U = simulate(c.f!, c.u₀, c.tRng, c.dt, c.pstar, c.dist, c.nr)
        prob = with_logger(NullLogger()) do
            WENDyProblem(tt, U, c.f!, 2; linearInParameters=Val(true), params=c.params)
        end
        oe = OutputErrorProblem(tt, U, c.f!, 2; params=c.params)
        p₀ = [0.5, 0.5]
        quiet(f) = with_logger(f, NullLogger())
        relErr(p̂) = norm(p̂[1:2] - c.pstar)/norm(c.pstar)
        p̂ = quiet(() -> solve(prob, p₀, c.params))
        @test relErr(p̂) < 0.05
        @test norm(∇wnll!(zeros(2), prob, p̂)) < 1e-3*norm(∇wnll!(zeros(2), prob, p₀))
        # approximate hessian
        p̂ₐ = quiet(() -> solve(prob, p₀, WENDyParameters(Kmax=25, approxHessian=true)))
        @test norm(p̂ₐ - p̂)/norm(p̂) < 1e-6
        # other weak form solvers
        for alg in (WENDy.ARCqK(), WENDy.IRLS(), WENDy.WLS(), WENDy.HybridWLSTrustRegion())
            @test relErr(quiet(() -> solve(prob, p₀, c.params; alg=alg))) < 0.1
            what, iters, wits = quiet(() -> solve(prob, p₀, c.params; alg=alg, return_wits=true))
            @test relErr(what) < 0.1 && iters > 0 && size(wits, 1) == 2
        end
        # output error least squares estimates [p; u₀] by default
        ŵ = quiet(() -> solve(oe, p₀))
        @test length(ŵ) == 3 && relErr(ŵ) < 0.05
        @test norm(oeResidual!(zeros(length(U)), oe, ŵ)) < norm(oeResidual!(zeros(length(U)), oe, vcat(p₀, oe.u₀)))
        @test length(quiet(() -> solve(oe, p₀, WENDyParameters(fsU0Free=false)))) == 2
        # hybrid: weak form MLE then output error
        ŵₕ = quiet(() -> solve(prob, p₀, c.params; alg=WENDy.HybridTrustRegionOELS(oe)))
        @test ŵₕ ≈ quiet(() -> solve(oe, p̂)) rtol=1e-8
        what, iters, wits = quiet(() -> solve(prob, p₀, c.params; alg=WENDy.HybridTrustRegionOELS(oe), return_wits=true))
        @test what ≈ ŵₕ rtol=1e-8
        @test size(wits, 1) == 3
    end
end
