using Test, WENDy, LinearAlgebra, Random, Logging, Statistics
using WENDy: Normal, LogNormal
using ForwardDiff
using OrdinaryDiffEq: ODEProblem, Rodas5P, solve as solve_ode

## Reference weak negative log-likelihood written directly from its definition, generic in the element type so
## ForwardDiff can differentiate it. It shares only the L(p) and r(p) kernels with the package.
Lₚ_ref(wform::WENDy.LinearWeakForm, p) = reshape(reshape(wform.L₁, :, length(p)) * p, size(wform.L₀))
Lₚ_ref(wform::WENDy.NonlinearWeakForm, p) = WENDy._Lₚ(p, wform.tt, wform.X, wform.Φ, wform.σ, wform.∇ₓf!)
function Lr_ref(prob::WENDyProblem, p::AbstractVector)
    L₀ = kron(Diagonal(prob.σ), prob.Φ̇) # L₀ = Φ̇ ∘ diag(σ)
    L₀ + Lₚ_ref(prob.ws.wform, p), WENDy._r(prob.ws.wform, p)
end
S_ref(params, L) = (1-params.diagReg)*(L*L') + params.diagReg*I
function ℓ_ref(prob::WENDyProblem, params::WENDyParameters, p::AbstractVector)
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
    sol = solve_ode(ODEProblem(f!, u₀, tRng, pstar), Rodas5P(); saveat=dt, reltol=1e-10, abstol=1e-10)
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
            WENDyProblem(tt, U, c.f!, J; linearInParameters=c.lip, noiseDist=c.dist, params=c.params)
        end
        p = c.pstar .* (1 .+ 0.1 .* randn(MersenneTwister(2), J))
        f(q) = ℓ_ref(prob, c.params, q)
        gradℓref = ForwardDiff.gradient(f, p)
        hessℓref = ForwardDiff.hessian(f, p)
        @testset "$(c.name)" begin
            # linear problems use the blocks of LLᵀ, nonlinear ones build S from L(p)
            @test prob.ws.wform isa (c.lip ? WENDy.LinearWeakForm : WENDy.NonlinearWeakForm)
            @test prob.ws.cov isa (c.lip ? WENDy.BlockCovariance : WENDy.FactorCovariance)
            @test prob.noiseDist == c.dist
            # a workspace that builds S from L(p) is always valid, so compare both
            wsL = WENDy.WENDyWorkspace(prob.ws.wform, c.params; cov=WENDy.FactorCovariance())
            @test ℓ(prob, p) ≈ f(p) rtol=1e-10
            @test WENDy.ℓ(wsL, p) ≈ f(p) rtol=1e-10
            gradℓ = zeros(J)
            @test ∇ℓ!(gradℓ, prob, p) ≈ gradℓref rtol=1e-8
            @test WENDy.∇ℓ(wsL, p) ≈ gradℓref rtol=1e-8
            @test ∇ℓ(prob, p) ≈ gradℓ
            hessℓ = zeros(J,J)
            @test ∇²ℓ!(hessℓ, prob, p) ≈ hessℓref rtol=1e-8
            @test issymmetric(hessℓ)
            @test WENDy.∇²ℓ(wsL, p) ≈ hessℓref rtol=1e-8
            @test ∇²ℓ(prob, p) ≈ hessℓ
            # approximate hessian (no log-det curvature)
            hessℓapproxref = approxH_ref(prob, c.params, p)
            @test ∇²ℓ!(hessℓ, prob, p; approx=true) ≈ hessℓapproxref rtol=1e-8
            @test issymmetric(hessℓ)
            @test WENDy.∇²ℓ(wsL, p; approx=true) ≈ hessℓapproxref rtol=1e-8
            # weak residual, its jacobian and covariance
            L, r = Lr_ref(prob, p)
            KD = prob.K*prob.D
            @test weakResidual!(zeros(KD), prob, p) ≈ r rtol=1e-12
            @test weakResidual(prob, p) ≈ r rtol=1e-12
            @test weakJacobian!(zeros(KD, J), prob, p) ≈ ForwardDiff.jacobian(q -> Lr_ref(prob, q)[2], p) rtol=1e-8
            @test weakJacobian(prob, p) ≈ weakJacobian!(zeros(KD, J), prob, p)
            @test weakCovariance(prob, p) ≈ S_ref(c.params, L) rtol=1e-12
            # the workspace is never stale
            p₂ = c.pstar .* (1 .+ 0.1 .* randn(MersenneTwister(3), J))
            gradℓ₂ref = ForwardDiff.gradient(f, p₂)
            hessℓ₂ref = ForwardDiff.hessian(f, p₂)
            for _ in 1:2 # alternate p and p₂ so every call follows a call at the other point
                @test ℓ(prob, p₂) ≈ f(p₂) rtol=1e-10
                @test ∇ℓ!(gradℓ, prob, p) ≈ gradℓref rtol=1e-8
                @test ∇²ℓ!(hessℓ, prob, p₂) ≈ hessℓ₂ref rtol=1e-8
                weakResidual!(zeros(KD), prob, p₂) # must not disturb the cached state at p
                @test ℓ(prob, p) ≈ f(p) rtol=1e-10
                @test ∇ℓ!(gradℓ, prob, p₂) ≈ gradℓ₂ref rtol=1e-8
                @test ∇²ℓ!(hessℓ, prob, p) ≈ hessℓref rtol=1e-8
            end
            # optimizers often mutate the same array in place
            q = copy(p)
            ∇ℓ!(gradℓ, prob, q)
            q .= p₂
            @test ∇ℓ!(gradℓ, prob, q) ≈ gradℓ₂ref rtol=1e-8
            # after ℓ and ∇ℓ at p, ∇²ℓ at p reuses the factorization rather than recomputing it
            ℓ(prob, p); ∇ℓ!(gradℓ, prob, p)
            Sfact = prob.ws.Sfact
            ∇²ℓ!(hessℓ, prob, p)
            @test prob.ws.Sfact === Sfact
            @test hessℓ ≈ hessℓref rtol=1e-8
        end
    end
    @testset "Logistic solvers" begin
        c = CASES[1]
        tt, U = simulate(c.f!, c.u₀, c.tRng, c.dt, c.pstar, c.dist, c.nr)
        prob = with_logger(NullLogger()) do
            WENDyProblem(tt, U, c.f!, 2; linearInParameters=true, params=c.params)
        end
        oe = OutputErrorProblem(tt, U, c.f!, 2; params=c.params)
        p₀ = [0.5, 0.5]
        quiet(f) = with_logger(f, NullLogger())
        relErr(p̂) = norm(p̂[1:2] - c.pstar)/norm(c.pstar)
        sol = quiet(() -> solve(prob, p₀, c.params))
        p̂ = sol.p̂
        @test sol isa WENDySolution && sol.retcode == :success && sol.alg == "TrustRegion" && isnothing(sol.û₀)
        @test sol.iterations > 0 && size(sol.iterates) == (2, sol.iterations + 1) && sol.iterates[:,1] == p₀ && sol.time > 0
        # earlier versions took Val(...) keywords
        probVal = quiet(() -> WENDyProblem(tt, U, c.f!, 2; linearInParameters=Val(true), noiseDist=Val(Normal), params=c.params))
        @test probVal.ws.wform isa WENDy.LinearWeakForm
        @test ℓ(probVal, p₀) ≈ ℓ(prob, p₀) rtol=1e-10 # test function selection is not bitwise reproducible
        @test relErr(p̂) < 0.05
        @test norm(∇ℓ!(zeros(2), prob, p̂)) < 1e-3*norm(∇ℓ!(zeros(2), prob, p₀))
        # approximate hessian
        p̂ₐ = quiet(() -> solve(prob, p₀, WENDyParameters(Kmax=25, approxHessian=true))).p̂
        @test norm(p̂ₐ - p̂)/norm(p̂) < 1e-6
        # other weak form solvers
        for alg in (WENDy.ARCqK(), WENDy.IRLS(), WENDy.WLS())
            s = quiet(() -> solve(prob, p₀, c.params; alg=alg))
            @test relErr(s.p̂) < 0.1 && s.iterations > 0 && size(s.iterates, 1) == 2 && s.iterates[:,1] == p₀
            @test s.retcode == :success && s.alg == string(nameof(typeof(alg)))
        end
        # output error least squares estimates p and u₀ by default
        oes = quiet(() -> solve(oe, p₀))
        @test length(oes.p̂) == 2 && length(oes.û₀) == 1 && relErr(oes.p̂) < 0.05 && size(oes.iterates, 1) == 2
        @test norm(oeResidual!(zeros(length(U)), oe, vcat(oes.p̂, oes.û₀))) < norm(oeResidual!(zeros(length(U)), oe, vcat(p₀, oe.u₀)))
        @test isnothing(quiet(() -> solve(oe, p₀, WENDyParameters(fsU0Free=false))).û₀)
        # hybrid: weak form MLE then output error
        hy = quiet(() -> solve(prob, p₀, c.params; alg=WENDy.HybridTrustRegionOELS(oe)))
        oeFromMLE = quiet(() -> solve(oe, p̂))
        @test hy.p̂ ≈ oeFromMLE.p̂ rtol=1e-8
        @test hy.û₀ ≈ oeFromMLE.û₀ rtol=1e-8
        @test hy.iterations == sol.iterations + oeFromMLE.iterations && size(hy.iterates, 1) == 2 && hy.iterates[:,1] == p₀
        # parameter covariances against their definitions
        G, S = weakJacobian(prob, p̂), weakCovariance(prob, p̂)
        Ĉref = inv(G' * (S \ G))
        @test Ĉ(prob, p̂) ≈ Ĉref rtol=1e-8
        @test Ĉ(prob, sol) == Ĉ(prob, p̂)
        @test C_cons(prob, p̂) ≈ pinv(G) * S * pinv(G)' rtol=1e-8
        @test C_cons(prob, p̂) ≈ (G \ S) / G' rtol=1e-8 # as the scripts computed it
        @test eigmin(Symmetric(C_cons(prob, p̂) - Ĉ(prob, p̂))) > -1e-10 * opnorm(Ĉ(prob, p̂)) # C_cons ⪰ Ĉ
        @test C_∇²ℓ(prob, p̂) ≈ inv(ForwardDiff.hessian(q -> ℓ_ref(prob, c.params, q), p̂)) rtol=1e-6
    end
end
