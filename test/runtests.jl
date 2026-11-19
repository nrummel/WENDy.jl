using Test, WENDy, LinearAlgebra, Random, Logging, Statistics
using WENDy: Normal, LogNormal
using ForwardDiff
using OrdinaryDiffEqRosenbrock: ODEProblem, Rodas5P, solve as solve_ode

## Reference weak negative log-likelihood written directly from its definition, generic in the element type so
## ForwardDiff can differentiate it. It shares only the L(p) and r(p) kernels with the package.
# a linear weak form evaluates ∇ₓf at p directly, rather than through its stored B
Lₚ_ref(wform::WENDy.LinearWeakForm, prob, p) = WENDy._Lₚ((; wform.dims, prob.tt, prob.X, prob.Φ, prob.σ, wform.∇ₓf!), p)
Lₚ_ref(wform::WENDy.NonlinearWeakForm, prob, p) = WENDy._Lₚ(wform, p)
function Lr_ref(prob::WENDyProblem, p::AbstractVector)
    L₀ = kron(Diagonal(prob.σ), prob.Φ̇) # L₀ = Φ̇ ∘ diag(σ)
    L₀ + Lₚ_ref(prob.ws.wform, prob, p), WENDy._r(prob.ws.wform, p)
end
S_ref(params, L) = (1-params.diagReg)*(L*L') + params.diagReg*I
function ℓ_ref(prob::WENDyProblem, params::WENDyParameters, p::AbstractVector)
    L, r = Lr_ref(prob, p)
    C = cholesky(Symmetric(S_ref(params, L)))
    z = C.L \ r
    (dot(z,z) + 2sum(log, diag(C.U)))/2 + length(r)/2*log(2pi)
end
## The approximate hessian is the hessian of the Mahalanobis distance m(p) = ½rᵀS⁻¹r
function mahal_ref(prob::WENDyProblem, params::WENDyParameters, p::AbstractVector)
    L, r = Lr_ref(prob, p)
    dot(r, S_ref(params, L) \ r)/2
end
approxH_ref(prob::WENDyProblem, params::WENDyParameters, p::AbstractVector) = ForwardDiff.hessian(q -> mahal_ref(prob, params, q), p)

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
    # linear in p but built as a nonlinear weak form: every second derivative is structurally zero
    (name="Lotka-Volterra (as nonlinear)", f! =lotkaVolterra!, u₀=[1.0,1.0], tRng=(0.0,10.0), dt=0.02, pstar=[1.5,1.0,1.0,3.0],
        lip=false, dist=Normal, nr=0.05, params=WENDyParameters(Kmax=25)),
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
            # the sizes, read once when the weak form is built and shared by the problem, weak form and workspace
            (; D, Mp1, K, KD) = prob.dims
            @test prob.dims.J == J && D == size(U, 2) && K == size(prob.Φ, 1) && KD == K*D && Mp1 == length(prob.tt)
            @test prob.dims === prob.ws.wform.dims
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
            @test ∇²m!(hessℓ, prob, p) ≈ hessℓapproxref rtol=1e-8
            @test issymmetric(hessℓ)
            @test WENDy.∇²m(wsL, p) ≈ hessℓapproxref rtol=1e-8
            # weak residual, its jacobian and covariance
            L, r = Lr_ref(prob, p)
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
            # the pieces cached at p (Lᵀy, vⱼ, the block contractions) give the same results whichever call comes first
            for w in (prob.ws, wsL), order in ((:g, :m, :h), (:h, :m, :g), (:m, :h, :g), (:m, :g, :h))
                WENDy.ℓ(w, p₂) # move away from p, dropping the caches
                for k in order
                    k == :g && @test WENDy.∇ℓ(w, p) ≈ gradℓref rtol=1e-8
                    k == :m && @test WENDy.∇²m(w, p) ≈ hessℓapproxref rtol=1e-8
                    k == :h && @test WENDy.∇²ℓ(w, p) ≈ hessℓref rtol=1e-8
                end
            end
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
        # IRLS stops at optimTimelimit with the last iterate
        s = quiet(() -> solve(prob, p₀, WENDyParameters(optimTimelimit=1e-9); alg=WENDy.IRLS()))
        @test s.retcode == :timelimit && s.iterations == 1 && size(s.iterates, 2) == 2 && s.p̂ == s.iterates[:,end]
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
    @testset "Exact noise level σ" begin
        quiet(f) = with_logger(f, NullLogger())
        # The estimate path with σ̂ overwritten by hand: the constructor's last steps on prob's (transformed) data and
        # test functions, with σ in place of estimate_std(X). Reusing prob's Φ matters: with more than Kmax test
        # functions, svds starts from a random vector, so two builds from the same data differ in Φ by rounding and signs.
        # With LogNormal noise X = log U already, and weakForm transforms the rhs, as in the constructor.
        function byHand(c, prob, σ)
            (; tt, X, Φ, Φ̇, b) = prob
            wform = WENDy.weakForm(c.f!, tt, X, Φ, Φ̇, σ, b, length(c.pstar); linearInParameters=c.lip, noiseDist=c.dist)
            quiet(() -> WENDyProblem(tt, X, Φ, Φ̇, σ, b, wform, c.params; noiseDist=c.dist))
        end
        # Lotka-Volterra (linear weak form) with some σ, Goodwin (nonlinear, LogNormal) with its true σ: the std of the log noise
        for (c, σ) in ((CASES[2], [0.07, 0.04]), (CASES[4], fill(CASES[4].nr, 3)))
            tt, U = simulate(c.f!, c.u₀, c.tRng, c.dt, c.pstar, c.dist, c.nr)
            J, D = length(c.pstar), length(σ)
            build(; kw...) = quiet(() -> WENDyProblem(tt, U, c.f!, J; linearInParameters=c.lip, noiseDist=c.dist, params=c.params, kw...))
            @testset "$(c.name)" begin
                est = build()
                @test !est.σKnown && est.σ == WENDy.estimate_std(est.X) # default unchanged
                given = build(σ=σ)
                ref = byHand(c, given, σ)
                @test given.σKnown && given.σ == σ && !ref.σKnown && ref.σ == σ
                @test given.X == est.X # the (transformed) data do not depend on σ
                p = c.pstar .* (1 .+ 0.1 .* randn(MersenneTwister(2), J))
                @test weakCovariance(given, p) == weakCovariance(ref, p)
                @test ℓ(given, p) == ℓ(ref, p) && ∇ℓ(given, p) == ∇ℓ(ref, p) && ∇²ℓ(given, p) == ∇²ℓ(ref, p)
                @test ∇²m(given, p) == ∇²m(ref, p) && Ĉ(given, p) == Ĉ(ref, p) && C_cons(given, p) == C_cons(ref, p)
                @test ℓ(given, p) ≈ ℓ_ref(given, c.params, p) rtol=1e-10 # the reference likelihood built from prob.σ
                for alg in (WENDy.TrustRegion(), WENDy.IRLS())
                    @test quiet(() -> solve(given, p, c.params; alg=alg)).p̂ == quiet(() -> solve(ref, p, c.params; alg=alg)).p̂
                end
                # a scalar is the same σ for every state
                s = build(σ=σ[1])
                @test s.σ == fill(σ[1], D) && s.σKnown
                # a wrong σ: S - δI = (1-δ)L(σ)L(σ)ᵀ and L is linear in σ, so scaling σ by c scales S - δI by c²
                # (only σ changes, through the weak form: the test functions are given's)
                withσ(σ′) = byHand(c, given, σ′)
                δ = c.params.diagReg
                @test weakCovariance(withσ(2σ), p) - δ*I ≈ 4(weakCovariance(given, p) - δ*I) rtol=1e-10
                @test ℓ(withσ(2σ), p) != ℓ(given, p)
                # changing σ in one state d changes L only in its columns (m, d), by the same factor
                σ₂ = copy(σ); σ₂[end] *= 3
                L(prob) = first(Lr_ref(prob, p))
                L₁, L₂, Mp1 = L(given), L(withσ(σ₂)), given.dims.Mp1
                @test L₂[:, 1:(D-1)*Mp1] == L₁[:, 1:(D-1)*Mp1]
                @test L₂[:, (D-1)*Mp1+1:end] ≈ 3L₁[:, (D-1)*Mp1+1:end] rtol=1e-12
                # validation
                @test_throws DimensionMismatch build(σ=vcat(σ, 1.0))
                @test_throws ArgumentError build(σ=vcat(σ[1:end-1], 0.0))
                @test_throws ArgumentError build(σ=-1.0)
                @test_throws ArgumentError build(σ=fill(NaN, D))
            end
        end
    end
    @testset "Fixed test functions" begin
        c = CASES[1]
        quiet(f) = with_logger(f, NullLogger())
        tt, U = simulate(c.f!, c.u₀, c.tRng, c.dt, c.pstar, c.dist, 0.0) # noise free reference data, 1001 points
        tf = quiet(() -> FixedTestFunctions(tt, U, c.params))
        K = length(tf.Σ)
        # on the reference grid: the default test functions of that data (rows up to sign)
        Φ₀, Φ̇₀ = quiet(() -> WENDy.getTestFunctionMatrices(tt, U, c.params; ll=Warn))
        Φ, Φ̇ = tf(tt)
        @test size(Φ) == size(Φ₀) == (K, length(tt))
        s = sign.(vec(sum(Φ .* Φ₀; dims=2)))
        @test Φ ≈ s .* Φ₀ atol=1e-12
        @test Φ̇ ≈ s .* Φ̇₀ rtol=1e-10
        @test WENDy.getTestFunctionMatrices(tt, tf) == (Φ, Φ̇)
        # the full set and the reference SVD: Φ̃_full ≈ U_K Σ_K Φ, Φ = Σ_K⁻¹U_KᵀΦ̃_full, Φ̇ = Σ_K⁻¹U_KᵀΦ̃̇_full
        Φf, Φ̇f = fullTestFunctionMatrices(tt, tf)
        Φf₀, Φ̇f₀ = quiet(() -> WENDy.getTestFunctionMatrices(tt, U, c.params; noSVD=true, ll=Warn))
        @test size(Φf) == (length(tf.radii), length(tt))
        @test Φf ≈ Φf₀ atol=1e-12
        @test Φ̇f ≈ Φ̇f₀ rtol=1e-10
        @test Φ ≈ (tf.U' * Φf) ./ tf.Σ atol=1e-12
        @test Φ̇ ≈ (tf.U' * Φ̇f) ./ tf.Σ rtol=1e-10
        @test tf.U' * tf.U ≈ I atol=1e-10
        @test issorted(tf.Σ; rev=true) && tf.Σ ≈ svdvals(Matrix(Φf))[1:K]
        # the bump's derivative
        for x in (-0.9, -0.3, 0.0, 0.5, 0.99)
            @test WENDy._bump′(x) ≈ ForwardDiff.derivative(WENDy._bump, x) rtol=1e-12
        end
        @test WENDy._bump(1.0) == WENDy._bump′(-1.5) == 0
        # the same continuous functions on every grid: a finer grid that contains tt gives the same values there
        tt2 = collect(range(tt[1], tt[end], 2length(tt) - 1))
        Φ2, Φ̇2 = tf(tt2)
        @test Φ2[:, 1:2:end] ≈ Φ atol=1e-12
        @test Φ̇2[:, 1:2:end] ≈ Φ̇ rtol=1e-10
        @test fullTestFunctionMatrices(tt2, tf)[1][:, 1:2:end] ≈ Φf atol=1e-12
        # Φ̇ is the derivative of Φ on each grid: ∑ₘ ψₖ g′ = -∑ₘ ψ̇ₖ g (the ψₖ vanish at both ends)
        for t in (collect(range(tt[1], tt[end], 501)), tt, tt2)
            Φt, Φ̇t = tf(t)
            @test norm(Φt * cos.(t) + Φ̇t * sin.(t)) < 1e-6 * norm(Φt * cos.(t))
        end
        # a grid that does not cover the interval is refused
        @test_throws AssertionError tf(tt[1:end-10])
        # radii given in time units (not on the grid), and K given
        tfr = quiet(() -> FixedTestFunctions(tt, [0.123, 0.5]; K=10))
        @test length(tfr.Σ) == 10 && size(tfr.U) == (length(tfr.radii), 10) && unique(tfr.radii) == [0.123, 0.5]
        @test all(tfr.centers .- tfr.radii .>= tt[1]) && all(tfr.centers .+ tfr.radii .<= tt[end])
        Φr, Φ̇r = tfr(tt)
        @test Φr * Φr' ≈ I atol=1e-10 # orthonormal rows on the reference grid
        # WENDyProblem: Φ, Φ̇ depend only on the grid, not on the data
        probs = map(1:2) do seed
            _, Un = simulate(c.f!, c.u₀, c.tRng, c.dt, c.pstar, c.dist, c.nr; seed)
            quiet(() -> WENDyProblem(tt, Un, c.f!, 2; linearInParameters=true, params=c.params, testFunctions=tf))
        end
        @test probs[1].Φ == probs[2].Φ == Φ && probs[1].Φ̇ == Φ̇ && probs[1].dims.K == K
        @test probs[1].X != probs[2].X
        # ... and across grids, only through the data
        tt3, U3 = simulate(c.f!, c.u₀, c.tRng, c.dt/2, c.pstar, c.dist, c.nr)
        prob3 = quiet(() -> WENDyProblem(tt3, U3, c.f!, 2; linearInParameters=true, params=c.params, testFunctions=tf))
        @test prob3.Φ == first(tf(tt3)) && prob3.dims.K == K && prob3.dims.Mp1 == length(tt3)
        @test norm(quiet(() -> solve(prob3, [0.5, 0.5], c.params)).p̂ - c.pstar)/norm(c.pstar) < 0.05
    end
end
