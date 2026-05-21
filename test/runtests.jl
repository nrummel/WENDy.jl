#!/usr/bin/env julia

using Random, Logging, LinearAlgebra, Printf # stdlib
using OrdinaryDiffEq: ODEProblem
using OrdinaryDiffEq: solve as solve_ode
using WENDy, Test
using WENDy: WLS, IRLS, TrustRegion, ARCqK
@testset "WENDy - Logistic Growth" begin

    # Generate data to test the WENDy Algorithms 
    Random.seed!(1)
    tRng            = (0.0, 10.0)
    pstar           = [1, 1]
    J               = length(pstar)
    p₀              = [2, 2] + randn(J)
    params          = WENDyParameters(;Kmax=1000)
    Mp1             = 101
    tt              = range(tRng..., length=Mp1)
    dt              = diff(tt)[1]
    u₀              = [0.01]
    D               = length(u₀)
    f!(du, u, p, t) = du[1] = p[1] * u[1] - p[2] * u[1]^2
    ode             = ODEProblem(f!, u₀, tRng, pstar)
    U_exact         = reduce(vcat, um' for um in solve_ode(ode, saveat=dt).u)
    nr              = 0.1
    U_normal        = U_exact + nr*randn(Mp1,D)
    U_lognormal     = U_exact .* exp.(nr*randn(Mp1,D))

    @testset "WENDy Additive Gaussian Noise" begin
        wendyProb  = WENDyProblem(tt, U_normal, f!, J; noiseDist=Val(Normal), ll=Warn)
        phat_tr    = solve(wendyProb, p₀, params; alg=TrustRegion());
        phat_arcqk = solve(wendyProb, p₀, params; alg=ARCqK());
        phat_irls  = solve(wendyProb, p₀, params; alg=IRLS());
        phat_wls   = solve(wendyProb, p₀, params; alg=WLS());
        @test norm(phat_tr - pstar) / norm(pstar) <= 0.1
        @test norm(phat_arcqk - pstar) / norm(pstar) <= 0.1
        @test norm(phat_irls - pstar) / norm(pstar) <= 0.1
        @test norm(phat_wls - pstar) / norm(pstar) <= 0.2
    end

    @testset "WENDy Multiplicative LogNormal Noise" begin
        wendyProb  = WENDyProblem(tt, U_lognormal, f!, J; noiseDist=Val(LogNormal), ll=Warn)
        phat_tr    = solve(wendyProb, p₀, params; alg=TrustRegion());
        phat_arcqk = solve(wendyProb, p₀, params; alg=ARCqK());
        phat_irls  = solve(wendyProb, p₀, params; alg=IRLS());
        phat_wls   = solve(wendyProb, p₀, params; alg=WLS());
        phat_tr = solve(wendyProb, p₀, params; alg=TrustRegion());
        @test norm(phat_tr - pstar) / norm(pstar) <= 0.1
        @test norm(phat_arcqk - pstar) / norm(pstar) <= 0.1
        @test norm(phat_irls - pstar) / norm(pstar) <= 0.1
        @test norm(phat_wls - pstar) / norm(pstar) <= 0.2
    end
    
    # Make an easier problem for OE-LS 
    U_easy  = U_exact + 0.01*randn(Mp1,D)
    @testset "OE-LS" begin
        oeProb   = WENDy.OutputErrorProblem(tt,U_easy,f!, J)
        puhat_OE = WENDy.solve(oeProb, p₀)
        u0hat_OE = puhat_OE[J+1:end]
        phat_OE  = puhat_OE[1:J]
        @test norm(phat_OE - pstar) / norm(pstar) <= 0.1
    end
end