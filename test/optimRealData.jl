@info "Loading dependencies"
@info " Loading generateNoise..."
includet("../src/wendyNoise.jl")
@info " Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info " Loading computeGradients..."
includet("../src/wendySymbolics.jl")
@info " Loading testFunctions..."
includet("../src/wendyTestFunctions.jl")
@info " Loading other dependencies..."
# std lib
using LinearAlgebra, Logging, Random
# external dependencies
using Optim, Tullio, BenchmarkTools
using Optim: minimizer, minimum, iterations, iteration_limit_reached
using LinearAlgebra, Tullio, BenchmarkTools, Random, Logging 
## 
# Specify the parameters/data/models
@info "Specifying the parameters/data/models..."
mdl                 = HINDMARSH_ROSE_MODEL
exampleFile         = joinpath(@__DIR__, "../data/HindmarshRose.bson")
ϕ                   = ExponentialTestFun()
diagReg            = 1e-10
noiseRatio         = 0.05
timeSubsampleRate = 2
mtParams           = 2 .^(0:3)
seed                = Int(1)
Kmin               = 10
Kmax               = Int(5e3)
pruneMeth           = SingularValuePruningMethod( 
    MtminRadMethod(),
    UniformDiscritizationMethod()
);
wTrue = Float64[ModelingToolkit.getdefault(p) for p in parameters(mdl)]
J = length(wTrue)
@info "Build julia functions from symbolic expressions of ODE..."
_,f!         = getRHS(mdl)
_,jacuf! = getJacobian(mdl);
_,jacwf! = getParameterJacobian(mdl);
Random.seed!(seed)
@info "Load data from file..."
data = BSON.load(exampleFile) 
tt_full = data[:t] 
U_full = data[:u] 
numRad = length(mtParams)
@info "Subsample data and add noise..."
tt = tt_full[1:timeSubsampleRate:end]
U = U_full[:,1:timeSubsampleRate:end]
U, noise, noise_ratio_obs, sigma = generateNoise(U, noiseRatio)
D, M = size(U)
@info "============================="
@info "Start of Algo..."
@info "Estimate the noise in each dimension..."
sig = estimate_std(U)
@info "Build test function matrices..."
V,Vp,Vfull = pruneMeth(tt,U,ϕ,Kmin,Kmax,mtParams);
K,_ = size(V)
@info "Build right hand side to NLS..."
b0 = reshape(-Vp * U', K*D);
# Get the cholesky decomposition of our current approximation of the covariance
function RTfun(U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, diagReg::Real, jacuf!::Function, w::AbstractVector)
    # Preallocate for L 
    D, M = size(U)
    K, _ = size(V)
    JuF = zeros(D,D,M)
    _L0 = zeros(K,D,D,M)
    _L1  = zeros(K,D,M,D)
    # Get _L1
    K,D,M,_ = size(_L1)
    @inbounds for m in 1:M
        jacuf!(view(JuF,:,:,m), w, view(U,:,m))
    end
    @tullio _L0[k,d2,d1,m] = V[k,m] * JuF[d2,d1,m] * sig[d1]
    @tullio _L0[k,d,d,m] += Vp[k,m]*sig[d]
    permutedims!(_L1,_L0,(1,2,4,3))
    L = reshape(_L1,K*D,M*D)
    # compute covariance
    S = L * L'
    # regularize for possible ill conditioning
    R = (1-diagReg)*S + diagReg*I
    # compute cholesky for lin solv efficiency
    cholesky!(Symmetric(R))
    return UpperTriangular(R)'
end
# 
function res(RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, b::AbstractVector, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Res Eval ++++"
        K, M = size(V)
        D, _ = size(U)
        @info " Evaluate F "
        dt = @elapsed a = @allocations begin 
            F = zeros(W, D, M)
            for m in 1:size(F,2)
                f!(view(F,:,m), w, view(U,:,m))
            end
        end
        @info "  $dt s, $a allocations"
        @info " Mat Mat mult "
        dt = @elapsed a = @allocations G = V * F'
        @info "  $dt s, $a allocations"
        @info " Reshape "
        dt = @elapsed a = @allocations g = reshape(G, K*D)
        @info "  $dt s, $a allocations"
        @info " Linear Solve "
        dt = @elapsed a = @allocations res = RT \ g
        @info "  $dt s, $a allocations"
        @info " Vec Vec add "
        dt = @elapsed a = @allocations res .-= b
        @info "  $dt s, $a allocations"
        @info "++++++++++++++++++"
        return res
    end
end

function ∇res(RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Jac Res Eval ++++"
        K, M = size(V)
        D, _ = size(U)
        J = length(w)
        @info " Evaluating jacobian of F"
        dt = @elapsed a = @allocations begin
            JwF = zeros(W,D,J,M)
            for m in 1:M
                jacwf!(view(JwF,:,:,m), w, view(U,:,m))
            end
        end
        @info "  $dt s, $a allocations"
        @info " Computing V ×_3 jacF with tullio"
        dt = @elapsed a = @allocations @tullio _JG[d,j,k] := V[k,m] * JwF[d,j,m] 
        @info "  $dt s, $a allocations"
        @info " permutedims"
        dt = @elapsed a = @allocations JG = permutedims(_JG,(3,1,2))
        @info "  $dt s, $a allocations"
        @info " Reshape"
        dt = @elapsed a = @allocations jacG = reshape(JG, K*D, J)
        @info "  $dt s, $a allocations"
        @info " Linsolve"
        dt = @elapsed a = @allocations ∇res = RT \ jacG
        @info "  $dt s, $a allocations"
        return ∇res
    end
end
# Define cost function 
function f(RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, b::AbstractVector, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W
    1/2 * norm(res(RT, U, V, b, f!, w; ll=ll))^2
end

function ∇f!(∇f::AbstractVector, RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, b::AbstractVector, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W
    ∇f .= ∇res(RT,U,V,jacwf!,w;ll=ll)' * res(RT, U, V, b, f!, w; ll=ll) 
    nothing
end
#
f(w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W = f(RT,U,V,b,f!,w;ll=ll)
∇f!(∇f::AbstractVector, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W = ∇f!(∇f,RT,U,V,b,jacwf!,w;ll=ll)
## 
@info "Test Allocations with with Float64"
w_rand = wTrue + norm(wTrue)* randn(J)
RT = RTfun(U,V,Vp,sig,diagReg,jacuf!,w_rand)
b = RT \ b0 
# because this problem is linear in w the jacobian is constant, 
# and we could solve this problem with backslash because 
# it is really a linear least squares problem
G = ∇res(RT,U,V,jacwf!,zeros(J))
w_star = G \ b 
##
res_w_rand = f(w_rand)
∇f_rand = zeros(J)
∇f!(∇f_rand,w_rand)
@assert !all(∇f_rand .== 0)
nothing 
##
@info " Benchmark f"
@time f(rand(J))
@info " Benchmark f! "
@time ∇f!(∇f_rand,rand(J))
nothing
##
@info "Test the NLS in isolation"
@time result = optimize(f, ∇f!, w_rand, LBFGS())
w_hat = minimizer(result)
f_hat = minimum(result)
iter = iterations(result)
iterFlag = iteration_limit_reached(result)
abserr = norm(w_star-w_hat)
relerr = abserr / norm(w_star)
@info "Absolute coeff error = $abserr"
@info "Relative coeff error = $relerr"
abserr = abs(f(w_star)- f_hat)
relerr = abs(f(w_star)- f_hat) / f(w_star)
@info "Absolute coeff error = $abserr"
@info "Relative coeff error = $relerr"
## Compute the non lin least square solution 
@info "Defining IRWLS_Nonlinear..."
function IRWLS_Nonlinear(U, V, Vp, b0, sig, diagReg, J, f!, jacuf!, jacwf!; ll=Logging.Info,maxIt=100, relTol=1e-10)
    with_logger(ConsoleLogger(stderr,ll)) do 
        @info "Initializing the linearization least squares solution  ..."
        D, M = size(U)
        K, _ = size(V)
        G0 = ∇res(Matrix{Float64}(I, K*D,K*D), U, V, jacwf!, w_rand)
        w0 = G0 \ b0 
        wit = zeros(J,maxIt)
        resit = zeros(J,maxIt)
        wnm1 = w0 
        wn = similar(w0)
        for n = 1:maxIt 
            RT = RTfun(U,V,Vp,sig,diagReg,jacuf!,wnm1)
            b = RT \ b0 
            G = ∇res(RT,U,V,jacwf!,zeros(J))
            w_star = G \ b 
            fn(w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W = f(RT,U,V,b,f!,w;ll=ll)
            ∇fn!(∇f::AbstractVector, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where W = ∇f!(∇f,RT,U,V,b,jacwf!,w;ll=ll)
            @info "Running local optimization method"
            dt = @elapsed a = @allocations result = optimize(fn, ∇fn!, w_rand, LBFGS())
            wn = minimizer(result)
            resn = wnm1-wn
            resit[:,n] = resn
            wit[:,n] = wn
            relErr = norm(w_star-wn) / norm(wn)
            @info """ Iteration $n
                iteration time     = $dt
                allocations        = $dt
                relative Error     = $relErr
                iterations         = $(iterations(result))
                max iteration flag = $(iteration_limit_reached(result))
                objective_value    = $(minimum(result))
            """
            if norm(resn) / norm(wnm1) < relTol
                resit = resit[:,1:n] 
                wit = wit[:,1:n] 
                @info "Convergence Criterion met!"
                return wn, wit, resit 
            end
            wnm1 = wn
        end
        @warn "Maxiteration met..."
        return wn, wit, resit 
    end
end 
##
@info "IRWLS (Nonlinear): "
@info "   Runtime info: "
@time what, wit, resit = IRWLS_Nonlinear(U, V, Vp, b0, sig, diagReg, J, f!, jacuf!, jacwf!; ll=Logging.Warn)
relErr = norm(wit[:,end] - wTrue) / norm(wTrue)
@info "   coeff rel err = $relErr"
@info "   iterations    = $(size(wit,2)-1)"