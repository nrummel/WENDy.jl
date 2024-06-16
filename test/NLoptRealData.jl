@info "Loading dependencies"
@info " Loading noise calcutation..."
includet("../src/wendyNoise.jl")
@info " Loading exampleProblems..."
includet("../src/wendyData.jl")
includet("../examples/exampleProblems.jl")
@info " Loading symbolic calcutations..."
includet("../src/wendySymbolics.jl")
@info " Loading testFunctions..."
includet("../src/wendyTestFunctions.jl")
@info " Loading other dependencies..."
# std lib
using LinearAlgebra, Logging, Random
# external dependencies
using Tullio, BenchmarkTools
using Optimization, OptimizationNLopt
using LinearAlgebra, Tullio, BenchmarkTools, Random, Logging, NaNMath
## 
# Specify the parameters/data/models
@info "Specifying the parameters/data/models..."
# ex                 = FITZHUG_NAGUMO_MODEL # HINDMARSH_ROSE_MODEL
# exampleFile         = joinpath(@__DIR__, "../data/FitzHug_Nagumo.bson")
# mdl                 = LOOP_MODEL # HINDMARSH_ROSE_MODEL
# exampleFile         = joinpath(@__DIR__, "../data/Loop.bson")
S = _MENDES_S_VALS[1]
P = _MENDES_P_VALS[1]
mdl = @mtkbuild MENDES_PROB_S_P = MendesModel(S=S,P=P)
exampleFile         = joinpath(@__DIR__, "../data/Mendes_S=$(S)_P=$P.bson")
@info "Running for Model $exampleFile)"
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
_,jacuf! = getJacobian(mdl);s
# _,jacwf! = getParameterJacobian(mdl);
function jacwf!(jwfm, w, um)
    jwfm .= 0
    jwfm[1] = (*)((*)((*)(w[25], (/)(w[3], (^)((+)((+)(1, (^)((/)(w[2], w[18]), w[24])), (^)((/)(w[19], w[1]), w[25])), 2))), (/)(w[19], (^)(w[1], 2))), (^)((/)(w[19], w[1]), (+)(-1, w[25])))
    jwfm[7] = (+)((/)((*)(w[15], um[4]), (*)(w[33], (+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])))), (/)((*)((*)((*)(-1, (+)(w[1], (*)(-1, um[7]))), w[15]), um[4]), (*)((^)(w[33], 2), (^)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), 2))))
    jwfm[9] = (/)((*)((*)((*)(-1, w[24]), (^)((/)(w[2], w[18]), (+)(-1, w[24]))), (/)(w[3], (^)((+)((+)(1, (^)((/)(w[2], w[18]), w[24])), (^)((/)(w[19], w[1]), w[25])), 2))), w[18])
    jwfm[10] = (/)((*)((*)((*)(-1, w[26]), (^)((/)(w[2], w[20]), (+)(-1, w[26]))), (/)(w[5], (^)((+)((+)(1, (^)((/)(w[21], um[7]), w[27])), (^)((/)(w[2], w[20]), w[26])), 2))), w[20])
    jwfm[11] = (/)((*)((*)((*)(-1, w[28]), (/)(w[7], (^)((+)((+)(1, (^)((/)(w[2], w[22]), w[28])), (^)((/)(w[23], um[8]), w[29])), 2))), (^)((/)(w[2], w[22]), (+)(-1, w[28]))), w[22])
    jwfm[16] = (+)((/)((*)((*)(-1, w[37]), (/)((*)((*)((+)(w[2], (*)(-1, um[8])), w[17]), um[6]), (*)((^)(w[37], 2), (^)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), 2)))), w[38]), (/)((*)(w[17], um[6]), (*)(w[37], (+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])))))
    jwfm[17] = (/)(1, (+)((+)(1, (^)((/)(w[2], w[18]), w[24])), (^)((/)(w[19], w[1]), w[25])))
    jwfm[25] = (*)(-1, um[1])
    jwfm[34] = (/)(1, (+)((+)(1, (^)((/)(w[21], um[7]), w[27])), (^)((/)(w[2], w[20]), w[26])))
    jwfm[42] = (*)(-1, um[2])
    jwfm[51] = (/)(1, (+)((+)(1, (^)((/)(w[2], w[22]), w[28])), (^)((/)(w[23], um[8]), w[29])))
    jwfm[59] = (*)(-1, um[3])
    jwfm[68] = (/)(um[1], (+)(w[30], um[1]))
    jwfm[76] = (*)(-1, um[4])
    jwfm[85] = (/)(um[2], (+)(w[31], um[2]))
    jwfm[93] = (*)(-1, um[5])
    jwfm[102] = (/)(um[3], (+)(w[32], um[3]))
    jwfm[110] = (*)(-1, um[6])
    jwfm[119] = (/)((*)((+)(w[1], (*)(-1, um[7])), um[4]), (*)(w[33], (+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33]))))
    jwfm[127] = (/)((*)((*)(-1, um[5]), (+)((*)(-1, um[8]), um[7])), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36]))))
    jwfm[128] = (/)((*)(um[5], (+)((*)(-1, um[8]), um[7])), (*)(w[35], (+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36]))))
    jwfm[136] = (/)((*)((+)(w[2], (*)(-1, um[8])), um[6]), (*)(w[37], (+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38]))))
    jwfm[137] = (*)((*)((*)(w[24], (^)((/)(w[2], w[18]), (+)(-1, w[24]))), (/)(w[3], (^)((+)((+)(1, (^)((/)(w[2], w[18]), w[24])), (^)((/)(w[19], w[1]), w[25])), 2))), (/)(w[2], (^)(w[18], 2)))
    jwfm[145] = (/)((*)((*)((*)(-1, w[25]), (/)(w[3], (^)((+)((+)(1, (^)((/)(w[2], w[18]), w[24])), (^)((/)(w[19], w[1]), w[25])), 2))), (^)((/)(w[19], w[1]), (+)(-1, w[25]))), w[1])
    jwfm[154] = (*)((*)((*)(w[26], (^)((/)(w[2], w[20]), (+)(-1, w[26]))), (/)(w[5], (^)((+)((+)(1, (^)((/)(w[21], um[7]), w[27])), (^)((/)(w[2], w[20]), w[26])), 2))), (/)(w[2], (^)(w[20], 2)))
    jwfm[162] = (/)((*)((*)((*)(-1, w[27]), (^)((/)(w[21], um[7]), (+)(-1, w[27]))), (/)(w[5], (^)((+)((+)(1, (^)((/)(w[21], um[7]), w[27])), (^)((/)(w[2], w[20]), w[26])), 2))), um[7])
    jwfm[171] = (*)((*)((*)(w[28], (/)(w[2], (^)(w[22], 2))), (/)(w[7], (^)((+)((+)(1, (^)((/)(w[2], w[22]), w[28])), (^)((/)(w[23], um[8]), w[29])), 2))), (^)((/)(w[2], w[22]), (+)(-1, w[28])))
    jwfm[179] = (/)((*)((*)((*)(-1, w[29]), (^)((/)(w[23], um[8]), (+)(-1, w[29]))), (/)(w[7], (^)((+)((+)(1, (^)((/)(w[2], w[22]), w[28])), (^)((/)(w[23], um[8]), w[29])), 2))), um[8])
    jwfm[185] = (*)((*)((*)(-1, (^)((/)(w[2], w[18]), w[24])), NaNMath.log((/)(w[2], w[18]))), (/)(w[3], (^)((+)((+)(1, (^)((/)(w[2], w[18]), w[24])), (^)((/)(w[19], w[1]), w[25])), 2)))
    jwfm[193] = (*)((*)((*)(-1, (/)(w[3], (^)((+)((+)(1, (^)((/)(w[2], w[18]), w[24])), (^)((/)(w[19], w[1]), w[25])), 2))), NaNMath.log((/)(w[19], w[1]))), (^)((/)(w[19], w[1]), w[25]))
    jwfm[202] = (*)((*)((*)(-1, (^)((/)(w[2], w[20]), w[26])), (/)(w[5], (^)((+)((+)(1, (^)((/)(w[21], um[7]), w[27])), (^)((/)(w[2], w[20]), w[26])), 2))), NaNMath.log((/)(w[2], w[20])))
    jwfm[210] = (*)((*)((*)(-1, NaNMath.log((/)(w[21], um[7]))), (^)((/)(w[21], um[7]), w[27])), (/)(w[5], (^)((+)((+)(1, (^)((/)(w[21], um[7]), w[27])), (^)((/)(w[2], w[20]), w[26])), 2)))
    jwfm[219] = (*)((*)((*)(-1, NaNMath.log((/)(w[2], w[22]))), (^)((/)(w[2], w[22]), w[28])), (/)(w[7], (^)((+)((+)(1, (^)((/)(w[2], w[22]), w[28])), (^)((/)(w[23], um[8]), w[29])), 2)))
    jwfm[227] = (*)((*)((*)(-1, NaNMath.log((/)(w[23], um[8]))), (/)(w[7], (^)((+)((+)(1, (^)((/)(w[2], w[22]), w[28])), (^)((/)(w[23], um[8]), w[29])), 2))), (^)((/)(w[23], um[8]), w[29]))
    jwfm[236] = (*)(-1, (/)((*)(w[9], um[1]), (^)((+)(w[30], um[1]), 2)))
    jwfm[245] = (*)(-1, (/)((*)(w[11], um[2]), (^)((+)(w[31], um[2]), 2)))
    jwfm[254] = (*)(-1, (/)((*)(w[13], um[3]), (^)((+)(w[32], um[3]), 2)))
    jwfm[263] = (*)((*)(-1, (+)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), (*)((*)(-1, w[33]), (/)(w[1], (^)(w[33], 2))))), (/)((*)((*)((+)(w[1], (*)(-1, um[7])), w[15]), um[4]), (*)((^)(w[33], 2), (^)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), 2))))
    jwfm[271] = (*)((*)(w[33], (/)((*)((*)((+)(w[1], (*)(-1, um[7])), w[15]), um[4]), (*)((^)(w[33], 2), (^)((+)((+)(1, (/)(um[7], w[34])), (/)(w[1], w[33])), 2)))), (/)(um[7], (^)(w[34], 2)))
    jwfm[279] = (*)((*)(-1, (+)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), (*)((*)(-1, w[35]), (/)(um[7], (^)(w[35], 2))))), (/)((*)((*)((*)(-1, w[16]), um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
    jwfm[280] = (*)((*)(-1, (+)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), (*)((*)(-1, w[35]), (/)(um[7], (^)(w[35], 2))))), (/)((*)((*)(w[16], um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
    jwfm[287] = (*)((*)(w[35], (/)(um[8], (^)(w[36], 2))), (/)((*)((*)((*)(-1, w[16]), um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
    jwfm[288] = (*)((*)(w[35], (/)(um[8], (^)(w[36], 2))), (/)((*)((*)(w[16], um[5]), (+)((*)(-1, um[8]), um[7])), (*)((^)(w[35], 2), (^)((+)((+)(1, (/)(um[7], w[35])), (/)(um[8], w[36])), 2))))
    jwfm[296] = (*)((*)(-1, (+)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), (*)((*)(-1, w[37]), (/)(um[8], (^)(w[37], 2))))), (/)((*)((*)((+)(w[2], (*)(-1, um[8])), w[17]), um[6]), (*)((^)(w[37], 2), (^)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), 2))))
    jwfm[304] = (*)((*)(w[37], (/)((*)((*)((+)(w[2], (*)(-1, um[8])), w[17]), um[6]), (*)((^)(w[37], 2), (^)((+)((+)(1, (/)(um[8], w[37])), (/)(w[2], w[38])), 2)))), (/)(w[2], (^)(w[38], 2)))
    nothing
end
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
function res(RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
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
        @info "  Is Real? $(eltype(F)<:Real), $(eltype(F))"
        @info "  $dt s, $a allocations"
        @info " Mat Mat mult "
        dt = @elapsed a = @allocations G = V * F'
        @info "  Is Real? $(eltype(G)<:Real), $(eltype(G))"
        @info "  $dt s, $a allocations"
        @info " Reshape "
        dt = @elapsed a = @allocations g = reshape(G, K*D)
        @info "  Is Real? $(eltype(g)<:Real), $(eltype(g))"
        @info "  $dt s, $a allocations"
        @info " Linear Solve "
        dt = @elapsed a = @allocations res = RT \ g
        @info "  Is Real? $(eltype(res)<:Real), $(eltype(res))"
        @info "  $dt s, $a allocations"
        @info " Vec Vec add "
        dt = @elapsed a = @allocations res .-= b
        @info "  Is Real? $(eltype(res)<:Real), $(eltype(res))"
        @info "  $dt s, $a allocations"
        @info "++++++++++++++++++"
        return res::AbstractVector{T}
    end
end

function ∇res(RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "++++ Jac Res Eval ++++"
        K, M = size(V)
        D, _ = size(U)
        J = length(w)
        @info " Evaluating jacobian of F"
        wcomplex = Complex.(w)
        Ucomplex = Complex.(U)
        dt = @elapsed a = @allocations begin
            JwF = Complex.(zeros(W,D,J,M))
            for m in 1:M
                try 
                jacwf!(view(JwF,:,:,m), wcomplex, view(Ucomplex,:,m))
                catch e
                    println("m = $m")
                    println("jwfm")
                    show(stderr, "text/plain", view(JwF,:,:,m))
                    println("w")
                    show(stderr, "text/plain", w)
                    println("um")
                    show(stderr, "text/plain", view(U,:,m)')
                    throw(e)
                    @assert false
                end
            end
        end
        @info "  Is Real? $(eltype(JwF)<:Real), $(eltype(JwF))"
        @info "  $dt s, $a allocations"
        @info " Computing V ×_3 jacF with tullio"
        dt = @elapsed a = @allocations @tullio _JG[d,j,k] := V[k,m] * JwF[d,j,m] 
        @info "  Is Real? $(eltype(_JG)<:Real), $(eltype(_JG))"
        @info "  $dt s, $a allocations"
        @info " permutedims"
        dt = @elapsed a = @allocations JG = permutedims(_JG,(3,1,2))
        @info "  Is Real? $(eltype(JG)<:Real), $(eltype(JG))"
        @info "  $dt s, $a allocations"
        @info " Reshape"
        dt = @elapsed a = @allocations jacG = reshape(JG, K*D, J)
        @info "  Is Real? $(eltype(jacG)<:Real), $(eltype(jacG))"
        @info "  $dt s, $a allocations"
        @info " Linsolve"
        dt = @elapsed a = @allocations ∇res = RT \ jacG
        @info "  Is Real? $(eltype(∇res)<:Real), $(eltype(∇res))"
        @info "  $dt s, $a allocations"
        return ∇res::AbstractMatrix{<:Complex}
    end
end
# Define cost function 
function f(RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, f!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    (1/2 * norm(res(RT, U, V, b, f!, w; ll=ll))^2)
end

function ∇f!(∇f::AbstractVector{<:Complex}, RT::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, b::AbstractVector{T}, jacwf!::Function, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W<:Real,T<:Real}
    ∇f .= ∇res(RT,U,V,jacwf!,w;ll=ll)' * res(RT, U, V, b, f!, w; ll=ll) 
    nothing
end
#
f(w::AbstractVector{W}, ::Any=nothing; ll::Logging.LogLevel=Logging.Warn) where W = f(RT,U,V,b,f!,w;ll=ll)
∇f!(∇f::AbstractVector{<:Real}, w::AbstractVector{W}, ::Any=nothing; ll::Logging.LogLevel=Logging.Warn) where W = ∇f!(∇f,RT,U,V,b,jacwf!,w;ll=ll)
## 
@info "Test Allocations with with Float64"
w_rand = wTrue + abs.(wTrue) .* randn(J)
RT = RTfun(U,V,Vp,sig,diagReg,jacuf!,w_rand)
b = RT \ b0 
# because this problem is linear in w the jacobian is constant, 
# and we could solve this problem with backslash because 
# it is really a linear least squares problem
G = ∇res(RT,U,V,jacwf!, w_rand)
w_star = G \ b 
##
res_w_rand = f(w_rand)
∇f_rand = zeros(J)
∇f!(∇f_rand,w_rand)
@assert !all(∇f_rand .== 0)
nothing 
##
@info " Benchmark f"
@time f(w_rand)
@info " Benchmark ∇f! "
@time ∇f!(∇f_rand,w_rand; ll=Info)
nothing
##
@info "Test the NLS in isolation"
optFun = OptimizationFunction(
    f;#,    Optimization.AutoForwardDiff(); 
    grad = ∇f!
)
problem = OptimizationProblem(optFun, w_rand; xtol_rel=1e-8, xtol_abs=1e-8)
a = @allocations sol = solve(problem,  Opt(:LD_LBFGS, J))
dt = sol.stats.time
w_hat = sol.u
f_hat = sol.objective
iter = sol.stats.iterations
abserr = norm(w_star-w_hat)
rselerr = abserr / norm(w_star)
@info "  $dt s, $a allocations"
@info "  iterations $iter"
@info "Absolute coeff error = $abserr"
@info "Relative coeff error = $relerr"
abserr = abs(f(w_star)- f_hat)
relerr = abs(f(w_star)- f_hat) / f(w_star)
@info "Absolute coeff error = $abserr"
@info "Relative coeff error = $relerr"
## Compute the non lin least square solution 
@info "Defining IRWLS_Nonlinear..."
function IRWLS_Nonlinear(U::AbstractMatrix{T}, V, Vp, b0, sig, diagReg, J, f!, jacuf!, jacwf!; ll=Logging.Info,maxIt=100, relTol=1e-10,w0=nothing) where T
    with_logger(ConsoleLogger(stderr,ll)) do 
        @info "Initializing the linearization least squares solution  ..."
        D, M = size(U)
        K, _ = size(V)
        R0 = Matrix{T}(I, K*D,K*D)
        f0(w::AbstractVector{W}, ::Any; ll::Logging.LogLevel=Logging.Warn) where W = f(R0,U,V,b,f!,w;ll=ll)
        ∇f0!(∇f::AbstractVector, w::AbstractVector{W}, ::Any; ll::Logging.LogLevel=Logging.Warn) where W = ∇f!(∇f,R0,U,V,b,jacwf!,w;ll=ll)
        optFun = OptimizationFunction(fn,grad=∇f0!)
        problem = OptimizationProblem(optFun, w_rand; xtol_rel=1e-8, xtol_abs=1e-8)
        try 
            a = @allocations sol = solve(problem,  Opt(:LD_LBFGS, J))
        catch ex
            @warn "Local Optimization method failed"
            return ex, w0, []
        wit = zeros(J,maxIt)
        resit = zeros(J,maxIt)
        wnm1 = w0 
        wn = similar(w0)
        for n = 1:maxIt 
            RT = RTfun(U,V,Vp,sig,diagReg,jacuf!,wnm1)
            b = RT \ b0 
            G = ∇res(RT,U,V,jacwf!,zeros(J))
            w_star = G \ b 
            fn(w::AbstractVector{W}, ::Any; ll::Logging.LogLevel=Logging.Warn) where W = f(RT,U,V,b,f!,w;ll=ll)
            ∇fn!(∇f::AbstractVector, w::AbstractVector{W}, ::Any; ll::Logging.LogLevel=Logging.Warn) where W = ∇f!(∇f,RT,U,V,b,jacwf!,w;ll=ll)
            @info "Iteration $n"
            @debug "  Running local optimization method"
            optFun = OptimizationFunction(fn; grad=∇fn!)
            problem = OptimizationProblem(optFun, w_rand; xtol_rel=1e-8, xtol_abs=1e-8)
            try 
                a = @allocations sol = solve(problem,  Opt(:LD_LBFGS, J))
            catch ex
                @warn "Local Optimization method failed"
                return ex, hcat(w0, wit[:,1:n-1]), resit[:,1:n-1],
            end
            dt = sol.stats.time
            wn = sol.u
            f_hat = sol.objective
            iter = sol.stats.iterations
            resn = wnm1-wn
            resit[:,n] .= resn
            wit[:,n] .= wn
            relErr = norm(w_star-wn) / norm(wn)
            @debug """ 
                iteration time  = $dt
                allocations     = $a
                relative Error  = $relErr
                iterations      = $(iter)
                ret code        = $(sol.retcode)
                objective_value = $(f_hat)
            """
            if norm(resn) / norm(wnm1) < relTol
                resit = resit[:,1:n] 
                wit = wit[:,1:n] 
                @info "  Convergence Criterion met!"
                return wn, hcat(w0,wit), resit 
            end
            wnm1 = wn
        end
        @warn "Maxiteration met for IRWLS"
        return wn, hcat(w0,wit), resit 
    end
end 
##
@info "IRWLS (Nonlinear): "
@info "   Runtime info: "
@time what, wit, resit = IRWLS_Nonlinear(U, V, Vp, b0, sig, diagReg, J, f!, jacuf!, jacwf!; ll=Logging.Info, w0=w_rand)
if typeof(what) <:AbstractVector 
    relErr = norm(wit[:,end] - wTrue) / norm(wTrue)
    @info "   coeff rel err = $relErr"
end
@info "   iterations    = $(size(wit,2)-1)"