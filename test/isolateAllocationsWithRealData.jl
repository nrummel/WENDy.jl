@info "Loading generateNoise..."
includet("../src/wendyNoise.jl")
@info "Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info "Loading computeGradients..."
includet("../src/wendySymbolics.jl")
@info "Loading testFunctions..."
includet("../src/wendyTestFunctions.jl")
@info "Loading other dependencies..."
using JuMP, Ipopt, LinearAlgebra, Tullio, BenchmarkTools, Random, Logging 
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
##
@info "Building functions..."
function RTfun(U::AbstractMatrix, V::AbstractMatrix, Vp::AbstractMatrix, sig::AbstractVector, diagReg::Real, jacuf!::Function, w::AbstractVector)
    ## Preallocate for L 
    D, M = size(U)
    K, _ = size(V)
    JuF = zeros(D,D,M)
    _L0 = zeros(K,D,D,M)
    _L1  = zeros(K,D,M,D)
    ## get _L1
    K,D,M,_ = size(_L1)
    @inbounds for m in 1:size(JuF,3)
        jacuf!(view(JuF,:,:,m), w, view(U,:,m))
    end
    @tullio _L0[k,d2,d1,m] = V[k,m] * JuF[d2,d1,m] * sig[d1]
    @tullio _L0[k,d,d,m] += Vp[k,m]*sig[d]
    permutedims!(_L1,_L0,(1,2,4,3))
    L = reshape(_L1,K*D,M*D)
    ## Preallocate for R
    
    S = L * L'
    R = (1-diagReg)*S + diagReg*I
    cholesky!(Symmetric(R))
    return UpperTriangular(R)'
end
# Define residual function to do as little allocation as possible 
function _res(RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, b::AbstractVector, f!::Function, ::Val{T}, w::AbstractVector{W}; ll::Logging.LogLevel=Logging.Warn) where {W,T}
    with_logger(ConsoleLogger(stderr, ll)) do 
        K, M = size(V)
        D, _ = size(U)
        @info "++++ Res Eval ++++"
        @info " Evaluate F "
        dt = @elapsed a = @allocations begin 
            F = zeros(T, D, M)
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
# Define Jacobian of residual with respect to the parameters w 
function _jacRes(RT::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, jacwf!::Function, ::Val{T}, w::AbstractVector{W}; showFlag::Bool=false, ll::Logging.LogLevel=Logging.Warn) where {W,T}
    with_logger(ConsoleLogger(stderr, ll)) do 
        showFlag && println(stderr, "Hello")
        K, M = size(V)
        D, _ = size(U)
        J = length(w)
        JwF = zeros(T,D,J,M)
        for m in 1:M
            jacwf!(view(JwF,:,:,m), w, view(U,:,m))
        end
        @tullio _JG[d,j,k] := V[k,m] * JwF[d,j,m] 
        JG = permutedims(_JG,(3,1,2))
        jacG = reshape(JG, K*D, J)
        return RT \ jacG
    end
end
@info "Test Allocations with with Float64"
w_rand = rand(J)
RT = RTfun(U,V,Vp,sig,diagReg,jacuf!,w_rand)
b = RT \ b0 
res_float64(w, ll) = _res(RT,U,V,b,f!,Val(Float64), w; ll=ll)
res_float64(w_rand, Logging.Info)
@btime res_float64($w_rand, $Logging.Warn);
nothing
## Compute the non lin least square solution 
@info "Defining IRWLS_Nonlinear..."
function IRWLS_Nonlinear(U, V, Vp, b0, sig, diagReg, J, f!, jacuf!, jacwf!; ll=Logging.Info,maxIt=100, relTol=1e-4)
    with_logger(ConsoleLogger(stderr,ll)) do 
        @info "Initializing the linearization least squares solution  ..."
        D, M = size(U)
        K, _ = size(V)
        G0 = _jacRes(Matrix{Float64}(I, K*D,K*D), U, V, jacwf!, Val(Float64), zeros(J))
        w0 = G0 \ b0 
        wit = zeros(J,maxIt)
        resit = zeros(J,maxIt)
        wnm1 = w0 
        wn = similar(w0)
    
        for n = 1:maxIt 
            RT = RTfun(U,V,Vp,sig,diagReg,jacuf!,wnm1)
            b = RT \ b0 
            res_AffExpr(w::AbstractVector) = _res(RT,U,V,b,f!,Val(AffExpr), w)
            jacRes_AffExpr(w::AbstractVector) = _jacRes(RT,U,V,jacwf!,Val(AffExpr),w;showFlag=true)
            w_star = _jacRes(RT, U, V, jacwf!, Val(Float64), zeros(J)) \ b 
            ##
            @info "Defining model for Ipopt"
            mdl = Model(Ipopt.Optimizer)
            @variable(mdl, w[j = 1:J], start = wnm1[j])
            @variable(mdl, r[k = 1:K*D])
            @operator(mdl, f, J, res_AffExpr, jacRes_AffExpr)
            @constraint(mdl, r == f(w))
            @objective(mdl, Min, sum(r.^2 ) ) 
            set_silent(mdl)
            @info "Running optimizer"
            dtn = @elapsed optimize!(mdl)
            wn = value.(w)
            resn = wn - w_star
            relErr = norm(resn) / norm(w_star)
            resit[:,n] = resn
            wit[:,n] = wn
            @info """ Iteration $n
                relative Error     = $dtn
                relative Error     = $relErr
                barrier_iterations = $(barrier_iterations(mdl))
                termination_status = $(termination_status(mdl))
                primal_status      = $(primal_status(mdl))
                objective_value    = $(objective_value(mdl))
            """
            if norm(wnm1 - wn) / norm(wnm1) < relTol
                resit = resit[:,1:n] 
                wit = wit[:,1:n] 
                @info "Convergence Criterion met!"
                break 
            elseif n == maxIt 
                @warn "Maxiteration met..."
            end
            wnm1 = wn
        end
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
nothing 