includet(joinpath(@__DIR__, "../src/WENDy.jl"))
includet(joinpath(@__DIR__, "../examples/hindmarshRose.jl"))
includet(joinpath(@__DIR__, "../examples/logisticGrowth.jl"))
using Test
using Plots, Logging, MAT
using LinearAlgebra: norm 
using Statistics: mean
using HypothesisTests: ApproximateTwoSampleKSTest, pvalue
gr()
using FiniteDiff
##
function testGenerateNoise(;dataFile::String=joinpath(@__DIR__, "../data/matlab_wendydata_Hindmarsh-Rose.mat"),ll::LogLevel=Warn, plotFlag=false) 
    with_logger(ConsoleLogger(stdout, ll)) do
        data = MAT.matread(dataFile)
        noiseRatio = data["noise_ratio"]
        U_exact = Matrix(data["xsub"]')
        U_matlab = Matrix(data["xobs"]')
        noise_matlab = Matrix(data["noise"]')
        sigma_matlab = data["sigma"]
        params = WENDyParameters(noiseRatio=noiseRatio)
        U, noise, noise_ratio_obs, sigTrue = generateNoise(U_exact, params, Val(Normal), isotropic=true)
        D, _ = size(U)
        for d in 1:D
            if plotFlag 
                p = histogram(
                    Any[noise[d,:],
                    noise_matlab[d,:]],
                    fillcolor=[:red :blue], 
                    fillalpha=0.5 
                )
                title!("Dimention $d")
                display(p)
            end
            pVal  = pvalue(ApproximateTwoSampleKSTest(noise[d,:],noise_matlab[d,:]))
            @info "pVal = $pVal (interpret as the prob that these come from same distributions)"
            if pVal < 0.05 
                return false 
            end
        end
        return true
    end
end
##
function testEstimateNoise(;dataFile::String=joinpath(@__DIR__, "../data/matlab_wendydata_Hindmarsh-Rose.mat"),ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "Testing Estimation of Noise"
        @info "\tLoading data from $dataFile"
        data = MAT.matread(dataFile)
        uobs = Matrix(data["xobs"]')
        std_est_matlab = typeof(data["sig_ests"]) <: AbstractArray ? data["sig_ests"][:] : [data["sig_ests"]]
        std_est = estimate_std(uobs, Val(Normal))
        relErr = norm(std_est-std_est_matlab)/norm(std_est)
        @info "\tThe two estimation methods are different by $(relErr)"
        return relErr < 1e2*eps()
    end
end
##
function testTestFunctionDiscritization(;dataFile::String=joinpath(@__DIR__,"../data/testFunDisc.mat"),ll::LogLevel=Warn,plotFlag::Bool=false)
    with_logger(ConsoleLogger(stderr, ll)) do
        data     = matread(dataFile)
        C_matlab = data["C"]
        maxd     = 3 
        m        = 25
        ϕ        = ExponentialTestFun()
        C        = _getTestFunctionWeights(ϕ,m,maxd)
        if plotFlag 
            plotTestFunction(x,C_matlab, C)
        end 
        return norm(C -C_matlab) / norm(C_matlab) < 1e2*eps()
    end
end
##
function testRadSelect(;dataFile::String=joinpath(@__DIR__,"../data/rad_select_test.mat"),ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        data = MAT.matread(dataFile)
        t0= Vector(data["t0"][:])
        y = data["y"]
        inc= Int(data["inc"])
        sub= data["sub"]
        q= data["q"]
        s= data["s"]
        m_min= Int(data["m_min"])
        m_max= Int(data["m_max"])
        # pow= data["pow"]
        mt_matlab= data["mt"]
        ϕ = ExponentialTestFun()
        mt = rad_select(t0,y',ϕ,m_max; inc=inc,
            m_min=m_min, sub=sub, q=q, s=s)
        return mt == mt_matlab
    end
end
##
function testBuildV(;dataFile::String=joinpath(@__DIR__,"../data/buildV.mat"), ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        data = matread(dataFile)
        Kmin     = Int(data["K_min"])
        Kmax     = Int(data["K_max"])
        xobs      = data["xobs"]'
        tobs      = Vector(data["tobs"][:]);
        ## matlab values
        mt_min_matlab = data["mt_min"]
        mt_max_matlab = data["mt_max"]
        mt_matlab     = data["mt"]
        V_matlab      = data["V"]
        Vp_matlab     = data["Vp"]
        V_full_matlab = data["Vfull"];
        ## Defaults
        mtParams  = 2 .^(0:3)
        pruneMeth  = SingularValuePruningMethod(
            MtminRadMethod(),
            UniformDiscritizationMethod()
        )
        ϕ          = ExponentialTestFun()
        D,M     = size(xobs)
        Kmin      = 10
        ## Compute mt_min/max
        mt_min, mt_max = _getMtMinMax(tobs, xobs, ϕ, Kmin)
        if mt_max != mt_max_matlab 
            @info "mt max does not match"
            return false 
        end 
        if mt_min != mt_min_matlab 
            @info "mt min does not match matlab"
            return false 
        end 
        ##
        mt = pruneMeth.radMeth(tobs, xobs, ϕ, mtParams, Kmin)
        if any(mt .!= mt_matlab) 
            @info "mt does not match matlab"
            return false
        end 
        ## Test discritization
        K = _getK(Kmax, D, length(mtParams), length(tobs))
        V_full = reduce(vcat,pruneMeth.discMeth(m,tobs,ϕ,0, K) for m in mt)
        if norm(V_full - V_full_matlab) / norm(V_full_matlab) >= 1e2*eps() 
            @info "The full V mat is bad"
            return false 
        end
        ##
        V,Vp = pruneMeth(tobs,xobs,ϕ,Kmin,Kmax,mtParams);
        for k in 1:size(V,1)
            vk = V[k,:]
            _vk = V_matlab[k,:]
            relErr = min(norm(vk - _vk) / norm(_vk),norm(vk + _vk) / norm(_vk))
            if relErr >= 1e4*eps()
                @info "The V mat is bad at $k with relErr = $relErr"
                return false
            end 
            vk = Vp[k,:]
            _vk = Vp_matlab[k,:]
            relErr = min(norm(vk - _vk) / norm(_vk),norm(vk + _vk) / norm(_vk))
            if relErr >= 1e4*eps()
                @info "The Vp mat is bad at $k with relErr = $relErr"
                return false
            end 
        end
        return true
    end
end
##
function test_VVp(;dataFile::String=joinpath(@__DIR__, "../data/Lw_hindmarsh_test.mat"), ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        data = HINDMARSH_ROSE
        params = WENDyParameters()
        data = matread(dataFile)
        tobs = Vector(data["tobs"][:])
        uobs = Matrix(data["xobs"]')
        G_matlab = data["G_0"]
        V_matlab = data["V"]
        Vp_matlab = data["Vp"]
        ##
        V,Vp,Vfull = params.pruneMeth(tobs,uobs,params.ϕ,size(G_matlab,2),params.Kmax,params.testFuctionRadii);

        for k in 1:size(V,1)
            vk = V[k,:]
            _vk = V_matlab[k,:]
            relErr = min(norm(vk - _vk) / norm(_vk),norm(vk + _vk) / norm(_vk))
            if relErr >= 1e4*eps()
                @info "The V mat is bad at $k with relErr = $relErr"
                return false
            end 
            vk = Vp[k,:]
            _vk = Vp_matlab[k,:]
            relErr = min(norm(vk - _vk) / norm(_vk),norm(vk + _vk) / norm(_vk))
            if relErr >= 1e4*eps()
                @info "The Vp mat is bad at $k with relErr = $relErr"
                return false
            end 
        end
        return true
    end
end
##
function WENDyParameters(matlab_data::Dict)
    noiseRatio=matlab_data["noise_ratio"]
    WENDyParameters(noiseRatio=noiseRatio)
end
function EmpricalWENDyData(matlab_data::Dict, ode::ODESystem, ::Val{LinearInParameters}=Val(true), ::Val{DistType}=Val(Normal)) where {LinearInParameters, DistType<:Distribution}
    t = matlab_data["tobs"][:]
    U = Matrix(matlab_data["xobs"]')
    EmpricalWENDyData{LinearInParameters,DistType}("", ode, t, U)
end
function _getMatlabProblem(matfile::String, ode::ODESystem, LineearInParameters::Bool)
    matlab_data=matread(matfile)
    params = WENDyParameters(matlab_data)
    data = EmpricalWENDyData(matlab_data, ode, Val(LineearInParameters))
    prob = WENDyProblem(data, params; matlab_data=matlab_data);
    w0 = matlab_data["w0"][:];
    return prob, params, w0, matlab_data
end
#
function testResidual(prob, params, w0, matlab_data)
    
    r_matlab = matlab_data["G_0"] * w0 - prob.b₀;
    r! = Residual(prob, params);
    r!(prob.b₀, w0)
    return norm(r!.r - r_matlab)/norm(r_matlab) < eps()*1e2
end
#
function testWeightedResidual(prob, params, w0, matlab_data)
    RT_matlab = matlab_data["RT"];
    r_matlab = RT_matlab \ (matlab_data["G_0"] * w0 - prob.b₀);
    r! = Residual(prob, params);
    r!(RT_matlab \ prob.b₀, w0; Rᵀ=RT_matlab)
    return norm(r!.r - r_matlab)/norm(r_matlab) < eps()*1e2
end
#
function testCovarianceFactor(prob, params, w0, matlab_data)
    L0_matlab = matlab_data["Lw"];
    L! = CovarianceFactor(prob, params);
    L!(w0)
    return norm(L!.L  - L0_matlab) / norm(L0_matlab) < eps()*1e2
end
#
function testGradientCovarianceFactor(prob, params, w0, matlab_data)
    L1_matlab = matlab_data["L1"];
    ∇L! = GradientCovarianceFactor(prob, params);
    ∇L!(w0)
    return norm(∇L!.∇L  - L1_matlab) / norm(L1_matlab) < eps()*1e2
end
#
function testCovariance(prob, params, w0, matlab_data)
    Sreg_matlab = matlab_data["Sw"];
    RT_matlab = matlab_data["RT"];
    R! = Covariance(prob, params);
    R!(w0)
    if norm(R!.Sreg - Sreg_matlab) / norm(R!.Sreg) > eps()*1e2
        @info "Regularized Covariance is differt"
        return false
    elseif norm(R!.R - RT_matlab) / norm(RT_matlab) > eps()*1e2
        @info "Cholesky Factor is wrong"
        return false
    end

    return true 
end
#
function testMahalanobisDistance(prob, params, w0, matlab_data)
    m0_matlab = 1/2*matlab_data["m"];
    m = MahalanobisDistance(prob, params);
    m0 = m(w0;efficient=true)
    return abs(m0 -m0_matlab) / abs(m0) < eps()*1e4
end
#
function testGradientMahalanobisDistance(prob, params, w0, matlab_data;ll=Warn)
with_logger(ConsoleLogger(stdout, ll)) do
    flag=true
    ∇m_matlab = 1/2*matlab_data["gradm"][:];
    ∇m! = GradientMahalanobisDistance(prob, params);
    ∇m!(w0)
    relErr = norm(∇m!.∇m -∇m_matlab) / norm(∇m_matlab)
    if relErr < 1e-10 
        @info "∇m in spec (relerr = $relErr) with MATLAB"
    else 
        flag = false
        @warn "H out of spec (relerr = $relErr) with MATLAB"
    end
    m = MahalanobisDistance(prob, params)
    ∇m_df = similar(∇m_matlab)
    FiniteDiff.finite_difference_gradient!(∇m_df,m,w0);
    relErr = norm(∇m!.∇m -∇m_df) / norm(∇m_df)
    if relErr < 1e-7
        @info "∇m in spec (relerr = $relErr) with finite differences to of the objective"
    else 
        flag = false
        @warn "H out of spec (relerr = $relErr) with finite differences to of the objective"
    end
    return flag
end
end
##
function testHessianMahalanobisDistance(prob, params, w0, matlab_data;ll=Warn)
    with_logger(ConsoleLogger(stdout, ll)) do
    ##
    flag = true
    @info "Building functions"
    m = MahalanobisDistance(prob, params)
    ∇m! = GradientMahalanobisDistance(prob, params)
    Hm! = HesianMahalanobisDistance(prob, params);
    ##
    @info "Evaluating analytic hessian"
    dt = @elapsed als = @allocations Hm!(w0)
    @info "   $dt s, $als allocations"
    ##
    @info "Evaluating first finite diff with m "
    H0_fd1 = similar(Hm!.H)
    dt = @elapsed als = @allocations FiniteDiff.finite_difference_hessian!(H0_fd1, m,w0);
    @info "   $dt s, $als allocations"
    ##
    @info "Evaluating second finite diff with grad m "
    function Hm_fd!(H,w) 
        FiniteDiff.finite_difference_jacobian!(H, ∇m!, w)
        @views H .= 1/2*(H + H')
        @views H .= Symmetric(H)
        nothing
    end 
    H0_fd2 = similar(H0_fd1)
    dt = @elapsed als = @allocations Hm_fd!(H0_fd2,w0)
    @info "   $dt s, $als allocations"
    ##
    relErr = norm(Hm!.H - H0_fd1) / norm(H0_fd1)
    if relErr < eps()*1e2 
        @info "H0 in spec (relerr = $relErr) with finite differences to of the objective"
    else 
        flag = false
        @warn "H out of spec (relerr = $relErr) with finite differences to of the objective"
    end
    relErr = norm(Hm!.H - H0_fd2) / norm(H0_fd1)
    if relErr < 1e-7 
        @info "H in spec (relerr = $relErr) with finite differences of the gradient" 
    else
        flag = false
        @warn "H out of spec (relerr = $relErr) with finite differences of the gradient"
    end
    return flag 
end
end
prob, params, w0, matlab_data = _getMatlabProblem(joinpath(@__DIR__, "../data/matlab_wendydata_Hindmarsh-Rose.mat"), HINDMARSH_ROSE_SYSTEM, true)
# testGradientMahalanobisDistance(prob, params, w0, matlab_data)  
testHessianMahalanobisDistance(prob, params, w0, matlab_data;ll=Info) 
## 
files = [joinpath(@__DIR__, "../data/matlab_wendydata_Hindmarsh-Rose.mat"), joinpath(@__DIR__, "../data/matlab_wendydata_Logistic_Growth.mat")]
odes = [HINDMARSH_ROSE_SYSTEM, LOGISTIC_GROWTH_SYSTEM]
@testset verbose=true begin 
for (file,ode) in zip(files, odes)
    @info "Running Tests for example $file"
    @info " testGenerateNoise "
    @test testGenerateNoise(dataFile=file) 
    @info " test_VVp "
    @test test_VVp(dataFile=file) 
    @info " testEstimateNoise "
    @test testEstimateNoise(dataFile=file) 
    for LinearInParameters in [true,false]
        prob, params, w0, matlab_data = _getMatlabProblem(file, ode, LinearInParameters)
        @info " $(LinearInParameters ? "Linear" : "Nonlinear") Problem/Methods"
        @info "  testResidual"
        @test testResidual(prob, params, w0, matlab_data)  
        @info "  testWeightedResidual"
        @test testWeightedResidual(prob, params, w0, matlab_data)  
        @info "  testCovarianceFactor"
        @test testCovarianceFactor(prob, params, w0, matlab_data)  
        @info "  testCovariance"
        @test testCovariance(prob, params, w0, matlab_data)  
        @info "  testMahalanobisDistance"
        @test testMahalanobisDistance(prob, params, w0, matlab_data)  
        @info "  testGradientCovarianceFactor"
        @test testGradientCovarianceFactor(prob, params, w0, matlab_data)  
        @info "  testGradientMahalanobisDistance"
        @test testGradientMahalanobisDistance(prob, params, w0, matlab_data)  
        @info "  testHessianMahalanobisDistance"
        @test testHessianMahalanobisDistance(prob, params, w0, matlab_data)  
    end
end
end
##
@testset "NLPMLE.jl" begin
    @testset "Compare to MATLAB HINDMARSH_ROSE..." begin
        @test testEstimateNoise()
        @test testRadSelect()
        @test testTestFunctionDiscritization()
        @test testBuildV()
    end
    @testset "Create Test DataSet..." begin
        @test testCreateTestData()
    end
end
