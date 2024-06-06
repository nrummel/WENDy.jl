using NLPMLE: create_test_data, _MENDES_S_VALS, _MENDES_P_VALS, _getTestFunctionWeights, MtminRadMethod, SingularValuePruningMethod, ExponentialTestFun, estiamte_std, UniformDiscritizationMethod, rad_select, _getK
using Test
using Plots, Logging, MAT
using LinearAlgebra: norm 
using Statistics: mean
gr()
##
function _plotTestData(ode_sol, ttlStr)
    return plot(ode_sol, title=ttlStr, xaxis="t", yaxis="u(t)")
end
##
function plotTestData(lg, hind, fitz, loop, mendes)
    display(_plotTestData(lg, "Logistic Growth"))
    display(_plotTestData(hind, "Hindmarsh Rose"))
    display(_plotTestData(fitz, "FitzHug-Nagumo"))
    display(_plotTestData(loop, "Loop Model"))
    for (i,S) in enumerate(_MENDES_S_VALS), 
            (j,P) in enumerate(_MENDES_P_VALS)
        display(_plotTestData(mendes[i,j], "Mendes Problem\nS=$S, P=$P"))
    end
end
##
function testCreateTestData(;ll::Logging.LogLevel=Logging.Warn, plotFlag=false) 
    lg, hind, fitz, loop, mendes = create_test_data(;ll=ll);
    if plotFlag
        plotTestData(lg, hind, fitz, loop, mendes)
    end
    return true
end
##
function testEstimateNoise(;
    dataFile::String=joinpath(@__DIR__,"../data/estimateNoiseTestProblem.mat"),
    ll::Logging.LogLevel=Logging.Warn 
)
    with_logger(ConsoleLogger(stderr, ll)) do
        @info "Testing Estimation of Noise"
        @info "\tLoading data from $dataFile"
        data = MAT.matread(dataFile)
        uobs = data["uobs"]
        std_est_matlab  = Vector(data["std_est_matlab"][:])
        std_est = estiamte_std(uobs)
        relErr = norm(std_est-std_est_matlab)/norm(std_est)
        @info "\tThe two estimation methods are different by $(relErr)"
        return relErr < 1e2*eps()
    end
end
##
function plotTestFunction(x,C_matlab, C)

    plot(x, C[1,:],label="ϕ")
    plot!(x, C_matlab[1,:],label="ϕ matlab")
    for d = 2:size(C,2)
        plot!(x, C[d,:],label="D^$(d-1)(ϕ)")
        plot!(x, C_matlab[d,:],label="D^$(d-1)(ϕ) matlab")
    end
    # plot!(yscale=:log10, minorgrid=true)
    xlims!(-1, 1)
    title!("Test Function Plot")
    xlabel!("time")
end
##
function testTestFunctionDiscritization(;dataFile::String=joinpath(@__DIR__,"../data/testFunDisc.mat"),ll::Logging.LogLevel=Logging.Warn,plotFlag::Bool=false)
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
function testRadSelect(;dataFile::String=joinpath(@__DIR__,"../data/rad_select_test.mat"),
    ll::Logging.LogLevel=Logging.Warn
)
    with_logger(ConsoleLogger(stderr, ll)) do
        data = MAT.matread(dataFile)
        t0= Vector(data["t0"][:])
        y= data["y"]
        inc= Int(data["inc"])
        sub= data["sub"]
        q= data["q"]
        s= data["s"]
        m_min= Int(data["m_min"])
        m_max= Int(data["m_max"])
        # pow= data["pow"]
        mt_matlab= data["mt"]
        ϕ = ExponentialTestFun()
        mt = rad_select(t0,y,ϕ,m_max; inc=inc,
            m_min=m_min, sub=sub, q=q, s=s)
        return mt == mt_matlab
    end
end
##
function testBuildV(;dataFile::String=joinpath(@__DIR__,"../data/buildV.mat"), ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        data = matread(dataFile)
        K_min     = Int(data["K_min"])
        K_max     = Int(data["K_max"])
        xobs      = data["xobs"]
        tobs      = Vector(data["tobs"][:]);
        ## matlab values
        mt_min_matlab = data["mt_min"]
        mt_max_matlab = data["mt_max"]
        mt_matlab     = data["mt"]
        V_matlab      = data["V"]
        Vp_matlab     = data["Vp"]
        V_full_matlab = data["Vfull"];
        ## Defaults
        mt_params  = 2 .^(0:3)
        radMeth    = MtminRadMethod()
        pruneMeth  = SingularValuePruningMethod(UniformDiscritizationMethod())
        ϕ          = ExponentialTestFun()
        Mp1, D     = size(xobs)
        K_min      = 10
        num_rad = length(mt_params);
        ## Compute mt_min/max
        mt_max = Int(max(floor((Mp1-1)/2)-K_min,1));
        mt_min = rad_select(tobs,xobs,ϕ,mt_max);
        if mt_max != mt_max_matlab 
            @info "mt max does not match"
            return false 
        end 
        if mt_min != mt_min_matlab 
            @info "mt min does not match matlab"
            return false 
        end 
        ##
        mt = zeros(num_rad, D)
        for (m, p) in enumerate(mt_params), d in 1:D
            mt[m,d] = radMeth(xobs, tobs, ϕ, mt_min, mt_max, p)
        end
        mt = Int.(ceil.(1 ./ mean(1 ./ mt, dims=2)))
        mt = mt[:]
        if any(mt .!= mt_matlab) 
            @info "mt does not match matlab"
            return false
        end 
        ## Test discritization
        K = _getK(K_max, D, num_rad, length(tobs))
        V_full = reduce(vcat,pruneMeth.discMethod(m,tobs,ϕ,0, K) for m in mt)
        if norm(V_full - V_full_matlab) / norm(V_full_matlab) >= 1e2*eps() 
            @info "The full V mat is bad"
            return false 
        end
        ##
        V,Vp = pruneMeth(mt,tobs,ϕ,K_min,K_max,D,num_rad);
        if norm(V - V_matlab) / norm(V_matlab) >= 1e2*eps() 
            @info "The V mat is bad"
            return false 
        end 
        if norm(Vp - Vp_matlab) / norm(Vp_matlab) >= 1e2*eps() 
            @info "The Vp mat is bad"
            return false
        end
        return true
        end
end
##
@testset "NLPMLE.jl" begin
    @testset "Compare to MATLAB Implementation..." begin
        @test testEstimateNoise()
        @test testRadSelect()
        @test testTestFunctionDiscritization()
        @test testBuildV()
    end
    @testset "Create Test DataSet..." begin
        @test testCreateTestData()
    end
end
