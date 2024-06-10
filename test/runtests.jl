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
        M, D     = size(xobs)
        K_min      = 10
        num_rad = length(mt_params);
        ## Compute mt_min/max
        mt_min, mt_max = getMtMinMax(tobs, xobs, ϕ, M, K, K_min)
        if mt_max != mt_max_matlab 
            @info "mt max does not match"
            return false 
        end 
        if mt_min != mt_min_matlab 
            @info "mt min does not match matlab"
            return false 
        end 
        ##
        mt = _getMt(tobs, xobs, ϕ, K_min)
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
        V,Vp = pruneMeth(tobs,xobs,ϕ,K_min,K_max,mt_params);
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

function L_matlab!(L, w, L0, L1)
    @tullio L[k,m] = L1[k,m,j] * w[j]
    L[:] += L0[:]
    nothing
end

function test_L_to_matlab(;dataFile::String=joinpath(@__DIR__, "../data/Lw_hindmarsh_test.mat"), ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConoleLogger(stderr, ll)) do
        ## Load example
        mdl = HINDMARSH_ROSE_MODEL
        data = matread(dataFile)
        tobs = Vector(data["tobs"][:])
        uobs = Matrix(data["xobs"]')
        L0_matlab = data["L0"]
        L1_matlab = data["L1"]
        V = data["V"]
        Vp = data["Vp"]
        _, _jacuF! = getJacobian(mdl)
        D, M = size(uobs)
        J = length(parameters(mdl))
        K = size(V, 1)
        ##
        sig = estimate_std(uobs)
        ##
        w_rand = rand(J)
        L_matlab = zeros(D*K,D*M)
        L = zeros(K,D,M,D)
        # allocate buffers for L!
        LL = zeros(K,M,D,D) 
        JJ = zeros(M,D,D)
        L0 = zeros(K,M,D,D);
        ##
        L0!(L0,Vp,sig)
        @assert norm(reshape(permutedims(L0,(1,3,2,4)), K*D,M*D) - L0_matlab) / norm(L0_matlab) < 1e2*eps() "L0 is bad"
        ##
        @time L_matlab!(L_matlab, w_rand, L0_matlab,L1_matlab)
        @time begin 
            L!(L, LL, JJ, w_rand, sig, L0, _jacuF!, uobs) 
            L_matrix = reshape(L, K*D, M*D)
        end;
        return  norm(L_matrix - L_matlab) / norm(L_matlab) < 1e2*eps()
    end
end

function test_residual(;dataFile::String=joinpath(@__DIR__, "../data/Lw_hindmarsh_test.mat"), ll::Logging.LogLevel=Logging.Warn)
    with_logger(ConsoleLogger(stderr, ll)) do
        mdl = HINDMARSH_ROSE_MODEL
        data = matread(joinpath(@__DIR__, "../data/Lw_hindmarsh_test.mat"))
        tobs = Vector(data["tobs"][:])
        uobs = Matrix(data["xobs"]')
        G_matlab = data["G_0"]
        b_matlab = data["b_0"][:]
        V = data["V"]
        Vp = data["Vp"]
        _, _F! = getRHS(mdl)

        D, M = size(uobs)
        J = length(parameters(mdl))
        K = size(V, 1)
        ##
        w_rand = rand(J)
        r = zeros(K*D)
        Gw = zeros(K, D)
        B = zeros(K,D)
        FF = zeros(M,D)
        Gw_matlab = G_matlab *w_rand  

        ##
        G!(Gw, FF, w_rand, V, _F!, uobs)
        B!(B, Vp, uobs)
        residual!(r,Gw,FF,B,w_rand,V,Vp,_F!,uobs)
        r_matlab = Gw_matlab - b_matlab 
        Gw_vec = reshape(Gw, K*D)
        b = reshape(B, K*D)
        return (norm(b- b_matlab)/norm(b_matlab) < 1e2*eps() &&
            norm(Gw_vec- Gw_matlab)/norm(Gw_matlab) < 1e2*eps() &&
            norm(r- r_matlab)/norm(r_matlab) < 1e2*eps() 
            )
    end
end
function test_VVp_Hindmarsh(;dataFile::String=joinpath(@__DIR__, "../data/Lw_hindmarsh_test.mat", ll::Logging.LogLevel=Logging.Warn))
    with_logger(ConsoleLogger(stderr, ll)) do
        mdl = HINDMARSH_ROSE_MODEL
        data = matread()
        tobs = Vector(data["tobs"][:])
        uobs = Matrix(data["xobs"]')
        G_matlab = data["G_0"]
        b_matlab = data["b_0"][:]
        V_matlab = data["V"]
        Vp_matlab = data["Vp"]
        ##
        V,Vp,Vfull = pruneMeth(tobs,uobs,ϕ,K_min,K_max,mt_params);

        return n(orm(V - diagm(sign.(diag(V * V_matlab'))) * V_matlab ) / norm(V_matlab) < 1e2*eps() 
            && norm(Vp - diagm(sign.(diag(Vp * Vp_matlab'))) * Vp_matlab ) / norm(Vp_matlab) < 1e2*eps())
    end
end

##
@testset "NLPMLE.jl" begin
    @testset "Compare to MATLAB Implementation..." begin
        @test testEstimateNoise()
        @test testRadSelect()
        @test testTestFunctionDiscritization()
        @test testBuildV()
        @test test_L_to_matlab()
        @test test_residual()
    end
    @testset "Create Test DataSet..." begin
        @test testCreateTestData()
    end
end
