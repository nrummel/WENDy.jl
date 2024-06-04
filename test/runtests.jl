using NLPMLE 
using Logging
using NLPMLE: create_test_data, _MENDES_S_VALS, _MENDES_P_VALS
using Test
## Set Up plotting
using Plots, PlotlyJS
using Plots: plot
gr()
function _plotTestData(ode_sol, ttlStr)
    return plot(ode_sol, title=ttlStr, xaxis="t", yaxis="u(t)")
end
function plotTestData(fitz, loop, mendes)
    display(_plotTestData(fitz, "FitzHug-Nagumo"))
    display(_plotTestData(loop, "Loop Model"))
    for (i,S) in enumerate(_MENDES_S_VALS), 
            (j,P) in enumerate(_MENDES_P_VALS)
        display(_plotTestData(mendes[i,j], "Mendes Problem\nS=$S, P=$P"))
    end
end
function testCreateTestData(;plotFlag=false) 
    fitz, loop, mendes = create_test_data(;ll=Logging.Info);
    if plotFlag
        plotTestData(fitz, loop,mendes)
    end
    return true
end
@testset "NLPMLE.jl" begin
    @testset "Create Test DataSet..." begin
        @test testCreateTestData(plotFlag=true)
    end
end
