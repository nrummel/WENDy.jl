## Load everything we need 
includet("util.jl")
##
ex = GOODWIN
# Recall you can change the lip or DistType like this 
# ex = SimulatedWENDyData(ex, Val(true), Val(Normal))
##
wendyProb, params, simParams = init(
    GOODWIN;
    noiseRatio = 0.20, 
    seed = 1, 
    timeSubsampleRate = 4, 
    μ = 2.0,
    optimMaxiters = 200, 
    optimTimelimit = 200.0,
    ll = Warn
);
## build costFunction
nll, l2Loss = buildCostFunctions(wendyProb, params; ll=Info);
## Pick a initializaiotn point
w0 = getW0(
    wendyProb, simParams; 
    allowNegW0=false
);
## run algorithms 
algoNames = [:tr_optim,:arc_sfn,:tr_jso,:arc_jso,:irwls,:fsnls_tr_optim]
results = runAlgos(wendyProb, params, nll, l2Loss, algoNames; return_wits=true)
## Make plots
# Residual parted out
_algoName = :tr_optim
display(
    plotResParts(
        results[_algoName].wits[], 
        wendyProb, 
        "Residual Broken Apart<br>Solver $(algo2Disp[_algoName]) : $(ex2Disp[ex.name])"
    )
)
## Plot different cost functions over the iterates, in this case we have nll
display(
    plotWits(
        w -> try
            nll.f(w)
        catch 
            NaN
        end,
        [results[algoName].wits[] for algoName in algoNames];
        names = [algo2Disp[algoName] for algoName in algoNames],
        title="Compare Solver Iterations NegLogLikelihood<br>$(ex2Disp[ex.name])"
    )
);
## now fwrSim l2
display(
    plotWits(
        w -> begin 
            try 
                a,f = forwardSolveRelErr(wendyProb, w);
                a
            catch
                NaN 
            end
        end,
        [results[algoName].wits[] for algoName in algoNames];
        names = [algo2Disp[algoName] for algoName in algoNames],
        title="Compare Solver Iterations Avg FS RelErr<br>$(ex2Disp[ex.name])"
    )
);
## Plot the solutions 
for algoName in [:init, :tr_optim]#algoNames 
    # continue
    try
        what = algoName == :init ? w0 : results[algoName].wits[][:,end]
        _p = plotDeepDive(
            wendyProb, what;
            title="$(algoName == :init ? "w₀ Solution" : algo2Disp[algoName])<br>$(ex2Disp[ex.name])",
            # file = joinpath(
            #     "/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig",
            #     "$(ex.name)_$(algoName)_solWData.png"
            # ),
            yaxis_type="linear"
        )
        display(_p)
    catch 
        @warn "$(algo2Disp[algoName]) failed to plot most likely, fs is imposible at what"
    end
end
## Automate the latex so we dont have to copy and past numbers over...
# if ex.name == "goodwin"
#     open(joinpath("/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig", "$(ex.name)_solWData.tex"), "w") do f
#         s = """
#         The plots above show the $(ex2Disp[ex.name]) model with truth specified as $(L"U^*") with the dashed lines, the data as $(L"U") with the markers, and the approximation of using the parameters that were the out put of the specified solver as $(L"\hat{U}") with the solid lines. The noise was multiplicative and distributed LogNormal with a SNR of $(params.noiseRatio*100)$(L"\%"). We initialized both algorithms with the same random point with $(L"\mu") = $μ. The data was subsampled down to $(wendyProb.M) points.
#         In this case the true parameters were \\\\
#         $(L"w^*") = [$(prod(j == J ? (@sprintf "%.4g" wj) : (@sprintf "%.4g, " wj) for (j,wj) in enumerate(wTrue)))], \\\\
#         and the initial guess for parameterns was \\\\
#         $(L"w_0") = [$(prod(j == J ? (@sprintf "%.4g" wj) : (@sprintf "%.4g, " wj) for (j,wj) in enumerate(w0)))] \\\\
#         While the FSNLS could not begin to iterate because $(L"w_0") was to far away from reasonable values, our two solver Trust Region and ARC could approximate the parameters at least to some degree. Here are the the metrics fro this run:

#         \\textbf{ARC}: 
#         \\begin{itemize}
#             \\item $(L"\frac{\|\hat{w} - w^*\|_2}{\|w^*\|}_2") = $(@sprintf "%.4g" results.arc.cl2[]*100)$(L"\%")
#             \\item $(L"\frac{\sum_{m=1}^M \|\hat{u}_m - u^*_m\|_2}{\sum_{m=1}^M \|u^*_m\|_2}") = $(@sprintf "%.4g" results.arc.fsl2[]*100)$(L"\%")
#             \\item $(L"m(\hat{w};U,T)") = $(@sprintf "%.4g" results.arc.mDist[])
#             \\item dt = $(@sprintf "%.4g" results.arc.dt[]) s
#             \\item iters = $(results.arc.iters[])
#         \\end{itemize}
#         \\textbf{Trust Region}: 
#         \\begin{itemize}
#             \\item $(L"\frac{\|\hat{w} - w^*\|_2}{\|w^*\|}") = $(@sprintf "%.4g" results.tr.cl2[]*100)$(L"\%")
#             \\item $(L"\frac{\sum_{m=1}^M \|\hat{u}_m - u^*_m\|_2}{\sum_{m=1}^M \|u^*_m\|_2}") = $(@sprintf "%.4g" results.tr.fsl2[]*100)$(L"\%")
#             \\item $(L"m(\hat{w};U,T)") = $(@sprintf "%.4g" results.tr.mDist[])
#             \\item dt    = $(@sprintf "%.4g" results.tr.dt[]) s
#             \\item iters = $(results.tr.iters[])
#         \\end{itemize}
#         """
#         write(f, s)
#     end
# end
##
# results[algoName].p2[] = plotResParts(results[algoName].wits[], wendyProb, "Parts Of Resdidual<br>Trust Region Solver<br>$(ex.name)")
##
# savefig( 
#     results[algoName].p2[], 
#     joinpath(
#         joinpath(
#             "/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig",
#             "$(ex.name)_$(algoName)_resPrts.png",
#         )
#     );
#     height=600,
#     width=700
# )

##
## 
# @info "Trust Region Solve with Analytic Hessian"
# relErr = norm(wts - wTrue) / norm(wTrue)
# @info """  
#     coef relErr =  $(relErr*100)%
#     fs relErr   =  $(fsRelErr*100)%
#     iter        =  $(iter_ts)
# """
# ##
# @info "  ARC with Analytic Hessian "
#     relErr = norm(warc - wTrue) / norm(wTrue)
#     @info """  
#         coef relErr =  $(relErr*100)%
#         fs relErr   =  $(fsRelErr*100)%
#         iter        =  $(size(wit_arc,2))
#     """
##
# D = wendyProb.D
# M = wendyProb.M
# F = zeros(D,M)
# myLF = zeros(D,M)
# LF = zeros(D,M)
# for m in 1:M 
#    @views ROBERTSON_f!(F[:,m], wendyProb.U_exact[:,m],  wTrue, nothing)
#    @views ROBERTSON_logf!(myLF[:,m], wendyProb.U_exact[:,m],  wTrue, nothing)
#    @views wendyProb.f!(LF[:,m], wTrue, wendyProb.U_exact[:,m])
# end
# ##
# trs = AbstractTrace[]
# for d in 1:D 
    # push!(
    #     trs,
    #     scatter(
    #         x = wendyProb.tt,
    #         y = wendyProb.U_exact[d,:],
    #         name="u[$d]"
    #     )
    # )
    # push!(
    #     trs,
    #     scatter(
    #         x = wendyProb.tt,
    #         y = wendyProb.U_exact[d,:],
    #         name="u[$d]"
    #     )
    # )
    # push!(
    #     trs,
    #     scatter(
    #         x = wendyProb.tt,
    #         y = F[d,:],
    #         name="f(u)[$d]"
    #     )
    # )
    # push!(
    #     trs,
    #     scatter(
    #         x = wendyProb.tt,
    #         y = log.(wendyProb.U_exact[d,:]),
    #         name="log(u[$d])"
    #     )
    # )
#     push!(
#         trs,
#         scatter(
#             x = wendyProb.tt,
#             y = LF[d,:],
#             name="(f(u)/u)[$d]"
#         )
#     )
#     push!(
#         trs,
#         scatter(
#             x = wendyProb.tt,
#             y = myLF[d,:],
#             name="my(f(u)/u)[$d]"
#         )
#     )
# end 
# plotjs(
#     trs,
#     Layout(
#         title="U_exact fo $(ex.name)"
#     )
# )
##

##
