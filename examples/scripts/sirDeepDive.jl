## Load everything we need 
includet("util.jl")
##
ex = SIR;
# Recall you can change the lip or DistType like this 
# ex = SimulatedWENDyData(ex, Val(true), Val(Normal))
##
wendyProb, params, simParams = init(
    ex;
    noiseRatio = 0.40, 
    seed = 1, 
    timeSubsampleRate = 4, 
    μ = 1.0,
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
results = runAlgos(wendyProb, params, nll, l2Loss, algoNames; return_wits=true);
## Make plots
# Residual parted out
_algoName = :arc_sfn
_p_resPrts = plotResParts(
    results[_algoName].wits[], 
    wendyProb, 
    "Residual Broken Apart<br>$(ex2Disp[ex.name]) : $(algo2Disp[_algoName])"
)
display(_p_resPrts);
## Plot different cost functions over the iterates, in this case we have nll
_p_wits_nll = plotWits(
    w -> try
        nll.f(w)
    catch 
        NaN
    end,
    [results[algoName].wits[] for algoName in algoNames];
    names = [algo2Disp[algoName] for algoName in algoNames],
    title="Negative Log-Likelihood<br>$(ex2Disp[ex.name])",
    yaxis_type="log"
);
display(_p_wits_nll);
## now fwrSim l2
_p_wits_fsl2 = plotWits(
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
    title="Avgerage Forward Solve Relative Error<br>$(ex2Disp[ex.name])"
)
display(_p_wits_fsl2);
## Plot the solutions 
_p_sols = Dict()
for algoName in vcat(:init, algoNames)
    # continue
    try
        what = algoName == :init ? w0 : results[algoName].wits[][:,end]
        _p_sols[algoName] = plotDeepDive(
            wendyProb, what;
            title="$(algoName == :init ? "w₀ Solution" : algo2Disp[algoName])<br>$(ex2Disp[ex.name])",
            yaxis_type="linear"
        )
    catch 
        @warn "$(algo2Disp[algoName]) failed to plot most likely, fs is imposible at what"
    end
end
display(_p_sols[:init])
display(_p_sols[:arc_sfn])
##
save_dir = "/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig/$(ex.name)"
Base.mkpath(save_dir)
relayout!(
    _p_resPrts,
    title_font_size=36,
    title_y=0.925,
    yaxis_domain=[0, 0.9],
    yaxis_tickfont_size=20,
    xaxis_tickfont_size=20,
    xaxis_title_text="Iteration",
    yaxis_title_font_size=20,
    xaxis_title_font_size=20,
    legend_font_size=20,
    legend_y = .85,
)
savefig(
    _p_resPrts,
    joinpath(save_dir, "$(ex.name)_resPrts.png"),
    width=1000, height=700
)
relayout!(
    _p_wits_fsl2,
    title_font_size=36,
    title_y=0.925,
    yaxis_domain=[0, 0.9],
    yaxis_tickfont_size=20,
    xaxis_tickfont_size=20,
    xaxis_title_text="Iteration",
    xaxis_title_font_size=20,
    legend_font_size=20,
    legend_y = .85,
)
relayout!(
    _p_wits_nll,
    title_font_size=36,
    title_y=0.925,
    yaxis_domain=[0, 0.9],
    yaxis_tickfont_size=20,
    xaxis_tickfont_size=20,
    xaxis_title_text="Iteration",
    xaxis_title_font_size=20,
    showlegend=false
    # legend_font_size=20,
    # legend_y = .85,
)
savefig(
    _p_wits_nll,
    joinpath(save_dir, "$(ex.name)_wits_nll.png"),
    width=1000, height=700
)
savefig(
    _p_wits_fsl2,
    joinpath(save_dir, "$(ex.name)_wits_fsl2.png"),
    width=1200, height=700
)
for algoName in [:init, :arc_sfn]
    _p = _p_sols[algoName]
    relayout!(
        _p,
        title_y=0.925,
        yaxis_domain=[0, 0.9],
        yaxis_tickfont_size=20,
        xaxis_tickfont_size=20,
        xaxis_title_font_size=20,
        legend_font_size=20,
        legend_y = .85,
    )
    algoName == :init && relayout!(_p, showlegend=false)
    width = algoName == :init ? 900 : 1000
    savefig(
        _p,
        joinpath(save_dir, "$(ex.name)_$(algoName)_solWData.png"),
        width=width, height=700
    )
end
## Automate the latex so we dont have to copy and past numbers over...
J = wendyProb.J 
wTrue = wendyProb.wTrue
open(joinpath(save_dir, "$(ex.name)_caption.tex"), "w") do f
    s = """
    The plots above show the solutions to the $(ex2Disp[ex.name]) differential equations. The plot on the left show the solution with the intial guessed parameters $(L"w_0") and the plot of the right shows the approximated parameteres form minimizing the negative log-likelihood with the $(L"ARC_qK") solver provided by SFN.jl. The true solution, $(L"U^*"), is shown with the dashed lines. The noisey data, $(L"U"), is displayed with the markers, and the approximation, $(L"\hat{U}"), using the specified parameters are shown with the solid lines. 
    
    The noise was multiplicative and distributed LogNormal with a SNR of $(simParams.noiseRatio*100)$(L"\%"). We initialized both algorithms with the same random point with $(L"\rho") = $(simParams.μ). The data was subsampled down to $(wendyProb.M) points.
    In this case the true parameters were \\\\
    $(L"w^*") = [$(prod(j == J ? (@sprintf "%.4g" wj) : (@sprintf "%.4g, " wj) for (j,wj) in enumerate(wendyProb.wTrue)))], \\\\
    and the initial guess for parameterns was \\\\
    $(L"w_0") = [$(prod(j == J ? (@sprintf "%.4g" wj) : (@sprintf "%.4g, " wj) for (j,wj) in enumerate(w0)))] \\\\

    While the FSNLS could not begin to iterate because $(L"w_0") was to far away from reasonable values, our solvers could approximate the parameters at least to some degree
"""
    write(f, s)
end
##
cl2_0 =  norm(w0 - wTrue) / norm(wTrue)
mean_fsl2_0, final_fsl2_0 = forwardSolveRelErr(wendyProb, w0)
_nll_0 = nll.f(w0)
table = Any[]
push!(table, Rule(:top), [
    "Algorithm",
    L"\tfrac{\|\hat{w} - w^*\|_2}{\|w^*\|}_2",
    L"$\tfrac{\sum_{m=1}^M \|\hat{u}_m - u^*_m\|_2}{\sum_{m=1}^M \|u^*_m\|_2}$", 
    L"\tfrac{\|\hat{u}_M - u^*_M\|_2}{\|u^*_M\|_2}",
    L"-\mathcal{L}(\hat{w};U,T)",
    "Run Time (s)",
    "Iterations"
])
push!(table, Rule(:mid), [ L"w_0",  "$(@sprintf "%.4g" cl2_0)", "$(@sprintf "%.4g" mean_fsl2_0)", "$(@sprintf "%.4g" final_fsl2_0)", "$(@sprintf "%.4g" _nll_0)", "", ""])
for algoName in algoNames
    _name = algo2Disp[algoName]
    _name = split(_name, "<br>")[1]
    push!(
        table, 
        Rule(), 
        vcat(
            _name,
            ["$(@sprintf "%.4g" results[algoName][metric][])" for metric in [:cl2, :mean_fsl2, :final_fsl2, :nll, :dt]],
            "$(results[algoName].iters[])"
        )
    ) 
end
push!(table, Rule(:bottom))
latex_tabular(
    joinpath(save_dir, "table.tex"), 
    Tabular("lcccccc"), 
    table
)


