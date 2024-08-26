## Load everything we need 
includet("util.jl")
##
ex = MULTIMODAL;
# Recall you can change the lip or DistType like this 
# ex = SimulatedWENDyData(ex, Val(true), Val(Normal))
##
algoNames = [:tr_optim,:arc_sfn,:arc_jso,:irwls,:fsnls_tr_optim, :fsnll_tr_optim] # :tr_jso
(
    wendyProb, params, simParams, 
    nll, l2Loss, nll_fwdSolve,
    w0, results
) = runExample(
    ex;
    algoNames=algoNames ,
    noiseRatio = .20, 
    seed = 2, 
    timeSubsampleRate = 8, 
    μ = 1.00,
    optimMaxiters = 200, 
    optimTimelimit = 200.0,
    allowNegW0=true
);
## Make plots
_p_resPrts, _p_wits_fsnll, _p_wits_nll, _p_wits_fsl2, _p_sols, p_l2, p_nll = makePlots(
    wendyProb, 
    nll, l2Loss, nll_fwdSolve,
    w0, results, algoNames
);
##
save_dir = "/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig/$(ex.name)"

## Automate the latex so we dont have to copy and past numbers over...
J = wendyProb.J 
wTrue = wendyProb.wTrue
open(joinpath(save_dir, "$(ex.name)_caption.tex"), "w") do f
    s = """
    The plots above show the solutions to the $(ex2Disp[ex.name]) differential equations. The plot on the left show the solution with the intial guessed parameters $(L"w_0") and the plot of the right shows the approximated parameteres form minimizing the negative log-likelihood with the $(L"ARC_qK") solver provided by SFN.jl. The true solution, $(L"U^*"), is shown with the dashed lines. The noisey data, $(L"U"), is displayed with the markers, and the approximation, $(L"\hat{U}"), using the specified parameters are shown with the solid lines. 
    
    The noise was additive and distributed Normal with a SNR of $(simParams.noiseRatio*100)$(L"\%"). We initialized both algorithms with the same random point with $(L"\rho") = $(simParams.μ). The data was subsampled down to $(wendyProb.M) points.
    In this case the true parameters were \\\\
    $(L"w^*") = [$(prod(j == J ? (@sprintf "%.4g" wj) : (@sprintf "%.4g, " wj) for (j,wj) in enumerate(wendyProb.wTrue)))], \\\\
    and the initial guess for parameterns was \\\\
    $(L"w_0") = [$(prod(j == J ? (@sprintf "%.4g" wj) : (@sprintf "%.4g, " wj) for (j,wj) in enumerate(w0)))] \\\\

    While most of our methods converge, oddly the trust region solver from JSOSolvers.jl fails horibly. Not sure what to make of that. In any case, FSNLS also fails. Also, it is interesting that IRWLS does well in this example as well. 
"""
    write(f, s)
end
##
cl2_0 = norm(w0 - wTrue) / norm(wTrue)
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


##
_wits = results.arc_sfn.wits[]
IX = (2,3)
_w1_min, _w1_max = extrema(_wits[IX[1],:])
_w2_min, _w2_max = extrema(_wits[IX[2],:])
_w1_min = min(_w1_min, wTrue[IX[1]]) - 1
_w1_max = max(_w1_max, wTrue[IX[1]]) + 1
_w2_min = min(_w2_min, wTrue[IX[2]]) - 1
_w2_max = max(_w2_max, wTrue[IX[2]]) + 1
p_nll = plotCostSurface(
    wendyProb, nll_fwdSolve, IX; wFix=w0,
    w1Rng=range(_w1_min, step=0.1, stop=_w1_max),
    w2Rng=range(_w2_min, step=0.1, stop=_w2_max)
);
xx = []
yy = []
zz = []
hovertext=[]
for i in 1:size(_wits,2)
    push!(xx, _wits[IX[1],i])
    push!(yy, _wits[IX[2],i])
    push!(zz, nll_fwdSolve.f(_wits[:,i]))
    push!(hovertext, """
        Iteration: $i 
        -ℒ = $(zz[end])
    """
    )
end
addtraces!(p_nll, scatter3d(x=xx, y=yy, z=zz, hovertext=hovertext))
addtraces!(p_nll,scatter3d(
    x=[wTrue[IX[1]]],
    y=[wTrue[IX[2]]],
    z=[nll_fwdSolve.f(wTrue)],
    text=["True Parameters"],
    textposition=["top left"],
    textfont=[attr(color="white")],
    mode="markers+text"
))
relayout!(
    p_nll, 
    title_text="Negative Log-Likelihood",
    title_y=.9,
    title_font_size=36,
    title_yanchor="center",
    scene=attr(
        zaxis=attr(title="-ℒ"),
        xaxis=attr(title="w₁"),
        yaxis=attr(title="w₂"),
        # xaxis_title_font_size=20,
        camera_eye=attr(x=-2, y=1, z=.1),
    ),
    margin=attr(t=30, r=0, l=20, b=10),
    annotations=[]
)
p_nll
# savefig(
#     p_nll,
#     joinpath(save_dir, "$(ex.name)_nllCostSpace_iters.png"),
#     width=800, height=700
# )
##
function _nll_fwdSolve(w)
    Uhat = forwardSolve(wendyProb, w)
    sig = wendyProb.sig
    U = wendyProb.U
    M = wendyProb.M
    sum(
        1/2*log(2*pi) 
        + J/2 * sum(log.(sig)) 
        + 1/2*dot(Uhat[:,m] - U[:,m], diagm(1 ./ sig), Uhat[:,m] - U[:,m])
        for m in 1:M
    )
end
nll_fwdSolve = WENDy.SecondOrderCostFunction(
    _nll_fwdSolve, 
    (g,w) -> ForwardDiff.gradient!(g, _nll_fwdSolve, w),
    (H,w) -> ForwardDiff.hessian!(H, _nll_fwdSolve, w),
)
g = zeros(J)
H = zeros(J,J)
@time "f nll_fwdSolve" nll_fwdSolve.f(wTrue)
@time "∇f! nll_fwdSolve" nll_fwdSolve.∇f!(g, wTrue)
@time "Hf! nll_fwdSolve" nll_fwdSolve.Hf!(H,wTrue)
p_nll_fwdSolve = plotCostSurface(
    wendyProb, nll, (3,4);
    w1Rng=range(0, step=0.5, stop=5),
    w2Rng=range(0, step=0.5, stop=5)
)
## Contour plot
IX = (3,4)
_wits = results.arc_sfn.wits[]
p_nll = plotContourSurface(
    wendyProb, nll, (3,4);
    w1Rng=range(0, step=0.1, stop=5), 
    w2Rng=range(0, step=0.1, stop=5),
    zaxis_type="log"
)

relayout!(
    p_nll, 
    title_text="Negative Log-Likelihood",
    # title_y=.9,
    # title_font_size=36,
    title_yanchor="center",
    # showcolorbar=false,
    xaxis=attr(
        title="w_$(IX[1])"
    ),
    yaxis=attr(
        title="w_$(IX[2])"
    ),
    margin=attr(t=30, r=0, l=20, b=10),
    annotations=[
        attr(
            x=_wits[IX[1],i],
            y=_wits[IX[2],i],
            z=log10.(nll.f(_wits[:,i])),
            text="Iter $i",
            font=attr(
                color="white"
            )
        )
        for i in 1:size(_wits,2)
    ] 
)
addtraces!(
    p_nll, 
    scatter( 
        x=_wits[IX[1],:],
        y=_wits[IX[2],:],
        z=log10.([nll.f(_wits[:,i]) for i in 1:size(_wits,2)]),
        mode="markers",
        hovertext=["w_$(IX[1])=$(_wits[IX[1],i])<br>w_$(IX[2])=$(_wits[IX[2],i])<br>log₁₀(-ℒ)=$(log10(nll.f(_wits[:,i])))" for i in 1:size(_wits,2)]
    )
)
p_nll