using FileIO, Dates, Printf, ProgressMeter
using Tullio: @tullio
using Latexify, ColorSchemes, Colors, LaTeXStrings # for plotting and saving captions to tex
using PlotlyJS
using PlotlyJS: savefig, scatter, Layout, AbstractTrace, attr
import PlotlyJS.plot as plotjs
Latexify.set_default(fmt = "%.4g")
try 
    using WENDy: _g!
catch 
    @info "wendy g! not loaded perhaps not modulizing wendy..."
end
##
PLOTLYJS_COLORS = [
    colorant"#1f77b4",  # muted blue
    colorant"#ff7f0e",  # safety orange
    colorant"#2ca02c",  # cooked asparagus green
    colorant"#d62728",  # brick red
    colorant"#9467bd",  # muted purple
    colorant"#8c564b",  # chestnut brown
    colorant"#e377c2",  # raspberry yogurt pink
    colorant"#7f7f7f",  # middle gray
    colorant"#bcbd22",  # curry yellow-green
    colorant"#17becf"   # blue-teal
]
# DATA_COLORS = colorschemes[:bam10]
# TRUTH_COLORS = colorschemes[:acton10]
# APPROX_COLORS = colorschemes[:navia10][end-4:end-1]
DATA_COLORS = colorschemes[:seaborn_deep]
TRUTH_COLORS = colorschemes[:seaborn_bright]
APPROX_COLORS = colorschemes[:seaborn_bright]
SOLVER_COLORS = colorschemes[:Dark2_8]
##
function plotjs(prob::WENDyProblem, title::String="", file::Union{Nothing, String}=nothing, yaxis_type::String="linear", showNoiseData::Bool=true)
    D = prob.D 
    trs = AbstractTrace[]
    for d in 1:D 
        showNoiseData && push!( 
            trs,
            scatter(
                x=prob.tt,
                y=prob.U[d,:],
                name="U_$d", 
                mode="markers" ,
                marker_color=DATA_COLORS[d]
            )
        )
        push!( 
            trs,
            scatter(
                x=prob.tt,
                y=prob.U_exact[d,:],
                name="U*_$d", 
                mode="lines" ,
                line_dash="dash",
                line_color=TRUTH_COLORS[d]
            )
        )
    end
    p = plotjs(
       trs, 
       Layout(
            title_text=title, 
            title_x=0.5,
            title_xanchor="center",
            yaxis_type=yaxis_type,
            showlegend=true, 
            xaxis_title="time (s)"
        )
    )
    !isnothing(file) && PlotlyJS.savefig(
        p,
        file;
        height=600,
        width=800
    )
    p
end
## Break apart the residual into its "Asymptotic" parts
function breakApartResidual(wendyProb::WENDyProblem{lip, DistType}, w::AbstractVector{<:Real}) where {lip, DistType<:Distribution}
    K,M,D,J = wendyProb.K, wendyProb.M, wendyProb.D, wendyProb.J
    tt, U_exact, U, V, Vp, E, b, f!, jacuf! = wendyProb.tt, wendyProb.U_exact, wendyProb.U, wendyProb.V, wendyProb.Vp, wendyProb.noise, wendyProb.b₀,wendyProb.f!, wendyProb.jacuf!
    wTrue = wendyProb.wTrue
    g_w_U = zeros(K*D)
    g_w_Ustar = zeros(K*D)
    g_wstar_Ustar = zeros(K*D)
    F = zeros(D, M)
    G = zeros(K, D)
    ## g_w_U
    _g!(
        g_w_U, w, 
        tt, U, V, 
        f!, 
        F,G); 
    ## g_w_Ustar
    _g!(
        g_w_Ustar, w, 
        tt, U_exact, V, 
        f!, 
        F,G); 
    ##
    _g!(
        g_wstar_Ustar, wTrue, 
        tt, U_exact, V, 
        f!, 
        F,G); 
    ##
    JuF = zeros(D,D,M)
    for m in 1:M 
        @views jacuf!(
            JuF[:,:,m],
            U[:,m], w, tt[m]
        )
    end
    @tullio _L₁[k,d2,m,d1] := JuF[d2,d1,m] * V[k,m] # increases allocation from 4 to 45 
    ∇g_w_U = reshape(_L₁,K*D,M*D)
    ##
    r₀ = g_w_Ustar - g_wstar_Ustar
    ##
    b_star = DistType == Normal ? reshape(-Vp * U_exact', K*D) : reshape(-Vp * log.(U_exact)', K*D)
    eⁱⁿᵗ =  g_wstar_Ustar - b_star
    r = g_w_U - b 
    E = DistType == Normal ? E : log.(E)
    b_ϵ =  reshape(-Vp * E', K*D)
    eᶿ = g_w_U - g_w_Ustar
    eᵋ = r - r₀ - eⁱⁿᵗ
    linPrt = ∇g_w_U * reshape(E, M*D) - b_ϵ
    hOrdTerms = eᵋ - linPrt
    @assert norm(r - eⁱⁿᵗ - r₀ - eᵋ) / norm(r) < 1e2*eps() "Splitting the residual did not work.... Assumptions are wrong"
    return (
        eᵋ=eᵋ, 
        eⁱⁿᵗ=eⁱⁿᵗ, 
        r₀=r₀, 
        linPrt=linPrt,
        hOrdTerms=hOrdTerms
    )
end
# Now plot it 
function plotResParts(wits, wendyProb, title)
    II = size(wits,2)
    res = zeros(5, II)
    for i in 1:II 
        prts = breakApartResidual(wendyProb, wits[:,i])
        res[1,i] = norm(prts.eᵋ)
        res[2,i] = norm(prts.eⁱⁿᵗ)
        res[3,i] = norm(prts.r₀)
        res[4,i] = norm(prts.linPrt)
        res[5,i] = norm(prts.hOrdTerms)
    end 
    SOLVER_COLORS = colorschemes[:Dark2_8]
    p = plotjs(
        [
            scatter(
                y=res[1,:],
                name="eᵋ",
                line_color=SOLVER_COLORS[1],
                legendgroup=1
            ), 
            scatter(
                y=res[2,:],
                name="eⁱⁿᵗ" ,
                line_color=SOLVER_COLORS[2],
                legendgroup=2
            ),
            scatter(
                y=res[3,:],
                name="r₀",
                line_color=SOLVER_COLORS[3],
                legendgroup=3
            ),
            scatter(
                y=res[4,:],
                name="∇G - b(ϵ)",
                line_color=SOLVER_COLORS[1],
                line_dash="dash",
                legendgroup=1
            ),
            scatter(
                y=res[5,:],
                name="O(ϵ²)",
                line_color=SOLVER_COLORS[1],
                line_dash="dot",
                legendgroup=1
            ),
        ],
        Layout(
            yaxis_type="log",
            yaxis_title_text="||⋅||₂",
            xaxis_title_text="Iteration",
            title_text=title,
            hovermode="x unified"
        )
    )
end
##
function plotSweep(data::Dict, metric::Symbol,odeName::String,displayName::String,yaxis_type)
    @info "Plotting $displayName $metric"
    results = data["results"] # zeros(nr,sub,mu,mc,J)
    optimparams = data["optimparams"]
    noiseRatios = data["noiseRatios"]
    timeSubsampleRates = data["timeSubsampleRates"]
    μs = data["μs"]
    metrics = [:what,:cl2,:fsl2,:mDist,:dt,:iters]

    metric2Str = (cl2="Coefficient Relative Error",fsl2="Forward Solve Relative Error",mDist="Mahalanobis Distance",dt="Run Time",iters="Number of Iterations")
    algo2Str = (tr="Trust Region",irwls="Iterative Reweighted<br>Least Squares",arc="Adaptive Cubic Regularization",fsnls_tr="Forward Solve Nonlinear<br>Least Squares")
    dashopts = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
    colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf"   # blue-teal
    ]
    markerSymbols=["circle", "triangle-up","triangle-down", "square","diamond"]
    trs = AbstractTrace[]
    l,u = (Inf,-Inf)
    for (i,name) in enumerate(keys(algos)), (ii,sub) in enumerate(timeSubsampleRates), (iii,μ) in enumerate(μs) 
        @views Y = results[name][metric][:,ii,iii,:]
        @views begin 
            y = [
                try 
                    median(filter(x->!isnan.(x), Y[k,:])) 
                catch 
                    NaN
                end
            for k in 1:size(Y,1)]
        end
        if !all(isnan.(y))
            l = min(l,floor(minimum(filter(x->!isnan.(x), log10.(y)))))
            u = max(u,ceil(maximum(filter(x->!isnan.(x), log10.(y)))))
        end
        @views stdy = [std(filter(x->!isnan.(x), Y[k,:])) for k in 1:size(Y,1)]
        numNaN = sum(isnan.(results[name][metric]))
        push!(
            trs,
            scatter(
                x=noiseRatios .+ 0.001*i,
                y=y,
                # error_y=attr(type="data", array=stdy,visible=true),
                marker=attr(
                    color=colors[i],
                    opacity=0.7,
                    size=20,
                    symbol=markerSymbols[i]
                ),
                line=attr(
                    dash=dashopts[i],
                    width=5
                ),
                name="$(algo2Str[name])<br>  Fail rate $(@sprintf "%.3g" (numNaN/length(results[name][metric][:])*100))%",#)<br>numPts = $(Int(1024/sub)), Fails=$numNaN" : "numPts = $(Int(1024/sub)), Fails=$numNaN",
                legendgroup=name,
                xaxis="x$(4-ii)",
                yaxis="y$(iii)",
                showlegend=iii==1 && ii==2
            )
        )
    end
    if isinf(l)
        l = -1
    end
    if u > 6
        u = 6
    end
    yaxis_tickvals = [1*10^e for e in l:u]
    yaxis_range = (l,u)
    p = plotjs(
        trs, 
        Layout(
            title=attr(
                text="$displayName<br>Median $(metric2Str[metric])",
                font_size=40,
                xanchor="center", 
                x=0.5,
                y=0.95
            ),
            xaxis1=attr(
                mirror="allticks",
                tickfont_size=20,
                tickvals=noiseRatios,
                title=attr(
                    font_size=20, 
                    text="Noise Ratio<br>Number of points $(Int(1024/timeSubsampleRates[3]))", 
                ),
                domain=[0.05,0.3]
            ),
            xaxis2=attr(
                mirror="allticks",
                tickfont_size=20,
                tickvals=noiseRatios,
                title=attr(
                    font_size=20, 
                    text="Noise Ratio<br>Number of points $(Int(1024/timeSubsampleRates[2]))", 
                ),
                domain=[0.35,0.6]
            ),
            xaxis3=attr(
                mirror="allticks",
                tickfont_size=20,
                tickvals=noiseRatios,
                title=attr(
                    font_size=20, 
                    text="Noise Ratio<br>Number of points $(Int(1024/timeSubsampleRates[1]))"), 
                domain=[0.65,0.9]
            ),
            yaxis1=attr(
                mirror="allticks",
                tickfont_size=20,
                type=yaxis_type,
                tickvals=yaxis_tickvals,
                range=yaxis_range,
                domain=[0.05,0.3],
                title=attr(
                    font_size=20, 
                    text="μ=$(μs[1])"
                )
            ),
            yaxis2=attr(
                mirror="allticks",
                tickfont_size=20,
                type=yaxis_type,
                tickvals=yaxis_tickvals,
                range=yaxis_range,
                domain=[0.35,0.6],
                title=attr(
                    font_size=20, 
                    text="μ=$(μs[2])"
                )
            ),
            yaxis3=attr(
                mirror="allticks",
                tickfont_size=20,
                type=yaxis_type,
                tickvals=yaxis_tickvals,
                range=yaxis_range,
                domain=[0.65,0.9],
                title=attr(
                    font_size=20, 
                    text="μ=$(μs[3])"
                )
            ),
            legend=attr(
                x=.925,
                y=0.5,
                yanchor="center",
                font=(
                    family="sans-serif",
                    size=20,
                    color="#000"
                ),
                bgcolor="#E2E2E2",
                bordercolor="#FFFFFF",
                borderwidth= 2,
            ),
            legendgrouptitle=attr(text="")
        )
    )
    savefile= joinpath("/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig","$(odeName)_$(String(metric)).png")
    savefig(
        p, savefile;
        width=1200,
        height=800,
    )
    p
end 
function plots_07_17_2024()
    for (file, odeName, displayName) in zip(
        (
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/mostRecent/hindmarshRose_15.07.2024.jld2",
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/mostRecent/logisticGrowth_15.07.2024.jld2", 
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/mostRecent/goodwin_15.07.2024.jld2",
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/mostRecent/robertson_19.07.2024.jld2",
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/mostRecent/sir_19.07.2024.jld2"
        ),
        (
            "hindmarshRose", 
            "logisticGrowth",
            "goodwin", 
            "robertson", 
            "sir"
        ),
        (
            "Hindmarsh-Rose",
            "LogisticGrowth",
            "Goodwin", 
            "Robertson", 
            "SIR"
        )
    )
        data = load(file)
        for metric in (:cl2,:fsl2,:mDist,:dt,:iters)
            # if metric != :fsl2
            #     continue 
            # end
            p = plotSweep(data, metric, odeName,displayName,"log")
            # display(p)
        end 
    end
end
##
function plotSols(examples=[GOODWIN, HINDMARSH_ROSE, LOGISTIC_GROWTH, ROBERTSON, SIR])
    for data in examples
        file = joinpath("/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig", "$(data.name)_sol.png")
        title = "$(data.name) Solution"
        yaxis_type = data.name == "robertson" ? "log" : "linear"
        plotjs(data;title_text=title, file=file, yaxis_type=yaxis_type)
    end
end
##
function plotDeepDive(wendyProb, what; title="", yaxis_type="linear", file=nothing)
    Uhat = forwardSolve(wendyProb, what, 
    reltol=1e-12
    )
    D = wendyProb.D
    M = wendyProb.M
    F = zeros(D,M)
    trs = AbstractTrace[]
    for d in 1:D 
        push!(
            trs,
            scatter(
                x = wendyProb.tt,
                y = wendyProb.U[d,:],
                mode="markers",
                maker_opacity=0.8,
                maker_size=1.5,
                marker_color=DATA_COLORS[d],
                name="U[$d]",
                legendgroup=d
            )
        )
        push!(
            trs,
            scatter(
                x = wendyProb.tt,
                y = wendyProb.U_exact[d,:],
                line_color=TRUTH_COLORS[d],
                line_width=3,
                line_dash="dot",
                name="U*[$d]",
                legendgroup=d
            )
        )
        push!(
            trs,
            scatter(
                x = wendyProb.tt[1:size(Uhat,2)],
                y = Uhat[d,:],
                mode="lines",
                line_color=APPROX_COLORS[d],
                line_width=3,
                line_opacity=.4,
                name="Û[$d]",
                legendgroup=d
            )
        )
    end 
    yaxis_range = if yaxis_type == "log"
        l,u = extrema(log10.(vcat(wendyProb.U_exact[:],Uhat[:],wendyProb.U[:])))
        l -= 0.5
        u += 0.5
        l,u
    else 
        # l,u = extrema(vcat(wendyProb.U_exact[:],Uhat[:],wendyProb.U[:]))
        # l -=
        "auto"
    end
    p = plotjs(
        trs,
        Layout(
            yaxis_type=yaxis_type,
            title_text=title,
            title_font_size=36,
            title_x=0.5,
            title_xanchor = "center",
            yaxis_range = yaxis_range,
            xaxis_title = "time (s)",
            hovermode = "x unified",
        ),
    )

    !isnothing(file) && savefig( 
        p, 
        file;
        height=600,
        width=700
    )

    p
end 
##
function plotWits(m::Function, wits_vec::AbstractVector{<:AbstractMatrix}; names::Union{Nothing, AbstractVector{<:AbstractString}}=nothing, title::String="", yaxis_type::String="linear")
    !isnothing(names) && @assert length(names) == length(wits_vec)
    trs = AbstractTrace[]
    costs = Vector(undef, length(wits_vec))
    for (i, wits) in enumerate(wits_vec)
        II = size(wits,2)
        costs[i] = zeros(II+1) 
        name = isnothing(names) ? "Solver $i" : names[i]
        @info name
        @showprogress desc="Computing Costs for $name" for ii in 1:II
            costs[i][ii+1] = m(wits[:,ii]) 
        end
        costs[i] = filter(x->!isnan(x), costs[i])
    end
    if yaxis_type == "log"
        min_cost = minimum(minimum(costs[i][:]) for i in 1:length(costs))
        if min_cost < 0 
            for i in 1:length(costs)
                costs[i][:] .-= min_cost - 1e-12
            end
        end
    end
    for (i, wits) in enumerate(wits_vec)
        name = isnothing(names) ? "Solver $i" : names[i]
        push!(
            trs,
            scatter(
                y=costs[i],
                name=name,
                mode="lines",
                marker_color=SOLVER_COLORS[i]
            )
        )
    end
    plotjs(
        trs,
        Layout(
            title_text=title, 
            hovermode="x unified",
            xaxis_type="log",
            yaxis_type=yaxis_type,
        )
    )
end
##
function plotCostSurface(
    wendyProb::WENDyProblem, nll::CostFunction, IX::NTuple{2,Int}=(1,2); 
    w1Rng::Union{Nothing, AbstractVector}=nothing, w2Rng::Union{Nothing, AbstractVector}=nothing, del::Real=2, step::Real=0.1, 
    wFix::Union{Nothing, AbstractVector{<:Real}}=nothing, zaxis_type::String="linear"
)
    ## Define the values we sweep over 
    isnothing(wFix) && (wFix = copy(wendyProb.wTrue))
    isnothing(w1Rng) && (w1Rng = range(wFix[IX[1]]-del, step=step, stop=wFix[IX[1]]+del))
    isnothing(w2Rng) && (w2Rng = range(wFix[IX[2]]-del, step=step, stop=wFix[IX[2]]+del))
    w = similar(wFix)
    mm = zeros(length(w1Rng), length(w2Rng))
    xx = similar(mm)
    yy = similar(mm)
    @showprogress "Getting surface values from cost fun..." for (n,w1) in enumerate(w1Rng), (nn,w2) in enumerate(w2Rng)
        w .= wFix 
        w[IX[1]] = w1
        w[IX[2]] = w2
        xx[n,nn] = w1
        yy[n,nn] = w2
        mm[n,nn] = nll.f(w)
    end
    ## use log scale on z-axis plotly seems busted here
    if zaxis_type == "log" 
        @views minVal = minimum(mm[:])
        if minVal < 0 
            mm .-= minVal - 1e-12
        end
        mm .= log10.(mm)
    end 
    ##
    p = plotjs(
        PlotlyJS.surface(
            x=xx,
            y=yy,
            z=mm,
            showscale=false        
        ),
        Layout(
            # zaxis=attr(type="log")
        )
    )
    
end