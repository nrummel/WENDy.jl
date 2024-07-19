using FileIO, Dates, Printf
using PlotlyJS: savefig, scatter, Layout, AbstractTrace, attr, plot as plotjs
using Plots: plot
function breakApartResidual(wendyProb::WENDyProblem{LinearInParameters, DistType}, w::AbstractVector{<:Real}) where {LinearInParameters, DistType<:Distribution}
    K,M,D = wendyProb.K, wendyProb.M, wendyProb.D
    tt, U_exact, U, V, Vp, E, b, f! = wendyProb.tt, wendyProb.U_exact, wendyProb.U, wendyProb.V, wendyProb.Vp, wendyProb.noise, wendyProb.b₀,wendyProb.f!
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
    r₀ = g_w_Ustar - g_wstar_Ustar
    ##
    b_star = DistType == Normal ? reshape(-Vp * U_exact', K*D) : reshape(-Vp * log.(U_exact)', K*D)
    eⁱⁿᵗ =  g_wstar_Ustar - b_star
    r = g_w_U - b 
    b_ϵ = DistType == Normal ? reshape(-Vp * E', K*D) : reshape(-Vp * log.(E)', K*D)
    eᶿ = g_w_U - g_w_Ustar
    eᶿmbᵋ = g_w_U - g_w_Ustar - b_ϵ
    @assert norm(r - eⁱⁿᵗ - r₀ - eᶿmbᵋ) / norm(r) < 1e2*eps() "Splitting the residual did not work.... Assumptions are wrong"
    return (eᶿmbᵋ=eᶿmbᵋ, eⁱⁿᵗ=eⁱⁿᵗ, r₀=r₀)
end

function plotResParts(wits, wendyProb, name)
    II = size(wits,2)
    res = zeros(3, II)
    for i in 1:II 
        prts = breakApartResidual(wendyProb, wits[:,i])
        res[1,i] = norm(prts.eᶿmbᵋ)
        res[2,i] = norm(prts.eⁱⁿᵗ)
        res[3,i] = norm(prts.r₀)
    end 
    p = plotjs(
        [
            scatter(
                y=res[1,:],
                name="eᶿ-bᵋ"
            ), 
            scatter(
                y=res[2,:],
                name="eⁱⁿᵗ" 
            ),
            scatter(
                y=res[3,:],
                name="r₀"
            ),
        ],
        Layout(
            yaxis_type="log",
            yaxis_title="||⋅||₂",
            xaxis_title="Iteration",
            title="Parts of residual for $name",
        )
    )
end
##
function plotSweep(data::Dict, metric::Symbol,odeName::String,displayName::String,yaxis_tickvals, yaxis_range,yaxis_type)
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
    p = plotjs(
        trs, 
        Layout(
            title=attr(
                text="$displayName<br>Median $(metric2Str[metric])",
                font_size=64,
                xanchor="center", 
                x=0.5,
                y=0.95
            ),
            xaxis1=attr(
                mirror="allticks",
                tickfont_size=30,
                tickvals=noiseRatios,
                title=attr(
                    font_size=36, 
                    text="Noise Ratio<br>Number of points $(Int(1024/timeSubsampleRates[3]))", 
                ),
                domain=[0.05,0.3]
            ),
            xaxis2=attr(
                mirror="allticks",
                tickfont_size=30,
                tickvals=noiseRatios,
                title=attr(
                    font_size=36, 
                    text="Noise Ratio<br>Number of points $(Int(1024/timeSubsampleRates[2]))", 
                ),
                domain=[0.35,0.6]
            ),
            xaxis3=attr(
                mirror="allticks",
                tickfont_size=30,
                tickvals=noiseRatios,
                title=attr(
                    font_size=36, 
                    text="Noise Ratio<br>Number of points $(Int(1024/timeSubsampleRates[1]))"), 
                domain=[0.65,0.9]
            ),
            yaxis1=attr(
                mirror="allticks",
                tickfont_size=30,
                type=yaxis_type,
                tickvals=yaxis_tickvals,
                range=yaxis_range,
                domain=[0.05,0.3],
                title=attr(
                    font_size=36, 
                    text="μ=$(μs[1])"
                )
            ),
            yaxis2=attr(
                mirror="allticks",
                tickfont_size=30,
                type=yaxis_type,
                tickvals=yaxis_tickvals,
                range=yaxis_range,
                domain=[0.35,0.6],
                title=attr(
                    font_size=36, 
                    text="μ=$(μs[2])"
                )
            ),
            yaxis3=attr(
                mirror="allticks",
                tickfont_size=30,
                type=yaxis_type,
                tickvals=yaxis_tickvals,
                range=yaxis_range,
                domain=[0.65,0.9],
                title=attr(
                    font_size=36, 
                    text="μ=$(μs[3])"
                )
            ),
            legend=attr(
                x=.925,
                y=0.5,
                yanchor="center",
                font=(
                    family="sans-serif",
                    size=48,
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
        width=3000,
        height=1200,
    )
    p
end 
function plots_07_17_2024()
    for (file, odeName, displayName) in zip(
        (
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/15.07.2024_final/hindmarshRose_15.07.2024.jld2",
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/15.07.2024_final/logisticGrowth_15.07.2024.jld2", 
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/15.07.2024_final/goodwin_15.07.2024.jld2",
            "/Users/user/Documents/School/WSINDy/WENDy.jl/results/15.07.2024_final/robertson_18.07.2024.jld2"
        ),
        ("hindmarshRose", "logisticGrowth","goodwin", "robertson"),
        ("Hindmarsh-Rose","LogisticGrowth","Goodwin", "Robertson")
    )
        data = load(file)
        for (metric,yaxis_tickvals,yaxis_range,yaxis_type) in zip(
            (:cl2,:fsl2,:mDist,:dt,:iters),
            (
                [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3], 
                [1e-2,1e-1,1e0,1e1], 
                [1,1e1,1e2,1e3,1e4,1e5,1e6], 
                [1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3], 
                [1,1e1,1e2,1e3]
            ),
            (
                [-3,3],
                [-2,1],
                [1,6], 
                [-4,3],
                [0,3]
            ),
            ("log","log","log","log","log")
        )
            plotSweep(data, metric, odeName,displayName,yaxis_tickvals,yaxis_range,yaxis_type)
        end 
    end
end
## plot the solution to ode using PlotlyJS because it is sexier
import PlotlyJS: plot as plotjs 
function plotjs(data::SimulatedWENDyData; 
    title::String="", file::Union{Nothing, String}=nothing, yaxis_type::String="linear"
)
    tt = data.tt_full
    U = data.U_exact
    D,M = size(U)
    ix = yaxis_type == "log" ? findall([all(U[:,m] .> 0) for m in 1:M]) : 1:M
    trs = AbstractTrace[]
    for d in 1:D
        push!(
            trs,
            scatter(
                x=tt[ix], 
                y=U[d,ix],
                name="u[$d]"
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
function plotSols(examples=[GOODWIN, HINDMARSH_ROSE, LOGISTIC_GROWTH, ROBERTSON])
    for data in examples
        file = joinpath("/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig", "$(data.name)_sol.png")
        title = "$(data.name) Solution"
        yaxis_type = data.name == "robertson" ? "log" : "linear"
        plotjs(data;title=title, file=file, yaxis_type=yaxis_type)
    end
end