## Load WENDy
using Revise
push!(
    LOAD_PATH, 
    joinpath(@__DIR__, "../src")
)
using WENDy
# includet(joinpath(@__DIR__, "../../src/WENDy.jl"))
## external deps 
using Base: with_logger
using Logging: LogLevel, Info, Warn, ConsoleLogger
using LinearAlgebra, Printf, Statistics
using Distributions: Normal, LogNormal, Distribution
using OrdinaryDiffEq: ODEProblem
using ModelingToolkit: modelingtoolkitize, @mtkbuild
using LaTeXTabulars
## load metrics and examples
includet(joinpath(@__DIR__, "../../metrics/wendyMetrics.jl"))
includet(joinpath(@__DIR__, "../../examples/odes/hindmarshRose.jl"))
includet(joinpath(@__DIR__, "../../examples/odes/logisticGrowth.jl"))
includet(joinpath(@__DIR__, "../../examples/odes/goodwin.jl"))
includet(joinpath(@__DIR__, "../../examples/odes/robertson.jl"))
includet(joinpath(@__DIR__, "../../examples/odes/sir.jl"))
includet(joinpath(@__DIR__, "../../examples/odes/multimodal.jl"))
##
function init(ex::WENDyData;
    noiseRatio::AbstractFloat=0.20, 
    seed::Int=1, 
    timeSubsampleRate::Int=4, 
    μ::Real=0.01,
    ll::LogLevel=Warn,
    kwargs...
)
    ## Simulate Noise and subsample
    simParams = SimulationParameters(;
        noiseRatio=noiseRatio, 
        seed=seed, 
        timeSubsampleRate=timeSubsampleRate,
        μ=μ
    )
    simulate!(ex, simParams;ll=ll)
    ## Build WENDy Problem
    params = WENDyParameters(;
        kwargs...
    )
    wendyProb = WENDyProblem(ex, params; ll=ll);
    wendyProb, params, simParams
end

function getW0(
    wendyProb::WENDyProblem, simParams::SimulationParameters;
    allowNegW0::Bool=false,
)
    J = wendyProb.J
    wTrue = wendyProb.wTrue
    w0 = if allowNegW0
        wTrue + simParams.μ * abs.(wTrue) .* randn(J);
    else
        w0 = -1 .* ones(J)
        while any(w0 .< 0)
            w0 = wTrue + simParams.μ * abs.(wTrue) .* randn(J);
        end
        w0
    end
    w0
end

function _call_algo(
    algo::Function, wendyProb::WENDyProblem, w0::AbstractVector{<:Real}, params::WENDyParameters, 
    nll::CostFunction, l2Loss::CostFunction, nll_fwdSolve::CostFunction; 
    return_wits::Bool=false, kwargs...
    )
    J = wendyProb.J
    try 
        (what, iters, wits) = nothing, nothing, nothing
        if return_wits
            (what, iters, wits) = algo(
                wendyProb, w0, params, nll, l2Loss, nll_fwdSolve;
                return_wits=return_wits, kwargs...
            ) 
        else 
            (what, iters) = algo(
                wendyProb, w0, params, nll, l2Loss, nll_fwdSolve;
                return_wits=return_wits, kwargs...
            ) 
            wits = zeros(J, 0)
        end 
        return (what, iters, wits)
    catch
        (NaN * ones(J), NaN, zeros(J,0)) 
    end
end

function runAlgos(
    wendyProb::WENDyProblem, params::WENDyParameters, nll::CostFunction, l2Loss::CostFunction, nll_fwdSolve::CostFunction, algoNames::AbstractVector{<:Symbol}; 
    return_wits::Bool=false, ll::LogLevel=Info, kwargs...
)
    with_logger(ConsoleLogger(stdout, ll)) do
        J = wendyProb.J
        wTrue = wendyProb.wTrue
        results = NamedTuple(
            algoName=>NamedTuple([
                :what=>Ref{Any}(NaN*ones(J)),
                :wits=>Ref{Any}(zeros(J,0)),
                :cl2=>Ref{Any}(NaN),
                :mean_fsl2=>Ref{Any}(NaN),
                :final_fsl2=>Ref{Any}(NaN),
                :fsnll=>Ref{Any}(NaN),
                :nll=>Ref{Any}(NaN),
                :dt=>Ref{Any}(NaN),
                :iters=>Ref{Any}(NaN)
            ])
        for algoName in algoNames);
        # Loop through algos and run this example
        @info "=================================="
        @info "Starting Optimization Routines"
        for algoName in algoNames
            algo = algos[algoName] 
            @info "  Running $algoName"
            alg_dt = @elapsed begin 
                (what, iters, wits) = _call_algo(
                    algo, wendyProb, w0, 
                    params, nll, l2Loss, nll_fwdSolve;
                    return_wits=return_wits, kwargs...
                )
            end
            cl2 = norm(what - wTrue) / norm(wTrue)
            mean_fsl2, final_fsl2 = try
                forwardSolveRelErr(wendyProb, what)
            catch 
                NaN, NaN 
            end
            fsnll = try
                nll_fwdSolve.f(what)
            catch 
                NaN 
            end
            _nll = try
                nll.f(what)
            catch 
                NaN 
            end
            @info """
              Results:
                    dt         = $alg_dt
                    cl2        = $(@sprintf "%.4g" cl2*100)%
                    mean_fsl2  = $(@sprintf "%.4g" mean_fsl2*100)%
                    final_fsl2 = $(@sprintf "%.4g" final_fsl2*100)%
                    fsnll      = $(@sprintf "%.4g" fsnll)
                    nll        = $(@sprintf "%.4g" _nll)
                    iters      = $iters
            """
            results[algoName].iters[]      = iters
            results[algoName].dt[]         = alg_dt
            results[algoName].cl2[]        = cl2
            results[algoName].mean_fsl2[]  = mean_fsl2
            results[algoName].final_fsl2[] = final_fsl2
            results[algoName].fsnll[]      = fsnll
            results[algoName].nll[]        = _nll
            results[algoName].what[]       = what
            results[algoName].wits[]       = wits
        end
        return results
    end
end
## 
algo2Disp = (
    init="Initial w₀ Solution",
    tr_optim="Trust Region (Optim)",
    arc_sfn="Adaptive Cubic Regularization (SFN)", 
    arc_jso="Adaptive Cubic Regularization (JSO)", 
    tr_jso="Trust Region (JSO)", 
    fsnls_tr_optim="Forward Solve Nonlinear Least Squares<br>(Trust Region Optim)",
    fsnll_tr_optim="Forward Solve Negative Log-Likelihood<br>(Trust Region Optim)",
    irwls="Iterative Reweighted Least Squares"
)
ex2Disp = Dict(
    "sir"=>"SIR",
    "goodwin"=>"Goodwin",
    "robertson"=>"Robertson",
    "hindmarshRose"=>"Hindmarsh Rose", 
    "logisticGrowth"=>"Logistic Growth", 
    "MULTIMODAL"=>"Multimodal Example", 
);
## Wrap solvers so they can all be called with the same inputs
algos = (
    fsnll_bfgs_optim = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> bfgs_Optim(nll_fwdSolve, w0, params; kwargs...),
    fsnll_tr_optim   = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> tr_Optim(nll_fwdSolve, w0, params; kwargs...),
    fsnls_bfgs_optim = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> bfgs_Optim(l2Loss, w0, params; kwargs...),
    fsnls_tr_optim   = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> tr_Optim(l2Loss, w0, params; kwargs...),
    fsnls_arc_sfn    = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> arc_SFN(l2Loss, w0, params; kwargs...),
    fsnls_tr_jso     = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> tr_JSO(l2Loss, w0, params; kwargs...),
    fsnls_arc_jso    = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> arc_JSO(l2Loss, w0, params; kwargs...),
    irwls            = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> IRWLS(wendyProb, w0, params; kwargs...),
    bfgs_optim       = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> bfgs_Optim(nll, w0, params; kwargs...),
    tr_optim         = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> tr_Optim(nll, w0, params; kwargs...),
    arc_sfn          = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> arc_SFN(nll, w0, params; kwargs...),
    tr_jso           = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> tr_JSO(nll, w0, params; kwargs...),
    arc_jso          = (wendyProb, w0, params, nll, l2Loss, nll_fwdSolve; kwargs...) -> arc_JSO(nll, w0, params; kwargs...),
);
##
function runExample(ex;
    algoNames = [:tr_optim,:arc_sfn,:tr_jso,:arc_jso,:irwls,:fsnls_tr_optim, :fsnll_tr_optim],
    noiseRatio = .20, 
    seed = 1, 
    timeSubsampleRate = 4, 
    μ = 0.50,
    optimMaxiters = 200, 
    optimTimelimit = 200.0,
    allowNegW0=true
)
    @info "==============================="
    @info "Running $(ex.name)"
    wendyProb, params, simParams = init(
        ex;
        noiseRatio=noiseRatio,
        seed=seed,
        timeSubsampleRate=timeSubsampleRate,
        μ=μ,
        optimMaxiters=optimMaxiters,
        optimTimelimit=optimTimelimit,
        ll = Warn
    );
    ## build costFunction
    nll, l2Loss, nll_fwdSolve = buildCostFunctions(wendyProb, params; ll=Info);
    ## Pick a initializaiotn point
    w0 = getW0(
        wendyProb, simParams; 
        allowNegW0=allowNegW0
    );
    ## run algorithms 
    J = wendyProb.J 
    wTrue = wendyProb.wTrue
    cl2_0 =  norm(w0 - wTrue) / norm(wTrue)
    mean_fsl2_0, final_fsl2_0 = try 
        forwardSolveRelErr(wendyProb, w0)
    catch 
        NaN, NaN
    end
    _nll_0 = nll.f(w0)
    _fsnll_0 = nll_fwdSolve.f(w0)
    mean_fsl2_star, final_fsl2_star = forwardSolveRelErr(wendyProb, wTrue)
    _fsnll_star = nll_fwdSolve.f(wTrue)
    _nll_star = nll.f(wTrue)
    @info """
    Initialization Info: 
        cl2        = $(@sprintf "%.4g" cl2_0*100)%
        mean_fsl2  = $(@sprintf "%.4g" mean_fsl2_0*100)%
        final_fsl2 = $(@sprintf "%.4g" final_fsl2_0*100)%
        fsnll      = $(@sprintf "%.4g" _fsnll_0)
        nll        = $(@sprintf "%.4g" _nll_0)
    Truth Info 
        mean_fsl2  = $(@sprintf "%.4g" mean_fsl2_star*100)%
        final_fsl2 = $(@sprintf "%.4g" final_fsl2_star*100)%
        fsnll      = $(@sprintf "%.4g" _fsnll_star)
        nll        = $(@sprintf "%.4g" _nll_star)
    """
    results = runAlgos(wendyProb, params, nll, l2Loss, nll_fwdSolve, algoNames; return_wits=true);
    return (
        wendyProb, params, simParams, 
        nll, l2Loss, nll_fwdSolve,
        w0, results
    )
end 

function makePlots(
    wendyProb, 
    nll, l2Loss, nll_fwdSolve,
    w0, results, algoNames
)
    # Residual parted out
    _algoName = :tr_optim
    _p_resPrts = plotResParts(
        results[_algoName].wits[], 
        wendyProb, 
        "Residual Broken Apart<br>$(ex2Disp[ex.name]) : $(algo2Disp[_algoName])"
    )
    display(_p_resPrts);
    ## Plot different cost functions over the iterates, in this case we have fsnll
    _p_wits_fsnll = plotWits(
        w -> try
            nll_fwdSolve.f(w)
        catch 
            NaN
        end,
        [results[algoName].wits[] for algoName in algoNames];
        names = [algo2Disp[algoName] for algoName in algoNames],
        title="Forward Solve Negative Log-Likelihood<br>$(ex2Disp[ex.name])",
        yaxis_type="log"
    );
    display(_p_wits_fsnll);
    ## our nll
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
    display(_p_sols[:irwls])
    display(_p_sols[:fsnll_tr_optim])
    display(_p_sols[:fsnls_tr_optim])
    ## Plot the cost surface
    p_l2 = plotCostSurface(
        wendyProb, l2Loss, (3,4);
        w1Rng=range(0, step=0.1, stop=5), 
        w2Rng=range(0, step=0.1, stop=5)
    );
    p_nll = plotCostSurface(
        wendyProb, nll, (3,4);
        w1Rng=range(0, step=0.1, stop=5), 
        w2Rng=range(0, step=0.1, stop=5)
    );

    (
        _p_resPrts, _p_wits_fsnll, _p_wits_nll, _p_wits_fsl2, _p_sols, p_l2, p_nll 
    )
end 

function savePlots(
    save_dir,
    _p_resPrts, _p_wits_nll, _p_wits_fsl2, _p_sols, p_l2, p_nll 
)
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
    relayout!(
        p_nll, 
        title_text="Negative Log-Likelihood",
        title_y=.9,
        title_font_size=36,
        title_yanchor="center",
        scene=attr(
            zaxis=attr(title="-ℒ"),
            xaxis=attr(title="w₃"),
            yaxis=attr(title="w₄"),
            # xaxis_title_font_size=20,
            camera_eye=attr(x=-1, y=2, z=.1),
        ),
        margin=attr(t=30, r=0, l=20, b=10),
    )
    relayout!(
        p_l2, 
        title_text="Forward Simulation L2 Loss",
        title_y=.9,
        title_font_size=36,
        title_yanchor="center",
        scene=attr(
            zaxis_title="||⋅||₂",
            xaxis_title="w₃",
            yaxis_title="w₄",
            xaxis_title_font_size=20,
            camera_eye=attr(x=-1, y=2, z=.1),
        ),
        margin=attr(t=30, r=0, l=20, b=10),
    )
    savefig(
        p_nll,
        joinpath(save_dir, "$(ex.name)_nllCostSpace.png"),
        width=800, height=700
    )
    savefig(
        p_l2,
        joinpath(save_dir, "$(ex.name)_l2CostSpace.png"),
        width=800, height=700
    )
end