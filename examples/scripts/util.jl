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
    nll::CostFunction, l2Loss::CostFunction; 
    return_wits::Bool=false, kwargs...
    )
    J = wendyProb.J
    try 
        (what, iters, wits) = nothing, nothing, nothing
        if return_wits
            (what, iters, wits) = algo(
                wendyProb, w0, params, nll, l2Loss;
                return_wits=return_wits, kwargs...
            ) 
        else 
            (what, iters) = algo(
                wendyProb, w0, params, nll, l2Loss;
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
    wendyProb::WENDyProblem, params::WENDyParameters, nll::CostFunction, l2Loss::CostFunction, algoNames::AbstractVector{<:Symbol}; 
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
                    params, nll, l2Loss;
                    return_wits=return_wits, kwargs...
                )
            end
            cl2 = norm(what - wTrue) / norm(wTrue)
            mean_fsl2, final_fsl2 = try
                forwardSolveRelErr(wendyProb, what)
            catch 
                NaN, NaN 
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
                    nll        = $(@sprintf "%.4g" _nll)
                    iters      = $iters
            """
            results[algoName].iters[]      = iters
            results[algoName].dt[]         = alg_dt
            results[algoName].cl2[]        = cl2
            results[algoName].mean_fsl2[]  = mean_fsl2
            results[algoName].final_fsl2[] = final_fsl2
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
    irwls="Iterative Reweighted Least Squares"
)
ex2Disp = Dict(
    "sir"=>"SIR",
    "goodwin"=>"Goodwin",
    "robertson"=>"Robertson",
    "hindmarshRose"=>"Hindmarsh Rose", 
    "logisticGrowth"=>"Logistic Growth", 
);
## Wrap solvers so they can all be called with the same inputs
algos = (
    fsnls_bfgs_optim = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> bfgs_Optim(l2Loss, w0, params; kwargs...),
    fsnls_tr_optim   = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> tr_Optim(l2Loss, w0, params; kwargs...),
    fsnls_arc_sfn    = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> arc_SFN(l2Loss, w0, params; kwargs...),
    fsnls_tr_jso     = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> tr_JSO(l2Loss, w0, params; kwargs...),
    fsnls_arc_jso    = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> arc_JSO(l2Loss, w0, params; kwargs...),
    irwls            = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> IRWLS(wendyProb, w0, params; kwargs...),
    bfgs_optim       = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> bfgs_Optim(nll, w0, params; kwargs...),
    tr_optim         = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> tr_Optim(nll, w0, params; kwargs...),
    arc_sfn          = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> arc_SFN(nll, w0, params; kwargs...),
    tr_jso           = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> tr_JSO(nll, w0, params; kwargs...),
    arc_jso          = (wendyProb, w0, params, nll, l2Loss; kwargs...) -> arc_JSO(nll, w0, params; kwargs...),
);