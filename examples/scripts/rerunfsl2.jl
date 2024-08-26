@info "Loading WENDy"
includet(joinpath(@__DIR__, "../src/WENDy.jl"))
includet(joinpath(@__DIR__, "../examples/hindmarshRose.jl"))
includet(joinpath(@__DIR__, "../examples/logisticGrowth.jl"))
includet(joinpath(@__DIR__, "../examples/goodwin.jl"))
includet(joinpath(@__DIR__, "../examples/robertson.jl"))
includet(joinpath(@__DIR__, "../examples/sir.jl"))
using Glob: glob 
using FileIO, JLD2, ProgressMeter,LoggingExtras,Dates
##
exName2Ex = Dict(
    "logisticGrowth"=>LOGISTIC_GROWTH, 
    "hindmarshRose"=>HINDMARSH_ROSE, 
    "robertson"=>ROBERTSON, 
    "sir"=>SIR, 
    "goodwin"=>GOODWIN, 
)

noiseRatios=[0.01,0.05,0.1,0.2]
timeSubsampleRates=[8,4,2]
μs =[0.1, 0.5, 1.]
numSamples = 11
##
_I,II,III,IV = length(noiseRatios),length(timeSubsampleRates),length(μs), numSamples
V = length(algos)
files = collect(glob("WSINDy/WENDy.jl/results/**/*.jld2"))
IX = length(files)*_I*II*III*IV*V
pBar = Progress(
    IX; 
    dt=0.5,
    barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
    barlen=20
)
function _dothis()
    ix = 0
    with_logger(
            MinLevelLogger(
                FileLogger(joinpath(@__DIR__,"../results/logfile.recomputeMetrics_$(Dates.format(now(),"d.mm.yyyy")).log")),
                Logging.Info
            )) do
            for file in files
            data = load(file)
            exName = split(splitpath(file)[end],"_")[1]
            ex = exName2Ex[exName]
            results = data["results"] # zeros(nr,sub,mu,mc,J)
            optimparams = data["optimparams"]
            noiseRatios = data["noiseRatios"]
            timeSubsampleRates = data["timeSubsampleRates"]
            μs = data["μs"]
            metrics = [:what,:cl2,:fsl2,:mDist,:dt,:iters]
            @showprogress desc="Recomputing for $exName" for algoName in keys(algos), (i,nr) in enumerate(noiseRatios), (ii,sub) in enumerate(timeSubsampleRates), (iii,μ) in enumerate(μs), iv in 1:size(results.tr.what,4) 
                next!(pBar)
                ix += 1
                wendyProb = WENDyProblem(
                    ex, 
                    WENDyParameters(
                        seed=iv,
                        noiseRatio=nr, 
                        timeSubsampleRate=sub,
                    ) 
                )
                what = copy(results[algoName].what[i,ii,ii,iv,:])
                if exName == "robertson"
                    what[end] = round(what[end])
                end
                fsl2 = try 
                    forwardSolveRelErr(wendyProb, what)
                catch 
                    NaN 
                end 
                cl2 = norm(what - wendyProb.wTrue) / norm(wendyProb.wTrue)
                @info """
                $exName:$algoName: nr=$nr, sub=$sub, μ=$μ, mc=$iv/11, ix=$ix/$IX
                    dt       = $(results[algoName].dt[i,ii,iii,iv])
                    iters    = $(results[algoName].iters[i,ii,iii,iv])
                    cl2 old  = $(@sprintf "%.4g" (results[algoName].cl2[i,ii,iii,iv])*100)%
                    cl2 new  = $(@sprintf "%.4g" (cl2)*100)%
                    mDist    = $(@sprintf "%.4g" (results[algoName].mDist[i,ii,iii,iv]))
                    fsl2 old = $(@sprintf "%.4g" results[algoName].fsl2[i,ii,iii,iv]*100)%
                    fsl2 new = $(@sprintf "%.4g" fsl2*100)%
                """
                # if isnan(fsl2) && !isnan(results[algoName].fsl2[i,ii,iii,iv]) || fsl2 > results[algoName].fsl2[i,ii,iii,iv]
                #     display(plotDeepDive(wendyProb, what))
                #     forwardSolveRelErr(wendyProb, what; verbose=true)
                #     @assert false
                # end
                results[algoName].fsl2[i,ii,iii,iv] = fsl2
            end
            mv(file, file*".bad_fsl2")
            @info "Saving to $file"
            save(
                file, 
                Dict(
                    "results"=>results, 
                    "optimparams"=>optimparams, 
                    "noiseRatios"=>noiseRatios, 
                    "timeSubsampleRates"=>timeSubsampleRates,
                    "μs"=>μs
                )
            )
        end 
    end 
end
_dothis()