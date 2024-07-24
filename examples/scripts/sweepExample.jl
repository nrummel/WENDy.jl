@info "Loading WENDy"
includet(joinpath(@__DIR__, "../src/WENDy.jl"))
includet(joinpath(@__DIR__, "../examples/hindmarshRose.jl"))
includet(joinpath(@__DIR__, "../examples/logisticGrowth.jl"))
includet(joinpath(@__DIR__, "../examples/goodwin.jl"))
includet(joinpath(@__DIR__, "../examples/robertson.jl"))
includet(joinpath(@__DIR__, "../examples/sir.jl"))
using ProgressMeter,Printf,Dates,LoggingExtras,FileIO,JLD2
##
@info "Setting sweep criterion"
Random.seed!(1)
noiseRatios=[0.01,0.05,0.1,0.2]
timeSubsampleRates=[8,4,2]
μs =[0.1, 0.5, 1.]
numSamples = 11
##
_I,II,III,IV = length(noiseRatios),length(timeSubsampleRates),length(μs), numSamples
V = length(algos)
##
metrics = [:what,:cl2,:fsl2,:mDist,:dt,:iters]
optimparams = (
    diagReg=1e-10,
    nlsAbstol=1e-8,
    nlsReltol=1e-8,
    nlsMaxiters=1000,
    optimAbstol=1e-8,
    optimReltol=1e-8,
    optimMaxiters=200,
    optimTimelimit=60.0
)
##
function do_it(ex::WENDyData)
    with_logger(TeeLogger(
        # Current global logger (stderr)
        global_logger(),
        # Accept any messages with level >= Info
        MinLevelLogger(
            FileLogger(joinpath(@__DIR__,"../results/logfile.$(ex.name)_$(Dates.format(now(),"d.mm.yyyy")).log")),
            Logging.Info
        ),
        # Accept any messages with level >= Debug
        MinLevelLogger(
            FileLogger(joinpath(@__DIR__,"../results/debug.$(ex.name)_$(Dates.format(now(),"d.mm.yyyy")).log")),
            Logging.Debug,
        ),
    )) do
        @info "Running sweep on $(ex.name)"
        # global results, wendyProb, params, m, ∇m!, Hm!, l2, ∇l2!, Hl2!,w0,ix,wTrue,what
        ## preallocate space for results to be stored
        J = length(ex.wTrue)
        results = NamedTuple(
            name=>NamedTuple(
                metric=> metric == :what ? zeros(_I,II,III,IV,J) : zeros(_I,II,III,IV)  
                for metric in metrics
            )
            for name in keys(algos)
        )
        total_its = _I*II*III*IV*V
        ##
        pbar = Progress(total_its; 
            dt=0.5,
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=10,
            enabled=false
            )
        ##
        ix = 1
        avg_it_time = 0
        for (i,nr) in enumerate(noiseRatios), (ii,sub) in enumerate(timeSubsampleRates), iv in 1:numSamples
            @info  "  nr=$nr, sub=$sub, mc=$iv/$numSamples"
            ProgressMeter.update!(pbar,ix;
                showvalues=[
                (:ex,"$(ex.name), nr=$nr, sub=$sub, mc=$iv/$numSamples"),
                (:step,"Build WENDy Problem")
                ]
            )
            dt = @elapsed a = @allocations begin 
                params = WENDyParameters(;
                    seed=iv, # TODO: Discuss if this is ok
                    noiseRatio=nr, 
                    timeSubsampleRate=sub,
                    diagReg=optimparams.diagReg,
                    nlsAbstol=optimparams.nlsAbstol,
                    nlsReltol=optimparams.nlsReltol,
                    nlsMaxiters=optimparams.nlsMaxiters,
                    optimAbstol=optimparams.optimAbstol,
                    optimReltol=optimparams.optimReltol,
                    optimMaxiters=optimparams.optimMaxiters,
                    optimTimelimit=optimparams.optimTimelimit
                )
                wendyProb = WENDyProblem(ex, params; ll=Warn);
            end
            ## solve with Maximum Likelihood Estimate
            ProgressMeter.update!(pbar,ix;
                showvalues=[
                (:ex,"$(ex.name), nr=$nr, sub=$sub, mc=$iv/$numSamples"),
                (:prevstep,"Build WENDy Problem, $dt s, $a allocations"),
                (:step,"Build objective functions")
                ]
            )
            dt = @elapsed a = @allocations begin 
                m = MahalanobisDistance(wendyProb, params);
                ∇m! = GradientMahalanobisDistance(wendyProb, params);
                Hm! = HesianMahalanobisDistance(wendyProb, params);
                l2(w::AbstractVector{<:Real}) = _l2(w,wendyProb.U,ex)
                ∇l2!(g::AbstractVector{<:Real},w::AbstractVector{<:Real}) = ForwardDiff.gradient!(g, l2, w) 
                Hl2!(H::AbstractMatrix{<:Real},w::AbstractVector{<:Real}) = ForwardDiff.hessian!(H, l2, w) 
            end
            @info "Creation of objective took $(@sprintf "%.4g" dt)"
            ProgressMeter.update!(pbar,ix;
                showvalues=[
                (:ex,"$(ex.name), nr=$nr, sub=$sub, mc=$iv/$numSamples"),
                (:prevstep,"Build objective functions, $dt s, $a allocations"),
                (:step,"Running objective functions (compilation)")
                ]
            )
            dt = @elapsed a = @allocations begin 
                J = wendyProb.J
                g_m = zeros(J)
                H_m = zeros(J,J) 
                g_fs = zeros(J)
                H_fs = zeros(J,J) 
                wTrue = wendyProb.wTrue
                m(wTrue)
                ∇m!(g_m,wTrue)
                Hm!(H_m,wTrue)
                l2(wTrue)
                ∇l2!(g_fs,wTrue)
                Hl2!(H_fs,wTrue);
                # @assert !all(g_fs .== 0) "Auto diff failed on fs"
                # @assert !all(H_fs .== 0) "Auto diff failed on fs"
            end
            @info "Compilaiton of objective took $(@sprintf "%.4g" dt)"
            ## 
            for (iii, μ) in enumerate(μs)
                @info  "  μ=$μ"
                w0 = wTrue + μ * abs.(wTrue) .* randn(J);
                for (v, (name, algo)) in enumerate(zip(keys(algos),algos)) 
                    if v==1 
                        ProgressMeter.update!(pbar,ix;
                            showvalues=[
                            (:ex,"$(ex.name), nr=$nr, sub=$sub, μ=$μ, mc=$iv/$numSamples"),
                            (:prevstep,"Running objective functions (compilation), $dt s, $a allocations"),
                            (:step,"Running algorithm $name")
                            ]
                        )
                    else
                        ProgressMeter.update!(pbar,ix;
                        showvalues=[
                        (:ex,"$(ex.name), nr=$nr, sub=$sub, μ=$μ, mc=$iv/$numSamples"),
                        (:prevstep,"Ran $(keys(algos)[v-1]), $(@sprintf "%.2g s" dt), $a allocations"),
                        (:___cl2,"$(@sprintf "%.2g%%" results[v-1].cl2[i,ii,iii,iv])"),
                        (:__fsl2,"$(@sprintf "%.2g%%" results[v-1].fsl2[i,ii,iii,iv])"),
                        (:_mDist,"$(@sprintf "%.2g " results[v-1].mDist[i,ii,iii,iv])"),
                        (:____dt,"$(@sprintf "%.2g s" results[v-1].dt[i,ii,iii,iv])"),
                        (:step,"Running algorithm $name")
                        ]
                    )
                    end
                    dt = @elapsed a = @allocations begin 
                        @info "Running $name"
                        alg_dt = @elapsed (what, iters) = try 
                            algo(wendyProb, params, w0, m, ∇m!, Hm!,l2,∇l2!,Hl2!) 
                        catch
                            (NaN*ones(J), NaN) 
                        end
                        cl2 = norm(what - wTrue) / norm(wTrue)
                        fsl2 = try
                            fsl2 = forwardSolveRelErr(wendyProb, what)
                        catch 
                            NaN 
                        end
                        mDist = try
                            m(what)
                        catch 
                            NaN 
                        end
                        @info """
                          $name Results: $(ex.name), nr=$nr, sub=$sub, μ=$μ, mc=$iv/$numSamples
                            dt    = $(@sprintf "%.4g" alg_dt)
                            cl2   = $(@sprintf "%.4g" cl2*100)%
                            fsl2  = $(@sprintf "%.4g" fsl2*100)%
                            mDist = $(@sprintf "%.4g" mDist)
                            iters = $iters
                        """
                        results[name].what[i,ii,iii,iv,:] = what
                        results[name].cl2[i,ii,iii,iv]    = cl2
                        results[name].fsl2[i,ii,iii,iv]   = fsl2
                        results[name].mDist[i,ii,iii,iv]  = mDist
                        results[name].dt[i,ii,iii,iv]     = alg_dt
                        results[name].iters[i,ii,iii,iv]  = iters
                    end
                    avg_it_time = ((ix-1) * avg_it_time + dt ) / ix
                    @info "This Iteration time $dt"
                    @info "Average Iteration time $avg_it_time"
                    tremain = avg_it_time * (total_its - ix)
                    days = floor(tremain/86400)
                    tremain -= days * 86400
                    hr = floor(tremain/3600)
                    tremain -= hr*3600
                    min = floor(tremain/60)
                    tremain -= min*60
                    @info "Iter = $(ix) / $total_its"
                    @info "Estimated time remaining $days days, $hr hr, $min min, $(@sprintf "%.4g" tremain) s"
                    ix +=1
                end
            end
        end
        ProgressMeter.finish!(pbar)
        file = joinpath(@__DIR__, "../results/$(ex.name)_$(Dates.format(now(),"d.mm.yyyy")).jld2")
        try
            @info "Saving Results to $file"
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
        catch 
            @info "Save failed... "
        end
        return results
    end
end

# results_hindmarsh = do_it(HINDMARSH_ROSE)
# results_logistic = do_it(LOGISTIC_GROWTH)
results_sir = do_it(SIR)
results_robertson = do_it(ROBERTSON)
