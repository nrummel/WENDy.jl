include(joinpath(@__DIR__, "../src/WENDy.jl"))
includet(joinpath(@__DIR__, "../src/wendyMetrics.jl"))
includet(joinpath(@__DIR__, "../examples/hindmarshRose.jl"))
includet(joinpath(@__DIR__, "../examples/logisticGrowth.jl"))
includet(joinpath(@__DIR__, "../examples/goodwin.jl"))
includet(joinpath(@__DIR__, "../examples/robertson.jl"))
includet(joinpath(@__DIR__, "../examples/sir.jl"))
using FiniteDiff, StaticArrays, Printf
##
# ex = HINDMARSH_ROSE;
ex = SIR;

params = WENDyParameters(;
    noiseRatio=0.05, 
    seed=1, 
    timeSubsampleRate=1,
    optimMaxiters=200, 
    optimTimelimit=200.0
)
μ = .1
wendyProb = WENDyProblem(ex, params; ll=Warn);
wTrue = wendyProb.wTrue
J = length(wTrue)
w0 = wTrue + μ * abs.(wTrue) .* randn(J);
## solve with Maximum Likelihood Estimate
@time "create m funs" begin 
    m = MahalanobisDistance(wendyProb, params);
    ∇m! = GradientMahalanobisDistance(wendyProb, params);
    Hm! = HesianMahalanobisDistance(wendyProb, params);
end
##
g = similar(w0)
H = zeros(J,J)
@time "mDist" m(w0)
@time "gradient" ∇m!(g, w0)
@time "hessian" Hm!(H, w0)
##
@info "Building Forward Solve L2 loss so that compile time does not affect results"
l2(w::AbstractVector{<:Real}) = _l2(w,wendyProb.U,ex)
∇l2!(g::AbstractVector{<:Real},w::AbstractVector{<:Real}) = ForwardDiff.gradient!(g, l2, w) 
Hl2!(H::AbstractMatrix{<:Real},w::AbstractVector{<:Real}) = ForwardDiff.hessian!(H, l2, w) 
##
@info "Run once so that compilation time is isolated here"
@time "l2 loss" l2(w0)
g_fs = similar(w0)
H_fs = zeros(J,J)
@time "gradient of l2" ∇l2!(g_fs,w0)
@time "hessian of l2" Hl2!(H_fs,w0);
@assert !all(g_fs .== 0) "Auto diff failed on fs"
@assert !all(H_fs .== 0) "Auto diff failed on fs"
##
results = NamedTuple(
    name=>NamedTuple([:what=>Ref{Any}(Vector(undef,J)), :wits=>Ref{Any}(Matrix(undef,0,0))])
for name in keys(algos))

for (name, algo) in zip(keys(algos),algos)
    @info "Running $name"
    # if name != :fsnls_tr
    #     continue 
    # end 
    alg_dt = @elapsed begin 
        (what, iters, wits) = begin 
            algo(
                wendyProb, params, w0, 
                m, ∇m!,Hm!, 
                l2, ∇l2!, Hl2!; 
                return_wits=true
            ) 
        catch
            (NaN * ones(J), NaN, nothing) 
        end 
    end
    cl2 = norm(what - wTrue) / norm(wTrue)
    fsl2 = try
        forwardSolveRelErr(wendyProb, what)
    catch 
        NaN 
    end
    mDist = try
        m(what)
    catch 
        NaN 
    end
    @info """
    Results:
        dt = $alg_dt
        cl2 = $(@sprintf "%.4g" cl2*100)%
        fsl2 = $(@sprintf "%.4g" fsl2*100)%
        mDist = $(@sprintf "%.4g" mDist)
        iters = $iters
    """
    results[name].what[] = what 
    results[name].wits[] = wits
end

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
D = wendyProb.D
M = wendyProb.M
F = zeros(D,M)
myLF = zeros(D,M)
LF = zeros(D,M)
for m in 1:M 
   @views ROBERTSON_f!(F[:,m], wendyProb.U_exact[:,m],  wTrue, nothing)
   @views ROBERTSON_logf!(myLF[:,m], wendyProb.U_exact[:,m],  wTrue, nothing)
   @views wendyProb.f!(LF[:,m], wTrue, wendyProb.U_exact[:,m])
end
##
trs = AbstractTrace[]
for d in 1:D 
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
    push!(
        trs,
        scatter(
            x = wendyProb.tt,
            y = LF[d,:],
            name="(f(u)/u)[$d]"
        )
    )
    push!(
        trs,
        scatter(
            x = wendyProb.tt,
            y = myLF[d,:],
            name="my(f(u)/u)[$d]"
        )
    )
end 
plotjs(
    trs,
    Layout(
        title="U_exact fo $(ex.name)"
    )
)
##
algo_name = :tr
what = results[algo_name].what[]
what[end] = round(what[end])
Uhat = forwardSolve(wendyProb, what)
trs = AbstractTrace[]
using Plots.Colors
colors = [
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
for d in 1:D 
    c = protanopic(colors[d],.25)
    push!(
        trs,
        scatter(
            x = wendyProb.tt,
            y = wendyProb.U[d,:],
            mode="markers",
            maker_opacity=0.7,
            maker_size=.5,
            marker_color="#$(hex(c))",
            name="U[$d]",
            legendgroup=d
        )
    )
    
    push!(
        trs,
        scatter(
            x = wendyProb.tt,
            y = Uhat[d,:],
            line_color="#$(hex(colors[d]))",
            line_width=5,
            name="Ũ[$d]",
            legendgroup=d
        )
    )
    push!(
        trs,
        scatter(
            x = wendyProb.tt,
            y = wendyProb.U_exact[d,:],
            line_color="#$(hex(colors[d+D]))",
            line_width=5,
            line_dash="dash",
            name="U^*[$d]",
            legendgroup=d
        )
    )
end 
relErr_data = norm(Uhat - wendyProb.U) / norm(wendyProb.U)
relErr_exact = norm(Uhat - wendyProb.U_exact) / norm(wendyProb.U_exact)
plotjs(
    trs,
    Layout(
        yaxis_type="log",
        title="Comparing Forward sim (With rounding) to data<br>$(ex.name) with Algorithm $algo_name<br>||Û-U||₂ / ||U||₂=$(@sprintf "%.4g" relErr_data)<br>||Û-U^*||₂ / ||U^*||₂=$(@sprintf "%.4g" relErr_exact)",
        yaxis_domain=[0,.8]
    )
)

##
