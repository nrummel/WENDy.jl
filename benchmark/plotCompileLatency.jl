## Plots and a table from benchmark/results/<sub>/timings.tsv and trace_<target>_<config>.txt there
##   julia --project=../../wendyScripts/analysis benchmark/plotCompileLatency.jl [sub]   (any env with Plots)
## Writes benchmark/figs/<sub>/{latency,compileShare,steady,packages,accuracy}.png. Only the steps that were run are plotted.
using Plots, Statistics, Printf, DelimitedFiles
using LinearAlgebra: norm
const SUB = isempty(ARGS) ? "" : ARGS[1]
const RES = joinpath(@__DIR__, "results", SUB)
const FIGS = joinpath(@__DIR__, "figs", SUB)
mkpath(FIGS)
raw = readdlm(joinpath(RES, "timings.tsv"), '\t', Any; quotes=false)
rows = [(; target=string(r[1]), config=string(r[2]), rep=Int(r[3]), step=string(r[4]), time=Float64(r[5]),
    compile=Float64(r[6]), retcode=string(r[8]), p̂=r[9] isa AbstractString && !isempty(r[9]) ? parse.(Float64, split(r[9], ',')) :
    r[9] isa Real ? [Float64(r[9])] : Float64[]) for r in eachrow(raw)]
targets = unique(r.target for r in rows)
configs = unique(r.config for r in rows)
sort!(configs, by=c -> c == "base" ? 0 : count(==('+'), c) + 1)
const STEPS = filter(s -> any(r -> r.step in (s, "$(s)_first"), rows), ["build_wendy", "build_oe", "wls", "irls", "mle", "oels", "hybrid"])
const METHOD_STEPS = filter(!startswith("build"), STEPS)
first_step(s) = s in ("build_wendy", "build_oe") ? s : "$(s)_first"
med(target, config, step, field) = (v = [getfield(r, field) for r in rows if r.target == target && r.config == config &&
    r.step == step]; isempty(v) ? NaN : median(v))

## 1. time to first solution per step, split into compilation and the rest (median over reps)
plts = map(targets) do target
    p = plot(; title=target, ylabel="seconds (first call)", legend=:topright, xrotation=30, size=(900, 450))
    n, w = length(configs), 0.8 / length(configs)
    for (i, c) in enumerate(configs)
        x = (1:length(STEPS)) .+ (i - (n + 1)/2) * w
        tot = [med(target, c, first_step(s), :time) for s in STEPS]
        comp = [med(target, c, first_step(s), :compile) for s in STEPS]
        bar!(p, x, tot; bar_width=w, label=c, color=i, alpha=0.35, linecolor=i)
        bar!(p, x, comp; bar_width=w, label="", color=i, linecolor=i)
    end
    xticks!(p, 1:length(STEPS), STEPS)
    p
end
savefig(plot(plts...; layout=(length(targets), 1), size=(950, 450length(targets)),
    plot_title="first call: solid = compilation, light = total"), joinpath(FIGS, "latency.png"))

## 2. total latency for a new f! (build + first call of every method), and the part that is compilation
plts = map(targets) do target
    tot = [sum(med(target, c, first_step(s), :time) for s in STEPS) for c in configs]
    comp = [sum(med(target, c, first_step(s), :compile) for s in STEPS) for c in configs]
    p = bar(configs, tot; label="total", alpha=0.35, color=1, title=target, ylabel="seconds", xrotation=20)
    bar!(p, configs, comp; label="compilation", color=1)
    p
end
savefig(plot(plts...; layout=(1, length(targets)), size=(450length(targets), 420)), joinpath(FIGS, "compileShare.png"))

## 3. steady state: second call relative to base (no run time regression)
plts = map(targets) do target
    p = hline([1.0]; color=:black, ls=:dash, label="", title=target, ylabel="second call / base", xrotation=30)
    for (i, c) in enumerate(filter(!=("base"), configs))
        r = [med(target, c, "$(s)_second", :time) / med(target, "base", "$(s)_second", :time) for s in METHOD_STEPS]
        scatter!(p, (1:length(r)) .+ 0.08(i - 2), r; label=c, color=i + 1, ms=6)
    end
    xticks!(p, 1:length(METHOD_STEPS), METHOD_STEPS)
    p
end
savefig(plot(plts...; layout=(1, length(targets)), size=(450length(targets), 400)), joinpath(FIGS, "steady.png"))

## 4. compile time for the target (after the warm-up), by the package that owns the compiled method, from the trace
## runs (Base.@trace_compile prints each compiled method with its time). Core.kwcall and Type{...} wrappers are skipped.
function pkgof(line)
    i = findfirst("Tuple{", line)
    isnothing(i) && return "other"
    for m in eachmatch(r"([A-Z][A-Za-z0-9_]*)\.", line[last(i):end])
        m.captures[1] in ("Core", "Type") || return m.captures[1]
    end
    "other"
end
msof(line) = (m = match(r"#=\s*([0-9.]+) ms =#", line); isnothing(m) ? 0.0 : parse(Float64, m.captures[1]))
counts = Dict{Tuple{String,String},Dict{String,Float64}}()
for target in targets, c in configs
    f = joinpath(RES, "trace_$(target)_$(c).txt")
    isfile(f) || continue
    d = Dict{String,Float64}()
    for l in eachline(f)
        occursin("precompile(", l) || continue
        k = pkgof(l)
        d[k] = get(d, k, 0.0) + msof(l) / 1000
    end
    counts[(target, c)] = d
end
if !isempty(counts)
    allpk = Dict{String,Float64}()
    for d in values(counts), (k, v) in d
        allpk[k] = get(allpk, k, 0.0) + v
    end
    top = first.(sort(collect(allpk), by=last, rev=true))[1:min(7, end)]
    grp(k) = k in top ? k : "other"
    pkgs = [filter(!=("other"), top); "other"]
    plts = map(targets) do target
        M = [sum((v for (k, v) in get(counts, (target, c), Dict{String,Float64}()) if grp(k) == pk); init=0.0) for c in configs, pk in pkgs]
        p = bar(; title=target, ylabel="compile seconds (traced run)", xrotation=20, legend=:outerright)
        bottom = zeros(length(configs))
        for (j, pk) in enumerate(pkgs)
            bar!(p, configs, bottom .+ M[:, j]; fillrange=bottom, label=pk, color=j)
            bottom .+= M[:, j]
        end
        p
    end
    savefig(plot(plts...; layout=(1, length(targets)), size=(650length(targets), 450)), joinpath(FIGS, "packages.png"))
    for target in targets, c in configs
        d = get(counts, (target, c), nothing)
        isnothing(d) && continue
        println(rpad("$target $c", 32), join(("$k $(round(v, digits=2))" for (k, v) in sort(collect(d), by=last, rev=true)[1:min(5, end)]), ", "))
    end
end

## 5. accuracy: relative difference of p̂ from base (first call), per method
plts = map(targets) do target
    p = plot(; title=target, ylabel="‖p̂ - p̂_base‖ / ‖p̂_base‖", yscale=:log10, xrotation=30, legend=:bottomright)
    for (i, c) in enumerate(filter(!=("base"), configs))
        d = map(METHOD_STEPS) do s
            a = [r.p̂ for r in rows if r.target == target && r.config == "base" && r.step == "$(s)_first"]
            b = [r.p̂ for r in rows if r.target == target && r.config == c && r.step == "$(s)_first"]
            isempty(a) || isempty(b) ? NaN : max(norm(first(b) - first(a)) / norm(first(a)), 1e-17)
        end
        scatter!(p, (1:length(d)) .+ 0.08(i - 2), d; label=c, color=i + 1, ms=6)
    end
    hline!(p, [1e-12]; color=:gray, ls=:dot, label="round-off")
    xticks!(p, 1:length(METHOD_STEPS), METHOD_STEPS)
    p
end
savefig(plot(plts...; layout=(1, length(targets)), size=(450length(targets), 400)), joinpath(FIGS, "accuracy.png"))

## table: median first call (compile) per step
for target in targets
    println("\n", target, ": median seconds, first call (compilation)")
    @printf("%-14s", "config")
    foreach(s -> @printf("%16s", s), [STEPS; "total"])
    println()
    for c in configs
        @printf("%-14s", c)
        T = C = 0.0
        for s in STEPS
            t, k = med(target, c, first_step(s), :time), med(target, c, first_step(s), :compile)
            T += t; C += k
            @printf("%16s", @sprintf("%.2f (%.2f)", t, k))
        end
        @printf("%16s\n", @sprintf("%.2f (%.2f)", T, C))
    end
    println("load ", round(med(target, "base", "load", :time), digits=1), " s, warm-up ",
        round(med(target, "base", "warmup", :time), digits=1), " s (base)")
end
