push!(LOAD_PATH, joinpath(@__DIR__, "../src/"))
using Documenter, WENDy
# CI_FLG = get(ENV, "CI", nothing) == "true"

makedocs(
    modules = [WENDy],
    format = Documenter.HTML(
        # prettyurls = CI_FLG,
        canonical = "http://weavejl.mpastell.com/stable/",
    ),
    sitename = "WENDy.jl",
    pages = [
        "index.md",
        "getting_started.md",
        "usage.md",
    ],
)

deploydocs(
    repo = "github.com/JunoLab/Weave.jl.git",
    push_preview = true,
)