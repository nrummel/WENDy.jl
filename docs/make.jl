using Documenter, WENDy

# DocMeta.setdocmeta(WENDy, :DocTestSetup, :(using WENDy); recursive=true)

makedocs(
    modules = [WENDy],
    checkdocs=:exports,
    # doctest=true, 
    # linktest=true, 
    # authors="Nicholas Rummel <nic.rummel@colorado.edu>",
    # repo="https://github.com/nrummel/WENDy.jl/blob/{commit}{path}#{line}",
    sitename="WENDy.jl",
    format = Documenter.HTML(
        prettyurls =  get(ENV, "CI", nothing) == "true",
        canonical = "https://nrummel.github.io/WENDy.jl/",
    ),
    pages = [
        "Home"=>"index.md",
        "Examples"=>"examples.md",
        "Reference"=>"reference.md",
    ],
)

deploydocs(
    repo = "github.com/nrummel/WENDy.jl",
    push_preview = true,
)