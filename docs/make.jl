using Documenter, WENDy

makedocs(
    modules = [WENDy],
    checkdocs=:exports,
    authors="Nicholas Rummel <nic.rummel@colorado.edu>",
    repo="https://github.com/nrummel/WENDy.jl",
    sitename="WENDy.jl",
    format = Documenter.HTML(
        prettyurls = true,
        canonical = "https://nrummel.github.io/WENDy.jl/",
    ),
    pages = [
        "Home"=>"index.md",
        "Getting Started"=>"gettingStarted.md",
        "Examples"=>"examples.md",
        "Reference"=>"reference.md",
    ],
)

deploydocs(
    repo = "github.com/nrummel/WENDy.jl",
    push_preview = true,
)