# WENDy.jl

*Weak form Estimation of Nonlinear Dynamics in Julia.*

|**Documentation** |
|:-----------------:|
| [![][docs-stable-img]][docs-dev-url] [![][docs-dev-img]][docs-dev-url] | 


## Installation
The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add https://github.com/nrummel/WENDy.jl
```

## Current features
- Estimation of parameters for ordinary differential equations
- Supports: 
    - Additive Gaussian Noise and Multiplicative LogNormal Noise.
    - Ordinary differential equations that are nonlinear in parameters
    - Ordinary differential equations that are inhomogeneous in time
    - Box constraints for parameter spaces
- Provides acceleration for problems that are linear in parameters 
- Directly calls robust optimization algorithms that are well suited to an non-convex problems.
- Creates efficient Julia functions for the likelihood function and its derivatives with minimal inputs from the end user.

## Publication
The work comes from the the [Stephen Becker's](https://amath.colorado.edu/faculty/becker) and the [MathBio Group's](https://www.colorado.edu/project/mathbio/) at University of Colorado Boulder. For further reading find our paper at [arxiv link](https://arxiv.org/).


[contrib-url]: https://documenter.juliadocs.org/dev/contributing/
[discourse-tag-url]: https://discourse.julialang.org/tags/documenter
[gitter-url]: https://gitter.im/juliadocs/users

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://nrummel.github.io/WENDy.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://nrummel.github.io/WENDy.jl/stable

[GHA-img]: https://github.com/JuliaDocs/Documenter.jl/workflows/CI/badge.svg
[GHA-url]: https://github.com/JuliaDocs/Documenter.jl/actions?query=workflows/CI

[codecov-img]: https://codecov.io/gh/JuliaDocs/Documenter.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaDocs/Documenter.jl

[issues-url]: https://github.com/JuliaDocs/Documenter.jl/issues

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/D/Documenter.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/D/Documenter.html