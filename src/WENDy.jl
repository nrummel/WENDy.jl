module WENDy
    ## external dependencies
    using Reexport
    # optimization algorithms
    using NonlinearSolve # nonlinear least squares
    import NonlinearSolve.solve
    using Optim # trust region
    using ManualNLPModels: NLPModel # necessary for ARC_qK
    using AdaptiveRegularization: ARCqKOp # ARC_qK
    # For solving and symbolicly representing diff eq 
    @reexport using OrdinaryDiffEq
    using Symbolics, ForwardDiff
    using OrdinaryDiffEq: ODESolution, OrdinaryDiffEqAlgorithm, solve, Rosenbrock23
    using Symbolics: jacobian, @variables
    # Other Necessities
    @reexport using Distributions: Normal, LogNormal, Distribution
    using ImageFiltering: imfilter, Inner # Convolution in Julia
    using FFTW: fft, ifft
    using Tullio: @tullio
    # stdlib
    using LinearAlgebra, Statistics, Random, Logging, Printf
    using Logging: Info, Warn, LogLevel
    ##
    include("wendyDataTypes.jl")
    include("wendyTestFunctions.jl")
    include("wendyNoise.jl")
    include("wendySymbolics.jl")
    include("wendyEquations.jl")
    include("wendyLinearEquations.jl")
    include("wendyNonlinearEquations.jl")
    include("wendyMethods.jl")
    include("wendyLinearMethods.jl")
    include("wendyNonlinearMethods.jl")
    include("wendyProblems.jl")
    include("wendyOptim.jl")

    export WENDyProblem, WENDyParameters, solve
end # module WENDy