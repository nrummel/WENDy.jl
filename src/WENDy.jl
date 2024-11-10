module WENDy
    ## external dependencies
    # optimization algorithms
    using NonlinearSolve # nonlinear least squares
    using Optim # trust region
    using ManualNLPModels: NLPModel # necessary for ARC_qK
    using AdaptiveRegularization: ARCqKOp # ARC_qK
    # For solving and symbolicly representing diff eq 
    using OrdinaryDiffEq, Symbolics
    using OrdinaryDiffEq: ODESolution, OrdinaryDiffEqAlgorithm, solve, RosenBrock23
    using Symbolics: jacobian
    # Other Necessities
    using Distributions: Normal, LogNormal, Distribution
    using ImageFiltering: imfilter, Inner # Convolution in Julia
    using FFTW: fft, ifft
    using Tullio: @tullio
    # stdlib
    using LinearAlgebra, Statistics, Random, Logging, Printf
    using Logging: Info, Warn, LogLevel
    ##
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