module WENDy
    ## external dependencies
    # optimization algorithms
    using NonlinearSolve # high performance nonlinear solver
    using Optim # Trust Region Solver
    using ManualNLPModels: NLPModel # necessary for ARC_qK
    using AdaptiveRegularization: ARCqKOp # ARC_qK
    # For solving and symbolicly representing diff eq 
    using OrdinaryDiffEq, Symbolics
    using OrdinaryDiffEq: ODESolution, OrdinaryDiffEqAlgorithm, solve
    using Symbolics: jacobian
    # Other Necessities
    using BSON
    using Distributions: Normal, LogNormal, Distribution
    using ImageFiltering: imfilter, Inner # Convolution in Julia
    using FFTW: fft, ifft
    using Tullio: @tullio
    # stdlib
    using LinearAlgebra, Statistics, Random, Logging, Printf
    using Logging: Info, Warn, LogLevel
    ##
    include("wendyData.jl")
    include("wendyTestFunctions.jl")
    include("wendyNoise.jl")
    include("wendySymbolics.jl")
    include("wendyProblems.jl")
    include("wendyDiffEq.jl")
    ## 
    include("wendyEquations.jl")
    include("wendyLinearEquations.jl")
    include("wendyNonlinearEquations.jl")
    include("wendyMethods.jl")
    include("wendyLinearMethods.jl")
    include("wendyNonlinearMethods.jl")
    include("wendyOptim.jl")
    export WENDyProblem, WENDyParameters, WENDyData, CostFunction, SimulatedWENDyData, SimulationParameters
    export WeakNLL, GradientWeakNLL, HesianWeakNLL
    export simulate!, buildCostFunctions, forwardSolveRelErr, forwardSolve
    export IRWLS, bfgs, trustRegion, arcqk, hybrid, nonlinearLeastSquares
end # module WENDy