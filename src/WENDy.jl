module WENDy
    ## external dependencies
    # optimization algorithms
    using NonlinearSolve # high performance nonlinear solver
    using Optimization, OptimizationOptimJL, Optim # Trust Region Solver
    using JSOSolvers, ManualNLPModels, AdaptiveRegularization
    using SFN # ARC and SFN solvers
    # For solving and symbolicly representing diff eq 
    using OrdinaryDiffEq, ModelingToolkit, Symbolics
    using OrdinaryDiffEq: ODESolution, OrdinaryDiffEqAlgorithm
    using Symbolics: jacobian
    using ModelingToolkit: build_function, modelingtoolkitize, @mtkbuild
    import ModelingToolkit: equations, parameters, unknowns, D_nounits, t_nounits
    using Symbolics: @variables
    # Other Necessities
    using BSON, MAT
    using Distributions: Normal, LogNormal, Distribution
    using ImageFiltering: imfilter, Inner # Convolution in Julia
    using FFTW: fft, ifft
    using Tullio: @tullio
    using ProgressMeter: @showprogress
    # stdlib
    using LinearAlgebra, Statistics, Random, Logging, Printf
    using Logging: Info, Warn, LogLevel
    ##
    include("wendyTestFunctions.jl")
    include("wendyData.jl")
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
    export WENDyProblem, WENDyParameters, WENDyData, CostFunction, SimulatedWENDyData, EmpricalWENDyData, SimulationParameters
    export WeakNLL, GradientWeakNLL, HesianWeakNLL
    export simulate!, buildCostFunctions, forwardSolveRelErr, forwardSolve
    export IRWLS, bfgs_Optim, tr_Optim, arc_SFN, tr_JSO, arc_JSO, hybrid, nonlinearLeastSquares
end # module WENDy