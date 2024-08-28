module WENDy
    # __precompile__(false)
    ## external dependencies
    # optimization algorithms
    using NonlinearSolve # high performance nonlinear solver
    using ForwardDiff # forward solve nonlinear lsq comparison
    using Optimization, OptimizationOptimJL, Optim # Trust Region Solver
    using JSOSolvers, ManualNLPModels, AdaptiveRegularization
    using SFN # ARC and SFN solvers
    # For solving and symbolicly representing diff eq 
    using OrdinaryDiffEq, ModelingToolkit, Symbolics
    using OrdinaryDiffEq: ODESolution, OrdinaryDiffEqAlgorithm
    using ModelingToolkit: t_nounits as t, D_nounits
    using ModelingToolkit: ODESystem
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
    export simulate!, buildCostFunctions, forwardSolveRelErr, forwardSolve
    export IRWLS, bfgs_Optim, tr_Optim, arc_SFN, tr_JSO, arc_JSO
end # module WENDy