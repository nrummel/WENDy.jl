# module WENDy
    # __precompile__(false)
    ## external dependencies
    # optimization algorithms
    using NonlinearSolve # high performance nonlinear solver
    using DiffEqParamEstim, ForwardDiff # forward solve nonlinear lsq comparison
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
    includet("wendyTestFunctions.jl")
    includet("wendyData.jl")
    includet("wendyNoise.jl")
    includet("wendySymbolics.jl")
    includet("wendyProblems.jl")
    includet("wendyDiffEq.jl")
    ## 
    includet("wendyEquations.jl")
    includet("wendyLinearEquations.jl")
    includet("wendyNonlinearEquations.jl")
    includet("wendyMethods.jl")
    includet("wendyLinearMethods.jl")
    includet("wendyNonlinearMethods.jl")
    includet("wendyOptim.jl")
#     export WENDyProblem, WENDyParameters, WENDyData, CostFunction, SimulatedWENDyData, EmpricalWENDyData, SimulationParameters
#     export simulate!, buildCostFunctions
#     export IRWLS, bfgs_Optim, tr_Optim, arc_SFN, tr_JSO, arc_JSO
# end # module WENDy