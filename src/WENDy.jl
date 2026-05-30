module WENDy
    # __precompile__(false)
    ## external dependencies
    # optimization algorithms
    using NonlinearSolve: NonlinearLeastSquaresProblem, NonlinearFunction, LevenbergMarquardt # nonlinear least squares
    using NonlinearSolve: solve as solve_lsq
    using Optim:optimize, NewtonTrustRegion, IPNewton, LBFGS, Options as Optim_Options # newton trust region for unconstrained optimization, and Interior Point Newton For Constrained Optimization
    using NLSolversBase: TwiceDifferentiable, TwiceDifferentiableConstraints # Necessary for IPNewton
    using JSOSolvers: tron # constrained trust region
    using ManualNLPModels: NLPModel # necessary for ARC_qK, JSOSolvers
    using AdaptiveRegularization: ARCqKOp # ARC_qK
    # For solving and symbolicly representing diff eq 
    using OrdinaryDiffEq: Rodas4P, ODEProblem, OrdinaryDiffEqAlgorithm 
    using OrdinaryDiffEq: solve as solve_ode
    using Symbolics, ForwardDiff
    using Symbolics: jacobian, @variables
    # Other Necessities
    using Distributions
    using ImageFiltering: imfilter, Inner # Convolution in Julia
    using FFTW: fft, ifft
    using Tullio: @tullio
    using Arpack: svds
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
    export WENDyProblem, WENDyParameters, solve, Normal, LogNormal
end # module WENDy