module WENDy
    ## external dependencies
    # optimization algorithms
    using NonlinearSolve: NonlinearLeastSquaresProblem, NonlinearFunction, LevenbergMarquardt # nonlinear least squares
    using NonlinearSolve: solve as solve_lsq, NonlinearSolve # (NonlinearSolve.ReturnCode for the solver status)
    using Optim: Optim, optimize, NewtonTrustRegion, Options as Optim_Options # unconstrained trust region
    using JSOSolvers: tron # constrained trust region
    using ManualNLPModels: NLPModel # necessary for ARC_qK, JSOSolvers
    using AdaptiveRegularization: ARCqKOp # ARC_qK
    # For symbolicly representing diff eq (forward solves for OutputErrorProblem are in ext/WENDyOrdinaryDiffEqExt.jl)
    using Symbolics, ForwardDiff
    using Symbolics: jacobian, @variables
    # Other Necessities
    using Distributions: Normal, LogNormal, Distribution
    using ImageFiltering: imfilter, Inner # Convolution in Julia
    using FFTW: fft, ifft
    using Tullio: @tullio
    using FunctionWrappers: FunctionWrapper # fixed-signature rhs of the output error sensitivity system
    using Arpack: svds
    # stdlib
    using LinearAlgebra, Statistics, Logging
    using Logging: Info, Warn, LogLevel
    ##
    include("wendyDataTypes.jl")
    include("wendyTestFunctions.jl")
    include("wendyNoise.jl")
    include("wendySymbolics.jl")
    include("wendyWeakLikelihood.jl")
    include("wendyProblems.jl")
    include("wendyOutputError.jl")
    include("wendyOptim.jl")

    export WENDyProblem, OutputErrorProblem, WENDyParameters, WENDySolution, solve, Normal, LogNormal
    export ℓ, ∇ℓ, ∇ℓ!, ∇²ℓ, ∇²ℓ!, ∇²m, ∇²m!, weakResidual, weakResidual!, weakJacobian, weakJacobian!, weakCovariance, Ĉ, C_cons, C_∇²ℓ, oeResidual!, oeJacobian!
end # module WENDy