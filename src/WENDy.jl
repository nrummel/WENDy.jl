module WENDy
    ## external dependencies
    # optimization algorithms
    using NonlinearSolve: NonlinearLeastSquaresProblem, NonlinearFunction, LevenbergMarquardt # nonlinear least squares
    using NonlinearSolve: solve as solve_lsq
    using Optim:optimize, NewtonTrustRegion, Options as Optim_Options # unconstrained trust region
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
    using Arpack: svds
    # stdlib
    using LinearAlgebra, Statistics, Logging
    using Logging: Info, Warn, LogLevel
    ##
    include("wendyDataTypes.jl")
    include("wendyTestFunctions.jl")
    include("wendyNoise.jl")
    include("wendySymbolics.jl")
    include("wendyEquations.jl")
    include("wendyLinearEquations.jl")
    include("wendyNonlinearEquations.jl")
    include("wendyWeakForms.jl")
    include("wendyWorkspace.jl")
    include("wendyProblems.jl")
    include("wendyOutputError.jl")
    include("wendyOptim.jl")

    export WENDyProblem, OutputErrorProblem, WENDyParameters, solve, Normal, LogNormal
    export ℓ, ∇ℓ, ∇ℓ!, ∇²ℓ, ∇²ℓ!, weakResidual, weakResidual!, weakJacobian, weakJacobian!, weakCovariance, oeResidual!, oeJacobian!
end # module WENDy