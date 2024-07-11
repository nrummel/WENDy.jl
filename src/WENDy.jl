# module WENDy

## external dependencies
# optimization algorithms
using NonlinearSolve # high performance nonlinear solver
using DiffEqParamEstim, ForwardDiff # forward solve nonlinear lsq comparison
using Optimization, OptimizationOptimJL # Trust Region Solver
using SFN # ARC and SFN solvers
# For solving and symbolicly representing diff eq 
using OrdinaryDiffEq, ModelingToolkit, Symbolics
using OrdinaryDiffEq: ODESolution
using ModelingToolkit: t_nounits as t, D_nounits
using ModelingToolkit: ODESystem
using Symbolics: @variables
# Other Necessities
using BSON, MAT
using Distributions: Normal, LogNormal, Distribution
using ImageFiltering: imfilter, Inner # Convolution in Julia
using FFTW: fft, ifft
using Tullio: @tullio
using LoopVectorization
# stdlib
using LinearAlgebra, Statistics, Random, Logging 
using Logging: Info, Warn, LogLevel
__precompile__(false)

includet("wendyTestFunctions.jl")
includet("wendyData.jl")
includet("wendyNoise.jl")
includet("wendySymbolics.jl")
includet("wendyProblems.jl")
includet("wendyDiffEq.jl")
includet("wendyEquations.jl")
includet("wendyLinearEquations.jl")
includet("wendyNonlinearEquations.jl")
includet("wendyMethods.jl")
includet("wendyLinearMethods.jl")
includet("wendyNonlinearMethods.jl")
includet("wendyOptim.jl")

# export #TODO: figure out this
# end # module