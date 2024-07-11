module WENDy

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
using BSON
using Distributions: Normal, LogNormal, Distribution
using ImageFiltering: imfilter, Inner # Convolution in Julia
using FFTW: fft, ifft
using Tullio: @tullio
using LoopVectorization
# stdlib
using LinearAlgebra, Statistics, Random, Logging 

__precompile__(false)

include("wendyEquations.jl")
include("wendyMethods.jl")
include("wendyNoise.jl")
include("wendySymbolics.jl")
include("wendyProblems.jl")
include("exampleProblems.jl")

export Wend
end # module