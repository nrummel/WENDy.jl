## 
using Revise
using DifferentialEquations, ModelingToolkit, BSON,MAT
using Random, Statistics, LinearAlgebra, Logging, StaticArrays # stdlib
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkit: @mtkmodel, @mtkbuild, ODESystem
using Symbolics: jacobian
includet("exampleProblems.jl")
includet("testFunctions.jl")
includet("computeGradients.jl")
##
_mdl = LogisticGrowthModel
@mtkbuild mdl = _mdl()
##
G = getRHS(mdl)
jacG = getParameterJacobian(mdl) 
jac_sym = _getParameterJacobian_sym(mdl)
jacG(1:2,[2])
##
w = parameters(mdl)
rhs_sym = _getRHS_sym(mdl)
jac_sym = jacobian(rhs_sym, w)
