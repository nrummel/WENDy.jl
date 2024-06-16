@info "Loading dependencies"
@info " Loading exampleProblems..."
includet("../examples/exampleProblems.jl")
@info " Loading symbolic calcutations..."
includet("../src/wendySymbolics.jl")
@info " Loading wendy equations..."
includet("../src/wendyEquations.jl")
@info " Loading IRWLS..."
includet("../src/wendyIRWLS.jl")
@info " Loading other dependencies"
using BenchmarkTools
##
ex = FITZHUG_NAGUMO
params = WENDyParameters()
wendyProb = WENDyProblem(ex, params;ll=Logging.Info)
nothing
##
iter = Nonlinear_IRWLS_Iter(wendyProb, params)
iterLin = Linear_IRWLS_Iter(wendyProb, params)
wTrue = wendyProb.wTrue
J = length(wTrue)
w0 = wTrue + abs.(wTrue) .* randn(J)
# w0 = iterLin(zeros(J))
##
@info "IRWLS"
dt = @elapsed a = @allocations what, wit, resit = IRWLS(wendyProb, params, iter, w0; trueIter=iterLin)
@info """   
    iterations  = $(size(wit,2)-1)
    time        = $(dt)
    allocations = $(a)
"""
if typeof(what) <:AbstractVector 
    relErr = norm(wit[:,end] - wTrue) / norm(wTrue)
    @info "   coeff rel err = $relErr"
end
