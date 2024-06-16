using NLPModels, NLPModelsJuMP, JuMP, JSOSolvers
using SparseArrays
model = Model()
x0 = [-1.2; 1.0]
@variable(model, x[i=1:2], start=x0[i])
@NLexpression(model, F1, x[1] - 1)
@NLexpression(model, F2, 10 * (x[2] - x[1]^2))

nls = MathOptNLSModel(model, [F1, F2], name="rosen-nls")

residual(nls, nls.meta.x0)
##
# Compare non-linear least squares solvers on the multi-dimensional Rosenbrock function

startpoint(n) = zeros(Float64, n)

################################################################################
# NLLS formulation
# using NLLSsolver, Static, StaticArrays

# struct MyRes <: NLLSsolver.AbstractResidual
#     ak::Vector{Float64}
#     bk::Float64
#     varind::Int64
# end

# Base.eltype(::MyRes) = Float64
# NLLSsolver.ndeps(::MyRes) = static(1) # Residual depends on 1 variable
# NLLSsolver.nres(::MyRes) = static(1) # Residual has length 2
# NLLSsolver.varindices(res::MyRes) = res.varind
# NLLSsolver.getvars(res::MyRes, vars::Vector) = (vars[res.varind]::Float64, )
# NLLSsolver.computeresidual(res::MyRes, x) = dot(res.ak,x) - res.bk

# function nllssolver_dense(n::Int, A::Matrix{Float64}, b::Vector{Float64})
#     costs = NLLSsolver.CostStruct{MyRes}()
#     costs.data[MyRes] = MyRes[MyRes(A[k,:],b[k], k) for k = 1:n]
#     return NLLSProblem(startpoint(n), costs)
# end

# function optimize(model::NLLSProblem)
#     result = NLLSsolver.optimize!(model, NLLSOptions(maxiters = 30000))
#     return result.niterations, result.startcost, result.bestcost
# end
################################################################################

################################################################################
# JSO formulation
using JuMP
using JSOSolvers, NLPModels, NLPModelsJuMP

struct JSOModel{T}
    model::MathOptNLSModel
    solver::T
end


function jso_dense(n::Int, G::Vector{Function}, gradG::Vector{Function}, b::Vector, solver=tron)
    model = Model()
    x0 = startpoint(n)
    @variable(model, x[i = 1:n], start = x0[i])
    register(model,:G, n, G, gradG; autodiff=false)
    @NLexpression(model, F, G(x...)-b)
    return JSOModel(MathOptNLSModel(model, F), solver)
end

function optimize(model::JSOModel)
    result = model.solver(model.model)
    return result.iter, NaN64, result.objective
end
##
# Run the test and display results
using Plots, Printf

tickformatter(x) = @sprintf("%g", x)

function runtest(name, sizes, solvers)
    result = Matrix{Float64}(undef, (4, length(sizes)))
    p = plot()
    K = 250
    # For each optimizer
    # for (label, constructor) in solvers
    #     # First run the optimzation to compile everything
    #     optimize(constructor(sizes[1],rand(K, sizes[1]),rand(K)))
    #     optimize(constructor(sizes[end], rand(K, sizes[end]),rand(K))) 
    #     # Do largest and smallest in case there are dense and sparse code paths
    # end
    # Go over each problem, recording the time, iterations and start and end cost
    for (i, n) in enumerate(sizes)
        A = rand(K, sizes[end],)
        b = rand(K)
        G(args...) = A*vcat(x...)
        gradG(args...) = A
        xstar = A\b
        for (label, constructor) in solvers
            # Construct the problem
            model = jso_dense(n,G,grad,b)
            # Optimize
            result[1,i] = @elapsed res = optimize(model)
            result[2,i] = res[1]
            result[3,i] = res[2]
            result[4,i] = res[3]
            if res[3] > 1.e-10
                cost = res[3]
                println("$label on size $n converged to a cost of $cost")
                result[1,i] = NaN64
            end
        end
        # Plot the graphs
        plot!(p, vec(sizes), vec(result[1,:]), label=label)
    end
    yaxis!(p, minorgrid=true, formatter=tickformatter)
    xaxis!(p, minorgrid=true, formatter=tickformatter)
    plot!(p, legend=:topleft, yscale=:log10, xscale=:log2)
    title!(p, "Speed comparison: $name")
    xlabel!(p, "Problem size")
    ylabel!(p, "Optimization time (s)")
    display(p)
end
################################################################################

runtest("dense problem",  [2 4 8 16 32 64],  ["JSO tron" => n->jso_dense(n, tron)])


##
"""
    memoize(foo::Function, n_outputs::Int)

Take a function `foo` and return a vector of length `n_outputs`, where element
`i` is a function that returns the equivalent of `foo(x...)[i]`.

To avoid duplication of work, cache the most-recent evaluations of `foo`.
Because `foo_i` is auto-differentiated with ForwardDiff, our cache needs to
work when `x` is a `Float64` and a `ForwardDiff.Dual`.
"""
function memoize(foo::Function, n_outputs::Int)
    last_x, last_f = nothing, nothing
    last_dx, last_dfdx = nothing, nothing
    function foo_i(i, x::T...) where {T<:Real}
        if T == Float64
            if x !== last_x
                last_x, last_f = x, foo(x...)
            end
            return last_f[i]::T
        else
            if x !== last_dx
                last_dx, last_dfdx = x, foo(x...)
            end
            return last_dfdx[i]::T
        end
    end
    return [(x...) -> foo_i(i, x...) for i in 1:n_outputs]
end
##
using Ipopt, JuMP
K = 92*3
n = 2
A = rand(K, n)
b = rand(K)
mdl = Model(Ipopt.Optimizer)
x0 = zeros(n)
@variable(mdl, x[i = 1:n], start = x0[i])
rr(x) = A*x - b
gradres(x) = A
@operator(mdl, residuals, n, rr, gradres)
@objective(mdl, Min, sum(residuals(x).^2) ) 
optimize!(mdl)
xstar = A\b
norm(xstar - value.(x)) /norm(xstar)
# F = Vector{NonlinearExpression}(undef,K)
# for k = 1:K 
#     rk(args...) = dot(A[k,:], vcat(args...)) - b[k]
#     gk(args...) = A[k,:]
#     hk(args) = Matrix{Float64}(I,n,n)
#     f = Symbol("f$k")
#     register(mdl,f,n,rk,gk,hk)
#     # exp_str = "\$(f$k)(" * prod("\$(x[$nn])," for nn in 1:n)[1:end-1] * ")"
#     # expr = Meta.parse(exp_str)
#     expr = Expr(:call, f, x...)
#     F[k] = @NLexpression(mdl, expr)
# end
# ##
# out = tron(MathOptNLSModel(mdl, F))
# A\b - out.solution



##
w0 = wTrue 
wstar = G0 \ b0


function res!(r::Vector{T}, G::Matrix{T}, b::Vector{T}, x::Vector{W}) where {T,W}
    r .= b
    mul!(r, G, x, 1, 1)
    nothing 
end
struct MyRes
    G0::Matrix 
    b::Vector 
    r::Vector 
end 
function MyRes(G0,b)
    MyRes(G0,b,similar(b))
end
function (s::MyRes)(x::Vector{W}) where W
    res!(s.r,s.G0,s.b,x)
    return s.r
end
res = MyRes(G0,b0)
G(x) = G0*x-b0
gradres(x) = G0
##
mdl = Model(Ipopt.Optimizer)
w0 = wTrue
@variable(mdl, w[i = 1:J], start = w0[i])
@variable(mdl, r[k = 1:K*D])
@operator(mdl, compRes, J, G, gradres)
@constraint(mdl, r == compRes(w))
@objective(mdl, Min, sum(r.^2) ) 
set_silent(mdl)
@time optimize!(mdl)
what = value.(w)
norm(wstar - what) /norm(wstar)
##


for i = 1:1000
    # L = Lgetter(wim1)
    # S0 =  L * L'
    # S = S0* (1-diagReg) + I * diagReg
    # S = cholesky(S)
    # S = I 
    # rim1 = r(wim1)
    # jacG = jac(wim1)
    # eim1 = 1/2*rim1'*(S\rim1)
    rim1 = A*wim1 - b 
    grad = A'*rim1
    eim1 = 1/2*norm(rim1)^2
    # @info "i = $i"
    # @info " eim1 = $eim1"
    for j = 1:10
        # wi = wim1 - alpha * jacG' * (S \ rim1)
        # ri = r(wi)
        # ei = 1/2*ri'*(S\ri) 
        wi = wim1 - alpha * grad 
        ri = A*wi - b
        ei = 1/2*norm(ri)^2
        # @info "  j = $j"
        # @info "   α  = $alpha"
        # @info "   ei = $ei"
        if ei < eim1 
            break 
        end 
        alpha *= 0.99
    end 
    if ei >= eim1
        @warn "line search failed "
        break 
    elseif norm(wi - wim1) / norm(wi) < 1e-4
        break
    end
    wim1 = wi  
end 
##
using JuMP, Ipopt, LinearAlgebra
# set up
K = 500
J = 10 
B = rand(K, J)
b = rand(K)
buf = similar(b)
res = similar(b)
wrand = rand(J) # for testing our methods
# Build random SPD matrix 
A = rand(K, K)
fact = qr(A)
Q= fact.Q
S = Q *diagm(100*rand(K))*Q'
S = (S+S')/2
cholesky!(S)
R = UpperTriangular(S)
##
function GG(w,B=B)
    B*w
end
function naiveF(w,R=R,b=b)
    return R' \ (GG(w)-b)
end
function F!(res, w, B, R, b, buf)
    buf .= b
    mul!(buf, B, w, 1, -1) # (Bw - b)
    ldiv!(res, R', buf) # (R^T)^{-1}(buf)
    nothing
end
##
@time F!(res, wrand, B, R, b, buf );
nres = naiveF(wrand)
@assert norm(res - nres)/norm(res) < 1e2*eps() "Inplace function is not working properly"
##
function F(w, B=B, R=R, b=b, buf=buf)
    r = zeros(K)
    F!(r, w, B, R, b, buf)
    return r
end
@time res = F(wrand)
@assert norm(res - nres)/norm(res) < 1e2*eps() "Function is not working properly"
##
function jacGG(w, R=R, B=B)
    return B
end
function jacF(w, R=R, B=B)
    return R' \ jacGG(w)
end
@assert norm(jacF(wrand) - R'\B) / norm(R'\B) < 1e2*eps() "Jacobian computation is not working properlys"
##
mdl = Model(Ipopt.Optimizer)
@variable(mdl, w[i = 1:J])
@variable(mdl, r[k = 1:K])
@operator(mdl, f, J, naiveF, jacF)
@constraint(mdl, r == f(w))
# @objective(mdl, Min, sum( (R'\(B*w -b)).^2 ) ) 
@objective(mdl, Min, sum(r.^2 ) ) 
# set_silent(mdl)
@time optimize!(mdl)
wi = value.(w)
wstar = (R' \ B) \ (R' \ b)
relErr = norm(wi - wstar) / norm(wstar)
@info """ How did we do?
    relative Error     = $relErr
    termination_status = $(termination_status(mdl))
    primal_status      = $(primal_status(mdl))
    objective_value    = $(objective_value(mdl))
 """