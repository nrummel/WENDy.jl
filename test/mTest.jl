using LinearAlgebra, Logging # stdlib 
try # handle you not having stuff installed 
    @info "Trying to load external dependencies...."
    using Revise 
    using Tullio, BSON, FiniteDiff, LoopVectorization, PlotlyJS, Crayons # minimum numbeer of dependencies
catch 
    @warn "External dependencies are not present adding them through package manager"
    using Pkg 
    for pkg in ["Revise", "Tullio", "BSON", "FiniteDiff","LoopVectorization", "PlotlyJS", "Crayons"]
        Pkg.add(pkg)
    end 
    using Revise 
    using Tullio, BSON, FiniteDiff, LoopVectorization, PlotlyJS, Crayons
end
includet(joinpath(@__DIR__, "../src/wendyEquations.jl"))
includet(joinpath(@__DIR__, "../ForStephen/_wendyEquations.jl")) # wrapper that allocate everything so we can call them as simple as possible
includet(joinpath(@__DIR__, "../ForStephen/gradientCheck.jl"))
## Load data from file
wTrue = [1.1]
J = length(wTrue)
μ = 0.1
w0 = wTrue + μ * abs.(wTrue) .* randn(J);
BSON.@load joinpath(@__DIR__, "../ForStephen/ExponentialData.bson") U V Vp b0 sig diagReg
b₀ = b0 # I didn't know how bson would like the unicode var name...
## Define Functions that only need w with the current data fixed
m(w::AbstractVector{<:Real}) = m(w, U, V, Vp, b0, sig, diagReg)
∇m!(∇m::AbstractVector{<:Real}, w::AbstractVector{<:Real}) = ∇m!(∇m, w, U, V, Vp, b0, sig, diagReg)
Hm!(H::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}) = Hm!(H, w, U, V, Vp, b0, sig, diagReg)
##
@info "Objective function function call"
@time m(w0)
## Gradients
@info "Comparing Gradient"
∇m0 = zeros(J)
@info "  Finite Diff for Gradient"
@time ∇m_fd = FiniteDiff.finite_difference_gradient(m, w0)
@info "  Analytic Gradient Computation"
@time ∇m!(∇m0, w0); 
fucker = zeros(J)
@time _∇m!_(fucker, w0); 
relErr = norm(∇m0 - ∇m_fd) / norm(∇m_fd)
@info "  relErr = $relErr"
# Define this function before so it can be reused when compile for acccurate timing 
function Hm_fd!(H,w,p=nothing) 
    FiniteDiff.finite_difference_jacobian!(H, ∇m!, w)
    @views H .= 1/2*(H + H')
    @views H .= Symmetric(H)
    nothing 
end 
## Hessian 
H0 = zeros(J,J)
@info "Comparing Hessian "
@info "  Analytic Hessian "
@time Hm!(H0, w0)
@info "  Finite Differences Hessian from objective"
Hfd = zeros(J,J)
@time FiniteDiff.finite_difference_hessian!(Hfd, m, w0)
@info "  Finite Differences Hessian from gradient"
Hfd2 = zeros(J,J)
@time Hm_fd!(Hfd2, w0)
@info "   Rel Error (finite diff obj vs finite diff grad) $(norm(Hfd - Hfd2) / norm(Hfd))"
@info "   Rel Error (analytic vs finite diff obj) $(norm(H0 - Hfd) / norm(Hfd))"
@info "   Rel Error (analytic vs finite diff grad) $(norm(H0 - Hfd2) / norm(Hfd2))"
@show Hfd 
@show H0
nothing
##
@info "Running Gradien Check for ∇m"
function g(w) 
    ∇m = zeros(J)
    ∇m!(∇m, w)
    return ∇m
end
p,_ = gradientCheck(m, g, w0; 
ll=Info, scaling=1e-8,makePlot=true)
PlotlyJS.relayout!(p, title="Gradient Maholinobis Distance")
display(p);
##
@info "Running Gradien Check for Hm"
v = [1]
f(w) = dot(g(w), v)
function gg(w) 
    H = zeros(J,J)
    Hm!(H, w)
    return H*v
end
function ggg(w) 
    H = zeros(J,J)
    FiniteDiff.finite_difference_jacobian!(H, ∇m!, w)
    @views H .= 1/2*(H + H')
    @views H .= Symmetric(H)
    return H*v 
end 
p,_ = gradientCheck(f, gg, w0; 
ll=Info, scaling=1e-8,makePlot=true)
PlotlyJS.relayout!(p, title="Hessian Maholinobis Distance")
display(p);
##
ww = range(1,3,200)
plot(
    [scatter(x=ww, y=[m([w]) for w in ww], name="Maholinobis Distance"),
    scatter(x=ww, y=[f([w])[1] for w in ww], name="Gradient Maholinobis Distance"),
    scatter(x=ww, y=[gg([w])[1] for w in ww], name="Hessian Maholinobis Distance"),
    scatter(x=ww, y=[ggg([w])[1] for w in ww], line_dash="dash", name="Hessian (fd) Maholinobis Distance")]
)