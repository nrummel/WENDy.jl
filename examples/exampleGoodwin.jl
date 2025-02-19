if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    @info "More than WENDy.jl is necessary for plotting and data generation"
    using Pkg; 
    Pkg.activate(@__DIR__)
    Pkg.develop(path=joinpath(@__DIR__, ".."))
end
using Random, Logging, LinearAlgebra
using PlotlyJS
using WENDy
using OrdinaryDiffEq: ODEProblem
using OrdinaryDiffEq: solve as solve_ode
## Define rhs, ic, time domain, and length of parameters
function f!(du, u, p, t)
    du[1] = p[1] / (2.15 + p[3]*u[3]^p[4]) - p[2] * u[1]
    du[2] = p[5]*u[1]- p[6]*u[2]
    du[3] = p[7]*u[2]-p[8]*u[3]
    nothing
end
tRng  = (0.0, 80.0)
dt    = 0.5
u₀    = [0.3617, 0.9137, 1.3934]
pstar = [3.4884, 0.0969, 1, 10, 0.0969, 0.0581, 0.0969, 0.0775]
J     = length(pstar)
D     = length(u₀)
## Generate data (one could use empircal data in practice)
ode = ODEProblem(
    f!, 
    u₀, 
    tRng, 
    pstar
)
tt    = tRng[1]:dt:tRng[end]
Mp1   = length(tt)
Ustar = reduce(vcat, um' for um in solve_ode(ode, saveat=dt).u)
nr   = 0.05
U     = Ustar.* exp.(nr*randn(size(Ustar)))
## Create wendy problem struct
# The test function radii range should be set specific to the problem
params = WENDyParameters(
    radiusMinTime=dt, 
    radiusMaxTime=tRng[end]/5
)
wendyProb = WENDyProblem(
    tt,
    U,
    f!,
    J;
    noiseDist=Val(LogNormal), # multiplicative LogNormal noise
    params=params, 
    ll=Logging.Info # turn on logging information
);
## Solve the wendy problm given an intial guess for the parameters 
J = length(pstar)
@info "Solving wendy problem ..."
p₀ = [3.0, 0.1, 4, 12, 0.1, 0.1, 0.1, 0.1]
relErr = norm(p₀ - pstar) / norm(pstar)
@info "Initializing with Relative Coefficient Error = $(relErr)"
@time phat = solve(wendyProb, p₀)
relErr = norm(phat - pstar) / norm(pstar)
@info "Relative Coefficient Error = $(relErr)"
## plot the resutls 
odeprob = ODEProblem(f!, u₀, tRng, phat)
sol = solve_ode(odeprob; saveat=dt)
Uhat = reduce(vcat, um' for um in sol.u)
colors = ["red", "blue", "green"]
plot(
    reduce(vcat, [
        scatter(x=tt, y=U[:,d], marker_color=colors[d], name="data", mode="markers", legendgroup=d, marker_opacity=0.5),
        scatter(x=tt, y=Uhat[:,d],line_color=colors[d], line_dash="dash", name="estimate", legendgroup=d, legendgrouptitle_text="u[$d]"),
        scatter(x=tt, y=Ustar[:,d],line_color=colors[d], name="truth", legendgroup=d )
    ] for d in 1:D),
    Layout(title="Goodwin Example",xaxis_title="time(s)", yaxis_title="State")
)