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
    du[1] = p[1] * u[1] - p[2] * u[1]^2
    nothing
end
tRng = (0.0, 10.0)
dt = 0.01
u₀ = [0.01]
pstar = [1.0, 1.0]
J = length(pstar);
## Generate data (one could use empircal data in practice)
ode = ODEProblem(
    f!, 
    u₀, 
    tRng, 
    pstar
)
tt = tRng[1]:dt:tRng[end]
Ustar = reduce(vcat, um for um in solve_ode(ode, saveat=dt).u)
nr = 0.1
U = Ustar + nr*randn(size(Ustar));
## Create wendy problem struct
params = WENDyParameters(Kmax=500)
wendyProb = WENDyProblem(
    tt, 
    U, 
    f!, 
    J, 
    linearInParameters=Val(true), 
    ll=Logging.Info,
    params=params
);
## Solve the wendy problm given an intial guess for the parameters 
p₀ = [0.5, 0.5]
relErr = norm(p₀ - pstar) / norm(pstar)
@info "Init Relative Coefficient Error = $(relErr)"
@time phat = solve(wendyProb, p₀)
relErr = norm(phat - pstar) / norm(pstar)
@info "Relative Coefficient Error = $(relErr)"
## plot the resutls 
odeprob = ODEProblem(f!, u₀, tRng, phat)
sol = solve_ode(odeprob; saveat=dt)
Uhat = reduce(vcat, um' for um in sol.u)
plot(
    [
        scatter(x=tt, y=U[:], name="data", mode="markers", marker_color="blue", marker_opacity=0.5),
        scatter(x=tt, y=Ustar[:], name="truth", line_color="blue", line_width=3),
        scatter(x=tt, y=Uhat[:], name="estimate", line_dash="dash", line_color="black", line_width=3),
    ],
    Layout(title="Results",xaxis_title="time(s)", yaxis_title="State")
)
