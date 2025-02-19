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
    β = (p[1] * exp(-p[1] * p[2])) / (1 - exp(-p[1] * p[2]))
    du[1] = -p[1] * u[1] + p[3] * u[2] + β * u[3] 
    du[2] = p[1] * u[1] - p[3] * u[2] - p[4] * (1 - exp(-p[5]  * t^2)) * u[2] 
    du[3] = p[4] * (1 - exp(-p[5]  * t^2)) * u[2] - β * u[3]
end
tRng  = (0.0, 50.0)
dt    = 0.1
u₀    = [1,0,0]
pstar = [0.2,1.5,0.074,0.113,0.0024]
constraints = [
    (1e-4,1.0), 
    (1e-4,2.0),
    (1e-4,1.0),
    (1e-4,1.0),
    (1e-4,1.0),
]
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
nr   = 0.1
U     = Ustar .* exp.(nr*randn(size(Ustar)))
## Create wendy problem struct
# because the time domain is larger in this case we want to change the radii of the test functions 
params = WENDyParameters(
    radiusMinTime  = 0.1,
    radiusMaxTime  = 25.0
)
wendyProb = WENDyProblem(
    tt, 
    U, 
    f!, 
    J;
    noiseDist=Val(LogNormal), # LogNormalNoise
    params=params,
    ll=Logging.Info, 
    constraints=constraints
);
## Solve the wendy problm given an intial guess for the parameters 
J = length(pstar)
p₀ = [0.5,1.0,0.5,0.5,0.05]
relErr = norm(p₀ - pstar) / norm(pstar)
@info "Initializing with Relative Coefficient Error = $(relErr)"
@info "Solving wendy problem ..."
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
        scatter(x=tt, y=U[:,d], marker_color=colors[d], name="data", mode="markers", legendgroup=d,marker_opacity=0.5 ),
        scatter(x=tt, y=Ustar[:,d],line_color=colors[d], name="truth", legendgroup=d,line_width=4 ),
        scatter(x=tt, y=Uhat[:,d],line_color="black", line_dash="dash", line_width=4, name="estimate", legendgroup=d, legendgrouptitle_text="u[$d]"),
    ] for d in 1:D),
    Layout(title="SIR",xaxis_title="time(s)", yaxis_title="State")
)