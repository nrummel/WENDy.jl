using Random, Logging, LinearAlgebra
using PlotlyJS
using WENDy
using OrdinaryDiffEq: ODEProblem
using OrdinaryDiffEq: solve as solve_ode
## Define rhs, ic, time domain, and length of parameters
function f!(du, u, p, t)
    du[1] = p[1] / (p[3] + u[3]^p[4]) - p[2] * u[1]
    du[2] = p[5]*u[1]- p[6]*u[2]
    du[3] = p[7]*u[2]-p[8]*u[3]
    nothing
end
tRng  = (0.0, 80.0)
dt    = 0.5
u₀    = [0.3617, 0.9137, 1.3934]
wstar = [3.4884, 0.0969, 2.15, 10, 0.0969, 0.0581, 0.0969, 0.0775]
J     = length(wstar)
D     = length(u₀)
## Generate data (one could use empircal data in practice)
ode = ODEProblem(
    f!, 
    u₀, 
    tRng, 
    wstar
)
tt    = tRng[1]:dt:tRng[end]
Mp1   = length(tt)
Ustar = reduce(vcat, um' for um in solve_ode(ode, saveat=dt).u)
snr   = 0.1
U     = Ustar.* exp.(snr*randn(size(Ustar)))
## Create wendy problem struct
wendyProb = WENDyProblem(tt, U, f!, J, Val(false)#=Nonlinear in parameters=#, Val(LogNormal)#=LogNormalNoise=#, 
    WENDyParameters(radiusMinTime=dt, radiusMaxTime=tRng[end]/5); # be sure to set the min and max testFunction radii to something reasonable
    ll=Logging.Info # turn on logging information
);
## Solve the wendy problm given an intial guess for the parameters 
J = length(wstar)
p₀ = wstar + 0.5*randn(J).*abs.(wstar)
relErr = norm(p₀ - wstar) / norm(wstar)
@info "Initializing with Relative Coefficient Error = $(relErr)"
@info "Solving wendy problem ..."
@time what = WENDy.solve(wendyProb, p₀)
relErr = norm(what - wstar) / norm(wstar)
@info "Relative Coefficient Error = $(relErr)"
## plot the resutls 
odeprob = ODEProblem(f!, u₀, tRng, what)
sol = solve_ode(odeprob; saveat=dt)
Uhat = reduce(vcat, um' for um in sol.u)
colors = ["red", "blue", "green"]
plot(
    reduce(vcat, [
        scatter(x=tt, y=Uhat[:,d],line_color=colors[d], line_dash="dash", name="estimate", legendgroup=d, legendgrouptitle_text="u[$d]"),
        scatter(x=tt, y=U[:,d], marker_color=colors[d], name="data", mode="markers", legendgroup=d ),
        scatter(x=tt, y=Ustar[:,d],line_color=colors[d], name="truth", legendgroup=d )
    ] for d in 1:D),
    Layout(title="Goodwin Example",xaxis_title="time(s)", yaxis_title="State")
)