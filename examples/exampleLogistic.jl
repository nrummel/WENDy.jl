using Random, Logging, LinearAlgebra
using PlotlyJS
using WENDy
## Define rhs, ic, time domain, and length of parameters
function f!(du, u, w, t)
    du[1] = w[1] * u[1] - w[2] * u[1]^2
    nothing
end
tRng = (0.0, 10.0)
dt = 0.1
u0 = [0.01]
wstar = [1.0, 1.0]
J = length(wstar)
## Generate data (one could use empircal data in practice)
ode = ODEProblem(
    f!, 
    u0, 
    tRng, 
    wstar
)
tt = tRng[1]:dt:tRng[end]
Ustar = reduce(vcat, um for um in solve(ode, saveat=dt).u)
U = Ustar + 0.01*randn(size(Ustar))
## Create wendy problem struct
wendyProb = WENDyProblem(tt, U, f!, J, Val(true), Val(Normal); ll=Logging.Info)
## Solve the wendy problm given an intial guess for the parameters 
w0 = [0.5, 0.5]
what = solve(wendyProb, w0)
relErr = norm(what - wstar) / norm(wstar)
@info "Relative Coefficient Error = $(relErr)"
## plot the resutls 
odeprob = ODEProblem(f!, u0, tRng, what)
sol = solve(odeprob; saveat=dt)
Uhat = reduce(vcat, um' for um in sol.u)
plot(
    [
        scatter(x=tt, y=Uhat[:], name="estimate"),
        scatter(x=tt, y=U[:], name="data", mode="markers"),
        scatter(x=tt, y=Ustar[:], name="truth")
    ],
    Layout(title="Results",xaxis_title="time(s)", yaxis_title="State")
)

