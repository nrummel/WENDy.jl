using Random, Logging, LinearAlgebra
using PlotlyJS
using WENDy
using OrdinaryDiffEq: ODEProblem
using OrdinaryDiffEq: solve as solve_ode
## Define rhs, ic, time domain, and length of parameters
function f!(du, u, w, t)
    β = (w[1] * exp(-w[1] * w[2])) / (1 - exp(-w[1] * w[2]))
    du[1] = -w[1] * u[1] + w[3] * u[2] + β * u[3] 
    du[2] = w[1] * u[1] - w[3] * u[2] - w[4] * (1 - exp(-w[5]  * t^2)) * u[2] 
    du[3] = w[4] * (1 - exp(-w[5]  * t^2)) * u[2] - β * u[3]
end
tRng  = (0.0, 50.0)
dt    = 0.1
u₀    = [1,0,0]
wstar = [0.2,1.5,0.074,0.113,0.0024]
constraints = [
    (1e-4,1.0), 
    (1e-4,2.0),
    (1e-4,1.0),
    (1e-4,1.0),
    (1e-4,1.0),
]
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
snr   = 0.01
U     = Ustar .* exp.(snr*randn(size(Ustar)))
## Create wendy problem struct
# because the time domain is larger in this case we want to change the radii of the test functions 
params = WENDyParameters(
    radiusMinTime  = 0.1,
    radiusMaxTime  = 25.0,
    Kmax=300
)
wendyProb = WENDyProblem(tt, U, f!, J, Val(false)#=Nonlinear in parameters=#, Val(LogNormal)#=LogNormalNoise=#, params; ll=Logging.Info, constraints=constraints);
## Solve the wendy problm given an intial guess for the parameters 
J = length(wstar)
w0 = wstar + 0.5*randn(J).*abs.(wstar)
relErr = norm(w0 - wstar) / norm(wstar)
@info "Initializing with Relative Coefficient Error = $(relErr)"
@info "Solving wendy problem ..."
@time what = WENDy.solve(wendyProb, w0)
relErr = norm(what - wstar) / norm(wstar)
@info "Relative Coefficient Error = $(relErr)"
## plot the resutls 
odeprob = ODEProblem(f!, u₀, tRng, what)
sol = solve_ode(odeprob; saveat=dt)
Uhat = reduce(vcat, um' for um in sol.u)
plot(
    reduce(vcat, [
        scatter(x=tt, y=Uhat[:,d], name="estimate"),
        scatter(x=tt, y=U[:,d], name="data", mode="markers"),
        scatter(x=tt, y=Ustar[:,d], name="truth")
    ] for d in 1:D),
    Layout(title="Results",xaxis_title="time(s)", yaxis_title="State")
)