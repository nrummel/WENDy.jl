using Random, Logging, LinearAlgebra
using PlotlyJS
using WENDy
## Define rhs, ic, time domain, and length of parameters
function f!(du, u, w, t)
    du[1] = w[1] / (w[3] + u[3]^w[4]) - w[2] * u[1]
    du[2] = w[5]*u[1]- w[6]*u[2]
    du[3] = w[7]*u[2]-w[8]*u[3]
    nothing
end
tRng  = (0.0, 80.0)
dt    = 0.5
u0    = [0.3617, 0.9137, 1.3934]
wstar = [3.4884, 0.0969, 2.15, 10, 0.0969, 0.0581, 0.0969, 0.0775]
J     = length(wstar)
D     = length(u0)
## Generate data (one could use empircal data in practice)
ode = ODEProblem(
    f!, 
    u0, 
    tRng, 
    wstar
)
tt    = tRng[1]:dt:tRng[end]
Mp1   = length(tt)
Ustar = reduce(vcat, um' for um in solve(ode, saveat=dt).u)
snr   = 0.01
U     = Ustar.* exp.(snr*randn(size(Ustar)))
## Create wendy problem struct
wendyProb = WENDyProblem(tt, U, f!, J, Val(false)#=Nonlinear in parameters=#, Val(LogNormal)#=LogNormalNoise=#; ll=Logging.Info)
## Solve the wendy problm given an intial guess for the parameters 
w0 = wstar + 0.2*abs.(wstar)
relErr = norm(w0 - wstar) / norm(wstar)
@info "Initializing with Relative Coefficient Error = $(relErr)"
@info "Solving wendy problem ..."
@time what, iter, wits = solve(wendyProb, w0; alg=:fslsq, return_wits=true)
relErr = norm(what - wstar) / norm(wstar)
@info "Relative Coefficient Error = $(relErr)"
## plot the resutls 
odeprob = ODEProblem(f!, u0, tRng, what)
sol = solve(odeprob; saveat=dt)
Uhat = reduce(vcat, um' for um in sol.u)
plot(
    reduce(vcat, [
        scatter(x=tt, y=Uhat[:,d], name="estimate"),
        scatter(x=tt, y=U[:,d], name="data", mode="markers"),
        scatter(x=tt, y=Ustar[:,d], name="truth")
    ] for d in 1:D),
    Layout(title="Results",xaxis_title="time(s)", yaxis_title="State")
)