Here we give three examples of ODE's and how to use WENDy to estimate the parameters. 

## Logistic Growth
The logistic growth equation is a well-known differential equation that has an exact solution. Classically, it was developed to describe population growth rate $$r$$ with a carrying capacity $$K$$, $$\dot{u} = ru(1-\tfrac{u}{K})$$. We have reparameterized this equation to be amenable to optimization. 
```math 
\dot{u}=p_1 u+p_2 u^2
```
For the sake of the example we will generate data by numerically solving the ODE, and then corrupting with noise. We do this through the standard library and OrdinaryDiffEq.jl. 

```@example logistic
using Random, LinearAlgebra
using OrdinaryDiffEq: ODEProblem
using OrdinaryDiffEq: solve as solve_ode
using WENDy
```

Now, we define the right hand side of the ODE, initial conditions, the time domain, and specify the true parameters:

```@example logistic
function f!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1]^2
    nothing
end
tRng = (0.0, 10.0)
dt = 0.01
u₀ = [0.01]
pstar = [1.0, 1.0]
J = length(pstar)

ode = ODEProblem(
    f!, 
    u₀, 
    tRng, 
    pstar
)
tt = tRng[1]:dt:tRng[end]
Ustar = reduce(vcat, um for um in solve_ode(ode, saveat=dt).u)
nr = 0.1 # noise ratio
U = Ustar + nr*randn(size(Ustar)) # corrupting the data with noise
nothing # hide
```
Now, that we have the data we are ready to build a WENDy Problem and then solve it. 
```@example logistic
wendyProb = WENDyProblem(
    tt, 
    U, 
    f!, 
    J
)
nothing # hide
```
The algorithm requires an initial guess for the parameter values. From this initial guess, it will then approximate the maximum likelihood estimator.
```@example logistic
p₀ = [0.5, 0.5]
@time phat = solve(wendyProb, p₀)
@show phat 
nothing # hide
```
The efficiency of the solver can be improved by specifying that the function $$f$$ is linear in parameters. This is done with the optional argument to the WENDyProblem. 
```@example logistic
wendyProb_linear = WENDyProblem(
    tt, 
    U, 
    f!, 
    J, 
    linearInParameters=Val(true), # f! is linear in parameters 
)
@time (wendyProb_linear, p₀)
nothing # hide
``` 
This problem can be visualized by looking at the data, the true solution of the ODE and the solution given by the estimated parameters.
```@example logistic
using PlotlyJS
odeprob = ODEProblem(f!, u₀, tRng, phat)
sol = solve_ode(odeprob; saveat=dt)
Uhat = reduce(vcat, um' for um in sol.u)
plot(
    [
        scatter(x=tt, y=U[:], name="data", mode="markers", marker_color="blue", marker_opacity=0.5),
        scatter(x=tt, y=Ustar[:], name="truth", line_color="blue", line_width=3),
        scatter(x=tt, y=Uhat[:], name="estimate", line_dash="dash", line_color="black", line_width=3),
    ],
    Layout(title="Logistic Growth",xaxis_title="time(s)", yaxis_title="u(t)")
)
``` 
## Goodwin 
A simple example of a system of differential equations which is nonlinear in parameters is the Goodwin model which describes negative feedback control processes . In particular there is a Hill function in the equation for $$u_1$$. The parameter $$p_3$$ appears in the denominator and $$p_4$$ is the Hill coefficient, and thus this serves as an example of how nonlinearity can effect the performance of the WENDy algorithm. 
```math 
\begin{aligned}
    \dot{u}_1 &= \frac{p_1}{2.15 + p_3 u_3^{p_4}} - p_2  u_1 \\
    \dot{u}_2 &= p_5u_1- p_6u_2 \\
    \dot{u}_3 &= p_7u_2-p_8u_3
\end{aligned}
```
We again can generate data to see how WENDy can estimate parameters. In this case it is realistic to for there to be LogNormal measurement error, so we choose this to be the distribution of the noise.
```@example goodwin
using Random, Logging, LinearAlgebra # hide
using PlotlyJS # hide
using WENDy # hide
using OrdinaryDiffEq: ODEProblem # hide
using OrdinaryDiffEq: solve as solve_ode # hide
## Define rhs, ic, time domain, and length of parameters
function f!(du, u, p, t)
    du[1] = p[1] / (2.15 + p[3] * u[3]^p[4]) - p[2] * u[1]
    du[2] = p[5]*u[1]- p[6]*u[2]
    du[3] = p[7]*u[2]-p[8]*u[3]
    nothing
end
tRng  = (0.0, 80.0)
dt    = 0.5
u₀    = [0.3617, 0.9137, 1.3934]
pstar = [3.4884, 0.0969, 1.0, 10, 0.0969, 0.0581, 0.0969, 0.0775]
J     = length(pstar)
D     = length(u₀)

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
U     = Ustar .* exp.(nr*randn(size(Ustar))) # LogNormal Noise
nothing # hide
```
Now we are ready to build the WENDy Problem and solve it for the unknown parameters. In this case, it is important to update the default hyper-parameters for the test function radii. In general, pick values for the test function radii that are larger than the step size, and smaller than the whole domain. In this case, the maximum test function radius is too small.  
```@example goodwin
params = WENDyParameters(
    radiusMinTime=dt, 
    radiusMaxTime=tRng[end]/5
); # be sure to set the min and max testFunction radii to something reasonable
wendyProb = WENDyProblem(
    tt, 
    U, 
    f!, 
    J;
    noiseDist=Val(LogNormal), # multiplicative LogNormal noise
    params=params, 
);
J = length(pstar)
nothing # hide
```
Here we are perturbing slightly from the true values for the initial guess. In practice one would provide this without knowledge of the true parameters. 
```@example goodwin
p₀ =  [3.0, 0.1, 4, 12, 0.1, 0.1, 0.1, 0.1]
phat = solve(wendyProb, p₀)
nothing # hide
```
We can visualize the quality of our estimated parameters by forward simulating and then plotting. 
```@example goodwin
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
```
## SIR 
The susceptible-infected-recovered (SIR) model is pervasive in epidemiology. This system describes an extension that allows for time delayed immunity (TDI) for parasitic deceases where there is a common source for infection.
```math
\begin{aligned}
    \dot{u}_{1} &= -p_{1}  u_{1} + p_{3}  u_{2} + \tfrac{p_1 e^{-p_1  p_2}}{1 - e^{-p_1  p_2}} u_{3} \\
    \dot{u}_{2} &= p_{1}  u_{1} - p_{3}  u_{2} \\
    & - p_{4}  (1 - e^{-p_{5}  t^2})  u_{2} \\
    \dot{u}_{3} &= p_{4}  (1 - e^{-p_{5}  t^2})  u_{2} - \tfrac{p_1 e^{-p_1  p_2}}{1 - e^{-p_1  p_2}}  u_{3}
\end{aligned}
```
Again we can build test data by first simulating with true parameters, and then corrupting with multiplicative LogNormal noise.
```@example sir
using Random, Logging, LinearAlgebra # hide
using PlotlyJS # hide
using WENDy # hide
using OrdinaryDiffEq: ODEProblem # hide
using OrdinaryDiffEq: solve as solve_ode # hide
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
nothing # hide
```

The time domain `[0,50]` so in this case it is best to adjust the parameters for the radii of the test functions. Also, we can define constraints for the parameters considered in optimization.
```@example sir
params = WENDyParameters(
    radiusMinTime  = 0.1,
    radiusMaxTime  = 25.0
)
constraints = [
    (1e-4,1.0), 
    (1e-4,2.0),
    (1e-4,1.0),
    (1e-4,1.0),
    (1e-4,1.0),
]
wendyProb = WENDyProblem(
    tt, 
    U, 
    f!, 
    J;
    noiseDist=Val(LogNormal), # LogNormalNoise
    params=params,
    constraints=constraints
)
nothing # hide
```
In this case we can perturb from truth as an example initial guess for the parameters. In practice, this guess would not be made from prior information of the true parameters.  
```@example sir
p₀ = pstar + 0.5*randn(J).*abs.(pstar) 
phat = solve(wendyProb, p₀)
```
We can visualize the quality of our estimated parameters by forward simulating and then plotting. 
```@example sir
odeprob = ODEProblem(f!, u₀, tRng, phat)
sol = solve_ode(odeprob; saveat=dt)
Uhat = reduce(vcat, um' for um in sol.u)
colors = ["red", "blue", "green"]
plot(
    reduce(vcat, [
        scatter(x=tt, y=U[:,d], marker_color=colors[d], name="data", mode="markers", legendgroup=d, marker_opacity=0.5 ),
        scatter(x=tt, y=Uhat[:,d],line_color=colors[d], line_dash="dash", name="estimate", legendgroup=d, legendgrouptitle_text="u[$d]"),
        scatter(x=tt, y=Ustar[:,d],line_color=colors[d], name="truth", legendgroup=d )
    ] for d in 1:D),
    Layout(title="SIR",xaxis_title="time(s)", yaxis_title="State")
)
```
