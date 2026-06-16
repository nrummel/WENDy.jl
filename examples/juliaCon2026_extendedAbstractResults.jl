if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    @info "More than WENDy.jl is necessary for plotting and data generation"
    using Pkg; 
    Pkg.activate(@__DIR__)
    Pkg.develop(path=joinpath(@__DIR__, ".."))
end
using Random, Logging, LinearAlgebra, Printf # stdlib
using PlotlyJS, Colors
using PlotlyKaleido: restart as restart_kaleido
# necessary to guarantee saving the LaTeX labels is done properly
restart_kaleido(plotly_version = "2.35.2", mathjax = true) 
using WENDy
using OrdinaryDiffEq: ODEProblem
using OrdinaryDiffEq: solve as solve_ode
## define priors
Random.seed!(1)
tRng   = (0.0, 10.0)
pstar  = [2.25, 7.0]
J      = length(pstar)
# Set initial guess for the parameters sufficiently far away from the true 
# parameters to demonstrate WENDy-MLE improved domain of convergence
p₀     = [3, 3] + randn(J) 
@info "p* = $pstar"
@info "p₀ = $p₀"
f!(du, u, p, t) = du[1] = p[1] * u[1] - p[2] * u[1]^2
params  = WENDyParameters(;Kmax=1000)
Mp1     = 101
tt      = range(tRng..., length=Mp1)
dt      = diff(tt)[1]
u₀      = [0.01]
D       = length(u₀)
ode     = ODEProblem(f!, u₀, tRng, pstar)
U_exact = reduce(vcat, um' for um in solve_ode(ode, saveat=dt).u)
nr      = 0.1
U       = U_exact .* exp.(nr*randn(Mp1,D)) # multiplicative log normal noise
# Get FS Error 
function fsErr(phat)
    u0star  = U_exact[1,:]
    odeprob = ODEProblem(f!, u0star, tRng, phat)
    sol     = solve_ode(odeprob; 
        reltol = params.optimReltol, abstol  = params.optimAbstol,
        saveat = dt,                 verbose = false
    )
    Uhat     = reduce(vcat, um' for um in sol.u)[:]
    fsRelErr = norm(Uhat - U_exact) / norm(U_exact)
    return fsRelErr, Uhat
end
# Output Error for comparison point
@info "========================================================================="
oeProb = WENDy.OutputErrorProblem(tt,U,f!, J)
puhat_OE = WENDy.solve(oeProb, p₀)
u0hat_OE = puhat_OE[J+1:end]
phat_OE = puhat_OE[1:J]
@info "Output Error"
relErr_OE = norm(phat_OE - pstar) / norm(pstar)
@info @sprintf "  Relation Coefficient error %.2g" relErr_OE
fsRelErr_OE, Uhat_OE = fsErr(phat_OE)
@info @sprintf "  Average Relative Forward Solver Error %.2g" fsRelErr_OE
# Build and solve wendy problem 
wendyProb = WENDyProblem(tt, U, f!, J; noiseDist=Val(LogNormal), ll=Warn)
phat_MLE, iters, P = solve(wendyProb, p₀, params; costFun=:wnll,return_wits=true);
# relative error 
relErr_MLE = norm(phat_MLE - pstar) / norm(pstar)
@info "WENDy MLE"
@info @sprintf "  Relation Coefficient error %.2g" relErr_MLE
fsRelErr_MLE, Uhat_MLE = fsErr(phat_MLE)
@info @sprintf "  Average Relative Forward Solver Error %.2g" fsRelErr_MLE
## covariance
S = WENDy.Covariance(wendyProb.data, params)
∇r = WENDy.JacobianResidual(wendyProb.data, params)
Ghat = ∇r(phat_MLE)  
Shat = S(phat_MLE; transpose=false, doChol=false)
cov_phat = (Ghat \ Shat) / (Ghat')
std_phat = sqrt.(diag(cov_phat))
@info "2σ confidence interval"
@info " p̂₁ : $(phat_MLE[1]) ± $(2*std_phat[1]) "
@info " p̂₂ : $(phat_MLE[2]) ± $(2*std_phat[2]) "
## Save example details to LaTeX 
open(joinpath(@__DIR__,"../paper", "LogisticGrowthExample.tex"), "w") do f
    write(f, """
   WENDy.jl obtains the A-MLE from an initial guess of \$\\params_0 = [$(@sprintf "%.3g, %.3g" p₀[1] p₀[2])]^\\top\$ and performing optimization of \\cref{eq:mle}. In this example, our estimated parameters have a coefficient relative error of $(@sprintf "%.2g" 100*relErr_MLE)\\% and a relative average forward solve error of $(@sprintf "%.2g" 100*fsRelErr_MLE)\\%. As a quick point of reference, a standard output error least squares algorithm resulted in a coefficient relative error $(@sprintf "%.2g" 100*relErr_OE)\\% and a relative average forward solve error of $(@sprintf "%.2g" 100*fsRelErr_OE)\\%. 
""")
end

## Visual Demonstration
PLOTLYJS_COLORS = [
    colorant"#1f77b4",  # muted blue
    colorant"#ff7f0e",  # safety orange
    colorant"#2ca02c",  # cooked asparagus green
    colorant"#d62728",  # brick red
    colorant"#9467bd",  # muted purple
    colorant"#8c564b",  # chestnut brown
    colorant"#e377c2",  # raspberry yogurt pink
    colorant"#7f7f7f",  # middle gray
    colorant"#bcbd22",  # curry yellow-green
    colorant"#17becf"   # blue-teal
]
CU_BOULDER_COLORS = [
    colorant"#000000", # black 
    colorant"#CFB87C", # gold 
    colorant"#565A5C", # dark grey 
    colorant"#A2A4A3", # light grey 
]
trs = AbstractTrace[]
push!( 
    trs,
    scatter(
        x=tt,
        y=U[:,1],
        name="\$\\{(\\mathrm{t}_m, \\mathrm{u}_m)\\}\$", 
        mode="markers" ,
        marker_color=PLOTLYJS_COLORS[1], 
        marker_opacity=0.5, 
        marker_size=10, 
        # legendgroup=d,
        # legendgrouptitle_text="State $d"
    )
)

push!(
    trs, 
    scatter(
        x=tt,
        y=Uhat_MLE,
        name="\$u(\\hat{\\mathbf{p}})\$",
        mode="lines",
        line=attr(
            color=CU_BOULDER_COLORS[2],
            width=5,
        )
    )
)

push!( 
    trs,
    scatter(
        x=tt,
        y=U_exact[:,1],
        name="\$u(\\mathbf{p}^*)\$", 
        mode="lines" ,
        line=attr(
            color=CU_BOULDER_COLORS[1],
            dash="dash",
            width=5,
            opacity=0.3
        ),
        # legendgroup=d,
    )
)
p1a = plot(
    trs, 
    Layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=attr(
            text="Logistic Growth", 
            x=0.5,
            xanchor="center",
            font_size=30
        ),
        # yaxis_type=yaxis_type,
        showlegend=true, 
        xaxis=attr(
            title=attr(
                text="\$t\$",
                font_size=20
            ),
            tick_font_size=15,
            showgrid=true, 
            zeroline=true
        ),
        yaxis=attr(
            title=attr(
                text="\$u\$",
                font_size=20
            ),
            tick_font_size=15,
            showgrid=true, 
            zeroline=true
        ),
        legend=attr(
            # x=.925,
            y=0.5,
            yanchor="center",
            font=(
                family="sans-serif",
                size=20,
                color="#000"
            ),
            bgcolor="#E2E2E2",
            bordercolor= "#636363",
            entrywidth= 150,        # Manually set this based on your longest equation
            entrywidthmode= "pixels", 
            borderpad= 20,          # Give extra "buffer" for tall fractions or exponents
            borderwidth= 1
        ),
        # margin_r= 150,
        hovermode="x unified"
    )
)



display(p1a)
# Save fig to file

savefig(
    p1a, 
    joinpath(@__DIR__, "../paper","LogisticGrowth_TrajectoryWithData.pdf"),
    height=400, 
    width=600
)