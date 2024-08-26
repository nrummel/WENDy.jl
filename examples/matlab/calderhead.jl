using LinearAlgebra, OrdinaryDiffEq, PlotlyJS, Distributions, ProgressMeter
using PlotlyJS: plot as plotjs
function rhs!(du,u,w,t)    
    du[1] = w[1] / (36 + w[2]*u[2]) - w[3]; 
    du[2] = w[4] * u[1] - w[5];
    nothing
end 

tt = 0:0.5:60;
u0 = [7,-10]; 

wTrue = Float64[72.0, 1.0, 2.0, 1.0, 1.0];
function solve(w)
    odeprob = ODEProblem{true, SciMLBase.FullSpecialize}(rhs!, u0, (tt[1],tt[end]), w)
    t_step = tt[2]-tt[1]
    sol = OrdinaryDiffEq.solve(
        odeprob, Rosenbrock23(); 
        saveat=t_step, verbose=false
    ); 
    Uhat = reduce(hcat, sol.u)
end

# add noise
U_exact = solve(wTrue)
vareps = 0.5 ;
epsx = sqrt(vareps) * randn(size(tt));
epsy = sqrt(vareps) * randn(size(tt));
U = similar(U_exact)
U[1,:] = U_exact[1,:] + epsx;
U[2,:] = U_exact[2,:] + epsy;
plot(
    vcat([[
        scatter(
            x=tt,
            y=U[d,:],
            mode="markers",
            name="Û[$d]"
        ),
        scatter(
            x=tt,
            y=U_exact[d,:],
            mode="lines",
            name="U*[$d]"
        ),
    ] for d in 1:size(U,1)] ...)

)

##loglikelihood S log-posterior under flat priors
xM = U[1,:]*ones(size(tt))';
yM = ones(size(tt))*U[2,:]';
mux = U_exact[1,:]*ones(size(tt))'
muy = ones(size(tt))*U_exact[2,:]'
pdfx = similar(xM)
pdfy = similar(xM)
for ix in eachindex(xM)
    pdfx[ix] = pdf(Normal(mux[ix], sqrt(vareps)), xM[ix])
    pdfy[ix] = pdf(Normal(muy[ix], sqrt(vareps)), yM[ix])
end
logp1 = - log.(pdfx) - log.(pdfy);
plotjs(
    surface(x=xM,y=yM,z=logp1),
    Layout(scene=attr(
        xaxis_title="x in time",
        yaxis_title="y in time",
        zaxis_title="log likelihood"
    ))
)
##


### LOOPS OVER k3 and k4:
function _their_nll(w)
    Uhat = solve(w); 
    x = Uhat[1,:];
    y = Uhat[2,:]; 
    logpx_ij = -sum((U[1,:]-x).^2)/(2*vareps) - n/2*log(2*pi*vareps);
    logpy_ij = -sum((U[2,:]-y).^2)/(2*vareps) - n/2*log(2*pi*vareps);
    -(logpx_ij+logpy_ij)
end 
function _my_nll(w)
    Uhat = solve(w); 
    M = size(Uhat, 2)
    @views sum(
        1/2 * dot(
            (Uhat[:,m] - U[:,m]), 
            diagm(1.0 ./ vareps .* ones(2)), 
            (Uhat[:,m] - U[:,m]))
        + 1/2 * log(2*pi*2*vareps)
    for m in 1:M)
end 

IX = (3,4)
del = 3
step = 0.1
kigrid = wTrue[IX[1]]-del:step:wTrue[IX[1]]+del;
kjgrid = wTrue[IX[2]]-del:step:wTrue[IX[2]]+del;
FSNLL= zeros(length(kigrid),length(kjgrid))
n = length(tt)
@showprogress "Computing nll on grid..." for (i,ki) in enumerate(kigrid), (j,kj) in enumerate(kjgrid)
    global FSNLL
    w = copy(wTrue)
    w[IX[1]] = ki
    w[IX[2]] = kj
    FSNLL[i,j] = _my_nll(w)
end

p_fsnll = plotjs(
    [
        scatter3d(
            x=[wTrue[IX[1]]],
            y=[wTrue[IX[2]]],
            z=[_my_nll(wTrue)],
            text=["truth"],
            mode="markers+text"
        ),
        surface(
            x=kigrid, 
            y=kjgrid, 
            z=FSNLL,
            cauto=false,
            cmin=minimum(log10.(FSNLL)) - 1,
            cmax=maximum(log10.(FSNLL)) + 1,
            showscale=false        
        )
    ],
    Layout(
        scene=attr(
            xaxis_title="k_$(IX[1])",
            yaxis_title="k_$(IX[2])",
            zaxis_title="-ℒ",
            zaxis_type="log",
            zaxis_autorange="reversed",
            camera_eye=attr(x=-2, y=1, z=.1),
        ),
        title_text="Forward Solve<br>Negative Log-Likelihood",
        title_y=.9,
        title_font_size=36,
        title_yanchor="center",
        margin=attr(t=30, r=0, l=20, b=10),
    )
)
##
save_dir = "/Users/user/Documents/School/WSINDy/NonLinearWENDyPaper/fig/MULTIMODAL"
PlotlyJS.savefig(
    p_fsnll,
    joinpath(save_dir, "MULTIMODAL_fsnllCostSpace.png"),
    width=800, height=700
)
## 
ex = WENDy.EmpricalWENDyData(
    "MULTIMODAL",
    MULTIMODAL_SYSTEM,
    tt, U
);   
params = WENDyParameters();
wendyProb = WENDyProblem(ex, params);
nll = WENDy.SecondOrderCostFunction(
    WENDy.MahalanobisDistance(wendyProb, params),
    WENDy.GradientMahalanobisDistance(wendyProb, params),
    WENDy.HesianMahalanobisDistance(wendyProb, params)
);
##
del = 3
step = 0.1
kigrid = wTrue[IX[1]]-del:step:wTrue[IX[1]]+del;
kjgrid = wTrue[IX[2]]-del:step:wTrue[IX[2]]+del;
NLL= zeros(length(kigrid),length(kjgrid))
@showprogress "Computing nll on grid..." for (i,ki) in enumerate(kigrid), (j,kj) in enumerate(kjgrid)
    global NLL
    w = copy(wTrue)
    w[IX[1]] = ki
    w[IX[2]] = kj
    NLL[i,j] = nll.f(w)
end
##
p_nll = plotjs(
    [
        scatter3d(
            x=[wTrue[IX[1]]],
            y=[wTrue[IX[2]]],
            z=[nll.f(wTrue)],
            text=["truth"],
            mode="markers+text"
        ),
        surface(
            x=kigrid, 
            y=kjgrid, 
            z=NLL,
            cauto=false,
            cmin=minimum(log10.(NLL)) - 1,
            cmax=maximum(log10.(NLL)) + 1,
            showscale=false        
        )
    ],
    Layout(
        scene=attr(
            xaxis_title="k_$(IX[1])",
            yaxis_title="k_$(IX[2])",
            zaxis_title="-ℒ",
            zaxis_type="log",
            zaxis_autorange="reversed",
            camera_eye=attr(x=-1, y=2, z=.1),
        ),
        title_text="Negative Log-Likelihood",
        title_y=.9,
        title_font_size=36,
        title_yanchor="center",
        margin=attr(t=30, r=0, l=20, b=10),
    )
)
savefig(
    p_nll,
    joinpath(save_dir, "MULTIMODAL_nllCostSpace.png"),
    width=800, height=700
)
## 
function _l2(w)
    Uhat = solve(w)
    M = size(Uhat,2)
    if M != size(U,2)
        return NaN 
    end
    @views sum(
        sum((Uhat[:,m] - U[:,m]).^2)
        for m in 1:M
    )
end
del = 3
step = 0.1
kigrid = wTrue[IX[1]]-del:step:wTrue[IX[1]]+del;
kjgrid = wTrue[IX[2]]-del:step:wTrue[IX[2]]+del;
L2 = zeros(length(kigrid),length(kjgrid))
@showprogress "Computing nll on grid..." for (i,ki) in enumerate(kigrid), (j,kj) in enumerate(kjgrid)
    global L2
    w = copy(wTrue)
    w[IX[1]] = ki
    w[IX[2]] = kj
    L2[i,j] = _l2(w)
end
##
p_l2 = plotjs(
    [
        scatter3d(
            x=[wTrue[IX[1]]],
            y=[wTrue[IX[2]]],
            z=[_l2(wTrue)],
            text=["truth"],
            mode="markers+text"
        ),
        surface(
            x=kigrid, 
            y=kjgrid, 
            z=L2,
            cauto=false,
            cmin=minimum(filter!(x->!isnan(x), log10.(L2[:]))) - 1,
            cmax=maximum(filter!(x->!isnan(x), log10.(L2[:]))) + 1,
            showscale=false        
        )
    ],
    Layout(
        scene=attr(
            xaxis_title="k_$(IX[1])",
            yaxis_title="k_$(IX[2])",
            zaxis_title="||⋅||₂",
            zaxis_type="log",
            zaxis_autorange="reversed",
            camera_eye=attr(x=-1, y=2, z=.1),
        ),
        title_text="Forward Sim L2 Loss",
        title_y=.9,
        title_font_size=36,
        title_yanchor="center",
        margin=attr(t=30, r=0, l=20, b=10),
    )
)
savefig(
    p_l2,
    joinpath(save_dir, "MULTIMODAL_l2CostSpace.png"),
    width=800, height=700
)
## 
w0 = wTrue + .50 * randn(size(wTrue)) .* abs.(wTrue)
what, iters, wits = WENDy.tr_Optim(nll, w0, params, return_wits=true)
_I = size(wits,2)
##
del = 3
step = 0.1
w_imin, w_imax = extrema(wits[IX[1], :])
w_jmin, w_jmax = extrema(wits[IX[2], :])
kigrid = floor(min(w_imin, wTrue[IX[1]]))-del:step:ceil(max(w_imax,wTrue[IX[1]]))+del;
kjgrid = floor(min(w_jmin, wTrue[IX[2]]))-del:step:ceil(max(w_jmax,wTrue[IX[2]]))+del;
NLL_iter= zeros(length(kigrid),length(kjgrid))
@showprogress "Computing nll on grid..." for (i,ki) in enumerate(kigrid), (j,kj) in enumerate(kjgrid)
    global NLL_iter
    w = copy(wTrue)
    w[IX[1]] = ki
    w[IX[2]] = kj
    NLL_iter[i,j] = nll.f(w)
end
##
shift = 10
p_nll_iter = plotjs(
    [
        scatter3d(
            x=[wTrue[IX[1]]],
            y=[wTrue[IX[2]]],
            z=[nll.f(wTrue) + shift],
            text=["truth"],
            mode="markers+text",
            name="Truth"
        ),
        scatter3d(
            x=[wits[IX[1],i] for i in 1:_I],
            y=[wits[IX[2],i] for i in 1:_I],
            z=[nll.f(wits[:,i]) + shift for i in 1:_I],
            # text=["Iter $i" for i in 1:_I],
            hovertext=["Iter $i" for i in 1:_I],
            mode="markers",
            name ="iterations"
        ),
        surface(
            x=kigrid, 
            y=kjgrid, 
            z=NLL_iter .+ shift,
            cauto=false,
            cmin=minimum(log10.(NLL_iter)) - 1,
            cmax=maximum(log10.(NLL_iter)) + 1,
            showscale=false        
        )
    ],
    Layout(
        scene=attr(
            xaxis_title="k_$(IX[1])",
            yaxis_title="k_$(IX[2])",
            zaxis_title="-ℒ",
            zaxis_type="log",
            zaxis_autorange="reversed",
            camera_eye=attr(x=-1, y=2, z=.1),
        ),
        title_text="Negative Log-Likelihood",
        title_y=.9,
        title_font_size=36,
        title_yanchor="center",
        margin=attr(t=30, r=0, l=20, b=10),
    )
)
p_nll_iter
##
PlotlyJS.savefig(
    p_nll_iter, 
    joinpath(save_dir, "MULTIMODAL_nllCostSpace_iters.png"),
    width=800, height=700
)