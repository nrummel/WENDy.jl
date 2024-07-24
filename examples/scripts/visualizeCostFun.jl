includet(joinpath(@__DIR__, "../src/WENDy.jl"))
##
ex = LOGISTIC_GROWTH;
params = WENDyParameters(;
    noiseRatio=0.2, 
    seed=2, 
    timeSubsampleRate=1,
    optimMaxiters=50, 
    optimTimelimit=30
)
wendyProb = WENDyProblem(ex, params; ll=Warn);
wTrue = wendyProb.wTrue
J = length(wTrue)
## solve with Maximum Likelihood Estimate
@info "Cost functions"
@time "create" begin 
    m = MahalanobisDistance(wendyProb, params);
    ∇m! = GradientMahalanobisDistance(wendyProb, params);
    Hm! = HesianMahalanobisDistance(wendyProb, params);
end
@time "m" m(wTrue)
@time "∇m!" ∇m!(wTrue)
@time "Hm!" Hm!(wTrue)
##
@info "Forward Solve L2 loss"
@time "create" begin 
    l2(w::AbstractVector{<:Real}) = _l2(w,wendyProb.U,ex)
    ∇l2!(g::AbstractVector{<:Real},w::AbstractVector{<:Real}) = ForwardDiff.gradient!(g, l2, w) 
    Hl2!(H::AbstractMatrix{<:Real},w::AbstractVector{<:Real}) = ForwardDiff.hessian!(H, l2, w) 
end
@info "Run once so that compilation time is isolated here"
@time "l2 loss" l2(wTrue)
g_fs = similar(wTrue)
H_fs = zeros(J,J)
@time "gradient of l2" ∇l2!(g_fs,wTrue)
@time "hessian of l2" Hl2!(H_fs,wTrue);
all(g_fs .== 0) && @warn "Auto diff failed on fs"
all(H_fs .== 0) && @warn "Auto diff failed on fs"
##
del = 5
step = 0.1
N = Int(ceil(2*del/step)) + 1
xx = zeros(N,N)
yy = zeros(N,N)
mm = zeros(N,N)
##
Progress
for (n,w1) in enumerate(range(wTrue[1]-del, step=step, stop=wTrue[1]+del))
    for (nn,w2) in enumerate(range(wTrue[2]-del, step=step, stop=wTrue[2]+del))
        xx[n,nn] = w1
        yy[n,nn] = w2
        # mm[n,nn] = m([w1,w2])
    end
end
p = plotjs(
    PlotlyJS.surface(
        x=xx,
        y=yy,
        z=log10.(mm)
    ),
    Layout(
        # zaxis=attr(type="log")
    )
)
