includet(joinpath(@__DIR__, "../src/WENDy.jl"))
includet(joinpath(@__DIR__, "../examples/hindmarshRose.jl"))
using FiniteDiff, StaticArrays, Printf
##
ex = HINDMARSH_ROSE;

params = WENDyParameters(;
    noiseRatio=0.05, 
    seed=1, 
    timeSubsampleRate=4,
)
ρ = .1 
wendyProb_l = WENDyProblem(ex, params; ll=Warn, forceNonlinear=false);
wendyProb_nl = WENDyProblem(ex, params; ll=Warn, forceNonlinear=true);
wTrue = wendyProb_l.wTrue
J = length(wTrue)
w0 = wTrue + ρ * abs.(wTrue) .* randn(J);
## solve with Maximum Likelihood Estimate
@time "create m funs" begin 
    m_l = WeakNLL(wendyProb_l, params);
    ∇m!_l = GradientWeakNLL(wendyProb_l, params);
    Hm!_l = HesianWeakNLL(wendyProb_l, params);
    m_nl = WeakNLL(wendyProb_nl, params);
    ∇m!_nl = GradientWeakNLL(wendyProb_nl, params);
    Hm!_nl = HesianWeakNLL(wendyProb_nl, params);
    Hm!_fd = (H,w) -> FiniteDiff.finite_difference_jacobian!(H, ∇m!_l, w)
end
##
w0s = SVector(w0...)
g_l = similar(w0)
g_nl = similar(w0)
H_l = zeros(J,J)
H_nl = similar(H_l)
g_l_s = @MVector zeros(J)
g_nl_s = @MVector zeros(J)
H_l_s = @MMatrix zeros(J,J)
H_nl_s = @MMatrix zeros(J,J)
H_l_s_fd = @MMatrix zeros(J,J)
@time "Linear gradient" ∇m!_l(g_l, w0)
@time "Nonlinear gradient" ∇m!_nl(g_nl, w0)
@time "Linear hessian" Hm!_l(H_l, w0)
@time "Nonlinear hessian" Hm!_nl(H_nl, w0)
@time "Linear gradient static" ∇m!_l(g_l_s, w0s)
@time "Nonlinear gradient static" ∇m!_l(g_nl_s, w0s)
@time "Linear hessian static" Hm!_l(H_l_s, w0s)
@time "Nonlinear hessian static" Hm!_nl(H_nl_s, w0s)
@time "Linear hessian Finite Difference static" Hm!_nl(H_l_s_fd, w0s)
##
@info "(comparative) relErr in ∇L: $(norm(Hm!_nl.∇L!.∇L - Hm!_l.∇L!.∇L) / norm(Hm!_l.∇L!.∇L))"
@info "(comparative) relErr in r: $(norm(Hm!_nl.r!.r - Hm!_l.r!.r) / norm(Hm!_l.r!.r))"
@info "(comparative) relErr in ∇r: $(norm(Hm!_nl.∇r!.∇r - Hm!_l.∇r!.∇r) / norm(Hm!_l.∇r!.∇r))"
@info "(comparative) relErr in L: $(norm(Hm!_nl.R!.L!.L - Hm!_l.R!.L!.L) / norm(Hm!_l.R!.L!.L))"
##
@info "(nl vs l) relErr in grad: $(norm(g_nl - g_l) / norm(g_l))"
@info "(nl vs l) relErr in hess: $(norm(H_nl - H_l) / norm(H_l))"
@info "(l vs l_s) relErr in grad: $(norm(g_l_s - g_l) / norm(g_l))"
@info "(l vs l_s) relErr in hess: $(norm(H_l_s - H_l) / norm(H_l))"
@info "(nl vs nl_s) relErr in nl grad: $(norm(g_nl_s - g_nl) / norm(g_nl))"
@info "(nl vs nl_s) relErr in nl hess: $(norm(H_nl_s - H_nl) / norm(H_nl))"
##
compResults = NamedTuple(
    lin=>NamedTuple(
        name=>NamedTuple([:what=>Ref{AbstractVector}(Vector(undef,J)), :wits=>Ref{AbstractMatrix}(Matrix(undef,0,0))])
    for name in keys(algos)) 
for lin in [:linear, :nonlinear])
for (linear, _m_, _∇m!_,_Hm!_, wendyProb) in [
    (:linear, m_l, ∇m!_l, Hm!_l,wendyProb_l), 
    (:nonlinear,m_nl,∇m!_nl,Hm!_nl,wendyProb_nl)
]
    @info "Building Forward Solve L2 loss so that compile time does not affect results"
    l2(w::AbstractVector{<:Real}) = _l2(w,wendyProb.U,ex)
    ∇l2!(g::AbstractVector{<:Real},w::AbstractVector{<:Real}) = ForwardDiff.gradient!(g, l2, w) 
    Hl2!(H::AbstractMatrix{<:Real},w::AbstractVector{<:Real}) = ForwardDiff.hessian!(H, l2, w) 
    ##
    @info "Run once so that compilation time is isolated here"
    g_fs = zeros(wendyProb_nl.J)
    H_fs = zeros(wendyProb_nl.J,wendyProb_nl.J)
    @time "l2 loss" l2(w0)
    @time "gradient of l2" ∇l2!(g_fs,w0)
    @time "hessian of l2" Hl2!(H_fs,w0);
    @assert !all(g_fs .== 0) "Auto diff failed on fs"
    @assert !all(H_fs .== 0) "Auto diff failed on fs"
    for (name, algo) in zip(keys(algos),algos)
        @info "Running $name"
        alg_dt = @elapsed begin 
            (what, iters, wits) = try 
                algo(wendyProb, params, w0, _m_, _∇m!_,_Hm!_,l2,∇l2!,Hl2!; return_wits=true) 
            catch
                (NaN * ones(J), NaN, nothing) 
            end 
        end
        cl2 = norm(what - wTrue) / norm(wTrue)
        fsl2 = try
            forwardSolveRelErr(wendyProb, what)
        catch 
            NaN 
        end
        mDist = try
            m_nl(what)
        catch 
            NaN 
        end
        @info """
        Results:
            dt = $alg_dt
            cl2 = $(@sprintf "%.2g" cl2*100)%
            fsl2 = $(@sprintf "%.2g" fsl2*100)%
            mDist = $mDist
            iters = $iters
        """
        compResults[linear][name].what[] = what 
        compResults[linear][name].wits[] = wits
    end
end
