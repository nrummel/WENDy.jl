##
includet("/Users/user/Documents/School/becker-misc/GradientTests/julia/gradientCheck.jl")

## Learn how to use tool
q = 20*randn(10)
p = rand()
f(x) = dot(q, x) + p
g(x) = q 
p,_ = gradientCheck(f, g, w0; 
ll=Info, scaling=1e-8,makePlot=true)
p
## check the weighted residual
_RT = Rw(wendyProb, params)
RT0 = _RT(w0)
b = LowerTriangular(RT0) \ wendyProb.b₀
_r = rw(wendyProb, params)
_∇r = ∇rw(wendyProb, params)
v = rand(length(b))
f(w) = dot(_r(RT0,b,w), v)
grad(w) = _∇r(RT0,w)' * v
p,_ = gradientCheck(f, grad, w0; 
ll=Info, scaling=1e-8,makePlot=true)
p
## 
∇m(w) = _∇m(w)
p,_ = gradientCheck(m, ∇m, w0; 
ll=Info, scaling=1e-8,makePlot=true)
PlotlyJS.relayout!(p, title="Maholinobis Distance")
p
## 
∇m(w) = FiniteDiff.finite_difference_gradient(_m, w)
p,_ = gradientCheck(m, ∇m, w0; 
ll=Info, scaling=1e-8,makePlot=true)
PlotlyJS.relayout!(p, title="Maholinobis Distance (using FiniteDiff)")
p
## Check hessian
## Hessian building blocks
    b₀ = wendyProb.b₀
    sig = wendyProb.sig
    U = wendyProb.U
    V = wendyProb.V
    _RT = Rw(wendyProb, params)
    _r = rw(wendyProb, params)
    _∇r = ∇rw(wendyProb, params)
    jacwjacuf! = wendyProb.jacwjacuf!
    heswjacuf! = wendyProb.heswjacuf!
    r = zeros(K*D)
    ∇r = zeros(K*D,J)
    S⁻¹r = zeros(K*D)
    S⁻¹∇r = zeros(K*D,J)
    JwJuF = zeros(D,D,J,M)
    _Lbuf0 = zeros(K,D,D,J,M)
    _Lbuf1 = zeros(K,D,M,D,J)
    ∇L = zeros(K*D,M*D,J)
    ∇Sw_buff = zeros(K*D,K*D,J)
    ∇S = zeros(K*D,K*D,J)
    HwJuF = zeros(D,D,J,J,M)
    Hbuf0 = zeros(K,D,D,J,J,M)
    Hbuf1 = zeros(K,D,M,D,J,J)
    ∇²L = zeros(K*D,M*D,J,J)
    ∂ⱼL∂ᵢLᵀ = zeros(K*D,K*D)
    ∂ⱼᵢLLᵀ = zeros(K*D,K*D)
    ∂ᵢⱼS = zeros(K*D,K*D)
    tmp = zeros(K*D,K*D)
    ∂ᵢSS⁻¹∂ⱼS = zeros(K*D,K*D)



## run gradient check
e1 = zeros(J)
e1[1]=1
f(w) = dot(_∇m(w),e1)
function g(w)
    H0 = zeros(J,J)
    _Hm!(H0, wtest, b₀, sig, U, V, _RT, _r, _∇r, jacwjacuf!, heswjacuf!, r, ∇r, S⁻¹r, S⁻¹∇r, JwJuF, _Lbuf1, _Lbuf0, ∇L, ∇Sw_buff, ∇S, HwJuF, Hbuf0, Hbuf1, ∇²L, ∂ⱼL∂ᵢLᵀ, ∂ⱼᵢLLᵀ, ∂ᵢⱼS, tmp, ∂ᵢSS⁻¹∂ⱼS)
    H0
end
p,_ = gradientCheck(f, g, w0; 
ll=Info, scaling=1e-8,makePlot=true)
PlotlyJS.relayout!(p, title="Maholinobis Distance Hessian")
p
##
e1 = zeros(J)
e1[1]=1
f(w) = dot(_∇m(w),e1)
function g(w) 
    H =zeros(J,J)
    FiniteDiff.finite_difference_hessian!(H, _m, w)
    return H*e1
end
p,_ = gradientCheck(f, g, w0; 
ll=Info, scaling=1e-8,makePlot=true)
PlotlyJS.relayout!(p, title="Maholinobis Distance Hessian(finite Differences)")
p
