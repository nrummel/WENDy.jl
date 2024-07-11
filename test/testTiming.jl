##
_m_ = mw(wendyProb, params);
_∇m!_ = ∇mw(wendyProb, params);
_Hm!_ = Hmw(wendyProb, params);
##
@info "Cost function call "
@time _m_(w0)   
## Gradient computation 
∇m0 = zeros(J)
@info "  Finite difference gradient"
@time ∇m_fd = FiniteDiff.finite_difference_gradient(_m_, w0)
@info "  Analytic gradient"
@time _∇m!_(∇m0, w0); 
relErr = norm(∇m0 - ∇m_fd) / norm(∇m_fd)
@info "  relErr = $relErr"
## Hessian computation 
function Hm_fd!(H,w,p=nothing) 
    FiniteDiff.finite_difference_jacobian!(H, _∇m!_, w)
    @views H .= 1/2*(H + H')
    @views H .= Symmetric(H)
    nothing 
end 
##
H0 = zeros(J,J)
@info "==============================================="
@info "==== Comparing Hess ====="
@info "  Analytic Hessian "
@time _Hm!_(H0, w0)
@info "  Finite Differences Hessian from _m_"
Hfd = zeros(J,J)
@time FiniteDiff.finite_difference_hessian!(Hfd, _m_, w0)
@info "  Finite Differences Hessian from _∇m_"
Hfd2 = zeros(J,J)
@time Hm_fd!(Hfd2, w0)
@info "   Rel Error (analytic vs finite diff obj) $(norm(H0 - Hfd) / norm(Hfd))"
@info "   Rel Error (analytic vs finite diff res) $(norm(H0 - Hfd2) / norm(Hfd2))"
@info "   Rel Error (finite diff obj vs finite diff res) $(norm(Hfd - Hfd2) / norm(Hfd))"
@info "==============================================="