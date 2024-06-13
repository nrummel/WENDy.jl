using GradDescent, LinearAlgebra

A = rand(5, 2)
b = rand(5)
obj(w) = 1/2* norm(A*w -b)^2 
grad(w) = A'*(A*w-b) 
##
w0 = rand(2)
wim1 = w0 
wi = w0
opt = Adam(α=1.0)
tol = 1e-4
epochs = 1000 
for i in 1:epochs
    # here we use automatic differentiation to calculate 
    # the gradient at a value
    # an analytically derived gradient is not required
    g = grad(wim1)
    δ = update(opt, g)
    wi = wim1 - δ
    @info "$(obj(wi))"
    if norm(wim1 - wi)/norm(wim1) < tol 
        @info "woot"
        break 
    end 
    wim1 = wi 
end
