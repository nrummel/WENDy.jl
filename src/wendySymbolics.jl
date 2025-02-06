"""
(private) Build a function for the the rhs specialized to 
the case of lognormal noise. This means the original rhs changes to 
    f̃(x) = f(x) ./ x 
"""
function _getf_sym(f!::Function, D::Int, J::Int, ::Val{LogNormal})
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    mymap = Dict([xd =>exp(xd) for xd in x])
    fu =[eq / x  for (x,eq) in zip(x,dx)]
    return [simplify(substitute(fud, mymap)) for fud in fu]
end
""" (private) Build a function for the the rhs specialized to normal noise """
function _getf_sym(f!::Function, D::Int, J::Int, ::Val{Normal})
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    return dx
end
""" Build a function for the the rhs """ 
function _getf(f!::Function, D::Int, J::Int, ::Val{DistType}) where {DistType<:Distribution}
    @variables p[1:J] x[1:D] t
    f_sym = _getf_sym(f!, D, J, Val(DistType))
    _,f! = build_function(f_sym, x, p, t;expression=false)
    return f!
end
""" Build a function for the jacobian wrt to the parameters of the rhs """ 
function _get∇ₚf(f!::Function, D::Int, J::Int) 
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₚf = jacobian(dx, p)
    return build_function(∇ₚf, x, p, t; expression=false)[end]
end
""" Build a function for the jacobian wrt to the state of the rhs"""
function _get∇ₓf(f!::Function, D::Int, J::Int) 
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₓf = jacobian(dx, x)
    return build_function(∇ₓf, x,p,t; expression=false)[end]
end
""" Build a function for the jacobian wrt to the parameters of the jacobian wrt to the state of the rhs"""
function _get∇ₚ∇ₓf(f!::Function, D::Int, J::Int)
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₓf = jacobian(dx, x)
    ∇ₚ∇ₓf = jacobian(∇ₓf, p)
    build_function(∇ₚ∇ₓf, x,p,t; expression=false)[end]
end
""" Build a function for the hessian wrt the parameters of the rhs """
function _getHₚf(f!::Function, D::Int, J::Int)
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₚf = jacobian(dx, p)
    Hₚf = jacobian(∇ₚf, p)
    return build_function(Hₚf, x,p,t; expression=false)[end]
end
""" Build a function for the hessian wrt the parameters of the jacobian wrt to the state of the rhs """
function _getHₚ∇ₓf(f!::Function, D::Int, J::Int)
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₓf = jacobian(dx, x)
    ∇ₚ∇ₓf = jacobian(∇ₓf, p)
    Hₚ∇ₓf = jacobian(∇ₚ∇ₓf, p)
    return build_function(Hₚ∇ₓf, x,p,t; expression=false)[end]
end