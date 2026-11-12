## Every generated function goes through _build: common subexpressions are pulled out (cse=true), which keeps the
## generated code, and its compile time, small for large expressions such as the second derivatives
_build(ex, args...) = build_function(ex, args...; expression=false, cse=true)
"""
(private) Build a function for the the rhs specialized to 
the case of lognormal noise. This means the original rhs changes to 
    f̃(x) = f(x) ./ x 
"""
function _getf_sym(f!::Function, D::Int, J::Int, ::Type{LogNormal})
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    mymap = Dict([xd =>exp(xd) for xd in x])
    fu =[eq / x  for (x,eq) in zip(x,dx)]
    return [simplify(substitute(fud, mymap)) for fud in fu]
end
""" (private) Build a function for the the rhs specialized to normal noise """
function _getf_sym(f!::Function, D::Int, J::Int, ::Type{Normal})
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    return dx
end
""" Build a function for the the rhs """ 
function _getf(f!::Function, D::Int, J::Int, noiseDist::Type{<:Distribution})
    @variables p[1:J] x[1:D] t
    f_sym = _getf_sym(f!, D, J, noiseDist)
    _,f! = _build(f_sym, x, p, t)
    return f!
end
""" Build a function for the jacobian wrt to the parameters of the rhs """ 
function _get∇ₚf(f!::Function, D::Int, J::Int) 
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₚf = jacobian(dx, p)
    return _build(∇ₚf, x, p, t)[end]
end
""" Build a function for the jacobian wrt to the state of the rhs"""
function _get∇ₓf(f!::Function, D::Int, J::Int) 
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₓf = jacobian(dx, x)
    return _build(∇ₓf, x, p, t)[end]
end
""" Build a function for the jacobian wrt to the parameters of the jacobian wrt to the state of the rhs"""
function _get∇ₚ∇ₓf(f!::Function, D::Int, J::Int)
    @variables p[1:J] x[1:D] t
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₓf = jacobian(dx, x)
    ∇ₚ∇ₓf = jacobian(∇ₓf, p)
    _build(∇ₚ∇ₓf, x, p, t)[end]
end
"""
The p-hessian of the scalar ψ(x, p, t; a, Ω) = ⟨a, f(x, p, t)⟩ + ⟨Ω, ∇ₓf(x, p, t)⟩ (a ∈ ℝᴰ, Ω ∈ ℝᴰˣᴰ), which is all the
weak likelihood needs of ∇²ₚf and ∇²ₚ∇ₓf (see _∂²ψ). Returns ∇²ₚψ!(out, x, p, t, a, Ω), writing the entries of the upper
triangle that are not structurally zero, and their linear indices in a J × J matrix: J(J+1)/2 outputs at most, instead
of D·J² + D²·J² for the two tensors. Every entry is differentiated and the structural zeros dropped, because
Symbolics.hessian_sparsity misses entries of a power with a parameter exponent (∂²/∂p₄² of u^p₄ in Goodwin).
"""
function _get∇²ₚψ(f!::Function, D::Int, J::Int)
    @variables p[1:J] x[1:D] t a[1:D] Ω[1:D,1:D]
    dx = Vector(undef, D)
    f!(dx, x, p, t)
    ∇ₓf = jacobian(dx, x)
    ψ = sum(a[d]*dx[d] for d in 1:D) + sum(Ω[d₂,d₁]*∇ₓf[d₂,d₁] for d₂ in 1:D, d₁ in 1:D)
    ∇ₚψ = Symbolics.gradient(ψ, collect(p))
    entries = [(i, j, Symbolics.derivative(∇ₚψ[i], p[j])) for j in 1:J for i in 1:j]
    filter!(e -> !Symbolics._iszero(e[3]), entries)
    idx = [LinearIndices((J, J))[i, j] for (i, j, _) in entries]
    # nothing to write when f is linear in p (as a nonlinear weak form)
    isempty(entries) && return ((out, x, p, t, a, Ω) -> nothing), idx
    _build([e[3] for e in entries], x, p, t, a, Ω)[end], idx
end
