## rhs of ode changes to f(u) ./ u for lognormal noise
function _getRHS_sym(f!::Function, D::Int, J::Int, ::Val{LogNormal})
    @variables w[1:J] u[1:D] t
    du = Vector(undef, D)
    f!(du, u, w, t)
    return [eq / u  for (u,eq) in zip(u,du)]
end

function _getRHS_sym(f!::Function, D::Int, J::Int, ::Val{Normal})
    @variables w[1:J] u[1:D] t
    du = Vector(undef, D)
    f!(du, u, w, t)
    return du
end

function getRHS(f!::Function, D::Int, J::Int, ::Val{DistType}) where {DistType<:Distribution}
    @variables w[1:J] u[1:D] t
    rhs_sym = _getRHS_sym(f!, D, J, Val(DistType))
    f,f! = build_function(rhs_sym, u, w, t;expression=false)
    return f, f!
end

function _getJacw_sym(f!::Function, D::Int, J::Int, ::Val{DistType}) where {DistType<:Distribution}
    @variables w[1:J]
    rhs_sym = _getRHS_sym(f!, D, J, Val(DistType))
    return jacobian(rhs_sym, w)
end

function getJacw(f!::Function, D::Int, J::Int, ::Val{DistType}) where {DistType<:Distribution}
    @variables w[1:J] u[1:D] t
    jac_sym = _getJacw_sym(f!, D, J, Val(DistType))
    jac,jac! = build_function(jac_sym, u,w,t; expression=false)
    return jac, jac!
end
## for normal noise ∇ᵤf 
function _getJacu_sym(f!::Function, D::Int, J::Int, ::Val{Normal})
    @variables u[1:D] 
    rhs_sym = _getRHS_sym(f!, D, J, Val(Normal))
    return jacobian(rhs_sym, u)
end
## for normal noise ∇_yf(u)/u where y = log(u)  
function _getJacu_sym(f!::Function, D::Int, J::Int, ::Val{LogNormal})
    @variables u[1:D] 
    rhs = _getRHS_sym(f!, D, J, Val(LogNormal))
    mymap = Dict([ud =>exp(ud) for ud in u])
    rhs_sym = [simplify(substitute(rhs_d, mymap)) for rhs_d in rhs]
    return jacobian(rhs_sym, u)
end

function getJacu(f!::Function, D::Int, J::Int, ::Val{DistType}) where {DistType<:Distribution}
    @variables w[1:J] u[1:D] t
    jac_sym = _getJacu_sym(f!, D, J, Val(DistType))
    jac,jac! = build_function(jac_sym, u,w,t; expression=false)
    return jac, jac!
end

function getJacwJacu(f!::Function, D::Int, J::Int, ::Val{DistType}) where {DistType<:Distribution}
    @variables w[1:J] u[1:D] t
    jacu_sym = _getJacu_sym(f!, D, J, Val(DistType))
    jacwjacu_sym = jacobian(jacu_sym, w)
    jac, jac! = build_function(jacwjacu_sym, u,w,t; expression=false)
    return jac, jac!
end

function getHeswJacu(f!::Function, D::Int, J::Int, ::Val{DistType}) where {DistType<:Distribution}
    @variables w[1:J] u[1:D] t
    jacu_sym = _getJacu_sym(f!, D, J, Val(DistType))
    jacwjacu_sym = jacobian(jacu_sym, w)
    heswjacu_sym = jacobian(jacwjacu_sym, w)
    hes, hes! = build_function(heswjacu_sym, u,w,t; expression=false)
    return hes, hes!
end

function getHesw(f!::Function, D::Int, J::Int, ::Val{DistType}) where {DistType<:Distribution}
    @variables w[1:J] u[1:D] t
    jacw_sym = _getJacw_sym(f!, D, J, Val(DistType))
    hesw_sym = jacobian(jacw_sym, w)
    hes, hes! = build_function(hesw_sym, u,w,t; expression=false)
    return hes, hes!
end