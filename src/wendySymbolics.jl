using Symbolics: jacobian
using ModelingToolkit: parameters, build_function, unknowns, ODESystem
## rhs is f(u) for normal noise
function _getRHS_sym(data::WENDyData{LinearInParameters, Normal}) where LinearInParameters
    return  [eq.rhs  for eq in equations(data.ode)]
end
## rhs of ode changes to f(u) ./ u for lognormal noise
function _getRHS_sym(data::WENDyData{LinearInParameters,LogNormal}) where LinearInParameters
    return  [eq.rhs / u  for (u,eq) in zip(unknowns(data.ode),equations(data.ode))]
end

function getRHS(data::WENDyData{LinearInParameters,DistType}) where {LinearInParameters, DistType<:Distribution}
    w = parameters(data.ode)
    u = unknowns(data.ode)
    rhs_sym = _getRHS_sym(data)
    _rhs,_rhs! = build_function(rhs_sym, w,u; expression=false)
    return _rhs,_rhs! 
end

function _getJacw_sym(data::WENDyData{LinearInParameters,DistType}) where {LinearInParameters, DistType<:Distribution}
    w = parameters(data.ode)
    rhs_sym = _getRHS_sym(data)
    return jacobian(rhs_sym, w)
end

function getJacw(data::WENDyData{LinearInParameters,DistType}) where {LinearInParameters, DistType<:Distribution}
    w = parameters(data.ode)
    u = unknowns(data.ode)
    jac_sym = _getJacw_sym(data)
    jac,jac! = build_function(jac_sym, w,u; expression=false)
    return jac, jac!
end
## for normal noise ∇ᵤf 
function _getJacu_sym(data::WENDyData{LinearInParameters,Normal}) where LinearInParameters
    u = unknowns(data.ode)
    rhs_sym = _getRHS_sym(data)
    return jacobian(rhs_sym, u)
end
## for normal noise ∇_yf(u)/u where y = log(u)  
function _getJacu_sym(data::WENDyData{LinearInParameters,LogNormal}) where LinearInParameters
    rhs = _getRHS_sym(data)
    mymap = Dict([u =>exp(u) for u in unknowns(data.ode)])
    rhs_sym = [simplify(substitute(rhs_d, mymap)) for rhs_d in rhs]
    u = unknowns(data.ode)
    return jacobian(rhs_sym, u)
end

function getJacu(data::WENDyData{LinearInParameters,DistType}) where {LinearInParameters, DistType<:Distribution}
    w = parameters(data.ode)
    u = unknowns(data.ode)
    jac_sym = _getJacu_sym(data)
    jac,jac! = build_function(jac_sym, w,u; expression=false)
    return jac, jac!
end

function getJacwJacu(data::WENDyData{LinearInParameters,DistType}) where {LinearInParameters, DistType<:Distribution}
    w = parameters(data.ode)
    u = unknowns(data.ode)
    jacu_sym = _getJacu_sym(data)
    jacwjacu_sym = jacobian(jacu_sym, w)
    jac, jac! = build_function(jacwjacu_sym, w,u; expression=false)
    return jac, jac!
end

function getHeswJacu(data::WENDyData{LinearInParameters,DistType}) where {LinearInParameters, DistType<:Distribution}
    w = parameters(data.ode)
    u = unknowns(data.ode)
    jacu_sym = _getJacu_sym(data)
    jacwjacu_sym = jacobian(jacu_sym, w)
    heswjacu_sym = jacobian(jacwjacu_sym, w)
    hes, hes! = build_function(heswjacu_sym, w,u; expression=false)
    return hes, hes!
end

function getHesw(data::WENDyData{LinearInParameters,DistType}) where {LinearInParameters, DistType<:Distribution}
    w = parameters(data.ode)
    u = unknowns(data.ode)
    jacw_sym = _getJacw_sym(data)
    hesw_sym = jacobian(jacw_sym, w)
    hes, hes! = build_function(hesw_sym, w,u; expression=false)
    return hes, hes!
end

## Maybe put size checks in these functions? 
# J = length(w)
# T = length(u)
# function f(w::SVector{J,<:Real},u::SVector{T,<:Real}, ::Val{J}, ::Val{T}) where {J,T} 
#     return _f(w,u)
# end
# function jac(w::AbstractVector{<:Real},u::AbstractVector{<:Real}) 
#     try
#         return f(SA[w...],SA[u...],Val(J),Val(T))
#     catch err 
#         if length(w) != J 
#             @error "length(w) = $(length(w)) != $J"
#             Base.throw_checksize_error(w, (J,))
#         elseif length(u) != T 
#             @error "length(u) = $(length(u)) != $T"
#             Base.throw_checksize_error(u, (T,))
#         else 
#             throw(err)
#         end
#     end
# end