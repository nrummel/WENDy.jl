using Symbolics: jacobian
using ModelingToolkit: parameters, build_function, unknowns, ODESystem
##
function _getRHS_sym(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    return  [T == Normal ? eq.rhs : log(eq.rhs) for eq in equations(mdl)]
end

function getRHS(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    w = parameters(mdl)
    u = unknowns(mdl)
    rhs_sym = _getRHS_sym(mdl, Val(T))
    _rhs,_rhs! = build_function(rhs_sym, w,u; expression=false)
    return _rhs,_rhs! 
end

function _getJacw_sym(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    w = parameters(mdl)
    rhs_sym = _getRHS_sym(mdl, Val(T))
    return jacobian(rhs_sym, w)
end

function getJacw(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    w = parameters(mdl)
    u = unknowns(mdl)
    jac_sym = _getJacw_sym(mdl, Val(T))
    jac,jac! = build_function(jac_sym, w,u; expression=false)
    return jac, jac!
end

function _getJacu_sym(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    u = unknowns(mdl)
    rhs_sym = _getRHS_sym(mdl, Val(T))
    return jacobian(rhs_sym, u)
end

function getJacu(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    w = parameters(mdl)
    u = unknowns(mdl)
    jac_sym = _getJacu_sym(mdl, Val(T))
    jac,jac! = build_function(jac_sym, w,u; expression=false)
    return jac, jac!
end

function getJacwJacu(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    w = parameters(mdl)
    u = unknowns(mdl)
    jacu_sym = _getJacu_sym(mdl, Val(T))
    jacwjacu_sym = jacobian(jacu_sym, w)
    jac, jac! = build_function(jacwjacu_sym, w,u; expression=false)
    return jac, jac!
end

function getHeswJacu(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    w = parameters(mdl)
    u = unknowns(mdl)
    jacu_sym = _getJacu_sym(mdl, Val(T))
    jacwjacu_sym = jacobian(jacu_sym, w)
    heswjacu_sym = jacobian(jacwjacu_sym, w)
    hes, hes! = build_function(heswjacu_sym, w,u; expression=false)
    return hes, hes!
end

function getHesw(mdl::ODESystem, ::Val{T}=Val(Normal)) where T<:Distribution
    w = parameters(mdl)
    u = unknowns(mdl)
    jacw_sym = _getJacw_sym(mdl, Val(T))
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