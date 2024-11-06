## rhs of ode changes to f(u) ./ u for lognormal noise
function _getRHS_sym(data::WENDyData{lip,LogNormal}) where lip 
    D = length(data.initCond)
    J = length(data.wTrue)
    @variables w[1:J] u[1:D] t
    du = Vector(undef, D)
    data.f!(du, u, w, t)
    return [eq / u  for (u,eq) in zip(u,du)]
end

function _getRHS_sym(data::WENDyData{lip,Normal}) where lip 
    D = length(data.initCond)
    J = length(data.wTrue)
    @variables w[1:J] u[1:D] t
    du = Vector(undef, D)
    data.f!(du, u, w, t)
    return du
end

function getRHS(data::WENDyData{lip,DistType}) where {lip, DistType<:Distribution}
    D = length(data.initCond)
    J = length(data.wTrue)
    @variables w[1:J] u[1:D] t
    rhs_sym = _getRHS_sym(data)
    f,f! = build_function(rhs_sym, u, w, t;expression=false)
    return f, f!
end

function _getJacw_sym(data::WENDyData{lip,DistType}) where {lip, DistType<:Distribution}
    J = length(data.wTrue)
    @variables w[1:J]
    rhs_sym = _getRHS_sym(data)
    return jacobian(rhs_sym, w)
end

function getJacw(data::WENDyData{lip,DistType}) where {lip, DistType<:Distribution}
    D = length(data.initCond)
    J = length(data.wTrue)
    @variables w[1:J] u[1:D] t
    jac_sym = _getJacw_sym(data)
    jac,jac! = build_function(jac_sym, u,w,t; expression=false)
    return jac, jac!
end
## for normal noise ∇ᵤf 
function _getJacu_sym(data::WENDyData{lip,Normal}) where lip
    D = length(data.initCond)
    @variables u[1:D] 
    rhs_sym = _getRHS_sym(data)
    return jacobian(rhs_sym, u)
end
## for normal noise ∇_yf(u)/u where y = log(u)  
function _getJacu_sym(data::WENDyData{lip,LogNormal}) where lip
    D = length(data.initCond)
    @variables u[1:D] 
    rhs = _getRHS_sym(data)
    mymap = Dict([u =>exp(u) for u in unknowns(data.odeprob)])
    rhs_sym = [simplify(substitute(rhs_d, mymap)) for rhs_d in rhs]
    return jacobian(rhs_sym, u)
end

function getJacu(data::WENDyData{lip,DistType}) where {lip, DistType<:Distribution}
    D = length(data.initCond)
    J = length(data.wTrue)
    @variables w[1:J] u[1:D] t
    jac_sym = _getJacu_sym(data)
    jac,jac! = build_function(jac_sym, u,w,t; expression=false)
    return jac, jac!
end

function getJacwJacu(data::WENDyData{lip,DistType}) where {lip, DistType<:Distribution}
    D = length(data.initCond)
    J = length(data.wTrue)
    @variables w[1:J] u[1:D] t
    jacu_sym = _getJacu_sym(data)
    jacwjacu_sym = jacobian(jacu_sym, w)
    jac, jac! = build_function(jacwjacu_sym, u,w,t; expression=false)
    return jac, jac!
end

function getHeswJacu(data::WENDyData{lip,DistType}) where {lip, DistType<:Distribution}
    D = length(data.initCond)
    J = length(data.wTrue)
    @variables w[1:J] u[1:D] t
    jacu_sym = _getJacu_sym(data)
    jacwjacu_sym = jacobian(jacu_sym, w)
    heswjacu_sym = jacobian(jacwjacu_sym, w)
    hes, hes! = build_function(heswjacu_sym, u,w,t; expression=false)
    return hes, hes!
end

function getHesw(data::WENDyData{lip,DistType}) where {lip, DistType<:Distribution}
    D = length(data.initCond)
    J = length(data.wTrue)
    @variables w[1:J] u[1:D] t
    jacw_sym = _getJacw_sym(data)
    hesw_sym = jacobian(jacw_sym, w)
    hes, hes! = build_function(hesw_sym, u,w,t; expression=false)
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