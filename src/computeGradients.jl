
function _getRHS_sym(mdl)
    return [eq.rhs for eq in equations(mdl)]
end


function getRHS(mdl)
    w = parameters(mdl)
    u = unknowns(mdl)
    J = length(w)
    D = length(u)
    rhs_sym = _getRHS_sym(mdl)
    _f,_ = build_function(rhs_sym, w,u; expression=false)
    function f(w::SVector{J,<:Real},u::SVector{D,<:Real}, ::Val{J}, ::Val{D}) where {J,D} 
        return _f(w,u)
    end
    function rhs(w::AbstractVector{<:Real},u::AbstractVector{<:Real}) 
        try
            return f(SA[w...],SA[u...],Val(J), Val(D))
        catch err 
            if length(w) != J 
                @error "length(w) = $(length(w)) != $J"
                Base.throw_checksize_error(w, (J,))
            elseif length(u) != D 
                @error "length(u) = $(length(u)) != $D"
                Base.throw_checksize_error(u, (D,))
            else 
                throw(err)
            end
        end
    end
    return rhs 
end

function _getParameterJacobian_sym(mdl)
    w = parameters(mdl)
    rhs_sym = _getRHS_sym(mdl)
    return jacobian(rhs_sym, w)
end

function getParameterJacobian(mdl)
    w = parameters(mdl)
    u = unknowns(mdl)
    J = length(w)
    D = length(u)
    jac_sym = _getParameterJacobian_sym(mdl)
    _f,_ = build_function(jac_sym, w,u; expression=false)
    function f(w::SVector{J,<:Real},u::SVector{D,<:Real}, ::Val{J}, ::Val{D}) where {J,D} 
        return _f(w,u)
    end
    function jac(w::AbstractVector{<:Real},u::AbstractVector{<:Real}) 
        try
            return f(SA[w...],SA[u...],Val(J),Val(D))
        catch err 
            if length(w) != J 
                @error "length(w) = $(length(w)) != $J"
                Base.throw_checksize_error(w, (J,))
            elseif length(u) != D 
                @error "length(u) = $(length(u)) != $D"
                Base.throw_checksize_error(u, (D,))
            else 
                throw(err)
            end
        end
    end
    return jac
end