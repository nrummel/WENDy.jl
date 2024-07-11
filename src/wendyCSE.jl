# using SymPy
using Symbolics.SymbolicUtils: Symbolic, issym, isterm, isadd, ispow, ismul, isdiv
using Symbolics: value

# function  _symbolicsEx2SymPyEx(f::Num, ode::ODESystem)
#     _symbolicsEx2SymPyEx(value(f), ode)
# end

# function  _symbolicsEx2SymPyEx(f::Symbolic, ode::ODESystem)
#     ukn = unknowns(ode)
#     u = [uu.f for uu in ukn]
#     subfstr = string(f)
#     for (uku, uu) in zip(ukn, u)
#         subfstr = replace(subfstr, string(uku)=>string(uu))
#     end
#     @show subfstr = replace(subfstr, "^"=>"**")
#     SymPy._sympy_.parse_expr(subfstr) 
# end

# function _symbolicsEx2SymPyEx(f::Array, ode::ODESystem)
#     sympyExp = [_symbolicsEx2SymPyEx(subf, ode) for subf in f]
# end

function coeff(p, ::Nothing)
    return 0 
end

function coeff(p::Number, ::Any)
    return 0
end

function coeff(p::Symbolic, sym::Num)
    coeff(p, value(sym))
end
function coeff(p::Num, sym::Symbolic)
    coeff(value(p), sym)
end

function coeff(p::Num, sym::Num)
    coeff(value(p), value(sym))
end

function coeff(p::Symbolic, sym::Symbolic; ll::LogLevel=Warn)
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "main call"
        if issym(p) || isterm(p)
            @info " is sym/term"
            return Int(isequal(p, sym))
        elseif ispow(p)
            @info " is pow"
            return Int(isequal(p, sym))
        elseif isadd(p)
            @info " is add"
            return sum(coeff(k, sym) * v for (k, v) in p.dict)
        elseif ismul(p)
            @info " is mul"
            args = unsorted_arguments(p)
            coeffs = map(a->coeff(a, sym), args)
            flags = falses(length(coeffs))
            for (i,c) in enumerate(coeffs)
                flag = iszero(c) 
                flag isa Bool && (flags[i] = flag)
            end
            if all(flags)
                return 0
            end 
            nz_coeff = coeffs[findall([!f for f in flags])]
            z_coeff = args[findall(flags)]
            return @views prod(Iterators.flatten((nz_coeff, z_coeff)))
        elseif isdiv(p)
            num, denom = unsorted_arguments(p) 
            if _occursin(sym, num) && _occursin(sym, denom)
                return Set([coeff(num, sym), coeff(denom, sym)])
            elseif _occursin(sym, denom)
                return coeff(denom, sym)
            end
            # TODO add check that denom in not zero
            # @assert 0 != denom  "coefficients in the denominator cannot be zero"
            return coeff(num, sym) / denom
        elseif p isa Number
            return 0
        elseif p isa Symbolic
            @info " is symbolic"
            return coeff(p, sym)
        end

        throw(DomainError(p, "Datatype $(typeof(p)) not accepted."))
    end
end


function _cse(f, ode::ODESystem)
    Set()
end
function _cse(f::Num, ode::ODESystem)
    _cse(value(f), ode)
end
function _cse(f::Symbolic, ode::ODESystem)
    w = parameters(ode)
    vars = intersect(w, get_variables(f))
    subexpr = Set()
    if isempty(intersect(w, vars))
        return f 
    end 
    for v in w 
        @show c = coeff(f, v)
        while !isempty(intersect(get_variables(c), w))
            println("here")
            for v in vars
                @show cc = coeff(c, v)
                if (cc == 0) isa Bool && cc == 0
                    continue 
                end 
                c = cc
            end
        end
        c isa Symbolic && push!(subexpr, c)
    end
    collect(subexpr)
end

function _cse(f::Array, ode::ODESystem)
    subexpr = Set()
    for subf in f 
        subexpr = union(subexpr,cse(subf, ode))
    end 
    collect(subexpr)
end

function _occursin(needle::Symbolic, haystack::Symbolic)
    if ismul(haystack)
        needle = issym(needle) ? [needle] : arguments(needle)
        haystack = arguments(haystack)
        return length(intersect(needle, haystack)) == length(needle)
    elseif isadd(haystack)
        any(_occursin(needle, a) for a in arguments(haystack))
    end
    isequal(needle, haystack)
end

function substitute_coeff(f::Num, p::Pair)
    substitute_coeff(value(f), p)
end

function substitute_coeff(f::Symbolic, p::Pair)
    if isadd(f)
        subf = arguments(f)
        f = 0 
        for s in subf 
            f += substitute_coeff(s, p)
        end
        return f
    end
    @assert ismul(f)
    if _occursin(p.first, f)
        needle = arguments(p.first)
        haystack = arguments(f)
        remainder = setdiff(haystack, needle)
        return isempty(remainder) ? p.second : prod(remainder) * p.second
    end
    return f
end

function cse(f::Num, ode::ODESystem)
    cse(value(f), ode)
end
function cse(f::Symbolic, ode::ODESystem)
    
end
function cse(f::Array, ode::ODESystem)
    subexp = Set()
    for subf in f 
        subexpr = union(_cse(subf, ode), subexp)
    end

    eff_f = similar(f)
    for (i, subf) in enumerate(f)
        substitute_coeff()
    end
end
# ## Code stolen from SymbiolicCodegen.jl 
# # but it is unmaintained, so I put it here for my use
# function cse(s::Symbolic, vars::AbstractVector{<:Symbolic})
#     dict = Dict()
#     vars = intersect(atoms(s), vars)
#     csestep(s, vars, vars, dict)
#     # r = @rule ~x => csestep(~x, vars, dict) 
#     # final = RW.Postwalk(RW.PassThrough(r))(s)
#     [[var.name => ex for (ex, var) in pairs(dict)]...]
# end

# csestep(s::Sym, vars, ogVars, dict) = s

# csestep(s, vars, ogVars, dict) = s

# function csestep(exp::Symbolic, vars, ogVars, dict)
#     @info "typeof(exp) $(typeof(exp)): $exp"
#     if isempty(intersect(atoms(exp), vars)) 
#         # If there are no vars of interest in this expr then leave
#         return exp
#     end
#     T = exprtype(exp)
#     if T == SYM || T == TERM  ## Base case 
#             # dict[] 
#         return 
#     end

#     op = operation(exp)
#     args = [csestep(arg, vars, ogVars, dict) for arg in arguments(exp)]

#     t = similarterm(exp, op, args)

#     if !haskey(dict, t) 
#         dict[t] = Sym{symtype(t)}(gensym())
#         @show t 
#     end
    
#     return dict[t]
# end
# function atoms(t::Symbolic)
#     if issym(t) || isterm(t)  ## Base case 
#         return [t] 
#     end 
#     f = operation(t)
#     args = arguments(t)

#     if hasproperty(f, :name) && f.name == :Sum
#         return setdiff(atoms(args[1]), [args[2]])
#     else
#         return union(atoms(f), union(atoms.(args)...))
#     end
# end

# atoms(x::Num) = atoms(value(x))
# atoms(x) = Set{Symbolic}()