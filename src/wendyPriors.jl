## if some parameters don't have priors but others do, we should pass a nothing
function _logPrior(prior::Nothing)
    zero_function(x) = 0
    return zero_function, zero_function, zero_function
end
## Build functions and their derivatives via symbolic computations
function _logPrior(prior::Distribution)
    @variables p
    π_sym = try
        -logpdf(prior,p)
    catch e
        @error "This function logpdf function probably is not symbolics friendly, please extend the logpdf function of interest"
        throw(e)
    end
    ∇ₚπ_sym = Symbolics.derivative(π_sym, p)
    ∇ₚ²π_sym = Symbolics.derivative(∇ₚπ_sym, p)
    
    _π = build_function(π_sym, p; expression=false)
    _∂π = build_function(∇ₚπ_sym, p; expression=false)
    _∂²π = build_function(∇ₚ²π_sym, p; expression=false)
    int = support(prior)
    function π(p)
        if p < int.lb || p > int.ub
            return 1e6
        end
        return _π(p)
    end
    function ∂π(p)
        if p < int.lb || p > int.ub
            return 1e6
        end
        return _∂π(p)
    end
    function ∂²π(p)
        if p < int.lb || p > int.ub
            return 1e6
        end
        return _∂²π(p)
    end
    return π, ∂π, ∂²π
end
## Some of the build in pdf functions are not symbolics friendly...
import Distributions: logpdf
## Beta 
function logpdf(dist::Beta, x::T) where {T <: Real}
    return Distributions.xlogy(dist.α - 1, x) + Distributions.xlog1py(dist.β - 1, -x) - Distributions.logbeta(dist.α, dist.β)
end
## Gamma 
function logpdf(dist::Gamma, x::T) where {T <: Real}
    α,θ = dist.α, dist.θ
    - x / θ +  (α -1) * log(x) - α *log(θ) - Distributions.loggamma(α)
end 
## Frechet https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
function logpdf(dist::Frechet, x::T) where {T <: Real}
    α, θ = dist.α, dist.θ
    return log(α) - log(θ) + (-1-α) * (log(x) - log(θ)) - (x / θ) ^ (-α)
end
## Heavy lifting function that combines the likelihood with the priors
function getNegativeLogPosterior(wnll::SecondOrderCostFunction, priors::Union{Nothing,AbstractVector{<:Distribution}})
    if isnothing(priors)
        return SecondOrderCostFunction(x->0,x->zeros(length(x)), x->zeros(length(x),length(x))), wnll
    end

    f = wnll.f; ∇f! = wnll.∇f!; ∇²f! = wnll.Hf!;

    priorFunList = [_logPrior(prior) for prior in priors]
    function Π(p) 
        v = sum(f[1](p[i]) for (i,f) in enumerate(priorFunList))
        return v
    end
    function ∇ₚΠ!(g, p) 
        for (i,f) in enumerate(priorFunList)
            g[i] = f[2](p[i]) 
        end
        return g 
    end
    function ∇ₚ²Π!(H, p) 
        H .= 0
        for (i,f) in enumerate(priorFunList)
            H[i,i] = f[3](p[i]) 
        end
        return H
    end

    function wnlp(p) 
        v = f(p) + sum(f[1](p[i]) for (i,f) in enumerate(priorFunList))
        return v
    end
    function ∇ₚwnlp!(g, p) 
        ∇f!(g,p)
        for (i,f) in enumerate(priorFunList)
            g[i] += f[2](p[i]) 
        end
        return g 
    end
    function ∇ₚ²wnlp!(H, p) 
        H .= 0
        ∇²f!(H,p)
        for (i,f) in enumerate(priorFunList)
            H[i,i] += f[3](p[i]) 
        end
        return H
    end
    
    return SecondOrderCostFunction(Π, ∇ₚΠ!, ∇ₚ²Π!), SecondOrderCostFunction(wnlp, ∇ₚwnlp!, ∇ₚ²wnlp!)
end
##
import Distributions: support 
function support(::Nothing)
    RealInterval{Float64}(-Inf,Inf)
end
## Update the constraints to respect the support of each prior
function _makeConstraintsRespectPriorSupport(J, constraints, priors)
    if isnothing(priors)
        return -Inf *ones(J), Inf*ones(J)
    end
    J = length(priors)
    l = zeros(J)
    u = zeros(J)
    for (j,prior_j) in enumerate(priors)  
        sup = support(prior_j)
        l[j] = sup.lb 
        u[j] = sup.ub
        if isnothing(constraints) || isnothing(constraints[j])
            continue
        end
        l[j] = max(int[1], l[j])
        u[j] = min(int[2], u[j])
    end
    l,u 
end