
abstract type AbstractOptimizer end 
"""
**Adam Optimizer**
```julia
    Adam(;α=0.001, β₁=0.9, β₂=0.999, ϵ=10e-8)
```

Algorithm:
```math
\\begin{align*}
    m_t =& \\beta_1 m_{t-1} + (1-\\beta_1)g_t\\\\
    v_t =& \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2\\\\
    \\hat{m}_t =& \\frac{m_t}{1-\\beta_1^t}\\\\
    \\hat{v}_t =& \\frac{v_t}{1-\\beta_2^t}\\\\
    \\Delta x_t =& \\frac{\\alpha}{\\sqrt{\\hat{v}_t}+\\epsilon}\\hat{m}_t\\\\
\\end{align*}
```
[Algorithm Reference](https://arxiv.org/abs/1412.6980)
"""
mutable struct Adam <: AbstractOptimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    α::Float64
    β₁::Float64
    β₂::Float64
    m_t::AbstractArray
    v_t::AbstractArray
end


function Adam(;α::Real=0.001, β₁::Real=0.9, β₂::Real=0.999, ϵ::Real=10e-8)
    @assert α > 0.0 "α must be greater than 0"
    @assert β₁ > 0.0 "β₁ must be greater than 0"
    @assert β₂ > 0.0 "β₂ must be greater than 0"
    @assert ϵ > 0.0 "ϵ must be greater than 0"

    Adam("Adam", 0, ϵ, α, β₁, β₂, [], [])
end

params(opt::Adam) = "ϵ=$(opt.ϵ), α=$(opt.α), β₁=$(opt.β₁), β₂=$(opt.β₂)"

function update(opt::Adam, g_t::AbstractArray{T}, ::Any, ::Any) where {T<:Real}
    # resize biased moment estimates if first iteration
    if opt.t == 0
        opt.m_t = zero(g_t)
        opt.v_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    # update biased first moment estimate
    opt.m_t = opt.β₁ * opt.m_t + (one(T) - opt.β₁) * g_t

    # update biased second raw moment estimate
    opt.v_t = opt.β₂ * opt.v_t + (one(T) - opt.β₂) * ((g_t) .^2)

    # compute bias corrected first moment estimate
    m̂_t = opt.m_t / (one(T) - opt.β₁^opt.t)

    # compute bias corrected second raw moment estimate
    v̂_t = opt.v_t / (one(T) - opt.β₂^opt.t)

    # apply update
    ρ = opt.α * m̂_t ./ (sqrt.(v̂_t .+ opt.ϵ))

    return ρ
end


function gradientDescent(w0::AbstractVector, F::Function, gradF!::Function; 
    opt::AbstractOptimizer=Adam(α=1e-2), maxIter::Int=100, tol::AbstractFloat=1e-8,
    jac::Union{JacGgetter, Nothing}=nothing, G::Union{GFun, Nothing}=nothing, R::Union{Nothing,AbstractMatrix}=nothing, b::Union{Nothing, AbstractVector}=nothing)
    J = length(w0)
    wit = zeros(J,maxIter)
    wit[:,1] .= w0
    ve = zeros(maxIter-1)
    fe = zeros(maxIter-1)
    g = similar(w0)
    for i in 2:maxIter 
        gradF!(g, wit[:,i-1])
        # δ = update(opt, g)
        r = R' \ G(wit[:,i-1]) - b
        jacG = jac(wit[:,i-1])
        tmp  = R' \ (jacG * r)
        @show α = dot(r,r) / dot(tmp, tmp )
        δ = α*g
        wit[:,i] .= wit[:,i-1] .- δ
        ve[i-1] = norm(wit[:,i-1] - wit[:,i])/ norm(wit[:,i-1])
        fe[i-1] = abs(F(wit[:,i-1]) - F(wit[:,i])) / abs(F(wit[:,i]))
        if  ve[i-1] < tol 
            @info """Convergence met for iteration convergence 
                num itr = $i
                rel itr = $(ve[i])
                rel obj = $(fe[i])
            """
            return wit[:,i], wit[:,1:i], ve[1:i], fe[1:i] 
        elseif fe[i-1] < tol
            @info """Convergence met for objective value convergence
                num itr = $i
                rel itr = $(ve[i])
                rel obj = $(fe[i])
            """
            return wit[:,i], wit[:,1:i], ve[1:i], fe[1:i] 
        end 
        if i == maxIter
            @warn """Did not converge
                num itr = $i
                rel itr = $(ve[end])
                rel obj = $(fe[end])
            """
        end
    end
    
    return wit[:,end], wit, ve, fe 
end