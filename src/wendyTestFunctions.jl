abstract type TestFunction <: Function end 
struct ExponentialTestFun <: TestFunction end 
function (ϕ::ExponentialTestFun)(x::Number,η::Real=9) 
    exp(-η*(1-x.^2).^(-1))
end
""" 
    Find where the error is with the secand line approximation is minimized in L1
"""
function _getcorner(yy::AbstractVector{<:Real}, xx::Union{AbstractVector{<:Real},Nothing}=nothing)
    N = length(yy)
    isnothing(xx) && (xx = 1:N)
    # scale by m aximum value for stability ?
    yy = (yy ./ maximum(abs.(yy)))*N
    errs = zeros(N)
    for k=1:N
        l1 = (yy[k] - yy[1]) / (xx[k] - xx[1]) * (xx[1:k] .- xx[k]) .+ yy[k] 
        l2 = (yy[end] - yy[k]) / (xx[end] - xx[k]) * (xx[k:end] .- xx[k]) .+ yy[k] 
        # relative L1 err
        errs[k] = ( 
            sum(abs.((l1-yy[1:k])./yy[1:k])) 
            + sum(abs.((l2-yy[k:end]) ./ yy[k:end]))
        )
    end
    replace!(errs, NaN=>Inf)
    _,ix = findmin(errs)
    return ix
end
"""
    Select the minimum radius by looking for where integration error 
    will be dominated by 
    for more detailsSee section 2.3 of Bort and Messenger 
"""
function _minRadius(
    U::AbstractMatrix, dt::Real, 
    radiusMin::Int, numRadii::Int, radiusMax::Int, testFunSubRate::Real, Kᵣ::Union{Nothing,Int}; 
    debug::Bool=false
)
    @assert 2 <= testFunSubRate < 4 "We only suppport scaling between 2 and 4"
    if radiusMin >= radiusMax
        return radiusMin 
    end
    Mp1, D = size(U)
    # look for a radius between radiusMin and radiusMax, but only check numRadii of them ...
    step = Int(ceil((radiusMax - radiusMin ) / numRadii))
    radii = radiusMin:step:radiusMax
    errs = zeros(length(radii))
    # Select the fourier modes from just the where roots of unit would hit on the subsampled grid
    IX = Int(floor((Mp1-1)/testFunSubRate)) 
    # the fft will will be in order F[0,], F[1], ... F[M/2], F[-M/2], .. F[-1]
    # we want the F[M/2]
    
    for (r,radius) in enumerate(radii)
        Φᵣ = _buildΦ(radius, Mp1, dt, Kᵣ)
        Φ = Φᵣ[:,:,1]
        K = size(Φ,1)
        @tullio ΦU[k,d,m] := Φ[k,m]*U[m,d]
        ΦU = reshape(ΦU, K*D, Mp1)
        Fhat_ΦU = fft(ΦU, (2,))
        errs[r] = norm(imag(Fhat_ΦU[:,IX]))
    end
    ix = _getcorner(log.(errs), radii) 
    
    return debug ? (ix, radii, errs) : radii[ix]
end
"""
    Compute the derivative of a test function 
"""
function _derivative(radius::Int, dt::AbstractFloat, order::Int)
    @variables t
    φ = ExponentialTestFun()
    φ′_sym = simplify((radius*dt)^(-order) * expand_derivatives(Differential(t)(φ(t))))
    return build_function(φ′_sym, t; expression=false)
end
"""
    Build chunks of the time domain given a radius of a test function and the number of time points. By default this will pack as many test functions as possible, but this can be reduced by setting K_r
"""
function _buildChunks(Mp1::Int, radius::Int, Kᵣ::Union{Int,Nothing}=nothing)
    diam = Int(2*radius+1)
    interiorLength = (Mp1-2*(radius+1))
    # If we dont want to manual set how many test functions 
    # there are, we can just add as many as possible. 
    # This is equivalent to making the gap=1 ↔ Kᵣ = I
    gap = isnothing(Kᵣ) ? 1 : max(Int(floor(interiorLength / Kᵣ)),1)
    # @assert isnothing(Kᵣ) || Kᵣ == Int(floor(interiorLength / gap))
    Kᵣ = Int(floor(interiorLength / gap))
    chunks = [(1:diam).+1 .+ k*gap for k in 0:(Kᵣ-1)] 
    mod(interiorLength, Kᵣ) 
    if mod(interiorLength, Kᵣ) != 0
        lastEndPt = chunks[end][end]
        intervalToFill =  Mp1-1 - lastEndPt 
        numToAdd = Int(ceil(intervalToFill / gap))
        chunks = vcat( 
            chunks,  
            [(Mp1-diam+1:Mp1) .- 1 .- k*gap for k in numToAdd-1:-1:0]
        )
    end
    return [c[2:end-1] for c in chunks] 
end
"""
    Build the Φ_full for a particular radius and test function φ. If asked provide the functions for each test function.
"""
function _buildΦ(radius::Int, Mp1::Int, dt::Real, Kᵣ::Union{Int, Nothing}=nothing; derivativeOrder::Int=0)
    @assert isnothing(Kᵣ) || mod(Kᵣ,2)==0 "Kᵣ must be even "
    diam = Int(2*radius+1) 
    f = derivativeOrder == 0 ? WENDy.ExponentialTestFun() : _derivative(radius, dt, derivativeOrder)
    chunks = _buildChunks(Mp1, radius, Kᵣ)
    # chunks may be slightly longer than Kᵣ because 
    # of rounding with discretization (or could be nothing)
    Kᵣ = length(chunks)
    Φᵣ = zeros(Kᵣ, Mp1) 
    xx = range(-1,1,diam)[2:end-1]
    ϕ = f.(xx)
    # normalize for stability
    ϕnorm  = norm(WENDy.ExponentialTestFun().(xx))
    ϕ /= ϕnorm
    for (k,c) in enumerate(chunks) 
        Φᵣ[k, c] .= ϕ
    end 
    return Φᵣ
end
""" """ 
function _computeDerivative_analytic(Φ̇_full::AbstractMatrix, F::SVD, K::Int)
    @info "    Computing Φ̇ with analytic Φ̇_full and svd(Φ_full)"
    (diagm(1 ./ F.S) * F.U' *  Φ̇_full)[1:K,:]
end
""" """
function _computeDerivative_fft(Φ::AbstractMatrix, dt::Real)
    @info "    Computing Φ̇ with the fft"
    _, Mp1 = size(Φ)
    Φ̇_fft = zeros(size(Φ))
    _Φ̇_fft = fft(Φ', (1, ))
    k = mod(Mp1,2)==0 ? 
        vcat(0:Mp1/2, -Mp1/2+1:-1) : 
        vcat(0:floor(Mp1/2), -floor(Mp1/2):-1)
    Φ̇_fft = imag(ifft((-2*pi/(Mp1*dt))*k .* _Φ̇_fft, (1,)))'
end
"""
    Radii (in grid points) of the test function families, picked from the data U by the default rule: the minimum
    radius from the fourier modes of Φ∘U (_minRadius), then radiiParams times it, below the maximum radius
"""
function _testFunctionRadii(
    tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, radiusMinTime::Real, radiusMaxTime::Real, numRadii::Int,
    testFunSubRate::Real, radiiParams::AbstractVector{<:Real}, Kᵣ::Union{Nothing,Int}
)
    @assert all(abs.(diff(tt) .- (tt[2] - tt[1])) .< 1e-6) "Must use uniform time grid diff(tt) = $(diff(tt))"
    dt = mean(diff(tt))
    Mp1, _ = size(U)
    @info "    Mp1 = $Mp1, dt = $dt"
    # dont let the radius be larger than the radius of the interior of the domain
    @info "    radiusMinTime = $radiusMinTime, radiusMaxTime = $radiusMaxTime"
    radiusMin = Int(max(ceil(radiusMinTime/dt), 2))
    radiusMax = Int(floor(radiusMaxTime/dt))
    _radiusMax = Int(floor((Mp1-2)/2))
    if radiusMax > _radiusMax
        @info "    We need to decrease the max radius for the time domain available"
        radiusMax = _radiusMax 
    end 
    @info "    pre-radiusMin=$radiusMin, radiusMax=$radiusMax"
    # select min radius by looking that M/testFunSubRate fourier mode of Φ∘U
    radiusMinMax = Int(floor(radiusMax/maximum(radiiParams)))
    if radiusMinMax < radiusMin
        @warn "Radius min and radius max are not chosen well"
        radiusMinMax = radiusMin*10
    end 
    @info "    radiusMinMax=$radiusMinMax"
    radiusMin =  _minRadius(U, dt, radiusMin, numRadii, radiusMinMax, testFunSubRate, Kᵣ)
    
    @info "    radiusMin=$radiusMin"
    radii = filter(r->r < radiusMax, Int.(floor.(radiiParams*radiusMin)))
    if length(radii) == 0 
        radii = [radiusMax]
    end
    return radii
end
"""
    SVD of Φ_full (the leading Kmax singular triplets when Φ_full is larger) and the number K of test functions
    kept: the largest K with condition number σ₁/σ_K ≤ maxTestFunCondNum, information ≤ minTestFunInfoNum and K ≤ Kmax
"""
function _reduceTestFunctions(Φ_full::AbstractMatrix{<:Real}, maxTestFunCondNum::Real, minTestFunInfoNum::Real, Kmax::Int)
    Kfull, Mp1 = size(Φ_full)
    Kmax = min(Kmax, Kfull, Mp1)
    F = minimum(size(Φ_full)) <= Kmax ? svd(Φ_full) : svds(Φ_full, nsv=Kmax)[1]
    # Choose K off how quickly the singular values decay wrt to the max
    condNumbers = F.S[1] ./ F.S
    sumAllS = sum(F.S) + (min(Kfull, Mp1)-Kmax) * F.S[end] # upper bound on the sum of all singular values
    infoNumbers = [sum(F.S[1:k]) / sumAllS for k in 1:Kmax] 
    K1 = findlast(condNumbers .<= maxTestFunCondNum)
    K2 = findlast(infoNumbers .<= minTestFunInfoNum)
    isnothing(K1) && (K1 = Inf)
    isnothing(K2) && (K2 = Inf)
    K = Int(floor(min(K1,K2,Kmax)))
    if K == Kmax
        @info "    Warning... K set to Kmax=$Kmax"
    end
    @info "    condNum is now $(condNumbers[K])"
    @info "    infoNum is now $(infoNumbers[K])"
    @info "    K=$(K)"
    return F, K
end
""" """
function getTestFunctionMatrices(
    tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, radiusMinTime::Real, radiusMaxTime::Real, numRadii::Int, testFunSubRate::Real, radiiParams::AbstractVector{<:Real}, maxTestFunCondNum::Real, minTestFunInfoNum::Real, Kmax::Int, Kᵣ::Union{Nothing,Int}; 
    analyticΦ̇::Bool=true, noSVD::Bool=false, ll::LogLevel=Info, debug::Bool=false
)
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "  Getting Test Function Matrices"
        radii = _testFunctionRadii(tt, U, radiusMinTime, radiusMaxTime, numRadii, testFunSubRate, radiiParams, Kᵣ)
        dt = mean(diff(tt))
        Mp1, _ = size(U)
        Φ_full = reduce(vcat, _buildΦ(r, Mp1, dt) for r in radii)
        Φ̇_full = reduce(vcat, _buildΦ(r, Mp1, dt; derivativeOrder=1) for r in radii)
        Kfull, _ = size(Φ_full)
        @info "    K_full=$Kfull"
        if noSVD
            @info "    Returning the Φ_full, Φ̇_full"
            return Φ_full, Φ̇_full 
        end
        F, K = _reduceTestFunctions(Φ_full, maxTestFunCondNum, minTestFunInfoNum, Kmax)
        Φ = F.Vt[1:K,:]
        Φ̇ = analyticΦ̇ ? _computeDerivative_analytic(Φ̇_full, F, K) : _computeDerivative_fft(Φ, dt)
        @assert size(Φ,2) == Mp1 && size(Φ̇,2) == Mp1
        return debug ? (radii, Φ_full, Φ̇_full, Φ, Φ̇) : (Φ, Φ̇)
    end
end
## Convience wrapper to get testfunctions with the parameter struct
function getTestFunctionMatrices(tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, params::WENDyParameters; kwargs...)
    getTestFunctionMatrices(tt, U, params.radiusMinTime,params.radiusMaxTime, params.numRadii, params.testFunSubRate,params.radiiParams,params.maxTestFunCondNum, params.minTestFunInfoNum,params.Kmax, params.Kᵣ; kwargs...)
end
## Fixed test functions: built once, independent of the noisy data and of the time grid
# the bump φ(x) = exp(-η/(1-x²)) on (-1,1) (ExponentialTestFun) and its derivative, zero outside
_bump(x::Real, η::Real=9) = abs(x) < 1 ? exp(-η/(1 - x^2)) : zero(float(x))
_bump′(x::Real, η::Real=9) = abs(x) < 1 ? -2η*x/(1 - x^2)^2 * exp(-η/(1 - x^2)) : zero(float(x))
"""
    FixedTestFunctions

A full set of K_full test functions φᵢ(t) = φ((t - cᵢ)/rᵢ)/sᵢ and the leading K singular triplets of their matrix at a
reference grid t̃, Φ̃_full ≈ U_K Σ_K V_Kᵀ, built once, independent of the noisy data and of the time grid. They define
K fixed continuous test functions ψ = Σ_K⁻¹U_Kᵀ φ, the same for every noise realization and every number of time
points M+1 on the interval (assumption (A3)). φ(x) = exp(-9/(1-x²)) is WENDy's bump, rᵢ and cᵢ are in time units and
sᵢ is fixed at the reference grid (the ℓ² norm of the samples there, as _buildΦ normalizes).

On any uniform grid tt that covers the interval:
- `Φ_full, Φ̇_full = fullTestFunctionMatrices(tt, tf)`: the full set and its exact derivatives sampled at tt
  (sparse, K_full × length(tt))
- `Φ, Φ̇ = tf(tt)` (or `getTestFunctionMatrices(tt, tf)`): Φ = Σ_K⁻¹U_KᵀΦ_full and Φ̇ = Σ_K⁻¹U_KᵀΦ̇_full
  (K × length(tt)), i.e. ψₖ(tₘ) and ψₖ′(tₘ), so Φ̇ is consistent with Φ on every grid. On t̃, Φ = V_Kᵀ, WENDy's usual Φ.
Pass tf to WENDyProblem with the keyword `testFunctions`.

# Constructors
    FixedTestFunctions(tt_ref, radii; maxTestFunCondNum=1e4, minTestFunInfoNum=0.95, Kmax=200, K=nothing)
the radii in time units, one family of translates per radius, centered at every point of tt_ref whose support stays
in [t̃₀ + Δt̃, t̃_M - Δt̃] (as WENDy places them). K by WENDy's cutoffs (condition number, information, Kmax), or given.

    FixedTestFunctions(tt_ref, U_ref, params=WENDyParameters(); noiseDist=Normal, K=nothing)
the radii picked from reference data U_ref (noise free, or any data the estimates should not depend on) by WENDy's
default rule, and the cutoffs from params: on tt_ref, tf(tt_ref) is getTestFunctionMatrices(tt_ref, U_ref, params)
(up to the signs of the singular vectors).

# Fields
- tspan : the interval [t̃₀, t̃_M] of the reference grid; every φᵢ is supported in it
- radii, centers, scale : rᵢ, cᵢ and sᵢ of each of the K_full functions of the full set
- U : U_K, K_full × K, the leading left singular vectors of Φ̃_full (the coefficients of ψ over the full set, times Σ_K)
- Σ : Σ_K, the leading K singular values of Φ̃_full
- Mp1ref : the number of points of the reference grid
"""
struct FixedTestFunctions
    tspan::Tuple{Float64,Float64}
    radii::Vector{Float64}
    centers::Vector{Float64}
    scale::Vector{Float64}
    U::Matrix{Float64}
    Σ::Vector{Float64}
    Mp1ref::Int
end
function Base.show(io::IO, tf::FixedTestFunctions)
    print(io, "FixedTestFunctions(K = $(length(tf.Σ)) of K_full = $(length(tf.radii)) on $(tf.tspan), radii = $(unique(tf.radii)), Mp1ref = $(tf.Mp1ref))")
end
# rᵢ, cᵢ, sᵢ of the full set for the given radii (time units) on the reference grid tt
function _fullSet(tt::AbstractVector{<:Real}, radii::AbstractVector{<:Real})
    dt = (tt[end] - tt[1]) / (length(tt) - 1)
    tol = 1e-8*dt
    rr, cc, ss = Float64[], Float64[], Float64[]
    for ρ in radii
        @assert ρ > 0 "radii must be positive"
        n = floor(Int, ρ/dt + 1e-8)
        s = sqrt(sum(abs2, _bump(j*dt/ρ) for j in -n:n))
        cs = filter(c -> c - ρ >= tt[1] + dt - tol && c + ρ <= tt[end] - dt + tol, tt)
        isempty(cs) && @warn "No test function of radius $ρ fits in the time interval"
        append!(rr, fill(ρ, length(cs))); append!(cc, cs); append!(ss, fill(s, length(cs)))
    end
    rr, cc, ss
end
# Φ_fullᵀ and Φ̇_fullᵀ (length(tt) × K_full, sparse): column i holds φᵢ and φᵢ′ at the points of tt in its support
function _fullSetᵀ(tt::AbstractVector{<:Real}, radii, centers, scale)
    Kfull, Mp1 = length(radii), length(tt)
    colptr = Vector{Int}(undef, Kfull + 1)
    windows = Vector{UnitRange{Int}}(undef, Kfull)
    colptr[1] = 1
    for i in 1:Kfull
        # open support (cᵢ - rᵢ, cᵢ + rᵢ): the bump vanishes at its ends
        w = searchsortedfirst(tt, centers[i] - radii[i]):searchsortedlast(tt, centers[i] + radii[i])
        windows[i] = w
        colptr[i+1] = colptr[i] + length(w)
    end
    nnz = colptr[end] - 1
    rowval, nzval, nzval′ = Vector{Int}(undef, nnz), Vector{Float64}(undef, nnz), Vector{Float64}(undef, nnz)
    Threads.@threads for i in 1:Kfull
        r, c, s = radii[i], centers[i], scale[i]
        for (j, m) in enumerate(windows[i])
            x = (tt[m] - c) / r
            ix = colptr[i] + j - 1
            rowval[ix] = m
            nzval[ix] = _bump(x) / s
            nzval′[ix] = _bump′(x) / (r*s)
        end
    end
    SparseMatrixCSC(Mp1, Kfull, colptr, rowval, nzval), SparseMatrixCSC(Mp1, Kfull, copy(colptr), copy(rowval), nzval′)
end
function _checkGrid(tt::AbstractVector{<:Real}, tf::FixedTestFunctions)
    @assert issorted(tt) && all(abs.(diff(tt) .- (tt[2] - tt[1])) .< 1e-6) "Must use uniform time grid diff(tt) = $(diff(tt))"
    span = tf.tspan[2] - tf.tspan[1]
    @assert tt[1] <= tf.tspan[1] + 1e-8*span && tt[end] >= tf.tspan[2] - 1e-8*span "The time grid [$(tt[1]), $(tt[end])] must cover the interval $(tf.tspan) of the fixed test functions"
end
"""
    fullTestFunctionMatrices(tt, tf::FixedTestFunctions)

Φ_full and Φ̇_full (sparse, K_full × length(tt)): the full set of tf and its exact derivatives sampled at the times tt
(uniform, covering tf.tspan). Φ = Σ_K⁻¹U_KᵀΦ_full and Φ̇ = Σ_K⁻¹U_KᵀΦ̇_full are given directly by tf(tt).
"""
function fullTestFunctionMatrices(tt::AbstractVector{<:Real}, tf::FixedTestFunctions)
    _checkGrid(tt, tf)
    Pᵀ, Ṗᵀ = _fullSetᵀ(tt, tf.radii, tf.centers, tf.scale)
    permutedims(Pᵀ), permutedims(Ṗᵀ)
end
function FixedTestFunctions(
    tt::AbstractVector{<:Real}, radii::AbstractVector{<:Real};
    maxTestFunCondNum::Real=1e4, minTestFunInfoNum::Real=0.95, Kmax::Int=200, K::Union{Nothing,Int}=nothing,
    ll::LogLevel=Warn
)
    with_logger(ConsoleLogger(stderr, ll)) do
        @assert issorted(tt) && all(abs.(diff(tt) .- (tt[2] - tt[1])) .< 1e-6) "The reference grid must be uniform"
        rr, cc, ss = _fullSet(tt, radii)
        Pᵀ, _ = _fullSetᵀ(tt, rr, cc, ss)
        @info "  Fixed test functions: K_full = $(length(rr)) on the reference grid, Mp1 = $(length(tt))"
        F, _K = _reduceTestFunctions(Matrix(Pᵀ'), maxTestFunCondNum, minTestFunInfoNum, isnothing(K) ? Kmax : max(K, Kmax))
        K = isnothing(K) ? _K : min(K, length(F.S))
        FixedTestFunctions((Float64(tt[1]), Float64(tt[end])), rr, cc, ss, Matrix(F.U[:, 1:K]), F.S[1:K], length(tt))
    end
end
function FixedTestFunctions(
    tt::AbstractVector{<:Real}, U::AbstractVecOrMat{<:Real}, params::WENDyParameters=WENDyParameters();
    noiseDist::Type{<:Distribution}=Normal, K::Union{Nothing,Int}=nothing, ll::LogLevel=Warn
)
    U = U isa AbstractVector ? reshape(U, length(U), 1) : U
    tt, X = _transformData(noiseDist, tt, U)
    radii = with_logger(ConsoleLogger(stderr, ll)) do
        _testFunctionRadii(tt, X, params.radiusMinTime, params.radiusMaxTime, params.numRadii, params.testFunSubRate,
            params.radiiParams, params.Kᵣ)
    end
    dt = (tt[end] - tt[1]) / (length(tt) - 1)
    FixedTestFunctions(tt, radii .* dt; maxTestFunCondNum=params.maxTestFunCondNum,
        minTestFunInfoNum=params.minTestFunInfoNum, Kmax=params.Kmax, K, ll)
end
"""
    getTestFunctionMatrices(tt, tf::FixedTestFunctions)
    tf(tt)

Φ = Σ_K⁻¹U_KᵀΦ_full and Φ̇ = Σ_K⁻¹U_KᵀΦ̇_full (K × length(tt)) at the times tt (uniform, covering tf.tspan): the fixed
test functions ψₖ and their exact derivatives sampled at tt. Φ_full is never formed densely.
"""
function getTestFunctionMatrices(tt::AbstractVector{<:Real}, tf::FixedTestFunctions; ll::LogLevel=Warn)
    _checkGrid(tt, tf)
    dt = (tt[end] - tt[1]) / (length(tt) - 1)
    K = length(tf.Σ)
    with_logger(ConsoleLogger(stderr, ll)) do
        minimum(tf.radii) / dt < 2 && @warn "The narrowest fixed test function (radius $(minimum(tf.radii))) has fewer than 2 grid points per radius at dt = $dt"
        K > length(tt) && @warn "K = $K test functions on $(length(tt)) time points: Φ is rank deficient"
        @info "  Fixed test functions: K = $K on Mp1 = $(length(tt))"
    end
    Φ_full, Φ̇_full = fullTestFunctionMatrices(tt, tf)
    Wᵀ = permutedims(tf.U ./ tf.Σ')         # Σ_K⁻¹U_Kᵀ (K × K_full)
    _mulSparse(Wᵀ, Φ_full), _mulSparse(Wᵀ, Φ̇_full)
end
# A*B for dense A and sparse B, threaded over the columns of B (faster than the generic method)
function _mulSparse(A::AbstractMatrix{<:Real}, B::SparseMatrixCSC)
    K = size(A, 1)
    out = zeros(K, size(B, 2))
    rv, nz = rowvals(B), nonzeros(B)
    Threads.@threads for m in axes(B, 2)
        for ix in nzrange(B, m)
            i, v = rv[ix], nz[ix]
            @inbounds @simd for k in 1:K
                out[k, m] += A[k, i]*v
            end
        end
    end
    out
end
(tf::FixedTestFunctions)(tt::AbstractVector{<:Real}) = getTestFunctionMatrices(tt, tf)
