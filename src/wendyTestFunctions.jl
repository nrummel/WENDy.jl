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
    Mp1, D = size(U)
    radii = radiusMin:min(radiusMin+numRadii, radiusMax)
    errs = zeros(length(radii))
    # Select the fourier modes from just the where roots of unit would hit on the subsampled grid
    IX = Int(floor((Mp1-1)/testFunSubRate)) 
    # the fft will will be in order F[0,], F[1], ... F[M/2], F[-M/2], .. F[-1]
    # we want the F[M/2]
    
    for (r,radius) in enumerate(radii)
        VV = _buildV(radius, Mp1, dt, Kᵣ)
        V = VV[:,:,1]
        K = size(V,1)
        @tullio ΦU[k,d,m] := V[k,m]*U[m,d]
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
    Build the V_full for a particular radius and test function φ. If asked provide the functions for each test function.
"""
function _buildV(radius::Int, Mp1::Int, dt::Real, Kᵣ::Union{Int, Nothing}=nothing; derivativeOrder::Int=0)
    @assert isnothing(Kᵣ) || mod(Kᵣ,2)==0 "Kᵣ must be even "
    diam = Int(2*radius+1) 
    f = derivativeOrder == 0 ? WENDy.ExponentialTestFun() : _derivative(radius, dt, derivativeOrder)
    chunks = _buildChunks(Mp1, radius, Kᵣ)
    # chunks may be slightly longer than Kᵣ because 
    # of rounding with discretization (or could be nothing)
    Kᵣ = length(chunks)
    Vr = zeros(Kᵣ, Mp1) 
    xx = range(-1,1,diam)[2:end-1]
    ϕ = f.(xx)
    # normalize for stability
    ϕnorm  = norm(WENDy.ExponentialTestFun().(xx))
    ϕ /= ϕnorm
    for (k,c) in enumerate(chunks) 
        Vr[k, c] .= ϕ
    end 
    return Vr
end
""" """ 
function _computeDerivative_analytic(Vp_full::AbstractMatrix, fact::SVD, K::Int)
    @info "    Computing Vp with analytic Vp_full and svd(V_full)"
    (diagm(1 ./ fact.S) * fact.U' *  Vp_full)[1:K,:]
end
""" """
function _computeDerivative_fft(V::AbstractMatrix, dt::Real)
    @info "    Computing Vp with the fft"
    _, Mp1 = size(V)
    Vp_fft = zeros(size(V))
    _Vp_fft = fft(V', (1, ))
    k = mod(Mp1,2)==0 ? 
        vcat(0:Mp1/2, -Mp1/2+1:-1) : 
        vcat(0:floor(Mp1/2), -floor(Mp1/2):-1)
    Vp_fft = imag(ifft((-2*pi/(Mp1*dt))*k .* _Vp_fft, (1,)))'
end
"""
"""
function getTestFunctionMatrices(
    tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, radiusMinTime::Real, radiusMaxTime::Real, numRadii::Int, testFunSubRate::Real, radiiParams::AbstractVector{<:Real}, maxTestFunCondNum::Real, Kmax::Int, Kᵣ::Union{Nothing,Int}; 
    analyticVp::Bool=true, noSVD::Bool=false, ll::LogLevel=Info, debug::Bool=false
)
    with_logger(ConsoleLogger(stderr, ll)) do 
        @info "  Getting Test Function Matrices"
        @assert all(diff(tt) .- (tt[2] - tt[1]) .< 1e-6) "Must use uniform time grid"
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
        radiusMin =  _minRadius(U, dt, radiusMin, numRadii, radiusMax, testFunSubRate, Kᵣ)
        # return _minRadius(U, dt, radiusMin, numRadii, radiusMax, testFunSubRate, Kᵣ, debug=true)
        # radiusMin = 4
        @info "    radiusMin=$radiusMin"
        radii = filter(r->r < radiusMax, Int.(floor.(radiiParams*radiusMin)))
        if length(radii) == 0 
            radii = [radiusMax]
        end
        V_full = reduce(vcat, _buildV(r, Mp1, dt) for r in radii)
        Vp_full = reduce(vcat, _buildV(r, Mp1, dt; derivativeOrder=1) for r in radii)
        @info "    K_full=$(size(V_full,1))"
        if noSVD
            @info "    Returning the V_full, Vp_full"
            return V_full, Vp_full 
        end
        fact = svd(V_full)
        # Choose K off how quickly the singular values decay wrt to the max
        condNumbers = fact.S[1] ./ fact.S
        K = findlast(condNumbers .<= maxTestFunCondNum) 
        if K > min(Mp1, Kmax)
            @info "    K=$K, but Kmax=$Kmax and Mp1=$Mp1"
            K = min(Mp1,Kmax)
        end
        @info "    K=$(K)"
        V = fact.Vt[1:K,:]
        Vp = analyticVp ? _computeDerivative_analytic(Vp_full, fact, K) : _computeDerivative_fft(V, dt)
        @assert size(V,2) == Mp1 && size(Vp,2) == Mp1
        return debug ? (radii, V_full, Vp_full, V, Vp) : (V, Vp)
    end
end
## Convience wrapper to get testfunctions with the parameter struct
function getTestFunctionMatrices(tt::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, params::WENDyParameters; kwargs...)
    getTestFunctionMatrices(tt, U, params.radiusMinTime,params.radiusMaxTime, params.numRadii, params.testFunSubRate,params.radiiParams,params.maxTestFunCondNum,params.Kmax, params.Kᵣ; kwargs...)
end