## Seed for reproducibility if asked 
function _seed(seed::Union{Nothing,Int})
    if (!isnothing(seed) && seed > 0) 
        Random.seed!(seed)
        @info "  Seeeding the noise with value $(seed)"
    else 
        @info "  no seeding"
    end
end
## Normal
function generateNoise(U_exact::AbstractMatrix{<:Real}, noiseRatio::Real, seed::Union{Nothing, Int}, isotropic::Bool, ::Val{Normal}) 
    @assert noiseRatio >= 0 "Noise ratio must be possitive"
    if noiseRatio == 0 
        return U_exact, zeros(size(U_exact)), zeros(size(U_exact,1))
    end
    # seed if asked
    _seed(seed)
    # corrupt data with noise
    U = similar(U_exact)
    noise = similar(U_exact)
    D = size(U,1)
    σ = zeros(D)
    mean_signals = isotropic ? mean(U_exact .^2) .* ones(size(σ)) : mean(U_exact .^2, dims=2) 
    for (d,signal) in enumerate(mean_signals)
        σ[d] = noiseRatio*sqrt(signal) # TODO: HERE
        dist = Normal(0, σ[d])
        noise[d,:] = rand(dist,size(U_exact,2))
        U[d,:] = U_exact[d,:] + noise[d,:]
    end
    return U,noise,σ
end
## Log normal 
function generateNoise(U_exact::AbstractMatrix{<:Real}, noiseRatio::Real, seed::Union{Nothing, Int}, isotropic::Bool, ::Val{LogNormal})
    @assert noiseRatio > 0 "Noise ratio must be possitive"
    if noiseRatio == 0 
        Mp1 = size(U_exact,2)
        return U_exact, Matrix{Float64}(I,Mp1,Mp1), zeros(size(U_exact,1))
    end
    # seed if asked
    _seed(seed)
    # corrupt data with noise
    U = similar(U_exact)
    noise = similar(U_exact)
    D = size(U,1)
    σ = zeros(D)
    σ .= noiseRatio
    for d in 1:D
        dist = LogNormal(0,σ[d]) # lognormal with logmean of 0, and 
        noise[d,:] = rand(dist,size(U_exact,2))
        U[d,:] = U_exact[d,:].*noise[d,:]
    end
    return U,noise,σ
end
## Convience function for calling with SimParams struct
function generateNoise(U_exact::AbstractMatrix{<:Real}, simParams::SimulationParameters, ::Val{DistType}) where DistType<:Distribution 
    generateNoise(U_exact, simParams.noiseRatio, simParams.seed, simParams.isotropic, Val(DistType))
end
## estimate the standard deviation of noise by filtering then computing rmse
function estimate_std(_Y::AbstractMatrix{<:Real}; k::Int=6) 
    D,Mp1 = size(_Y) 
    std = zeros(D)
    for d = 1:D
        f = _Y[d,:]
        C = fdcoeffF(k,0,-k-2:k+2)
        filter = C[:,end]
        filter = filter / norm(filter,2)
        std[d] = sqrt(mean(imfilter(f,filter,Inner()).parent.^2))
    end
    return std
end
"""
Compute coefficients for finite difference approximation for the
derivative of order k at xbar based on grid values at points in x.
This function returns a row vector c of dimension 1 by n, where n=length(x),
containing coefficients to approximate u^{(k)}(xbar), 
the k'th derivative of u evaluated at xbar,  based on n values
of u at x(1), x(2), ... x(n).  
If U is a column vector containing u(x) at these n points, then 
c*U will give the approximation to u^{(k)}(xbar).
Note for k=0 this can be used to evaluate the interpolating polynomial 
itself.
Requires length(x) > k.  
Usually the elements x(i) are monotonically increasing
and x(1) <= xbar <= x(n), but neither condition is required.
The x values need not be equally spaced but must be distinct.  
This program should give the same results as fdcoeffV.m, but for large
values of n is much more stable numerically.
Based on the program "weights" in 
B. Fornberg, "Calculation of weights in finite difference formulas",
SIAM Review 40 (1998), pp. 685-691.
Note: Forberg's algorithm can be used to simultaneously compute the
coefficients for derivatives of order 0, 1, ..., m where m <= n-1.
This gives a coefficient matrix C(1:n,1:m) whose k'th column gives
the coefficients for the k'th derivative.
In this version we set m=k and only compute the coefficients for
derivatives of order up to order k, and then return only the k'th column
of the resulting C matrix (converted to a row vector).  
This routine is then compatible with fdcoeffV.   
It can be easily modified to return the whole array if desired.
From  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)
"""
function fdcoeffF(k,xbar,x)
    n = length(x)
    if k >= n
       @error "*** length(x) must be larger than k"
    end
    
    m = k   # change to m=n-1 if you want to compute coefficients for all
             # possible derivatives.  Then modify to output all of C.
    c1 = 1
    c4 = x[1] - xbar
    C = zeros(n,m+1)
    C[1,1] = 1
    for i=1:n-1
        i1 = i+1
        mn = min(i,m)
        c2 = 1
        c5 = c4
        c4 = x[i1] - xbar
        for j=0:i-1
            j1 = j+1
            c3 = x[i1] - x[j1]
            c2 = c2*c3
            if j==i-1
                for s=mn:-1:1
                    s1 = s+1
                    C[i1,s1] = c1*(s*C[i1-1,s1-1] - c5*C[i1-1,s1])/c2
                end
                C[i1,1] = -c1*c5*C[i1-1,1]/c2
            end
            for s=mn:-1:1
                s1 = s+1
                C[j1,s1] = (c4*C[j1,s1] - s*C[j1,s1-1])/c3
            end
            C[j1,1] = c4*C[j1,1]/c3
            end
        c1 = c2
        end            # last column of c gives desired row vector
    return C
end
    