## estimate the standard deviation of noise by filtering then computing rmse
function estimate_std(_Y::AbstractMatrix{<:Real}; k::Int=6) 
    _, D = size(_Y) 
    std = zeros(D)
    for d = 1:D
        f = _Y[:,d]
        C = _fdcoeffF(k,0,-k-2:k+2)
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
From  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)
"""
function _fdcoeffF(k,xbar,x)
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
    