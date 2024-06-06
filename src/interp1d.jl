using Interpolations

function interp1(xpt, ypt, x; method="linear", extrapvalue=nothing)

    if extrapvalue == nothing
        y = zeros(x)
        idx = trues(x)
    else
        y = extrapvalue*ones(x)
        idx = (x .>= xpt[1]) .& (x .<= xpt[end])
    end
    
    if method == "linear"
        intf = interpolate((xpt,), ypt, Gridded(Linear()))
        y[idx] = intf[x[idx]]

    elseif method == "cubic"
        itp = interpolate(ypt, BSpline(Cubic(Natural())), OnGrid())
        intf = scale(itp, xpt)
        y[idx] = [intf[xi] for xi in x[idx]]
    end
    
    return y
end
