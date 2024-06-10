module NLPMLE
__precompile__(false)
include("exampleProblems.jl")
include("computeGradients.jl")
include("generateNoise.jl")
include("interp1d.jl")
include("testFunctions.jl")
using Random 
# Write your package code here.

function runAlgo(;
    exampleFile::String=joinpath(@__DIR__, "../data/LogisticGrowth.mat"),
    mdl::ODESystem=LOGISTIC_GROWTH_MODEL,
    ϕ::Function=ExponentialTestFun(),
    noise_ratio::Real=0.05,
    time_subsample_rate::Int=1,
    mt_params::AbstractVector = 2 .^(0:3),
    radMeth::RadMethod=MtminRadMethod(),
    pruneMeth::TestFunctionPruningMethod=SingularValuePruningMethod(UniformDiscritizationMethod())
    seed::Real=1.0,
)
    BSON.@load exampleFile t u 
    @assert M == length(t) "Number of time points should match dependent variable array"
    @assert mod(time_subsample_rate,2) ==0 || time_subsample_rate == 1 "Subsample rate should be divisible by 2"
    ## Subsample the data
    tobs = t[1:time_subsample_rate:end]
    uobs = u[:,1:time_subsample_rate:end]
    M, D = size(uobs)
    num_rad = length(mt_params)
    ## Add noise 
    Random.seed!(seed)
    uobs = generateNoise(uobs, noise_ratio)
    ## Start of algorithm 
    estimated_std = estimated_std(uobs)
    K_min = length(w)
    K_max = 5e3
    mt_max = maximum(floor((M-1)/2)-K_min,1) 
    mt_min = rad_select(tobs,xobs,ϕ,mt_max)
    
    mt = zeros(num_rad, D)
    for (n,mtp) in enumerate(mt_params),d in 1:N
        mt[n,d] = radMeth(xobs, tobs, ϕ, mtp, mt_min, mt_max)
    end
    mt = ceil(1 ./ mean(1 ./ mt, dims=2))

end






end # module