module NLPMLE
__precompile__(false)
## External imports
using Revise
using DifferentialEquations, ModelingToolkit, BSON
using Random, Statistics, LinearAlgebra, Logging, StaticArrays # stdlib
## Import specfics
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkit: @mtkmodel, @mtkbuild, ODESystem
using Symbolics: jacobian
using ImageFiltering: imfilter, Inner # can produce the same as conv in matlab 
## include other code
using FFTW: fft, ifft
include("exampleProblems.jl")
include("computeGradients.jl")
include("generateNoise.jl")
include("interp1d.jl")
include("testFunctions.jl")
# Write your package code here.

function runAlgo(;
    exampleFile::String=joinpath(@__DIR__, "../data/LogisticGrowth.mat"),
    noise_ratio::Real=0.05,
    subsample::Int=1,
    mt_params::AbstractVector = 2 .^(0:3),
    K_max::Real=5e3,
    radMeth::RadMethod=MtminRadMethod(),
    seed::Real=1.0,
    ϕ::Function=ExponentialTestFun(),
    toggle_VVp_svd::Union{Int, Nothing}=nothing
)
    BSON.@load exampleFile t u 
    Mp1, D = size(u)
    num_test_fun_params = length(mt_params)
    @assert Mp1 == length(t) "Number of time points should match dependent variable array"
    @assert mod(subsample,2) ==0 || subsample == 1 "Subsample rate should be divisible by 2"
    ## Subsample the data
    tobs = t[1:subsample:end]
    uobs = u[:,1:subsample:end]
    ## Add noise 
    Random.seed!(seed)
    uobs = generateNoise(uobs, noise_ratio)
    ## Start of algorithm 
    estimated_std = estimated_std(uobs)
    K_min = length(w)
    K_max = 5e3
    mt_max = maximum(floor((M-1)/2)-K_min,1) 
    mt_min = rad_select(tobs,xobs,ϕ,mt_max)
    
    mt = zeros(num_test_fun_params, D)
    for (n,mtp) in enumerate(mt_params),d in 1:N
        mt[n,d] = radMeth(xobs, tobs, ϕ, mtp, mt_min, mt_max)
    end
    mt = ceil(1 ./ mean(1 ./ mt, dims=2))

end






end # module