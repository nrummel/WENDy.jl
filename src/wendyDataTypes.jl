"""
Hyper-parameters for the WENDy Algorithm

# Constructor 
    WENDyParameters(;   
        diagReg::Real=1.0e-10,
        radiusMinTime::Real=0.01,
        radiusMaxTime::Real=5.0,
        numRadii::Int=100,
        radiiParams::AbstractVector{<:Real}=2 .^(0:3),
        testFunSubRate::Real=2.0,
        maxTestFunCondNum::Real=1e4,
        minTestFunInfoNum::Real=0.95,
        Kmax::Int=200,
        Kᵣ::Union{Nothing,Int}=100,
        fsAbstol::Real=1e-8,
        fsReltol::Real=1e-8,
        nlsAbstol::Real=1e-8,
        nlsReltol::Real=1e-8,
        nlsMaxiters::Int=1000,
        optimAbstol::Real=1e-8,
        optimReltol::Real=1e-8,
        optimMaxiters::Int=500,
        optimTimelimit::Real=200.0,
        fsAlg=nothing,
        fsU0Free::Bool=true,
        approxHessian::Bool=false
    )

# Fields
- diagReg::Real = 1.0e-10 : Regularization constant for the covariance computations
- radiusMinTime::Real = 0.01 : Minimum test function radius (in time units matching _tt)
- radiusMaxTime::Real = 5.0 : Minimum test function radius (in time units matching _tt)
- numRadii::Int = 100 : Maximum number of radii to be checked in the min radii detection sub-algorithm
- radiiParams::AbstractVector{<:Real} = 2 .^(0:3) : Multiplied by the minRadius to give a list of radii to use when building the test function  matrix
- testFunSubRate::Real = 2.0 : Corresponds to how much we should sup-sample in the min radii detection sub-algorithm 
- maxTestFunCondNum::Real = 1e4 : Maximum Condition number of the test function matrix after svd reduction
- minTestFunInfoNum::Real = 0.95 : Minimum information (σₖ/σ₁)in the test function matrix after svd reduction
- Kmax::Int = 200 : Hard maximum size on the test function matrix
- Kᵣ::Union{Nothing,Int} = 100 : how many test function to spread through the time domain in the min radii detection sub-algorithm
- fsAbstol::Real = 1e-8 : forward solve absolute tolerance for solving ordinary differential equation
- fsReltol::Real = 1e-8 : forward solve relative tolerance for solving ordinary differntial equation
- nlsAbstol::Real = 1e-8 : nonlinear least squares absolute tolerance (only used in the IRLS WENDy algorithm)
- nlsReltol::Real = 1e-8 : nonlinear least squares relative tolerance (only used in the IRLS WENDy algorithm)
- nlsMaxiters::Int = 1000 : nonlinear least squares maximum iterations (only used in the IRLS WENDy algorithm)
- optimAbstol::Real = 1e-8 : absolute tolerance (used by all other optimization algorithms)
- optimReltol::Real = 1e-8 : relative tolerance (used by all other optimization algorithms)
- optimMaxiters::Int = 200 : maximum iterations (used by all other optimization algorithms)
- optimTimelimit::Real = 500.0 : maximum time in seconds (used by all other optimization algorithms)
- fsAlg = nothing : forward solve algorithm (any OrdinaryDiffEq ODE solver) used by the output error least squares problem; nothing uses Rodas5P() (see OutputErrorProblem)
- fsU0Free::Bool = true : Specifies if the forward solve algorithm should also optimize over the initial condition
- approxHessian::Bool = false : Second order solvers (TrustRegion, ARCqK and the hybrid methods) use an approximate hessian of the weak negative log-likelihood that leaves out the curvature of its log-determinant term. This skips the most expensive part of the hessian (it is the exact hessian of the Maholinobis distance when the problem is linear in parameters), turning these into quasi-Newton methods. Often several times faster when that term is small relative to the rest of the hessian, but it can converge to a worse point when it is not, so check the result against the exact hessian on a new problem
"""
@kwdef struct WENDyParameters   
    diagReg::Real                       = 1.0e-10
    radiusMinTime::Real                 = 0.01
    radiusMaxTime::Real                 = 5.0
    numRadii::Int                       = 100
    radiiParams::AbstractVector{<:Real} = 2 .^(0:3)
    testFunSubRate::Real                = 2.0
    maxTestFunCondNum::Real             = 1e4
    minTestFunInfoNum::Real             = 0.95
    Kmax::Int                           = 200
    Kᵣ::Union{Nothing,Int}              = 100
    fsAbstol::Real                      = 1e-8
    fsReltol::Real                      = 1e-8
    nlsAbstol::Real                     = 1e-8
    nlsReltol::Real                     = 1e-8
    nlsMaxiters::Int                    = 1000
    optimAbstol::Real                   = 1e-8
    optimReltol::Real                   = 1e-8
    optimMaxiters::Int                  = 500
    optimTimelimit::Real                = 200.0
    fsAlg                               = nothing
    fsU0Free::Bool                      = true
    approxHessian::Bool                 = false
end 
"""
    WENDySolution

What every solver returns:
- p̂::Vector{Float64} : the parameter estimate
- û₀::Union{Nothing,Vector{Float64}} : the estimated initial condition (output error least squares and the hybrid
  solver when params.fsU0Free), otherwise nothing
- iterations::Int : number of iterations
- iterates::Matrix{Float64} : the parameters at each iterate (J × iterates, starting at p₀)
- retcode::Symbol : :success, :maxiters, :timelimit or :failed, from the solver's own status
- time::Float64 : seconds spent in the solver
- alg::String : the solver, e.g. "TrustRegion"

The parameter covariance at the estimate is available from Ĉ, C_cons and C_∇²ℓ.
"""
struct WENDySolution
    p̂::Vector{Float64}
    û₀::Union{Nothing,Vector{Float64}}
    iterations::Int
    iterates::Matrix{Float64}
    retcode::Symbol
    time::Float64
    alg::String
end
