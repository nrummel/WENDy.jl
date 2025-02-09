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
        fsAlg::OrdinaryDiffEqAlgorithm=Rodas4P(),
        fsU0Free::Bool=true
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
- fsAlg::OrdinaryDiffEqAlgorithm = Rodas4P() :  forward solve algorithm used by the forward solve nonlinear least squares algorithm
- fsU0Free::Bool = true : Specifies if the forward solve algorithm should also optimize over the initial condition
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
    fsAlg::OrdinaryDiffEqAlgorithm      = Rodas4P()
    fsU0Free::Bool                      = true
end 
##
struct WENDyInternals{lip, DistType}
    J::Int # number of parameters (to be estimated)
    b₀::AbstractVector{<:AbstractFloat}
    sig::AbstractVector{<:AbstractFloat}
    tt::AbstractVector{<:AbstractFloat}
    X::AbstractMatrix{<:AbstractFloat} 
    V::AbstractMatrix{<:AbstractFloat}
    Vp::AbstractMatrix{<:AbstractFloat}
    f!::Function 
    ∇ₓf!::Function
    # Only valid when the problem is linear 
    G::AbstractMatrix{<:Real}
    # Only necessary when the problem is non linear
    ∇ₚf!::Function
    ∇ₚ∇ₓf!::Function
    Hₚf!::Function
    Hₚ∇ₓf!::Function
end
## IRLS 
abstract type IRLSIter end 
##
struct LinearIRLSIter <: IRLSIter
    b₀::AbstractVector{<:AbstractFloat}
    G0::AbstractMatrix{<:AbstractFloat}
    Rᵀ!::Function 
end 
##
struct NonLinearIRLSIter <: IRLSIter
    b₀::AbstractVector
    Rᵀ!::Function 
    r!::Function
    ∇r!::Function
    reltol::AbstractFloat
    abstol::AbstractFloat
    maxiters::Int
end 
## 
abstract type CostFunction end
##
struct FirstOrderCostFunction <: CostFunction
    f::Function 
    ∇f!::Function 
end
##
struct SecondOrderCostFunction <: CostFunction
    f::Function 
    ∇f!::Function 
    Hf!::Function 
end
##
struct LeastSquaresCostFunction <: CostFunction 
    r!::Function 
    ∇r!::Function 
    KD::Int
end 
"""
    WENDyProblem{lip, DistType}(...)

A WENDyProblem struct pre-computes and allocates data structures for efficient solving of the parameter inverse problem

# Constructor 
    WENDyProblem(
        _tt::AbstractVector{<:Real}, 
        U::AbstractVecOrMat{<:Real}, 
        _f!::Function, 
        J::Int; 
        linearInParameters::Val{lip}=Val(false), 
        noiseDist::Val{DistType}=Val(Normal), 
        params::WENDyParameters=WENDyParameters(), 
        constraints::Union{Nothing,AbstractVector{Tuple{<:Real,<:Real}}}=nothing, 
        ll::LogLevel=Warn
    )
## Arguments 
- _tt::AbstractVector{<:Real} : vector of times (equispaced)
- U::AbstractVecOrMat{<:Real} : Corrupted state variable data 
- _f!::Function : Right hand-side of the differential equation
    Must be of the form f!(du, u, p, t)
- J::Int : number of parameters (to be estimated)
- linearInParameters::Val{lip}=Val(false) : (optional) specify whether the right hand side is `linear in parameters' for improved computational efficiency
- noiseDist::Val{DistType}=Val(Normal) : (optional) specify the distribution of the measurement noise. Choose either Val(Normal) for additive Gaussian noise of Val(LogNormal) for multiplicative LogNormal noise.
- params::WENDyParameters : (optional) struct of hyper-parameters for the WENDy Algorithm (see the doc for WENDyParameters)
- constraints=nothing : (optional) Linear box constraints for each parameter, ∀j ∈ [1, ⋯,J], ℓⱼ ≤ pⱼ ≤ uⱼ. Accepts constraints as a list of tuples, [(ℓ₁,u₁), ⋯]. Note: this only is compatible with the TrustRegion solver.
- ll::LogLevel=Warn : (optional) see additional algorithm information by setting ll=Info

# Fields
- D::Int : number of state variables
- J::Int : number of parameters (to be estimated)
- Mp1::Int : (Mp1+1) number of data points in time 
- K::Int : number of test functions 
- u₀::AbstractVector{<:Real} : Initial Condition of the ODE (Necessary for the forward solver)
- constraints : vector of tuples containing linear constraints for each parameter
- data : Internal data structure 
- oels::LeastSquaresCostFunction : Cost function for the comparison method
- wlsq::LeastSquaresCostFunction : Cost function for the weak form least squares problem
- wnll::SecondOrderCostFunction : Cost function for the weak form negative log-likelihood 

"""
struct WENDyProblem{lip, DistType}
    D::Int # number of state variables
    J::Int # number of parameters (to be estimated)
    Mp1::Int # (Mp1+1) number of data points in time 
    K::Int # number of test functions 
    u₀::AbstractVector{<:Real}
    constraints::Union{Nothing,AbstractVector{Tuple{<:Real,<:Real}}}
    data::WENDyInternals{lip, DistType}
    # Cost functions 
    oels::LeastSquaresCostFunction
    wlsq::LeastSquaresCostFunction
    wnll::SecondOrderCostFunction 
end 