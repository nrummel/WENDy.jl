##
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
    optimMaxiters::Int                  = 200
    optimTimelimit::Real                = 200.0
    fsAlg::OrdinaryDiffEqAlgorithm      = Rosenbrock23()
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
## IRWLS 
abstract type IRWLS_Iter end 
##
struct Linear_IRWLS_Iter <: IRWLS_Iter
    b₀::AbstractVector{<:AbstractFloat}
    G0::AbstractMatrix{<:AbstractFloat}
    Rᵀ!::Function 
end 
##
struct NLS_iter <: IRWLS_Iter
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
##
struct WENDyProblem{lip, DistType}
    D::Int # number of state variables
    J::Int # number of parameters (to be estimated)
    Mp1::Int # (Mp1+1) number of data points in time 
    K::Int # number of test functions 
    u0::AbstractVector{<:Real}
    constraints::Union{Nothing,AbstractVector{Tuple{<:Real,<:Real}}}
    data::WENDyInternals{lip, DistType}
    # Cost functions 
    fslsq::LeastSquaresCostFunction
    wlsq::LeastSquaresCostFunction
    wnll::SecondOrderCostFunction 
end 