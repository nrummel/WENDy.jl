## Forward solves for the output error least squares problem (see OutputErrorProblem). Loaded with
## OrdinaryDiffEqRosenbrock, the sublibrary with the default Rodas5P, which `using OrdinaryDiffEq` also loads.
module WENDyOrdinaryDiffEqExt
    using WENDy: WENDy, OutputErrorProblem, WENDyParameters
    using OrdinaryDiffEqRosenbrock: ODEProblem, ODEFunction, Rodas5P, SciMLBase
    import OrdinaryDiffEqRosenbrock as OrdinaryDiffEq
    using OrdinaryDiffEqRosenbrock.OrdinaryDiffEqCore: DEVerbosity, SciMLLogging
    using LinearAlgebra: I

    # params.fsAlg = nothing uses Rodas5P, recommended for small stiff systems at tight tolerances
    _alg(::Nothing) = Rodas5P()
    _alg(alg) = alg

    function WENDy._forwardSolve(oe::OutputErrorProblem, p::AbstractVector, u₀::AbstractVector, params::WENDyParameters)
        tRng = (oe.tt[1], oe.tt[end])
        dt = (oe.tt[end] - oe.tt[1]) / (length(oe.tt) - 1)
        sol = OrdinaryDiffEq.solve(ODEProblem(oe.f!, u₀, tRng, p), _alg(params.fsAlg);
            reltol=params.fsReltol, abstol=params.fsAbstol,
            saveat=dt, verbose=DEVerbosity(SciMLLogging.None())
        )
        reduce(vcat, um' for um in sol.u)
    end
    # The forward sensitivity system (see WENDy.OESensitivity). Its rhs, jacobian and time gradient are FunctionWrappers
    # with fixed Float64 signatures, wrapped once more in these concrete callables, and the problem is FullSpecialize, so
    # the solver is compiled for these types only, whatever f! is.
    struct _Rhs; fw::WENDy._ZVec; end
    struct _Jac; fw::WENDy._ZMat; end
    (r::_Rhs)(dz, z, p, t) = r.fw(dz, z, p, t)
    (j::_Jac)(Jz, z, p, t) = j.fw(Jz, z, p, t)
    function WENDy._sensitivitySolve(oe::OutputErrorProblem, p::Vector{Float64}, u₀::Vector{Float64}, params::WENDyParameters)
        (; D, J, Mp1, sens) = oe
        tRng = (Float64(oe.tt[1]), Float64(oe.tt[end]))
        dt = (tRng[2] - tRng[1]) / (Mp1 - 1)
        z₀ = [u₀; zeros(D*J); vec(Matrix{Float64}(I, D, D))]
        fun = ODEFunction{true, SciMLBase.FullSpecialize}(_Rhs(sens.rhs!); jac=_Jac(sens.jac!), tgrad=_Rhs(sens.tgrad!))
        sol = OrdinaryDiffEq.solve(ODEProblem(fun, z₀, tRng, p), _alg(params.fsAlg);
            reltol=params.fsReltol, abstol=params.fsAbstol,
            saveat=dt, verbose=DEVerbosity(SciMLLogging.None())
        )
        length(sol.u) == Mp1 || return nothing
        Û = Matrix{Float64}(undef, Mp1, D)
        S = Array{Float64}(undef, D, J + D, Mp1)
        for (m, z) in enumerate(sol.u)
            Û[m, :] = view(z, 1:D)
            S[:, :, m] = reshape(view(z, D+1:length(z)), D, J + D)
        end
        Û, S
    end
end
