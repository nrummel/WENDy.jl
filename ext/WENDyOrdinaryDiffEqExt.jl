## Forward solves for the output error least squares problem (see OutputErrorProblem), loaded with OrdinaryDiffEq
module WENDyOrdinaryDiffEqExt
    using WENDy: WENDy, OutputErrorProblem, WENDyParameters
    using OrdinaryDiffEq: ODEProblem, Rodas5P, DEVerbosity, SciMLLogging
    import OrdinaryDiffEq

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
end
