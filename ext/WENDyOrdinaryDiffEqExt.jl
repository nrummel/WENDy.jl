## Forward solves for the output error least squares problem (see OutputErrorProblem), loaded with OrdinaryDiffEq
module WENDyOrdinaryDiffEqExt
    using WENDy: WENDy, OutputErrorProblem, WENDyParameters
    using OrdinaryDiffEq: ODEProblem, ODEFunction, Rodas5P, DEVerbosity, SciMLLogging
    import OrdinaryDiffEq

    # params.fsAlg = nothing uses Rodas5P, recommended for small stiff systems at tight tolerances
    _alg(::Nothing) = Rodas5P()
    _alg(alg) = alg

    # with WENDy._OE_SYMBOLIC[] the problem carries the symbolic state jacobian and time gradient of f!, so the
    # Rosenbrock method does not differentiate f! itself
    _odeFunction(oe::OutputErrorProblem) =
        isnothing(oe.∇ₓf!) ? oe.f! : ODEFunction{true}(oe.f!; jac=oe.∇ₓf!, tgrad=oe.∂ₜf!)

    function WENDy._forwardSolve(oe::OutputErrorProblem, p::AbstractVector, u₀::AbstractVector, params::WENDyParameters)
        tRng = (oe.tt[1], oe.tt[end])
        dt = (oe.tt[end] - oe.tt[1]) / (length(oe.tt) - 1)
        sol = OrdinaryDiffEq.solve(ODEProblem(_odeFunction(oe), u₀, tRng, p), _alg(params.fsAlg);
            reltol=params.fsReltol, abstol=params.fsAbstol,
            saveat=dt, verbose=DEVerbosity(SciMLLogging.None())
        )
        reduce(vcat, um' for um in sol.u)
    end
end
