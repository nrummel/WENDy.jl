using Symbolics: jacobian
##
@mtkbuild mdl = FitzHugNagumoModel()
RHS = [eq.rhs for eq in equations(mdl)]
jacobian(RHS, parameters(mdl))