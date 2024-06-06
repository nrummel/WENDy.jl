
## Two types of test functions of ϕ
abstract type TestFunction end 
struct ExponentialTestFun <: TestFunction end 
function (ϕ::ExponentialTestFun)(x::Number,η::Real=9) 
    exp(-η*(1-x.^2).^(-1))
end
struct QuadradicTestFun <: TestFunction end
function (ϕ::QuadradicTestFun)(x::Number,η::Real=9) 
    (1-x.^2).^η
end
##
function rad_select(t0::AbstractVector,y::AbstractMatrix,
    ϕ::TestFunction, m_max::Int;inc::Real=1,
    sub::Real=2.1,q::Real=0.0,s::Real=1.0,m_min::Int=2
)::Int
    M,nstates = size(y)
    
    if isnothing(ϕ)
        return m_min
    elseif m_max<=m_min
        return m_min
    end
    dt = mean(diff(t0))
    t = 0:dt/inc:(M-1)*dt
    ms = m_min:m_max
    errs = zeros(length(ms))
    for (ix,m) in enumerate(ms)
        l = 2*inc*m-1
        t_phi = range(-1+dt/inc,1-dt/inc,Int(l))
        Qs = 1:Int(floor(s*inc*m)):Int(length(t)-2*inc*m)
        errs_temp = zeros(nstates,length(Qs))
        for Q in 1:length(Qs)
            phi_vec = zeros(length(t))
            phi_vec[Qs[Q]:Qs[Q]+length(t_phi)-1] = ϕ.(t_phi)
            phi_vec /= norm(phi_vec,2)
            for nn=1:nstates
                phiu_fft = (dt/sqrt(M*dt))*fft(phi_vec .* y[:,nn])
                alias = phiu_fft[1:Int(floor(M/sub)):Int(floor(inc*M/2))]
                errs_temp[nn,Q] = 2*(2*pi/sqrt(M*dt))*sum((0:length(alias)-1).*imag(alias[:]'))
            end
        end
        errs[ix] = sqrt(mean(errs_temp[:].^2))
    end
    b = getcorner(log.(errs),ms) # This corresponds to when 'pow' is empty
    return ms[b]
end
function fftshift(arr::AbstractArray)
    return circshift(arr, Int(floor(length(arr)/2)))
end
## Find where the error is with the secand line approximation is minimized in L1
function getcorner(U::AbstractVector{<:Real},xx::Union{AbstractVector{<:Real},Nothing}=nothing)
    NN = length(U)
    U = U / maximum(abs.(U))*NN
    errs = zeros(NN)
    for k=1:NN
        L1,L2,U1,U2 = build_lines(U,k,xx)
        # relative L1 err
        errs[k] = sum(abs.((L1-U1)./U1)) + sum(abs.((L2-U2)./U2))
    end
    replace!(errs, NaN=>Inf)
    _,ix = findmin(errs)
    return ix
end
## build secant lines splitting U
function build_lines(U::AbstractVector{<:Real},k::Int,xx::Union{AbstractVector,Nothing}=nothing)
   NN = length(U)
   subinds1 = 1:k
   subinds2 = k:NN
   x1 = isnothing(xx) ? subinds1 : xx[subinds1]
   x2 = isnothing(xx) ? subinds2 : xx[subinds2]
   U1 = U[subinds1]
   U2 = U[subinds2]
   m1,b1,L1 = lin_regress(U1,x1)
   m2,b2,L2 = lin_regress(U2,x2)
    return L1,L2,U1,U2
end
## fit a line with the secant of the end points
function lin_regress(U::AbstractVector{<:Real},x::AbstractVector{<:Real})
    m = (U[end]-U[1])/(x[end]-x[1])
    a = m*x[1]
    b = U[1] - a
    L = U[1] .+ m.*x .- a
    return m,b,L
end 
## Define RadMethods
abstract type RadMethod end 
struct DirectRadMethod <:RadMethod end
struct TimeFracRadMethod<:RadMethod end
struct MtminRadMethod <:RadMethod end

function (meth::DirectRadMethod)(xobs::AbstractVecOrMat{<:Real}, tobs::AbstractVector{<:Real}, ϕ::TestFunction, mtmin::Real, mtmax::Real, p::Real)
    return min(mtmax, p)
end 
function (meth::DirectRadMethod)(xobs::AbstractVecOrMat{<:Real}, tobs::AbstractVector{<:Real}, ϕ::TestFunction, mtmin::Real, mtmax::Real, p::Real)
    return min(mtmax, floor(length(tobs)*p))
end
function (meth::MtminRadMethod)(xobs::AbstractVecOrMat{<:Real}, tobs::AbstractVector{<:Real}, ϕ::TestFunction, mtmin::Real, mtmax::Real, p::Real)
    return min(mtmax, p*mtmin)
end
## Helper to get derivative of test functions
function _computeDerivative(ϕ::TestFunction, deg::Int) 
    if deg == 0 
        return ϕ
    end 
    @variables t
    Dt = Differential(t)^(deg)
    Df_sym = Dt(ϕ(t))
    Df_sym_exp = expand_derivatives(Df_sym)
    return build_function(Df_sym_exp,t;expression=false)
end

## Compute the values of the test function and its derivatives on a grid from [-1,1] with m points with weights that correspond to trapazoid rule. (normalized to have norm 1)
function _getTestFunctionWeights(ϕ::TestFunction, m::Int, maxd::Int)
    @variables t
    xf = range(-1,1,2*m+1);
    x = xf[2:end-1];
    Cfs = zeros(maxd+1,2*m+1);
    for (j,d)= enumerate(0:maxd)
        # Take the derivative and turn it into a function
        Df = _computeDerivative(ϕ,d)
        Cfs[j,2:end-1] = Df.(x)
        replace!(Cfs[j,2:end-1],NaN=>Df(eps()),missing=>Df(eps()))
        # handle infinite valuse by perturbing the evaluation point
        inds = findall(isinf.(abs.(Cfs[j,:])));
        for ix in inds
            # Cfs[j,ix] = Df(xf[ix]-sign(xf[ix])*eps()); ## What dan has
            Cfs[j,ix] = Df(x[ix]-sign(x[ix])*eps()); ## TODO: Possible bug
        end
    end
    return Cfs ./ norm(Cfs[1,:], 2);
end
## Define how to subsample the test function discritization
abstract type TestFunDiscritizationMethod end 
struct UniformDiscritizationMethod <:TestFunDiscritizationMethod end 
struct RandomDiscritizationMethod <:TestFunDiscritizationMethod end
struct SpecifiedDiscritizationMethod <:TestFunDiscritizationMethod
    ix::AbstractVector{<:Int}
end  
function _prepare_discritization(mt::Int,t::AbstractVector{<:Real},ϕ::TestFunction,max_derivative::Int)
    dt = mean(diff(t));
    M = length(t);
    Φ = _getTestFunctionWeights(ϕ,mt,max_derivative);
    return dt, M, Φ
end 

function (meth::UniformDiscritizationMethod)(mt::Int,t::AbstractVector{<:Real},ϕ::TestFunction,max_derivative::Int, K::Real)
    dt, M, Φ = _prepare_discritization(mt,t,ϕ,max_derivative)
    gap     = Int(max(1,floor((M-2*mt)/K)));
    dd      = 0:gap:M-2*mt-1;
    dd      = dd[1:min(K,end)];
    V       = zeros(length(dd),M);
    for j=1:length(dd)
        for (i,d) = enumerate(0:max_derivative)
            V[j,gap*(j-1)+1:gap*(j-1)+2*mt+1,i] = Φ[i,:]*(mt*dt)^(-d)*dt;
        end
    end
    return V
end

function (meth::RandomDiscritizationMethod)(mt::Int,t::AbstractVector{<:Real},ϕ::TestFunction,max_derivative::Int, K::Real)
    dt, M, Φ = _prepare_discritization(mt,t,ϕ,max_derivative)
    gaps = randperm(M-2*mt,K);        
    V = zeros(K,M);
    for j=1:K
        for (i,d) = enumerate(0:max_derivative)
            V[j,gaps[j]:gaps[j]+2*mt,i] = Φ[i,:]*(mt*dt)^(-d)*dt;
        end
    end
    return V
end

function (meth::SpecifiedDiscritizationMethod)(mt::Int,t::AbstractVector{<:Real},ϕ::TestFunction,max_derivative::Int, ::Real)
    dt, M, Φ = _prepare_test_fun_svd(mt,t,ϕ,max_derivative)
    center_scheme = unique(max.(min.(meth.ix,M-mt),mt+1));
    K = length(center_scheme);
    V = zeros(K,M,max_derivative+1);
    for j=1:K
        for (i,d) = enumerate(0:max_derivative)
            V[j,center_scheme[j]-mt:center_scheme[j]+mt,i] = Φ[i,:]*(mt*dt)^(-d)*dt;
        end
    end        
    return V
end
## Pruning Methods
abstract type TestFunctionPruningMethod end 
struct NoPruningMethod <: TestFunctionPruningMethod
    discMethod::TestFunDiscritizationMethod
end 
struct SingularValuePruningMethod <: TestFunctionPruningMethod 
    discMethod::TestFunDiscritizationMethod
    val::UInt 
    function SingularValuePruningMethod(discMethod::TestFunDiscritizationMethod=UniformDiscritizationMethod(),val::UInt=UInt(0))
        return new(discMethod, val)
    end
end

function _getK(K_max::Int, D::Int, num_rad::Int, Mp1::Int)
    return Int(min(floor(K_max/(D*num_rad)), Mp1))
end

function (meth::NoPruningMethod)(mt::AbstractVector{<:Real}, t::AbstractVector{<:Real}, ϕ::TestFunction,K_min::Int,K_max::Int,D::Int,num_rad)
    K = _getK(K_max, D, num_rad, length(t))
    V = cat(discMeth(m,t,ϕ,1,K) for m in mt;dims=1)
    return V[:,:,1], V[:,:,2]
end

function (meth::SingularValuePruningMethod)(mt::AbstractVector{<:Real},t::AbstractVector{<:Real},ϕ::TestFunction, K_min::Int,K_max::Int,D::Int,num_rad::Int)
    if length(mt) == 1
        return NoPruningMethod(meth.discMethod)(mt, t, ϕ, K)
    end
    K = _getK(K_max, D, num_rad, length(t))
    V = reduce(vcat,meth.discMethod(m,t,ϕ,0, K) for m in mt)
    Mp1 = length(t);
    dt = mean(diff(t));
    svd_fact = svd(V';full=false);
    U = svd_fact.U 
    sings = svd_fact.S;
    if meth.val == 0 
        # default is to find the corner adaptively
        corner_data = cumsum(sings)/sum(sings);
        corner_data[1:15]'
        ix = getcorner(corner_data);
        ix = min(max(K_min,ix),K);
    else 
        # one can specify the "corner" or where the svd values start falling off
        ix = findfirst(cumsum(sings.^2)/sum(sings.^2)>meth.val^2 > 0 );
        if isnothing(ix)
            ix = min(K,size(V,1));
        end
    end
    inds = 1:ix;
    Vt = U[:,inds]*dt;
    V = Matrix(Vt')
    Vp_hat = fft(Vt,(1,)); # the second argument specifies that we want to do the fft across columns like in matlab
    if mod(Mp1,2)==0
        k = vcat(0:Mp1/2, -Mp1/2+1:-1);
    else
        k = vcat(0:floor(Mp1/2), -floor(Mp1/2):-1);
    end
    Vp_hat = -((2*pi/Mp1/dt)*k).*Vp_hat;
    # For odd derivatives there is a loss of symmetry
    if mod(Mp1,2)==0
        # TODO : is this a bug? should be   Vp_hat[Int(Mp1/2),:] .= 0 
        Vp_hat[Int(Mp1/2)] = 0;        
    end
    Vp = Matrix(imag(ifft(Vp_hat,(1,)))');
    return V, Vp
end
