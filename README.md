[![Build Status](https://github.com/nrummel/NLPMLE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nrummel/NLPMLE.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Motivation
Parameter estimation in this context is a sub-step of the system identification problem. System Identification is a basically when you have a dynamical system of interest, and you would like to model it analytically/numerically, so you want to fit a model to the data. Some models simply want to predict/extrapolate from the current data to then correct when more data comes online. In this case the model can be "wrong" if the uncertainty in the model estimate can be well characterize. Some such models such as ARMA and its derivatives as well as Kalman Filters and its derivatives.

```ad-question
title: Is there any apitite in CHARMNET?
I have more background in kalman filtering than almost anything else, I think it could be iteresting to use (W)SINDy in this context. 
##### Recent work
- [[Rosafalco et al. - 2024 - EKF-SINDy Empowering the extended Kalman filter w.pdf|EKF w/ SINDy]]
- [[Zotero/storage/4UWIK4TA/Zhang et al. - 2023 - Reduced-order Koopman modeling and predictive cont.pdf|Koopman + SINDy for Control]] 
```

If it is desired to however simulate the system at different initial conditions, then a variety of methods  provide a somewhat black box approach. Dynamic Mode Decomposition, Truncated Koopman Methods, sometimes ARMA derivatives if the time discretization is constant or scalable, and other less principled ML methods such as Neural Networks. A good (but old) survey of these topics is well documented in [[Ljung_L_System_Identification_Theory_for_User-ed2.pdf|Ljung]]. 

If however we have reason to believe the system can be modeled by a differential equation **of a particular form**, then learning the parameters of the system could allow for a more interpretable model. Furthermore, the error of the simulating or extrapolating the model can be well characterized. This is the the motivation for SINDy and variants.  

Unfortunately the form of equations that SINDy can learn, is quite limited. We want to work on a more general system to extend the existing framework and allow for WSINDy to tackle a broader class of parameter estimation problems.
## Review of Linear Parameter Estimation
This is the problem that the familiar FSNLLS,  SINDy, WSINDy, and WENDy all are looking at
Given the ODE (the PDE case can be thought of by taking a tensor product over the independent variables):
$$\dot{U}(t) = \sum\limits_{j=1}^J w_jf_j(U(t))$$
### SINDy Framework
The seminal [[Brunton et al. - 2016 - Discovering governing equations from data Sparse .pdf|paper by Brunton]]. Given data on a grid with additive noise
$$\{t_m, u_m\}_{m=1}^M = \{t_m, u^*_m +\varepsilon_m\}_{m=1}^M \text{ where } \varepsilon_m \stackrel{idd}{\sim} N(0,\sigma^2)$$
We build the linear system 
$$\dot{\tilde{u}} = \Theta w $$
where $\Theta_{jm} = f_j(u_m)$ and $\dot{\tilde{u}}_m = \tilde{D}[u_1,\cdots,u_M]$ is an approximate of the $\dot{u}(t_m)$. This works quite well but it has been shown that even with considerable effort to "de-noise" or compensate for noise in the data that the operator $\tilde{D}$ can the exacerbate effects of noise.
The following estimation problem is solved in variety of ways:
$$w = \underset{\nu}{\operatorname{armin}} \tfrac{1}{2}\|\Theta \nu - \dot{\tilde{u}}\|_2^2 +\lambda\|\nu\|_*$$
The $\|\cdot\|_*$ is a regularizer on $w$ chosen to promote sparse solutions. This in practice can be the 1 norm or 0 norm. This can per solved via LASSO or orthogonal matching pursuit (varieties of alternative) respectively. 
### WSINDy 
. This extends the SINDy framework but now instead of looking at the strong for we have 
$$\langle \dot{\Phi},u\rangle=\langle\Phi,\Theta\rangle w$$
where $\langle \Phi, \cdot \rangle_k \approx \int_{t_1}^{t_J} \varphi_k(\tau) (\cdot(\tau)) d\tau$ where $\varphi_k \in C_c^\infty[t_1,t_J]$. These are the familiar test functions for the weak form of the differential equation. The boon here is that the derivative can be passed to the test functions to forgo the approximating the derivative. This along with other properties intrinsic to the weak form make it more robust to noisy data. 
Again this is posed as a regularized least squares
$$w = \underset{\nu}{\operatorname{armin}} \tfrac{1}{2}\|\langle\Phi,\Theta\rangle \nu - \dot{\Phi},u\rangle \|_2^2+\lambda\|\nu\|_*$$
References: 
- [[Messenger and Bortz - 2021 - Weak SINDy Galerkin-Based Data-Driven Model Selec.pdf|WSINDy ODE]] 
- [[Zotero/storage/KBUXC8MA/Messenger and Bortz - 2021 - Weak SINDy For Partial Differential Equations.pdf|WSINDy For PDE]] 
- [[WENDy_ode2023.pdf|WENDy for ODE]]
### WENDy
 This algorithm aims to work from a statistical framework instead of looking at the squared error of the coefficients, we instead isolated the error that is related to noise and come up with an estimator based on the asymptotic distribution of the residual. After much derivation and some assumption due to integration error and model mismatch error being negligible we find
$$S_w^{-1/2}(\langle\Phi,\Theta\rangle w - \dot{\Phi}) \stackrel{asymp}{\sim} N(0, I)$$
where $L_w = \underbrace{\dot{\Phi}}_{\stackrel{\Delta}{=} L_0} + \underbrace{\langle \Phi, \nabla_u \Theta \rangle}_{\stackrel{\Delta}{=} L_1} \times_3 w$ 
and $S_w = L_wL_w^T$.   $S_w^{1/2}$ is the Cholesky Factorization of $S_w$.
Looking at the negative log-likelihood
$$-\mathcal{L}(w;u) \stackrel{asymp}{=} (Gw-b)^TS_w^{-1}(Gw-b) + \log(\det(S_w))$$
where $G =(\langle\Phi,\Theta\rangle$ and $b=\dot{\Phi}$. Assuming that we have already identified the $\{f_j\}_{j=1}^J$ that correspond to a "good" model. The regularization is no longer used. Because most of the the Eigen values of $S_w$ are very close to 0. Then we see that we can approximate the maximum likelihood estimator of $w$ by looking at the following
$$w = \underset{\nu}{\operatorname{armin}} f(\nu) \text{ where } f(w)\stackrel{\Delta}{=} (G\nu-b)^TS_\nu^{-1}(G\nu-b) $$
This is a **non-linear and non-convex optimization problem**! This minimization is done currently by an iterative re-weighted least squares routine which is a fixed point method. 
$$w_{n+1} = F(w_n) =  (G^T S_{w_n}^{-1}G)^{-1} S_{w_n}^{-1/2}G^Tb$$
This can be convergent, but is not always when initialization point $w_0$ is outside of the contractive region of the map $F$. 
```ad-note
title: Local Optimization Effort 
The cost function can be minimized in others ways. For instance a local optimization method can used instead, such a varieties of gradient descent, 2$^\text{nd}$ order methods, interior point methods. 

Deriving the gradient of the $f$ we see 
$$\begin{align*}
\nabla_w f &= 2S_w^{-1}(Gw-b) + (Gw-b)^T(\nabla_w S_w^{-1})(Gw-b)  \\
\nabla_w S_w^{-1} &= S_w^{-1}\times_3(L_0 \times_1 L_1^T + L_1 \times_3 L_0^T)\times_4S_w^{-1}
\end{align*}$$
##### Results
I implemented a large sweep comparing using this local optimization solver compared to IRWLS over all of the examples in the WENDy paper. 
- In general the MLE and IRWLS gives similar approximations of $w$ while the MLE is slower
- This does address the Hindmarsh-Rose behavior at high noise and lower number of data points by giving a improved answer to the OLS solution while the IRWLS diverges. 
```

```ad-note
title: Global Optimization Effort
While the local optimization methods gives a more robust solution to the IRWLS in general. There is no gaurentee that it is converging to a global or local minimum. So I have been developing a Branch and Bound algorithm for the MLE.
##### Summary of Work
- I have developed [[Bounding MLE|numerous lower bounds]] to make the algorithm more efficient
- I have implemented this in MATLAB
- The method coverges to a superior minimum when given an initial search space bases on a confidence interval derived from the local optimization method's estimate.
![[manufactured_BnB_dim12.png]]
```

## Non-Linear Parameter Estimation
Instead of considering 
$$\dot{U}(t) = \sum\limits_{j=1}^{J}w_jf_j(U(t))$$
We now look at 
$$\dot{U}(t) =  F(U(t), w)$$
### The Residual's Distribution 
If we consider data of the familiar form 
$$\{t_m, u_m\}_{m=1}^M = \{t_m, u^*_m +\varepsilon_m\}_{m=1}^M \text{ where } \varepsilon_m \stackrel{idd}{\sim} N(0,\sigma^2)$$
The residual in the weak form is 
$$\begin{align*}
	r(w) &= \langle \Phi, F(u, w)\rangle - \langle \dot{\Phi}, u \rangle \\
	&\stackrel{\Delta}{=} G(u,w) - b(u) \\
\end{align*}$$
Notice that while $b$ is linear in $u$ by the nature of the inner product $G$ is not! Now, for ease of notation, we define $G(w) = G(u, w)$, $G^*(u,w) = G(u^*,w)$, $b^* = b(u^*)$ , and $b^\epsilon = b(\epsilon)$. Now consider $w^*$ to be the true weight and $w$ to be the approximated weights.
$$\begin{align*}
	r(w) &= G(w) - b \\
	&= G(w)+ G^*(w) - G^*(w) +G^*(w^*) -G^*(w^*) - b^* - b^\epsilon\\
	&= \underbrace{G(w) - G^*(w)}_{e^\Theta} + \underbrace{G^*(w) -G^*(w^*)}_{e^{\text{int}}} + \underbrace{G^*(w^*) - b^*}_{r^*} -b^\epsilon
\end{align*}$$
The WENDy paper states that in the linear case that $e^\text{int}$ is due to numerical integration. This still should be largely true, there could be some more numerical error because evaluation $G(\cdot)$ may have non-negligible noise, but for the sake of simplicity perhaps we can say this is still small. The $r^*$ is the residual evaluated at the true weights, so in general this should be small if enough data is present so that evaluation of $G$ can be done to high accuracy. Ok so maybe we still can say that as the number of points gets large
$$r(w) \rightarrow e^\Theta -b^\epsilon$$
Inspecting this quantity, we see that we need to linearize $G$ about the true data $u^*$:
$$\begin{align*}
	e^\Theta-b^\epsilon &= G^*(w) + \langle \nabla_u G^*(w), \epsilon\rangle + H(u^*,w,\epsilon) - G^*(w) -b^\epsilon \\
	&= \langle \nabla_u G^*(w), \epsilon\rangle  -b^\epsilon + H(u^*,w,\epsilon)\\
	&= \langle \nabla_u G^*(w) - \dot{\Phi}, \epsilon\rangle
\end{align*}$$
```ad-warning
title: Wait a second
We can't evaluate $\nabla_u G^*$! We in practice approximate with $\nabla_u G$. Is this part of the assymptotic argument... seems sketchy...
```
Notice that because $\epsilon_m \stackrel{iid}{\sim} N(0,\sigma^2) \Leftrightarrow \epsilon \sim N(0, \sigma^2 I)$. Now if we say that $|\epsilon| \ll 1$ then because $H = O(\epsilon^2)$ and can be throw out, we have linear function of a multivariate gaussian, which is itself multivariate Gaussian
$$S_w^{-1/2} (e^\Theta - b^\epsilon) \stackrel{asymp}{\sim} N(0, I)$$
But now  $L_w = \underbrace{\dot{\Phi}}_{\stackrel{\Delta}{=} L_0} + \underbrace{\langle \Phi, \nabla_u F(u,w) \rangle}_{\stackrel{\Delta}{=} L_1(w)}$,
$S_w = L_wL_w^T$, and $S_w^{-1/2}$ is the Cholesky factorization. This now has a negative log likelihood:
$$-\mathcal{L}(w; u) = \underbrace{(G(w)-b)^TS_w^{-1}(G(w)-b)}_{f(w;u)} + \log(\det(S_w))$$Neglecting  the $\log\det$ term due to its lack of contribution in general, we obtain 
$$\begin{align*}
\nabla_w f &=  2S_w^{-1}(G(w)-b) \\
&+ (G(w)-b)^T(\nabla_wS_w^{-1})(G(w)-b) \\
\nabla_w S_w^{-1} &= S_w^{-1}\times_3 (L_0 \times_1 \nabla_w L_1 ^T + \nabla_w L_1 \times_3 L_0^T)\times_4S_w^{-1}
\end{align*}$$
```ad-summary
title: Highlighting Non-linearity
#### Linear Case
- $S_w$ is quadradic in $w$
$$\begin{align*}
	L_w &=  L_0 + L_1 w \\
	\nabla_w L_w &=  L_1
\end{align*}$$ 
- This equation is linear, so evaluating $F$ is a Mat-Vec
$$F(u,w) = \Theta w \Rightarrow G(w) = Gw$$
#### Nonlinear Case
- $S_w$ is the outer product of a constant and another non-linear term in $w$:
$$\begin{align*}
	L_w &=  L_0 + L_1(w) \\
	\nabla_w L_w &=  \nabla_w L_1(w)
\end{align*}$$ 
- This equation is nonlinear, so evaluating $F$ is no longer a mat vec this feeds down stream and means that evaluating the RHS of 
$$G(w) = \langle \Phi,F(u,w)\rangle$$
So evaluating the cost function require computing the inner product with the test functions. 
```
### Picking an initial $w_0$
Two obvious options: 
- We can use the McClaurin or Taylor series of $F$ and solve the least squares problem.	$$\begin{align*}
		F(u,w) &\approx F(u, 0) + \langle\nabla_w F(u,0), w\rangle\\
		r(w) &\approx \underbrace{\langle \Phi, u\rangle - F(u, 0)}_y - \underbrace{\langle\nabla_w F(u,0), w\rangle}_{Aw}\\
		\|r(w)\|_2 &\approx \|Aw - y\|_2 \\
		w &= \underset{\nu}{\operatorname{argmin}} \|r(w)\|_2\\
		&= (A^TA)^{-1}A^Tb
	\end{align*}$$
- Randomly pick in a desired domain or prior distribution? 
## Test Problems
### FitzHugh–Nagumo Equations
$$
\begin{align*} 
		\dot{V} &= c\left(V-\frac{V^3}{3}+R\right),  \\
		\dot{R} &= -\frac{1}{c}(V-a+b R) 
	\end{align*}
$$
As mentions this can be solved with linear in parameters with an equality constraint $w = (a,b,c,d), d = 1/c$, so perhaps not a necessary problem 
## From Other Literature
Here is a thesis from a more recent work on the development of the Forward Solve Non-linear Least Squares [Calver Thesis Ch2.6](https://tspace.library.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=43). This chapter covers possible examples. A common theme is that they look for **time lags** which I don't know if our frame work would allow, but here are a couple relevent equations:
### Goodwin Example - *Loop model* 
$$
\begin{aligned} 
& y_1^{\prime}(t)=\frac{a}{A+y_3(t)^\sigma}-b y_1(t), \\ 
& y_2^{\prime}(t)=\alpha y_1(t)-\beta y_2(t), \\ 
& y_3^{\prime}(t)=\gamma y_2(t)-\delta y_3(t),\end{aligned}
$$
### Mendes Problem
$$
\begin{aligned} 
y_1^{\prime}(t) & =\frac{k_1}{1+\left(\frac{P}{q_1}\right)^{q_2}+\left(\frac{q_3}{S}\right)^{q_4}}-k_2 y_1 \\
 y_2^{\prime}(t) & =\frac{k_3}{1+\left(\frac{P}{q_5}\right)^{q_6}+\left(\frac{q_7}{y_7}\right)^{q_8}}-k_4 y_2 \\ y_3^{\prime}(t) & =\frac{k_5}{1+\left(\frac{P}{q_9}\right)^{q_{10}}+\left(\frac{q_{11}}{y_8}\right)^{q_{12}}}-k_6 y_3 \\ y_4^{\prime}(t) & =\frac{k_7 y_1}{y_1+q_{13}}-k_8 y_4 \\ y_5^{\prime}(t) & =\frac{k_9 y_2}{y_2+q_{14}}-k_{10} y_5 \\ y_6^{\prime}(t) & =\frac{k_{11} y_3}{y_3+q_{15}}-k_{12} y_6 \\ y_7^{\prime}(t) & =\frac{k_{13} y_4\left(\frac{1}{q_{16}}\right)\left(S-y_7\right)}{1+\left(\frac{S}{q_{16}}\right)+\left(\frac{y_7}{q_{17}}\right)}-\frac{k_{14} y_5\left(\frac{1}{q_{18}}\right)\left(y_7-y_8\right)}{1+\left(\frac{y_7}{q_{18}}\right)+\left(\frac{y_8}{q_{19}}\right)} \\ y_8^{\prime}(t) & =\frac{k_{14} y_5\left(\frac{1}{q_{18}}\right)\left(y_7-y_8\right)}{1+\left(\frac{y_7}{q_{18}}\right)+\left(\frac{y_8}{q_{19}}\right)}-\frac{k_{15} y_6\left(\frac{1}{q_{20}}\right)\left(y_8-P\right)}{1+\left(\frac{y_8}{q_{20}}\right)+\left(\frac{P}{q_{21}}\right)} \end{aligned}
$$
### Michaelis-Menten Equation 
```ad-warning
title: Not Immediately Relevant 
I touched base with Bortz this am because the reference he pointed me to did not have a form of the equations that was non-linear in parameters:
$\begin{gathered}\frac{d S}{d \tau}=-k_1 S E+k_{-1} C_s, \quad \frac{d M}{d \tau}=-k_3 M E+k_{-3} C_m \\ \frac{d C_s}{d \tau}=k_1 S E-\left(k_{-1}+k_2\right) C_s, \quad \frac{d C_m}{d \tau}=k_3 M E-k_{-3} C_m\end{gathered}$
The *Hill Equation* does have a nonlinear parameter that is an exponent, but this is not a differential equation
$V=\frac{V_m S^n}{K^n+S^n}$
There is an example where this $n$ shows up in the RHS of the ODE but it is unclear what good choices of parameters would be? I am still talking with Bortz about how to proceed here
Defining $s_1=K_m^{-1} S_1, s_2=$ $K_e^{1 / n} S_2, t=\left(k_2 E_0 / K_m\right) \tau$, to obtain
$$
\frac{d s_1}{d t}=\alpha-f\left(s_1, s_2\right), \quad \frac{d s_2}{d t}=\gamma f\left(s_1, s_2\right)-\beta s_2,
$$
where $\alpha=a /\left(k_2 E_0\right), \beta=b K_m /\left(k_2 E_0\right), \gamma=\frac{k_{-1}}{k_2} K_e^{1 / n} K_m^{-1}$, and
$$
f\left(s_1, s_2\right)=\frac{s_1}{s_2^{-n}+1+s_1} .
$$
```
## Good Resource on What is **State of the Art**
It does look like forward solve least squares is state of the art, but using the adjoint method to approach the optimization plays an important role. See [Calver Thesis Ch4](https://tspace.library.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=78)

## Alternatives?
```ad-warning
title: Outstanding Questions

- How do we choose $w_0$ close enough to the $w$ so that $\tilde{F} \approx F$ Has anyone tried this? 
- Other alternatives: 
	- Shooting Methods (FSNLS) [Calver Disertation Presentation](https://www.cs.toronto.edu/~calver/presentations/caims2019_calver.pdf)
	- Adjoint methods [[Johnson - Notes on Adjoint Methods for 18.335.pdf|Johnson's Notes]]
	- Bayesian Collocation Methods [[Campbell - Bayesian Collocation Tempering and Generalized Pro.pdf|Campbell]]
- Good test problems?
	- _FitzHugh–Nagumo Equations_ from [[Ramsay et al. - 2007 - Parameter Estimation for Differential Equations a.pdf#page=2|Ramsay]]
	$$\begin{align*} 
		\dot{V} &= c\left(V-\frac{V^3}{3}+R\right),  \\
		\dot{R} &= -\frac{1}{c}(V-a+b R) 
	\end{align*}$$
	where $w = (a,b,c)$
	- _Continuously Stirred Tank Reactor Equations_ from [[Ramsay et al. - 2007 - Parameter Estimation for Differential Equations a.pdf#page=4|Ramsay]]
```

## Identifiably of Parameters
[Julia Implementation](https://docs.sciml.ai/ModelingToolkit/stable/tutorials/parameter_identifiability/) 