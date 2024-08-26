## Abstract
System identification is a ubiquitous challenge in the sciences, engineering, control and applied mathematics. Even when a model has been identified, a common struggle estimating unknown parameters, especially in the presence of noise. Historically, it is necessary to use shooting base methods to minimize a residual in forward simulation error. This can require forward simulation of the dynamics at each iteration of an optimization routine. We instead move the system into its weak form, and derive an asymptotic distribution of the coefficient residual. Then, we minimize the negative log-likelihood to give an estimate of the parameters. The novel use of the weak form allows for robustness to noise and the regressing against coefficient error allows us to remove the necessity for forward simulation. 
## Motivation
Parameter estimation in this context is a sub-step of the system identification problem. System Identification is a basically when you have a dynamical system of interest, and you would like to model it analytically/numerically, so you want to fit a model to the data. Some models simply want to predict/extrapolate from the current data to then correct when more data comes online. In this case the model can be "wrong" if the uncertainty in the model estimate can be well characterize. Some such models such as ARMA and its derivatives as well as Kalman Filters and its derivatives.

If it is desired to however simulate the system at different initial conditions, then a variety of methods  provide a somewhat black box approach. Dynamic Mode Decomposition, Truncated Koopman Methods, sometimes ARMA derivatives if the time discretization is constant or scalable, and other less principled ML methods such as Neural Networks. A good (but old) survey of these topics is well documented in [[Ljung_L_System_Identification_Theory_for_User-ed2.pdf|Ljung]]. 

If however we have reason to believe the system can be modeled by a differential equation **of a particular form**, then learning the parameters of the system could allow for a more interpretable model. Furthermore, the error of the simulating or extrapolating the model can be well characterized. This is the the motivation for SINDy and variants.  

Unfortunately the form of equations that SINDy can learn, is quite limited. We want to work on a more general system to extend the existing framework and allow for WSINDy to tackle a broader class of parameter estimation problems.
## Review of Prior Work
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
This extends the SINDy framework but now instead of looking at the strong for we have 
$$-\langle \dot{\Phi},u\rangle=\langle\Phi,\Theta\rangle w$$
where $\langle \Phi, \cdot \rangle_k \approx \int_{t_1}^{t_J} \varphi_k(\tau) (\cdot(\tau)) d\tau$ where $\varphi_k \in C_c^\infty[t_1,t_J]$. These are the familiar test functions for the weak form of the differential equation. The boon here is that the derivative can be passed to the test functions to forgo the approximating the derivative. This along with other properties intrinsic to the weak form make it more robust to noisy data. 
Again this is posed as a regularized least squares
$$w = \underset{\nu}{\operatorname{armin}} \tfrac{1}{2}\|\langle\Phi,\Theta\rangle \nu +\langle \dot{\Phi},u\rangle \|_2^2+\lambda\|\nu\|_*$$
References: 
- [[Messenger and Bortz - 2021 - Weak SINDy Galerkin-Based Data-Driven Model Selec.pdf|WSINDy ODE]] 
- [[Zotero/storage/KBUXC8MA/Messenger and Bortz - 2021 - Weak SINDy For Partial Differential Equations.pdf|WSINDy For PDE]] 
### WENDy
 This algorithm proposed in [[WENDy_ode2023.pdf|Bortz and Messenger 2023]] aims to work from a statistical framework instead of looking at the squared error of the coefficients. Assuming that we have already identified the $\{f_j\}_{j=1}^J$ that correspond to a "good" model. The regularization is no longer used. We instead isolated the error that is related to noise and come up with an estimator based on the asymptotic distribution of the residual. After much derivation and some assumption due to integration error and model mismatch error being negligible we find
$$S_w^{-1/2}(\langle\Phi,\Theta\rangle w +\langle \dot{\Phi},u\rangle) \stackrel{asymp}{\sim} N(0, I)$$
where $L_w = \underbrace{ \mathbb{I}_D \otimes \dot{\Phi}}_{\stackrel{\Delta}{=} L_0} + \underbrace{\nabla_u \Theta \circ_2 \Phi, }_{\stackrel{\Delta}{=} L_1} \times_3 w$ 
and $S_w = L_wL_w^T$.   $S_w^{1/2}$ is the Cholesky Factorization of $S_w$.
They lean on other generalized least squares literature to suggest that using iterative re-weighted least squares (IRWLS) as a possible way of finding a better solution. This is a fixed point method with a step of 
$$w_{n+1} = F(w_n) =  (G^T S_{w_n}^{-1}G)^{-1} S_{w_n}^{-1/2}G^Tb$$
This can be convergent, but is not always when initialization point $w_0$ is outside of the contractive region of the map $F$. Currently, $w_0 = \operatorname{argmin}\|Gw-b\|_2$ the ordinary least squares solution. 
#### Maximum Likelihood
This past spring I went through the trouble of deriving the likelihood [[WENDy_MLE.pdf| MLE derivation]] for the WENDy residual (assuming the convergence to the Normal). This mostly involves just taking lots of derivative. Looking at the negative log-likelihood
$$-\mathcal{L}(w;u) \stackrel{asymp}{=} (Gw-b)^TS_w^{-1}(Gw-b) + \log(\det(S_w))$$
where $G =(\langle\Phi,\Theta\rangle$ and $b=\dot{\Phi}$. Because most of the the Eigen values of $S_w$ get small the $\log\det$ term can be neglected and we instead minimize the Mahalonobis Distance. Then we see that we can approximate the maximum likelihood estimator of $w$ by looking at the following
$$w = \underset{\nu}{\operatorname{armin}} m(\nu) \text{ where } m(w)\stackrel{\Delta}{=} (G\nu-b)^TS_\nu^{-1}(G\nu-b) $$
This is a **non-linear and non-convex optimization problem**! This in part explains why IRWLS is not likely to be convergent when noise level/non-linearity (wrt data) in $G$ becomes high. 
#### Optimization Effort 
What I propose is instead we instead minimize the negative log-likihood. Which allows us to use more general optimization methods, such as 1$^{\text{st}}$ order methods like BFGS,  2$^\text{nd}$ order methods like trust region, ARC, and possibly interior point methods. 

In order to use these methods gradient and hessian information is necessary. This can be done through Automatic differentiation or approximated finite differences, but both of those options tend to be too slow when the amount of data/test functions/parameters becomes too high. Instead analytic derivatives are necessary.
##### Gradient Information
When the right hand side of the ODE is linear in $w$ we obtain the following derivatives:
$$\partial_{w_j} m = 2g_j^T S_w^{-1} (Gw-b) + (Gw-b)^T(\partial_{w_j} S_w^{-1})(Gw-b)$$
where
$$  
\begin{align*}
	 g_j &:= G[:,j]\\
	(L_1)_j &:= L_1[:,:,j]\\
\partial_{w_j} S_w^{-1} &= S_w^{-1} \partial_{w_j}S_w S_w^{-1}\\
&= -S_w^{-1} \big( L_w(L_1)_j^T \big) S_w^{-1}\\
\end{align*}$$
##### Hessian Information
$$\begin{align*}
\partial_{w_iw_j} m &= 2g_j^T \big(\partial_{w_i} S_w^{-1}\big) (Gw-b) \\
&+ 2g_j^T S_w^{-1}g_i\\
&+ 2g_i^T (\partial_{w_j} S_w^{-1}) (Gw-b)\\
&+ (Gw-b)^T(\partial_{w_iw_j} S_w^{-1})(Gw-b)
\end{align*}$$
where
$$\begin{align*}
	\partial_{w_iw_j} S_w^{-1} &= \partial_i ( - S^{-1} \partial_j S S^{-1})\\
	&= (- S^{-1} \partial_i S S^{-1}) (- \partial_j S S^{-1}) - S^{-1} \partial_{i} (\partial_j S S^{-1})\\
	&= S^{-1} \partial_i S S^{-1} \partial_j S S^{-1} - S^{-1} \partial_{ij} S S^{-1} \\
	&+ S^{-1} \partial_j S S^{-1}  \partial_i S S^{-1}\\
	\partial_{w_iw_j} S_w &= \partial_i (\partial_j L L^T + L \partial_jL^T)\\
	&=  (L_1)_j (L_1)_i^T + (L_1)_i (L_1)_j^T \\
\end{align*}$$
We notice that $\partial_{w_iw_j}S_w^{-1}$ is constant with respect to $w$ so this can be cached so that we do not need to compute this when the hessian is computed. 
##### Results for the Linear Case
I implemented a large sweep comparing using this local optimization solver compared to IRWLS over all of the examples in the WENDy paper. 
- In general the MLE and IRWLS gives similar approximations of $w$ while the MLE is slower
- This does address the Hindmarsh-Rose behavior at high noise and lower number of data points by giving an improved answer to the OLS solution while the IRWLS diverges. 
##### Global Optimization Effort
While the local optimization methods gives a more robust solution to the IRWLS in general. There is no gaurentee that it is converging to a global or local minimum. So I have been developing a Branch and Bound algorithm for the MLE.
- I have developed [[Bounding MLE|numerous lower bounds]] to make the algorithm more efficient
- I have implemented this in MATLAB
- The method coverges to a superior minimum when given an initial search space bases on a confidence interval derived from the local optimization method's estimate.
	![[manufactured_BnB_dim12.png]]
## Non-Linear WENDy
One obvious extension of the work already done, is to now consider the case when the right hand side of the ODE $F$ is no longer linear in parameters. To highlight what this means, instead of considering 
$$\dot{U}(t) = \sum\limits_{j=1}^{J}w_jf_j(U(t))$$
We now look at 
$$\dot{U}(t) =  F(U(t), w)$$
### Residual's Distribution 
This is a straight forward extension of what was done previously because it relied on an expansion about the data, and did not rely on the an expansion in the parameters. I summarize the derivation below If we consider data of the familiar form 
$$\{t_m, u_m\}_{m=1}^M = \{t_m, u^*_m +\varepsilon_m\}_{m=1}^M \text{ where } \varepsilon_m \stackrel{idd}{\sim} N(0,\sigma^2)$$
The residual in the weak form is 
$$\begin{align*}
	r(w) &= \langle \Phi, F(u, w)\rangle + \langle \dot{\Phi}, u \rangle \\
	&\stackrel{\Delta}{=} G(u,w) - b(u) \\
\end{align*}$$
Notice that while $b$ is linear in $u$ by the nature of the inner product $G$ is not! Now, for ease of notation, we define $G(w) = G(u, w)$, $G^*(u,w) = G(u^*,w)$, $b^* = b(u^*)$ , and $b^\epsilon = b(\epsilon)$. Now consider $w^*$ to be the true weight and $w$ to be the approximated weights.
$$\begin{align*}
	r(w) &= G(w) - b \\
	&= G(w)+ G^*(w) - G^*(w) +G^*(w^*) -G^*(w^*) - b^* - b^\epsilon\\
	&= \underbrace{G(w) - G^*(w)}_{e^\Theta} + \underbrace{G^*(w) -G^*(w^*)}_{r^*} + \underbrace{G^*(w^*) - b^*}_{e^{\text{int}}} -b^\epsilon
\end{align*}$$
Investigating the potential sources of error: 
- $e^\text{int}$ is due to numerical integration because the only source of error is building $G^*(w),b^*$ with perfect data that satisfies the ODE. 
- $r^*$ is the difference of the RHS of the weak form evaluated at the true weights vs the approximated one. $$\begin{align*}
		\begin{cases}r^* = \|\nabla_w G^*(w-w^*)\| ,& \text{ when $G$ is linear in parameters}\\
		r^*\le \|\nabla_w G_*\|\|w-w^*\| ,& \text{ when $G$ is non-linear in parameters}
\end{cases}
	\end{align*}$$As  $w^*\rightarrow w$ (i.e. method is convergent),  $r^* \rightarrow 0$. 

We still can say that as the number of points gets large and the method converges that:
$$r(w) \rightarrow e^\Theta -b^\epsilon$$
Inspecting this quantity, we see that we need to linearize $G$ about the  data $u$:
$$\begin{align*}
	e^\Theta-b^\epsilon &= G(u,w) - G(u^*,w) + \langle \dot{\Phi}, \epsilon \rangle \\
	&= G(u, w) + \langle \dot{\Phi}, \epsilon \rangle \\
	&- \Big(G(u,w) - \langle \nabla_u G(u,w), \epsilon \rangle + H(u^*,w,\epsilon) \Big) \\
	&\approx \langle \nabla_u G(u,w) + \dot{\Phi}, \epsilon\rangle
\end{align*}$$
Notice that because $\epsilon_m \stackrel{iid}{\sim} N(0,\sigma^2) \Leftrightarrow \epsilon \sim N(0, \sigma^2 I)$. Now if we say that $|\epsilon| \ll 1$ then because $H = O(\epsilon^2)$ and can be throw out. 
Assuming that $w \rightarrow w^*$ and $M \gg 1$, then we have
$$\begin{align*}
	\mathbb{E}[r(w^*)|u] &= \mathbb{E}[e^\Theta -b^\epsilon |u]\\
	&= \langle \Phi \nabla_u F(w,u) \mathbb{E}[\epsilon | u] \rangle + \dot{\Phi} \mathbb{E}[\epsilon | u] \\
	&= 0 \\ 
	\mathbb{Var}[r(w^*)|u] &= E\big[r(w^*) r(w^*)^T\big]\\
	&=  \\
	S_w^{-1/2} (e^\Theta - b^\epsilon) &\stackrel{asymp}{\sim} N(0, I)\\
\end{align*}$$
But now  $L_w =  \underbrace{\mathbb{I}_D \circ \dot{\Phi}}_{\stackrel{\Delta}{=} L_0} + \underbrace{\nabla_u F(u,w) \circ \Phi}_{\stackrel{\Delta}{=} L_1(w)}$,
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
#### Fleshing out $L_w$ 
Recall that 
$$\begin{align*}
F&: \mathbb{R}^{J \times D} \rightarrow \mathbb{R}^D\\
F(w,u) &= \begin{bmatrix}f_1(w,u)\\
\vdots \\
f_D(w,u)\end{bmatrix}\\
\nabla_uF &: \mathbb{R}^{J \times D} \rightarrow \mathbb{R}^{D\times D}\\\\
\nabla_uF &=  \begin{bmatrix} \partial_{u_1} f_1(w,u) & \cdots & \partial_{u_D} f_1\\
\vdots  & \ddots & \vdots\\
\partial_{u_1} f_D(w,u) & \cdots & \partial_{u_D}f_D(w,u) \end{bmatrix}\\
G&:\mathbb{R}^{J \times D} \rightarrow \mathbb{R}^{K \times D}\\
G(w,u) &= \langle \Phi, F(w,u)\rangle = \Phi^T F(w,u)\\
\nabla_u G&:\mathbb{R}^{J \times D} \rightarrow \mathbb{R}^{K \times D \times D}\\
\nabla_u G(w,u) &= \langle\Phi, \nabla_uF(w,u)\rangle
\end{align*}
$$
### Picking an initial $w_0$
Two obvious options: 
- Solve the non-linear least squares problem neglecting the covariance. This is equivalent to the what was done in the previous WENDy paper or currently what we think of as the "WSINDy" $w_0$. This can be done with a nonlinear least squares solver. In Julia, we use [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/):$$\begin{align*}
		\min_{w} \frac{1}{2} \|G(w)-b\|_2^2\\\\
		G(w) &= \Phi F(w)\\
		\nabla_w G(w) &= \Phi \nabla_w F(w)
	\end{align*}$$
- Randomly pick in a desired domain or prior distribution? I think it is reasonable to allow the use of prior knowledge or constraints on the parameter values. Basically what I mean is we add more and more noise to the the true parameters until we cannot anymore. 
	$$w_0 = w^* + \eta \omega$$
	where $$  \eta \in [1e-4, 1e3],\; \omega \sim N(0, \operatorname{diag}[\operatorname{abs}.(w^*)])$$
### Nonlinear MLE
In the NonLinear case, we still assume that the residual has an asymptotic, conditional distribution of 
$$r\sim N(0, \operatorname{diag}[\sigma_1, \cdots, \sigma_D])$$This leads to a Log-Likelihood of
$$\mathcal{l}(w; u) = (G(w) - b)^TS_w^{-1}(G(w)-b) + \log(\det(S^{-1}_w))$$
For now we are dropping the $\log\det$ term because is seems to hurt performance, so we are minimizing the Mahalanobis Distance:

Now to use optimization methods we need derivative information for speed and accuracy. 
When the right hand side of the ODE is linear in $w$ we obtain the following derivatives:
$$\begin{align*}
\partial_{w_j} m &= 2\partial_{w_j}G_w^T S_w^{-1} (G_w-b) \\
&+ (G_w-b)^T(\partial_{w_j} S_w^{-1})(G_w-b)
\end{align*}$$
where
$$  
\begin{align*}
	 \partial_{w_j}G_w &:= V \partial_{w_j} F(w)\\
	(L_1)_j &:= L_1[:,:,j]\\
	\partial_{w_j} S_w^{-1} &= S_w^{-1} \partial_{w_j}S_w S_w^{-1}\\
	&= -S_w^{-1} \big( L_w(\partial_{w_j}L_w)_j^T \big) S_w^{-1}\\
	\partial_{w_j}L_w &= V \circ \partial_{w_j}\nabla_u F(w)+ L_0
\end{align*}$$
##### Hessian Information
$$\begin{align*}
	\partial_{w_iw_j} m &= 2\partial_{w_i w_j}G_w^T  S_w^{-1}(Gw-b) \\
	&+2\partial_{w_j}G_w^T \big(\partial_{w_i} S_w^{-1}\big) (Gw-b) \\
	&+ 2\partial_{w_j}G_w^T S_w^{-1}g_i\\
	&+ 2g_i^T (\partial_{w_j} S_w^{-1}) (Gw-b)\\
	&+ (Gw-b)^T(\partial_{w_iw_j} S_w^{-1})(Gw-b)
\end{align*}$$
where
$$\begin{align*}
	\partial_{w_iw_j} S_w &= \partial_i ( - S_w^{-1} \partial_j S_w S_w^{-1})\\
	&= (- S_w^{-1} \partial_i S_w S_w^{-1}) (- \partial_j S_w S_w^{-1}) - S_w^{-1} \partial_{i} (\partial_j S_w S_w^{-1})\\
	&= S_w^{-1} \partial_i S_w S_w^{-1} \partial_j S_w S_w^{-1} - S_w^{-1} \partial_{ij} S_w S_w^{-1} \\
	&+ S_w^{-1} \partial_j S_w S_w^{-1}  \partial_i S_w S_w^{-1}\\
	\partial_{w_iw_j} S_w &= \partial_i (\partial_j L_w L_w^T + L_w \partial_jL_w^T)\\
	&= \partial_{ij} L_w L_w^T+  \partial_j L_w \partial_i L_w^T \\
	&+ \partial_i L_w \partial_j L_w^T + L_w \partial_{ij}L^T\\
	\partial_{w_iw_j} L_w &= V\circ \partial_{w_iw_j} \nabla_u F(w)
\end{align*}$$
Evaluating this gradient/hessian is computationally expensive, but the following are common sense things that can help: 
- Compute $S^{-1} (G(w)-b)$ and reuse 
- Compute $\{\partial_j S \}_{j=1}^J$ and reuse
- Compute $\{\partial_{ji} S \}_{j=1,i=1}^J$ and reuse
## Comparable Methods 
- [Two-Stage-method-(Non-Parametric-Collocation)](https://docs.sciml.ai/DiffEqParamEstim/stable/methods/collocation_loss/) Here is a built in method 
- Vanilla FSNLS
## Example ODE's
Choice of example problems is guided by what was previously done in  the previous paper WENDy, benchmarked problems in Julia, and what was suggested by colleagues and other literature. The main categories that we want to show are 
- ODE's that are being used right now in domain sciences, and how the algorithm could be plugged into a current workflow to show benefit. 
- Stress testing the algorithms derivation by having non-linear both in $u$ and in $w$. We need to show how and why this algorithm would fail. This could give us new insights in how to improve the algorithm moving forward, but also could could give guiding advice to end users (domain scientists) so they can take the results with a grain of salt when they apply to different systems.
- Stiff ODE's: this should highlight the benefit of not having to forward simulate. Even with fast jacobian tricks like sparsity, coloring, and static data structures. The fact is that needing to forward simulate more than once can be a real down side for an end user. This could be as simple as  making a very large system of ordinary differential equations with very few parameters , meaning $D >> 1$ while $J \in [1,20]$. This makes me think of Systems of PDE that are solved with finite differences (or method of lines). The road blocks here are going to be computing the different necessary differential operators quickly: $\partial_{w_j} F, \partial_{w_ju_k} F, \partial_{w_i w_j} F, \partial_{w_i w_j u_k}$. 
### Non-Stiff ODE
- Fitzhugh-Nagumo 
	Found in [SciML Benchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/NonStiffODE/FitzhughNagumo_wpd/) and  [[Ramsay et al. - 2007 - Parameter Estimation for Differential Equations a.pdf#page=17| Ramsay's paper]], and after discussing with Bortz this seems like an "easy problem", but can be a good choice to warm up.
	$$
	\begin{align*} 
			\dot{V} &= c\left(V-\frac{V^3}{3}+R\right),  \\
			\dot{R} &= -\frac{1}{c}(V-a+b R) 
		\end{align*}
	$$
	As mentioned, this can be solved with linear in parameters with an equality constraint $w = (a,b,c,d), d = 1/c$, so perhaps not a necessary problem 
- Lotka-Volterra 
	Found in [SciML Benchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/NonStiffODE/LotkaVolterra_wpd/) and the [[WENDy_ode2023.pdf#page=16|WENDy Paper]], so this probably is good choice at least to 
- [[Calver - Parameter Estimation for Systems of Ordinary Diffe.pdf#page=47|Goodwin]] - *Loop model*$$\begin{aligned} 
	& y_1^{\prime}(t)=\frac{a}{A+y_3(t)^\sigma}-b y_1(t), \\ 
	& y_2^{\prime}(t)=\alpha y_1(t)-\beta y_2(t), \\ 
	& y_3^{\prime}(t)=\gamma y_2(t)-\delta y_3(t),\end{aligned}$$
- Nonlinear Pendulum 
#### Others 
- See [[Calver - Parameter Estimation for Systems of Ordinary Diffe.pdf |Calver Thesis]]
	- Kermack-McKendrick model - time shifted
	- Hutchinson’s model - time shift\ed
	- Calcium Ion Example - log normal noise but cool
	- [[Calver - Parameter Estimation for Systems of Ordinary Diffe.pdf#page=37|Barnes Problem]] - possibly too simplistic, but predator prey
- See [SciML Benchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput): 
	- lPleiades - Non homogeneous
	- Rigid Body - Possibly non homogeneous
	- [Three Body](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/NonStiffODE/ThreeBody_wpd/) - Could fit but we would need to assume that we have measurements on velocity which is perhaps unrealistic?
	- 100 Independent Linear  - Lots of parameters and forward simulation should be crazy fast, so why would this be a good use case.
- [[Britton - 2003 - Essential Mathematical Biology.pdf#page=191|Michaelis-Menten Equation]] Bortz mention this because the Hill-parameters could be a good choice, but this is would require us extending the simple linear in parameter equations in the book. Seem's like a side quest. 
### Stiff ODE 
- [VanDerPol](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/StiffODE/VanDerPol/) - Could be a good fit. Seems similar to the Hindmarsh-Rose but in two dims instead of three. But would have a non-linear parameter so that is cool. 
- [[WENDy_ode2023.pdf#page=16|Hindmarsh-Rose]] - Show benefit of the MLE compared to the IRWLS and FSNLS because of the robustness of the trust region solver to high noise and  poor initializatioin
#### Others 
- [Mackey Glass](https://en.wikipedia.org/wiki/Mackey%E2%80%93Glass_equations "https://en.wikipedia.org/wiki/Mackey%E2%80%93Glass_equations")  Bortz suggested we have a look at these but there is a time shift
- [Brusselator](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/StiffODE/Bruss/)  - This would be cool, but we need to build out more of an infrastructure to compute or give these derivatives quickly. Also, it is non-homogeneous
- HIRES - Log Normal noise
- OREGO - Log Normal noise
- POLLU - Log Normal noise and we would need to have a a better way to compute derivative. 
- ROBER - Log Normal noise
### Biological Differential Equations
- [BCR](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/Bio/BCR/) Seems like a bitch to get working because of how complicated all the tools are to get working together, but this is a nice idea of how we could show benefit if we had a domain science partner who needed a better param estimation and we could demonstrate how to get this to work in a real world application.  
- [Bidkhori2012](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/Bio/Bidkhori2012/) [Original Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0048004) Again could be cool to show benefit on real world example. Seems like this could be a slightly easier path because it is a benchmark, but there would be log normal noise. 
- Egfr_net - Also LogNormal noise, possibly way too many parameters 
- Fceri_gamma2 - LogNormal noise
- Multisite2 - LogNormal noise
- Multistate1  - LogNormal noise but smaller and could be a good model 
## Resources
- Parameter Estimation Algorithms 
	- [SciML Benchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/) - A good place to start for test problem selection. Give us a path forward to simulate the odes quickly and accurately in Julia, but we also have them broken into categories, and here is a very brief paper we can sight [[Rackauckas and Nie - 2018 - Confederated Modular Differential Equation APIs fo.pdf|SciML Benchmark Paper.]]
		- [Calver Thesis Ch4](https://tspace.library.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=78) It does look like forward solve least squares is state of the art, but using the adjoint method to approach the optimization plays an important role.  [Calver Thesis Ch2.6](https://tspace.library.utoronto.ca/bitstream/1807/95761/3/Calver_Jonathan_J_201906_PhD_thesis.pdf#page=43) covers possible examples. A common theme is that they look for **time lags** which I don't know if our frame work would allow, but here are a couple relevant equations:
	- Shooting Methods (FSNLS) [Calver Disertation Presentation](https://www.cs.toronto.edu/~calver/presentations/caims2019_calver.pdf)
	- Adjoint methods [[Johnson - Notes on Adjoint Methods for 18.335.pdf|Johnson's Notes]]
	- Bayesian Collocation Methods [[Campbell - Bayesian Collocation Tempering and Generalized Pro.pdf|Campbell]]
- Theoretical Backing For WENDy
	- [[Zotero/storage/QK273B4H/Bollerslev and Wooldridge - 1992 - Quasi-maximum likelihood estimation and inference .pdf#page=39|Bollerslev Lemma A.2]] - Dan suggested this for us to show that we have a unique minimizer. Seems very powerful also has tons of citations
	- [[Russo and Laiu - 2024 - Convergence of weak-SINDy Surrogate Models.pdf|WSINDy Convergence Paper]] - Dan said this may not be worth reading but we can talk about it. 
- Optimization
	- [ARCqK Paper](https://arxiv.org/abs/2112.02089) - This is not ending up being a good solver but Cooper is invested so make time to read this
	- [[Dussault and Orban - 2021 - Scalable adaptive cubic regularization methods.pdf|ARC Paper]] - This may end up being faster/more reliable than the good ol trust region 
	- [[Pal et al. - 2024 - NonlinearSolve.jl High-Performance and Robust Sol.pdf|NonlinearSolve.jl]] - We use this to extend the IRWLS and possibly initialize our point [docs](https://docs.sciml.ai/NonlinearSolve/stable/) 
	- [Trust Region](https://julianlsolvers.github.io/Optim.jl/stable/algo/newton_trust_region/) Optim.jl provides both a way to call the traditional trust region solvers both the Newton and Krylov-Newton methods. They point the interested reader to the [[Nocedal and Wright - 2006 - Numerical optimization.pdf#page=87|Nocedal Book]].
	- [[deep learning goodfellow.pdf|Deep Learning Book]] - Ian Goodfellow et all
	- [AutoDiff SciML book](https://book.sciml.ai/notes/08-Forward-Mode_Automatic_Differentiation_(AD)_via_High_Dimensional_Algebras/) - Chris Rackauckas 

## Outstanding questions
### LogNormal noise
As I understand it the derivation of distribution of the residual, and thus the Likelihood, and thus the optimization routine... relies heavily on not only normality of the noise but also thee noise being additive. For a system with LogNormal (LN) noise,  we could (naively) make the residual have a normal distribution through some algebra
$$\begin{align*}\\
	u_m &= \hat{u}_m\epsilon_m, \epsilon_m \sim LN(0,\sigma)\\
	y_m := \log(u_m) &= \log(\hat{u}_m) + \log(\epsilon_m),  \epsilon_m \sim N(0,\sigma)\\
	r(w) &=  G(U, w) - b(U) \\
	&= G\Big(\exp\big(\log(U)\big), w\Big) - b\Big(\exp\big(\log(U)\big)\Big)\\
	&:= G(Y, w) - b(Y)
\end{align*}$$
Now if we do our asymptotic expansion wrt $y$ instead of $u$ everything would be the same. 
$$\begin{align*}
	r(w) &\underset{M\rightarrow \infty, w\rightarrow w^*}{\rightarrow} e^\Theta-b^\epsilon \\
	&= G(y,w) - G(y^*,w) + \langle \dot{\Phi}, \epsilon \rangle \\
	&= G(y, w) + \langle \dot{\Phi}, \epsilon \rangle \\
	&- \Big(G(y,w) - \langle \nabla_y G(y,w), \epsilon \rangle + H(y^*,w,\epsilon) \Big) \\
	&\underset{\epsilon\rightarrow 0}{\rightarrow} \langle \nabla_y G(y,w) + \dot{\Phi}, \epsilon\rangle
\end{align*}$$
```ad-warning 
title: Caution Power Series 
It is well known since we were children that a power series expansion of exponential/periodic functions are not the best series expansion... Could we do something smarter.
```

Frechet Derivative
### Estimated/Computing the Different Parts of Residual
We want to see what happens as the our optimization methods converge, and analyze why the fail. Our algorithm relies on a few things in order for the distribution of the residual to be estimated well by a normal. 
- Normality of the noise, $\epsilon \sim N(0, \sigma)$, because we generate the noise, this is a non issue for our tests, but when we have empirical data we should have a way of evaluating this. 
- The quadrature error to be low (In general this should be ok, but when the number of time points becomes low $M \in [1,100]$ then we could have problems because $|e^\text{int}| \approx |e^\theta - b^\epsilon|$ )
- If the method is not convergent, or the initialization point $w_0$ is to far away from the true weights then the nonlinear terms in the residual become large $|H(u^*,w,\epsilon)|\geq \big|\langle\nabla_uG(u,w),\epsilon\rangle\big|$ . Then the distribution is not valid. 
# Parameter Identifiably 
[http://biorxiv.org/lookup/doi/10.1101/2024.05.09.593464](http://biorxiv.org/lookup/doi/10.1101/2024.05.09.593464)
[http://arxiv.org/abs/2405.20591](http://arxiv.org/abs/2405.20591)
$$
\begin{align*}
r(w) = Gw-b &\sim N(0,S(\hat{w}))\\\\
w &\sim N(\underbrace{G^{-1}b}_{\approx \hat{w}}, G^{-1}S(\hat{w})G^{-1})
\end{align*}$$