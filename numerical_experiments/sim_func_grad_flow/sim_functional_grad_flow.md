# Numerical simulation of functional gradient flow

We represent the flow map $f_\tau(t,u,x)$ on a discrete tensor grid, with
$t,u\in[0,1]$ and $x\in[-2,2]^d$. The state has
$N_t N_u N_x^d d$ entries. Diagnostics use the domain $t\leq u$.

We use the affine velocity field $v(x)=Ax+b$, whose exact flow map
$f^*(t,u,x)$ is evaluated through an augmented matrix exponential. The initial
map is
\[
f_0(t,u,x)=f^*(t,u,x)+(u-t)\delta(t,u,x),
\]
where $\delta$ is a smooth trigonometric perturbation with amplitude controlled
by `eps`. Multiple initial maps use deterministic phase and amplitude variations.
This construction satisfies $f_0(t,t,x)=x$.

We simulate the Eulerian and Lagrangian functional gradient flows,
\[
\partial_\tau f_\tau
=(u-t)\left(\partial_t f_\tau+D_xf_\tau\,v(x)\right),
\qquad
\partial_\tau f_\tau
=-(u-t)\left(\partial_u f_\tau-v(f_\tau)\right).
\]
Derivatives use centered finite differences in the interior and second-order
one-sided differences at grid boundaries. We integrate in optimization time
$\tau$ with fourth-order Runge–Kutta and enforce $f_\tau(t,t,x)=x$ after each
step. The implementation uses PyTorch tensors on CPU or CUDA, with float64 as
the default precision. A grid-spacing and velocity-based heuristic selects the
time step unless `dtau` is specified.

At checkpoints, we compare the numerical state with the explicit solutions
\[
f_\tau^{\mathrm E}(t,u,x)
=f_0\!\left(s_{t\to u}^\tau,u,f^*(t,s_{t\to u}^\tau,x)\right),
\qquad s_{t\to u}^\tau=u-e^{-\tau}(u-t),
\]
\[
f_\tau^{\mathrm L}(t,u,x)
=f^*\!\left(s_{u\to t}^\tau,u,f_0(t,s_{u\to t}^\tau,x)\right),
\qquad s_{u\to t}^\tau=t+e^{-\tau}(u-t).
\]
We report the root mean squared Euclidean error and maximum Euclidean error
across grid points with $t\leq u$; these statistics are averaged over initial
maps when several are used. Scatter plots pool checkpoint values across initial
maps and grid points, using scalar values for $d=1$ and vector norms for $d>1$.
A deterministic subset of at most 40,000 points is plotted, and Pearson
correlation is computed on that subset. Pointwise plots compare
$\|f_\tau(t,u,x)-f^*(t,u,x)\|_2$ with the corresponding explicit prediction at
fixed probes.

The default grid sizes $(N_t=N_u,N_x)$ are $(61,121)$ for $d=1$, $(31,31)$ for
$d=2$, and $(21,9)$ for $d=3,4,5$. The default final optimization time is $0.2$.
Tensor-grid memory grows exponentially with $d$; grid resolution and time step
control the discretization error.
