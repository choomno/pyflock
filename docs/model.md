# Model and conventions

The state variable $z_k$ is displacement from the prescribed formation spacing.
Coherent motion is $z_k(t)=a+bt$. The interior equations are

$$
\ddot z_k=g_x\sum_j\rho_{x,j}z_{k+j}
             +g_v\sum_j\rho_{v,j}\dot z_{k+j},
\qquad \sum_j\rho_{x,j}=\sum_j\rho_{v,j}=0.
$$

A stencil lists offsets in increasing order, $(-2,-1,0,1,2)$.
Nearest-neighbour interactions have zero outer entries. In that case, with
diagonal weights normalized to one, $\rho_{\bullet,-1}=-1-\rho_{\bullet,1}$.
The usual restoring and damping gains satisfy $g_x,g_v<0$.

## Circle and path

On the circle, $N$ is the **total number of agents**, indexed modulo $N$.
On the path, $N$ is the **number of followers**: agents are $0,\ldots,N$,
with leader 0 and increasing index pointing toward the tail. Positive signal
velocity means propagation toward increasing agent index, measured in agents
per unit time.

For the leader experiment, initially $z_k=0$; followers have zero velocity,
and $\dot z_0=v_0$. The leader then follows $z_0(t)=v_0t$.
The plotted displacement is $y_k=z_k-z_0$, so the tail response is $y_N$.
An initial velocity impulse on the circle is a different experiment and must
be specified separately.

The path boundary is part of the model. For nearest neighbours its last equation is

$$
\ddot z_N=g_x\beta_x(z_N-z_{N-1})
             +g_v\beta_v(\dot z_N-\dot z_{N-1}).
$$

`drop` sets $\beta_x=-\rho_{x,-1}$, $\beta_v=-\rho_{v,-1}$: discard missing
neighbours and adjust the diagonal so each row sums to zero. This is the
*variable mass* condition in *Transients*, Definition 1. `regular` sets both
$\beta$'s to one. For next-nearest neighbours, `drop` applies the same row
rule at every incomplete neighbourhood. The historical `fold` rule is another
closure; it should be identified explicitly when used.

## Fourier analysis

For the circle mode $e^{ik\theta}$, define

$$
\nu_\bullet(\theta)=g_\bullet\sum_j\rho_{\bullet,j}e^{ij\theta},
\qquad \lambda^2-\nu_v\lambda-\nu_x=0.
$$

These quadratic roots give the exact finite-circle eigenvalues at
$\theta=2\pi m/N$. Define moments
$I_{\bullet,n}=g_\bullet\sum_j\rho_{\bullet,j}j^n$.
If $I_{x,1}=0$ and the two real slopes are distinct, their local expansions are

$$
\lambda_\pm(\theta)=-ic_\pm\theta-D_\pm\theta^2+O(\theta^3),
\quad
c_\pm=-\frac{I_{v,1}}2
\pm\sqrt{\frac{I_{v,1}^2}{4}+\frac{I_{x,2}}2}.
$$

Thus phase velocity is $-\operatorname{Im}\lambda/\theta$.
In the historical MATLAB naming, `fm` is $-c_+$ and `fp` is $-c_-$;
`qm` is $-D_+$ and `qp` is $-D_-$, respectively.
Retaining the quadratic term gives the formal envelope equation
$u_t+cu_s=Du_{ss}$. Its Gaussian spreading alone does not prove an error
estimate for a reflected wave or a tail extremum.

## Stability and tail observables

Circle stability means convergence to coherent motion: the translation and
uniform-velocity modes give a double zero eigenvalue; all others must lie in
the open left half-plane. Path stability is checked on the follower matrix
with the prescribed leader removed. For fixed $N$, this is distinct from
control of transient amplification as $N\to\infty$.

For a stable, flock-stable nearest-neighbour path with symmetric position
coupling, negative gains and $c_-<0<c_+$, *Transients*, Theorem 3, predicts
signed tail extrema for large $N$ and a fixed number of reflections:

$$
T_k=\frac N{c_+}+(k-1)P,\qquad
A_k=-\left(\frac{c_-}{c_+}\right)^{k-1}\frac{Nv_0}{c_+},
\quad P=N\left(\frac1{c_+}-\frac1{c_-}\right).
$$

The full oscillation period is $2P$. The paper's attenuation coefficient is
$\alpha=A_3/A_1=(c_-/c_+)^2$; the ratio of consecutive absolute extrema is
$|c_-/c_+|$. Report measured extremum times and signed amplitudes with their
sampling or interpolation procedure. A maximum over a finite simulation
window is not automatically the maximum over all time.
The prediction uses the paper's bulk-wave and boundary conjectures and
regularity assumptions; a numerical comparison tests those assumptions too.
