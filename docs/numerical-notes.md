# Source corrections and numerical limits

## NNN coefficients and boundaries

*NNN-agents*, Figure 7, prints
$\rho_x=(-2,-15/4,1,-21/4,5/2)$, whose sum is $-15/2$.
The surviving MATLAB case uses $+15/4$, restoring zero row sum and zero first
position moment. `nnn-types` uses that correction and records it in its JSON.

The paper's *fixed mass* boundary clamps a missing neighbour to the endpoint.
It does not include the final-row rescaling in the package's historical `fold`
rule. The NNN study implements the clamp separately. Its velocity endpoint uses
velocity weights, correcting the displayed position-weight subscript typo.

The original NNN configuration files and their exact selection rule were not
recovered. New ensembles save their complete coefficient arrays and selection
procedure. Lyons's original random seeds are also absent; his examples are
reconstructed parameter families with saved samples.

## The energy bound

Herman–Knorn–Ahlén (2017), equation (8), uses
$y^T D y\geq\sigma_{\min}(D)\|y\|^2$ for a nonsymmetric damping matrix.
For $N=32$, $h_p=1/2$, and unit damping coefficients, let

$$B=I-\operatorname{superdiag}(1),\qquad D=(B+\tfrac12|B|)B^T.$$

For a unit eigenvector $y$ of $S=(D+D^T)/2$ with least eigenvalue,

$$y^TDy=0.00853588766<0.04893368534=\sigma_{\min}(D).$$

Thus this proof step fails within the stated homogeneous parameter regime.
The valid symmetric-part estimate is

$$H(t)\leq H(0)+\frac{1}{2\lambda_{\min}(S)}
                    \int_0^t\|d(s)\|^2\,ds,$$

when $S$ is positive definite.

The integrated bound in equation (7) also admits a reproducible numerical
counterexample. Set $N=128$, $h_p=1/2$, unit masses and gains, zero initial state,
and $T=512$. Define

$$\mathcal A=\begin{pmatrix}0&I\\-BB^T&-D\end{pmatrix},\quad
F=\binom{0}{I},\quad C=\operatorname{diag}(B^T,I).$$

Let $W_T=\int_0^T e^{s\mathcal A}FF^T e^{s\mathcal A^T}\,ds$.
For the largest eigenvalue $\mu$ of $CW_TC^T$ and a corresponding unit eigenvector
$q$, apply

$$d(t)=F^T e^{(T-t)\mathcal A^T}C^Tq/\sqrt\mu\quad(0\leq t\leq T),$$

and zero afterwards. Its squared $L^2$ norm is one. The final energy is
$43.74150560$, whereas the printed bound is $40.78143255$: a 7.2584% excess.
These parameters satisfy the theorem's bounded positive coefficients and
nonincreasing damping assumptions. The input acts on all followers.

The experiment saves the input, state, accumulated input norm, and generating
vectors. The plotted comparison applies equation (7) to the input truncated at
each observation time. Adjoint and forced DOP853 integration agrees with the Gramian prediction
to $1.1\times10^{-12}$ relative state error. A separate check using the source's
momentum and spacing equations and a spline interpolant of the saved input
gives the same excess. This is numerical evidence against the finite coefficient
as printed; it does not settle the asymptotic scaling claims. Singular-value
scaling and the original single-site pulse are computed separately.

## Narrow resonances

The matched-coupling example at $\rho=.55$, $N=100$ has a narrow low-frequency
resonance. A coarse frequency grid misses it; ordinary double precision also
loses accuracy there. Sensitive peak neighbourhoods use a 70-digit recurrence.
Reported peaks are refined samples over the recorded band, not certified
supremum norms.
