# pyflock

A Python port of the MATLAB/C++ flock simulation code (Carlos Cantos, J. J. P. Veerman and
co-workers) behind the *signal velocity* and *transients* line of papers.

## Model

Agents on a line or a circle, each a damped linear oscillator coupled to up to two
neighbours on each side:

```
zddot_k = gx * sum_j rx[j+2] z_{k+j}  +  gv * sum_j rv[j+2] zdot_{k+j},   j = -2..2
```

A stencil is the 5-vector `r = [r_-2, r_-1, r_0, r_+1, r_+2]`, exactly as in the MATLAB.
Decentralisation requires `sum(rx) = sum(rv) = 0`.

## Install

```
pip install -e .[plot,test]
```

## Use

```python
import pyflock as pf

st = pf.nn_stencil(gx=-1.0, gv=-1.0, alpha_x1=-0.5, alpha_v1=-0.3)

pf.signal_velocities(st)      # (c_+, c_-)
pf.coefficients(st)           # per branch: c (velocity), D (diffusion), beta (dispersion)

res = pf.simulate_line(st, N=200, tmax=800.0)
pf.transient_summary(res.t, res.last_relative)     # A_k, T, gamma
pf.theory_amplitudes(st, N=200)                    # leading-order predictions
```

Curated parameter sets transcribed from `flock45sim_function.m`:

```python
pf.case_names()
st = pf.case("type_I")        # "STABLE on CIRCLE, UNSTABLE on LINE"
```

## What replaced what

| MATLAB / C++ | pyflock |
|---|---|
| `flock.m` (dispersion plots) | `pyflock.dispersion` + `pyflock.plotting.spectrum_figure` |
| `flock45.m` (line simulator) | `pyflock.simulate_line` |
| moment algebra in `flock45sim*.m` | `Stencil.moment_x/v`, `Stencil.a`, `dispersion.matlab_fq` |
| `flockplot*.m` (convergence study of \|A_1\| vs N) | `pyflock.plotting.convergence_figure` |
| `flocksim_extractor.m` | `pyflock.analysis.transient_summary` |
| `flocks_workflow.m` | plain loops over `pyflock.cases` |
| `varycolor.m`, `linspecer.m` | matplotlib colormaps |
| `CPP/` sweep infrastructure | not needed — see below |

## Why there is no sweep infrastructure

The originals integrate `zdot = M z` with an explicit Runge–Kutta method (`ode45`, odeint)
even though **M is constant**. The C++ layer exists to make that fast enough to sweep.

pyflock uses exact propagators instead: a matrix exponential on the line, and on the circle
an FFT diagonalisation of the block-circulant structure that solves each Fourier mode in
closed form. Both are exact to machine precision and need no tolerances, which removes the
reason the sweep layer existed. `method="rk"` is retained for cross-checking.

## Boundary conventions

`flock45.m` contains two boundary treatments; the live one is `bc="drop"`.

- `drop` — discard out-of-range neighbours, set the diagonal to minus the sum of the
  remaining weights so rows still sum to zero. This is `flock45.m`'s "second version",
  and for nearest-neighbour stencils it is the *variable mass* condition of
  Cantos–Hammond–Veerman Definition 1.1.
- `fold` — add a missing neighbour's weight onto the nearest in-range one and rescale the
  final row; the commented-out "first version".
- `regular` — nearest-neighbour only: `zddot_N = gx(z_N - z_{N-1}) + gv(zdot_N - zdot_{N-1})`,
  i.e. `chi = 1` in Definition 1.2.

The choice matters. It is the most likely cause of a port silently disagreeing with
published figures, so it is a parameter rather than a hard-coded default.

## Notes on the original code

- `flock45sim_function.m` computes `fm, fp, qm, qp`. These are exactly the first two Taylor
  coefficients of the eigenvalue branches: `f_sigma = mu_sigma` (so `c = -f`, the signal
  velocity) and **`q_sigma = -D_sigma`, the diffusion coefficient** — verified to machine
  precision in `tests/`. `dispersion.matlab_fq` reproduces them.
- `flock.m`'s commented block defines `a = 0.25*Iv1^2 + 0.5*Ix2`, which is `Stencil.a`.
- `Flocks Matlab/nu(phi).mw` is a Maple 15 worksheet running `series`/`factor` on the same
  quantities in the variables `gx, gv, ax1, av1, bx1, bv1, bv2` — the symbolic source of the
  published expansion.

## Figures

```python
from pyflock import plotting as P
P.overview_figure(st, N=150).savefig("overview.png", dpi=140, bbox_inches="tight")
```

| function | the question it answers |
|---|---|
| `spectrum_figure` | Is it stable, and how good is the low-frequency approximation? Exact branches with the order-1/2/3 Taylor curves overlaid. |
| `circle_vs_line_figure` | Does the boundary change the spectrum? Both spectra on one pair of axes, with the two `max Re` values. |
| `spacetime_figure` | **Start here.** Where does the disturbance go and how fast? Heat map over (agent, time) with the predicted signal path drawn on top. |
| `trajectory_figure` | What do individual orbits look like? All agents, coloured by position in the flock. |
| `transient_figure` | How big are the A_k? The last agent's orbit with extrema marked and Theorem 3 overlaid. |
| `packet_figure` | Does the pulse spread diffusively? Profile at several times against the predicted Gaussians. |
| `convergence_figure` | What does the second-order term buy? Relative error of A_1 vs N, leading-order and corrected. |
| `overview_figure` | All four diagnostics for one stencil, on one page. |
| `stability_map_figure` | Where is the line flock stable? N·max Re λ over (q, N), the predicted boundary marked on it, and the theory error in the same units. |

Colour follows the job it does: categorical for branch/geometry identity (with a
distinct marker or dash per series, so identity is never colour-alone), a single-hue
sequential ramp for agent index, and a diverging map with a neutral midpoint for signed
fields. The palette is validated for colour-vision deficiency.

`spacetime_figure` clips colour at the 99.5th percentile by default so that a late
blow-up does not wash out early structure; when it clips, the colourbar says so and
reports the true maximum. Pass `robust=False` for raw limits.

## Tests

`pytest tests` — checks covering the stencil algebra, the branch labelling (including
the `by_imag` convention `flock.m` uses), the boundary rows hard-coded against
`flock45.m`, exactness of both propagators, and the curated cases.

One case is worth singling out: `type_I` is labelled "STABLE on CIRCLE, UNSTABLE on LINE",
and that turns out to mean *asymptotically* unstable — on the line its largest eigenvalue
real part is strictly positive (≈ +0.022 at N = 20, decaying roughly like 1/N), while the
circle is stable for every N tested. The periodic and path problems disagree about
asymptotic stability, not merely about transient size.


That eigenvalue is well-conditioned (κ ≈ 16 at N = 60), so the instability is not round-off. The same
is not true of every stencil: strongly directional couplings make the line matrix non-normal enough
that double-precision eigenvalues lose meaning. `two_positive_velocities`, whose signal velocities are
both positive, has κ ≈ 2×10¹³ at N = 60, so its computed eigenvalues are good only to about ±0.04.
`pf.eigen_reliability(M)` returns the condition number and that round-off bound, and
`circle_vs_line_figure` uses the leaderless matrix (so the leader's double zero does not mask the
stability margin) and prints a warning on the plot when the line eigenvalues are unreliable.


## Least-stable eigenvalue on the line

For nearest-neighbour flocks with symmetric position coupling, the slowest mode on the line decays
(or grows) at a rate fixed by the signal velocities and diffusion coefficients alone:

```
Re ν ≈ ln|c₋/c₊| / P  −  π² N (D₊/c₊³ + D₋/|c₋|³) / P³,      P = N (1/c₊ + 1/|c₋|)
```

The first term is the amplitude gain per round trip of a signal reflected at both ends. The second is
diffusive damping of that round-trip mode, whose frequency π/P is shared by both legs.
`pf.least_stable_eigenvalue(st, N)` implements it; `method="linear"` gives the small-asymmetry limit
`g_v (q/(2N) + π²/(16N²))`, with q = 1 + 2α_v1.

Checked against the leaderless line matrix for three gain pairs, q ∈ [−0.6, 0.8] and both `drop` and
`regular` boundary conditions, the wave form is within 0.3% at N = 400 and 2.6% at N = 50. Its error
shrinks like 1/N and is smallest for `regular`, and the leading-order behaviour does not depend on the
boundary condition. The linear form is good only for small |q g_v|: at g_v = −3 its error is about 3%
at |q| = 0.1 and 45% at |q| = 0.4.

It follows that the line is stable only for q > q* ≈ −π²/(8N) (`pf.critical_velocity_asymmetry`):
weighting the agent ahead helps, and any fixed backward asymmetry destabilises a long enough flock.
`pf.stability_map` sweeps a stencil family over (q, N), and `plotting.stability_map_figure` compares
the result with this theory.

The linear form appears, in different notation, in unpublished 2014 notes by R. De Dios, He Hao and
J. J. P. Veerman (`DeDios2/` in the project folder); the typed draft there states an incorrect
version. The theory does not cover asymmetric position coupling.
