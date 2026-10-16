# Flocks: propagation, boundaries and transients

One-dimensional arrays of coupled linear oscillators. This archive studies how a
leader's change of velocity travels through a flock, what happens at its boundaries,
and how the response changes with the number of agents.

The original research material includes MATLAB/C++ programs and papers by Cantos,
Hammond, Veerman and collaborators. A previous Python port supplied the starting
point. The 2026 reconstruction checks its numerical core and replaces its figure
workflow with reproducible experiments. Historical code and unpublished notes are
evidence to assess; their formulas are not assumed correct.

![Propagation and selected trajectories](docs/img/propagation.png)

## Reproduce

Python 3.12 was used for the checked environment. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
python -m pyflock.experiments all
python -m pytest
```

The experiments save PDF, SVG and PNG figures, numerical arrays, and a JSON
record of parameters, sources, software versions and checksums under `figures/`.
To regenerate just one, or change its appearance without recomputing:

```sh
python -m pyflock.experiments reflection
python -m pyflock.experiments reflection --redraw
```

The default convergence experiment uses six parameter pairs and three flock sizes.
The larger historical study has 30 pairs, two boundaries and six sizes up to 3,200
followers. Run it separately with:

```sh
python -m pyflock.experiments convergence --profile paper --jobs 4 --output figures-paper
```

The expanded suite adds published spectral and pulse examples, NNN wave types,
matched-coupling amplification, force disturbances, and heterogeneous arrays from
Lyons's thesis. For example:

```sh
python -m pyflock.experiments nnn-types --profile paper --output figures-expanded
python -m pyflock.experiments lyons_response --profile paper --jobs 2 --output figures-expanded
python -m pyflock.experiments nnn-ensemble --profile paper --jobs 6 --output figures-expanded
```

The last command runs 24,000 cases in explicitly defined stable families. It
matches the historical class counts, sizes and boundaries; its configurations and
some gains differ. The larger studies checkpoint and resume matching results.
Reduce `--jobs` to limit CPU use.

See the [gallery](docs/gallery.md) for the question and interpretation of each figure.
The [expanded gallery](docs/gallery-expanded.md) covers the additional experiments.
The [coverage map](docs/coverage.md) records source fidelity and remaining gaps.

## Model

With offsets $j=-2,\ldots,2$,

$$\ddot z_k=g_x\sum_j\rho_{x,j}z_{k+j}+g_v\sum_j\rho_{v,j}\dot z_{k+j}.$$

Each stencil sums to zero. On a path, $N$ counts the followers; agent $0$ is the
leader. The default experiment starts in formation and gives the leader velocity
one. Periodic systems use $N$ total agents. The heterogeneous module follows
Lyons's convention: $N$ total agents and positive gain magnitudes with an overall
minus sign. Its records and axes state that convention explicitly.

```python
import pyflock as pf

st = pf.nn_stencil(-1, -1, alpha_x1=-0.5, alpha_v1=-0.3)
res = pf.simulate_line(st, N=160, tmax=1000, nt=1601, bc="regular")
print(pf.signal_velocities(st))
print(pf.transient_summary(res.t, res.last_relative))
```

[Model and conventions](docs/model.md) defines the boundary rules and observables.
[Sources](docs/sources.md) connects the experiments to the literature.

## What the comparisons establish

Periodic dispersion and time evolution can be checked independently by Fourier
reduction and finite matrices. Path wave predictions depend on additional
conjectures and asymptotic hypotheses. The gallery states this distinction.
Spectral stability for fixed $N$ also differs from amplification as $N$ grows.

The previous port's Gaussian amplitude correction and small-eigenvalue estimates
remain available as exploratory calculations. They are not used as correctness
criteria for the reconstructed experiments. Matrix exponentials and FFT evolution
also have floating-point error; independent-method and resolution checks accompany
the computations.
