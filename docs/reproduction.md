# Reproducing an experiment

Install the checked environment using the README, then run from an empty output
directory. For example:

```sh
python -m pyflock.experiments reflection --profile paper --output figures-paper
python -m pyflock.experiments reflection --output figures-paper --verify
python -m pyflock.experiments reflection --output figures-paper --redraw
```

The first command computes numerical arrays and exports the figure. `--verify`
checks the saved record and its files. `--redraw` verifies the numerical input and
source snapshot, then renders the saved arrays with the current renderer. It
does not rerun the simulation. Each command returns a nonzero exit status on
failure.

The [coverage map](coverage.md) identifies the paper figure, source parameters,
adaptations and missing inputs. `--profile quick` is the default; `--profile paper`
selects the recorded larger study. Use `--jobs` to set concurrency for studies
that support it. Checkpoints resume matching computations; changing inputs or
the numerical implementation invalidates the relevant cache.

## Files and evidence

| File | Role |
| --- | --- |
| `experiment.json` | Experiment identity, parameters, initial data, boundary, methods, sources, software and file checksums. |
| `data.npz` | Numerical arrays used by the renderer, including saved gain samples where relevant. |
| `figure.pdf`, `figure.svg`, `figure.png` | Exports from those arrays. |
| `source/` | Package snapshot associated with the computation. |
| CSV files, where present | Extracted measurements for inspection and independent analysis. |

Verification checks file integrity against the record. It does not establish
the governing model, numerical accuracy or a theorem. Those need the source
comparison, independent checks and hypotheses in the
[mathematical reference](mathematical-reference/mathematical-reference.pdf).

The small [gallery](gallery.md) and [expanded gallery](gallery-expanded.md)
explain the observables and comparisons. Committed PNGs are reader previews;
full numerical results are regenerated locally and excluded from distributions.
The paper profiles include large studies and are separate from the CI suite.

## Reviewing a result

Compare a fresh run with the recorded coefficients, initial data, boundary,
dimensions and numerical measurements. Numerical tolerances should reflect
sampling, conditioning and approximation error separately. Floating-point
arrays and figure bytes can differ across numerical libraries and fonts.
Inspect both raster and PDF renderings after a change affecting figures.

The test suite checks Fourier and matrix identities, analytic modes, independent
integration and transfer calculations, measurement rules, file verification,
and the installed command line interface. CI records these checks; it does not
rerun every paper-scale experiment or replace visual review.
