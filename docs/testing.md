# Tests and continuous integration

Install the development environment from the repository root:

```sh
python -m pip install -r requirements-dev.txt -e .
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The pinned environment uses Python 3.12. The package supports Python 3.10 or
newer; the CI matrix also exercises Python 3.10 and 3.14 on Linux and Python
3.12 on Windows using compatible dependency releases. Local results and hosted
CI results are separate evidence; workflow runs record the latter.

## What the tests check

Numerical tests compare Fourier symbols with finite matrices, matrix
exponentials with independent ODE integration, and asymptotic coefficients with
their defining moment equations. Boundary tests check the actual endpoint rows.
Experiment tests check parameters, observables, recorded samples, checkpoint
compatibility, data integrity, and figure construction. Small arrays make these
checks suitable for each commit; reproducing a paper-size ensemble is a separate
computation documented in the [coverage map](coverage.md).

Floating-point tolerances concern the stated finite computations. Passing tests
or a high coverage percentage does not prove a limiting theorem or replace the
hypotheses in the [mathematical reference](mathematical-reference/mathematical-reference.pdf).

## Line and branch coverage

The gate requires **at least 99.5% line coverage and 99.5% branch coverage** in
each of two groups: the computational and plotting API, and the experiment
modules. The measured source is every Python file in `pyflock/`. Both
percentages are checked before rounding. There are no source-line or branch
exclusions; the checker rejects missing files and excluded lines.

```sh
python -m coverage erase
python -m coverage run -m pytest -q
python -m coverage combine
python -m coverage json -o coverage.json
python ci/check_coverage.py coverage.json
```

The configuration measures Python subprocesses and multiprocessing workers.
`python -m coverage html` produces an annotated report in `htmlcov/`.
The mathematical assertions, independent expected values and failure cases in
the tests determine what the exercised code establishes.

## Distribution checks

```sh
python -m build
python ci/check_distribution.py dist
```

The wheel contains the maintained Python package. The source distribution also
contains public documentation, tests, CI checks and examples; generated
experiment collections and private working instructions are excluded.

Install the wheel in a separate environment with its runtime dependencies and
run `python ci/check_installed.py` from the checkout. It uses isolated Python
processes in a temporary directory to test the public numerical API, fresh
generation, verification and saved-data redraw. It also checks that corrupt
data is rejected. An editable installation fails this check.

`python ci/check_sdist.py dist` extracts the source distribution, builds its
wheel and runs its tests. `--no-isolation` uses already-installed build tools
for an offline check. These checks require no source PDFs, research workspace
or precomputed figures. An isolated build may download its build dependencies.

## Documentation and figures

```sh
python ci/check_docs.py
python ci/build_docs.py
```

The documentation check validates local links. The TeX build runs two passes
and checks the log before replacing the compiled mathematical reference.
GitHub Actions compiles an independent copy and retains the PDF as an artifact.

New or changed figures require visual inspection of the final exports in
addition to numerical checks. Reproducing arrays, redrawing figures from saved
arrays, and verifying file integrity are different operations; the experiment
record identifies the inputs and software used for each.

The workflow runs tests, coverage, distribution checks and documentation checks
on pushes, pull requests and manual dispatch. It uses read-only repository
permissions, pinned action revisions and job time limits.
