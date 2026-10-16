# Exact counterexamples to the energy bound

The [write-up](../docs/energy-bound-counterexamples.pdf) checks the hypotheses
and disproves Herman–Knorn–Ahlén (2017), Theorem 4(iii), equation (7), in two ways:

- [Five followers, nonzero initial state](energy-short-time/README.md): an analytic
  proof with an explicit rational terminal time.
- [128 followers, zero initial state](energy-zero-state/README.md): a rational
  piecewise cubic trajectory and exact matrix factorizations. Its energy-bound
  violation is strictly between 7.2581% and 7.25855%.

From the repository root:

```sh
python3 certificates/energy-short-time/verify.py
python3 certificates/energy-zero-state/verify.py
```

Both verifiers use only the Python standard library. Proof decisions use exact
integer and rational arithmetic. Numerical solvers proposed the zero-state data;
verification treats those data as untrusted coefficients and checks the resulting
trajectory, input cost, terminal energy and singular-value comparisons exactly.
The optional proposal script is not used by either verifier.
