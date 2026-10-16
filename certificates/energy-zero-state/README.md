# Exact zero-state energy counterexample

This directory gives an exact rational counterexample to Theorem 4(iii),
equation (7), in Herman, Knorn and Ahlén, *Automatica* 82 (2017), 13–20,
DOI [10.1016/j.automatica.2017.04.010](https://doi.org/10.1016/j.automatica.2017.04.010).

Run from this directory:

```sh
python3 verify.py
```

The verifier uses only the Python standard library. Every proof calculation
uses integers or `fractions.Fraction`; decimal strings are explanatory.
To regenerate the complete JSON, including every exact LDL pivot:

```sh
python3 verify.py --output certificate.json
```

## System and admissible forcing

Take 128 followers, a stationary leader, all masses and position/velocity
coupling gains equal to one, position asymmetry zero and velocity asymmetry
one half. These satisfy the source's Assumptions 2–3, including nonincreasing
damping gains. With equilibrium displacement coordinates,

\[
B=I-\operatorname{superdiag}(1),\quad K=BB^T,\quad
D=(B+\tfrac12|B|)B^T,\quad
x''+Dx'+Kx=d.
\]

Thus momentum is \(p=x'\), spacing error is \(\Delta=-B^Tx\), and
\(H=\tfrac12(\|x'\|^2+\|B^Tx\|^2)\), exactly as in the source.

`trajectory.txt.gz` specifies integer positions and velocities divided by
\(2^{40}\) at 2,049 knots \(t_j=j/4\). Between knots, define \(x\) by
cubic Hermite interpolation and **define** \(d=x''+Dx'+Kx\). The verifier
checks zero initial position and velocity and exact agreement of position
and velocity at every endpoint. Hence \(x\) is \(C^1\), and its bounded,
piecewise polynomial force satisfies the ODE almost everywhere. Extend the
force by zero after \(T=512\), continuing the state by the homogeneous ODE.
The total input energy is consequently the integral on \([0,512]\).

This is an exact piecewise polynomial witness near the earlier numerical
trajectory; it is not the original Gramian-defined force. Numerical
calculations originally proposed the knot coefficients. Their
accuracy plays no role in this proof: the distributed integers define the
trajectory exactly. `propose.py` is an optional, untrusted NumPy conversion
tool and is never imported or needed by `verify.py`.

## Exact contradiction

Exact polynomial integration and endpoint evaluation give

\[
J=\int_0^\infty\|d(t)\|^2dt
=\frac{16247963052961153388411898917}{16247963015620616108051005440},
\qquad
H(512)=\frac{6610029438766194687931881}{151115727451828646838272}.
\]

The verifier proves \(\sigma_{\min}(D)>s=3/250\) by an exact pentadiagonal
LDL factorization of \(250000(D^TD-s^2I)\): all 128 pivots are positive.
It reconstructs the factorization exactly. Equation (7), with \(H(0)=0\),
would imply \(H(512)\le J/(2\sigma_{\min}(D))<125J/3\), whereas

\[
H(512)-\frac{125}{3}J
=\frac{20227142829655916997564034147}{9748777809372369664830603264}>0.
\]

Even the separately verified inequalities \(H(512)>43\) and
\(J<101/100\) suffice, since \((125/3)(101/100)=505/12<43\).

Optional exact LDL checks strengthen the comparison:
\(245209/20000000<\sigma_{\min}(D)<24521/2000000\).
The lower threshold has 128 positive pivots; the upper threshold has 127
positive and one negative pivot, giving the upper bound by Sylvester's law
of inertia. The actual energy excess over the source bound is therefore
strictly between **7.2581% and 7.25855%**, which rounds to **7.26%**. The full exact gaps and ratio
bounds are in `certificate.json`. This contradicts the finite bound (7);
it does not by itself decide the source's asymptotic scaling claims.

The SHA-256 of the uncompressed exact integer text is
`34df4a2bb31dcdddad56e3e9a5bdb1ba6cc3d4b7c627c0d52eba44a73d6cfc7a`.
The verifier checks this and the compressed-file checksum before calculating.
