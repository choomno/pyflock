# Exact short-time energy counterexample

This certificate refutes **Theorem 4(iii), equation (7)** of Herman, Knorn and Ahlén, *Disturbance scaling in bidirectional vehicle platoons with different asymmetry in position and velocity coupling*, Automatica 82 (2017), 13–20, [DOI: 10.1016/j.automatica.2017.04.010](https://doi.org/10.1016/j.automatica.2017.04.010).

Run `python3 verify.py` from this directory. It requires only Python's standard library and checks all matrix identities and numerical inequalities using exact rational arithmetic. The analytical implications are given below; this is not a formal proof-assistant certificate.

## Model and initial data

Use five followers, unit masses and unit position/velocity gains, symmetric position coupling, and velocity asymmetry \(h_p=1\). The leader is stationary; the final follower uses its forward error only. These choices satisfy the paper's Assumptions 2–3 and have nonnegative forward/backward velocity weights.

Let \(B=I-\operatorname{superdiag}(1)\), \(D=2B^T\), and \(y=(p,\Delta)\). Equations (3)–(5) give
\[
\dot y=\mathcal A y+f,
\qquad \mathcal A=\begin{pmatrix}-D&B\\-B^T&0\end{pmatrix},
\qquad f=(d,0),\qquad H(y)=\tfrac12\|y\|_2^2.
\]
Choose \(p(0)=(3,5,6,5,3)\), \(\Delta(0)=0\), and \(d(t)=p(0)/2\) on \([0,T]\), zero afterwards, where \(T=1/20000\). Exactly,
\[
\|p(0)\|^2=104,\quad p(0)^TDp(0)=28,\quad \|d\|^2=26,
\quad H(0)=52,\quad \dot H(0)=24.
\]
The complete squared force \(L^2\) norm is \(26T\).

## Exact bound for the source coefficient

The inverse \(C=B^{-1}\) is the upper triangular matrix of ones. Thus \(G=C^TC\) has entries \(G_{ij}=\min(i,j)\), and
\[
\operatorname{tr}(G^2)=155,\qquad D^{-1}D^{-T}=G/4.
\]
For a positive semidefinite matrix \(P\), \(\lambda_{\max}(P)^2\le\operatorname{tr}(P^2)\). Consequently
\[
\sigma_{\min}(D)^4\ge\frac{16}{155}>\left(\frac59\right)^4,
\]
where the strict comparison is the integer inequality \(104976>96875\). The coefficient in equation (7) therefore satisfies
\[
C_{\rm source}=\frac{1}{2\sigma_{\min}(D)}<\frac9{10}.
\]

## Explicit positive excess

Consider the auxiliary solution with the same constant force continued to \(t=1/100\). It agrees with the actual truncated-force solution through \(T\). Since \(\|B\|_2\le2\), \(\|D\|_2\le4\), and \(\|\mathcal A\|_2\le6\), variation of constants gives
\[
\|y(t)\|<e^{6t}(11+6t)\le\frac{553}{47}<12
\quad(0\le t\le1/100).
\]
Here \(\|y(0)\|<11\), \(\|f\|<6\), and \(e^u\le(1-u)^{-1}\) for \(0\le u<1\). Hence
\[
\|\dot y\|<78,\qquad \|\ddot y\|<468,
\qquad |\ddot H|\le78^2+12\cdot468=11700.
\]
Set \(F(t)=H(t)-H(0)-(9/10)26t\). Then \(F(0)=0\), \(F'(0)=3/5\), and Taylor's integral remainder yields
\[
F(T)\ge\frac35T-5850T^2=\frac{123}{8000000}>0.
\]
Since \(C_{\rm source}<9/10\), the energy at \(T\) strictly exceeds the right-hand side of equation (7), even when that right-hand side uses the full force norm over \([0,\infty)\).

This proves failure with nonzero initial data, which the theorem permits. It does not certify the separate zero-initial-state numerical example, or refute asymptotic stability, passivity, or the singular-value scaling in Lemma 6.
