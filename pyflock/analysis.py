"""
Transient analysis: the quantities A_k, T and gamma of Cantos-Hammond-Veerman.

Following that paper, the transient is summarised by the orbit of the LAST agent
relative to the leader, w(t) = z_N(t) - z_0(t):

    A_k     the k-th local extremum of w
    T       the period, the time between successive extrema of the same sign
    gamma   the attenuation A_3/A_1

Leading-order theory (their Theorem 3) predicts, with r = c_-/c_+ < 0,

    A_k = -r^{k-1} N v0/c_+ ,   T_k = N/c_+ + (k-1) P ,   P = N(1/c_+ - 1/c_-)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dispersion import coefficients, signal_velocities
from .stencil import Stencil

__all__ = [
    "Extremum",
    "StabilityMap",
    "corrected_amplitudes",
    "critical_velocity_asymmetry",
    "eigen_reliability",
    "extrema",
    "least_stable_eigenvalue",
    "relative_error_law",
    "stability_map",
    "theory_amplitudes",
    "transient_summary",
    "velocity_asymmetry_family",
]


@dataclass
class Extremum:
    t: float
    value: float
    index: int


def extrema(
    t: np.ndarray, w: np.ndarray, nmax: int = 8, refine: bool = True
) -> list[Extremum]:
    """Local extrema of w(t), in time order, with optional parabolic refinement.

    A flat extremum is reported once at the midpoint of its time interval,
    without interpolation. A plateau in a monotone trace is not an extremum.
    Endpoints are omitted because both neighbouring slopes must be known.
    """
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    if (
        t.ndim != 1
        or w.shape != t.shape
        or t.size < 3
        or not np.isfinite(t).all()
        or not np.isfinite(w).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "t and w must be finite vectors of equal length >= 3, with t strictly increasing"
        )
    _positive_integer(nmax, "nmax")
    d = np.diff(w)
    nonzero = np.flatnonzero(d)
    turns = np.flatnonzero(np.diff(np.sign(d[nonzero])) != 0)
    out: list[Extremum] = []
    for turn in turns:
        start, end = nonzero[turn] + 1, nonzero[turn + 1]
        i = int((start + end) // 2)
        tt, ww = float(t[start] + (t[end] - t[start]) / 2), float(w[i])
        if refine and start == end:
            left, right = t[i] - t[i - 1], t[i + 1] - t[i]
            slope_left = (w[i] - w[i - 1]) / left
            slope_right = (w[i + 1] - w[i]) / right
            quadratic = (slope_right - slope_left) / (left + right)
            linear = slope_left + quadratic * left
            if quadratic != 0:
                shift = -linear / (2 * quadratic)
                if -left <= shift <= right:
                    tt = float(t[i] + shift)
                    ww = float(w[i] + linear * shift + quadratic * shift**2)
        out.append(Extremum(tt, ww, int(i)))
        if len(out) >= nmax:
            break
    return out


def transient_summary(t: np.ndarray, w: np.ndarray, nmax: int = 8) -> dict:
    """A_k, their times, the period T and the attenuation gamma = A_3/A_1."""
    ex = extrema(t, w, nmax=nmax)
    A = [e.value for e in ex]
    T_k = [e.t for e in ex]
    out = {"A": A, "T_k": T_k, "n_extrema": len(ex)}
    if len(A) >= 3:
        out["gamma"] = A[2] / A[0] if A[0] != 0 else float("nan")
        out["T"] = T_k[2] - T_k[0]
    if len(A) >= 2:
        out["half_period"] = T_k[1] - T_k[0]
    return out


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")


def _transient_domain(st):
    """Check the local hypotheses of the nearest-neighbour reflection formulas."""
    if not st.is_nearest_neighbour:
        raise ValueError("reflection formulas cover nearest-neighbour stencils only")
    st.require_decentralized()
    stn = st.normalised()
    if abs(stn.bx1) > 1e-12:
        raise ValueError("reflection formulas require symmetric position coupling")
    if stn.gx >= 0 or stn.gv >= 0:
        raise ValueError("reflection formulas require negative normalized gains")
    cp, cm = signal_velocities(stn)
    if not cp > 0 > cm:
        raise ValueError(
            "reflection formulas require opposite nonzero signal velocities"
        )
    co = coefficients(stn)
    if co["+"]["D"] < 0 or co["-"]["D"] < 0:
        raise ValueError(
            "reflection formulas require nonnegative diffusion coefficients"
        )
    return stn, cp, cm, co


def _amplitude_inputs(N, v0, kmax):
    _positive_integer(N, "N")
    _positive_integer(kmax, "kmax")
    if not np.isfinite(v0):
        raise ValueError("v0 must be finite")


def theory_amplitudes(st: Stencil, N: int, v0: float = 1.0, kmax: int = 6) -> dict:
    """Conditional leading-order path predictions of Cantos-Hammond-Veerman.

    The local stencil checks do not establish the path stability and large-N
    approximation hypotheses used in the paper.
    """
    _amplitude_inputs(N, v0, kmax)
    _, cp, cm, _ = _transient_domain(st)
    r = cm / cp
    P = N * (1.0 / cp - 1.0 / cm)
    A = [-(r ** (k - 1)) * N * v0 / cp for k in range(1, kmax + 1)]
    T_k = [N / cp + (k - 1) * P for k in range(1, kmax + 1)]
    return {
        "c_plus": cp,
        "c_minus": cm,
        "r": r,
        "P": P,
        "A": A,
        "T_k": T_k,
        "T": 2 * P,
        "gamma": r**2,
    }


def corrected_amplitudes(st: Stencil, N: int, v0: float = 1.0, kmax: int = 6) -> dict:
    """Exploratory magnitudes with a Gaussian rounding correction.

    This ansatz is not a theorem or a validated error estimate. It uses the same
    nearest-neighbour hypotheses as ``theory_amplitudes`` and may become negative
    outside its proposed large-N regime.

    The leading-order theory treats the disturbance as a rigid travelling wave, so the
    orbit of the last agent is a zig-zag with sharp corners.  The theta^2 term of
    lambda(theta) spreads the arriving pulse, rounding each corner and reducing |A_k|.

    An arrival that has made k forward legs and k-1 backward ones has accumulated an
    arrival-TIME variance

        s_k^2 = 2 N [ k D_+/c_+^3 + (k-1) D_-/|c_-|^3 ]

    (each leg's spatial spread becomes a permanent time delay at that leg's own speed),
    and rounding a corner where the velocity jumps by du reduces the extremum by
    du * s_k * phi(x*), with Phi(x*) = 1/(1-r) fixed by the reflection coefficient.
    Hence

        |A_k| = |r|^(k-1) [ N v0/c_+  -  (1-r) v0 phi(x*) s_k ].
    """
    from scipy.stats import norm

    _amplitude_inputs(N, v0, kmax)
    _, cp, cm, co = _transient_domain(st)
    Dp, Dm = co["+"]["D"], co["-"]["D"]
    r = cm / cp
    xs = norm.ppf(1.0 / (1.0 - r))
    phi = norm.pdf(xs)
    A, S = [], []
    for k in range(1, kmax + 1):
        s2 = 2.0 * N * (k * Dp / cp**3 + (k - 1) * Dm / abs(cm) ** 3)
        s = np.sqrt(s2)
        sharp = abs(r) ** (k - 1) * N * abs(v0) / cp
        A.append(sharp - abs(r) ** (k - 1) * (1.0 - r) * abs(v0) * phi * s)
        S.append(s)
    return {
        "A": A,
        "s_k": S,
        "r": r,
        "x_star": float(xs),
        "phi": float(phi),
        "c_plus": cp,
        "c_minus": cm,
        "D_plus": Dp,
        "D_minus": Dm,
    }


def relative_error_law(st: Stencil, v0: float = 1.0) -> float:
    """Proposed N^(-1/2) error constant of the exploratory Gaussian correction."""
    from scipy.stats import norm

    if not np.isfinite(v0) or v0 == 0:
        raise ValueError("a relative amplitude error requires finite nonzero v0")
    _, cp, cm, co = _transient_domain(st)
    Dp = co["+"]["D"]
    r = cm / cp
    return float(
        (1.0 - r) * norm.pdf(norm.ppf(1.0 / (1.0 - r))) * np.sqrt(2.0 * Dp / cp)
    )


def eigen_reliability(M: np.ndarray) -> dict:
    """The computed largest-real-part eigenvalue and a sensitivity heuristic.

    Line flocks can be non-normal enough that double-precision eigenvalues are meaningless:
    first-order sensitivity is estimated by kappa * eps * ||M||_2, where
    kappa = 1/|y^H x| for unit left and right eigenvectors. ``reliable`` means this
    estimate is below 10% of |Re lambda|. It is not a certified error bound or a
    proof of the sign of the spectral abscissa, especially near multiple roots.
    Pass a leaderless matrix (``system_matrix_line_leaderless``):
    the leader's Jordan block at 0 is defective and would fail the check for the wrong reason.
    """
    from scipy.linalg import eig

    w, vl, vr = eig(M, left=True, right=True)
    i = int(np.argmax(w.real))
    overlap = abs(np.vdot(vl[:, i], vr[:, i]))
    kappa = 1.0 / overlap if overlap else float("inf")
    tol = kappa * np.finfo(float).eps * np.linalg.norm(M, 2)
    return {
        "eigenvalue": complex(w[i]),
        "kappa": float(kappa),
        "tol": float(tol),
        "reliable": bool(tol < 0.1 * abs(w[i].real)),
        "heuristic": True,
    }


# ------------------------------------------------------ least-stable eigenvalue on the line
def least_stable_eigenvalue(
    st: Stencil, N: int, method: str = "wave", detail: bool = False
):
    """Heuristic long-wave estimate of the path spectral abscissa.

    Covers nearest-neighbour stencils with SYMMETRIC position coupling and gx, gv < 0 (the
    flocks that are stable on the circle).  The velocity asymmetry q = 1 + 2*alpha_v1 is free.

    method='wave' (default)::

        Re nu ~ ln|c_-/c_+| / P  -  pi^2 N (D_+/c_+^3 + D_-/|c_-|^3) / P^3,
        P = N (1/c_+ + 1/|c_-|)

    The first term is the amplitude gain per round trip of a signal reflected at both ends
    (reflection ratio c_-/c_+).  The second is diffusive damping of that round-trip mode: its
    frequency w = pi/P is shared by both legs, and each leg damps at D w^2/c^2 for time N/c.

    method='linear'::

        Re nu ~ g_v (q/(2N) + pi^2/(16 N^2))       (normalised stencil, r_0 = 1)

    the small-asymmetry limit of the wave form.

    These formulas are asymptotic proposals, not the spectrum of a specified finite
    path matrix. Boundary rules affect finite-N errors; the function takes no
    boundary argument. Agreement for selected parameters does not certify stability.

    detail=True returns a dict with the separate terms.
    """
    if method not in ("wave", "linear"):
        raise ValueError("method must be 'wave' or 'linear'")
    _positive_integer(N, "N")
    stn, cp, cm, co = _transient_domain(st)

    q = stn.bv1
    if method == "linear":
        total = stn.gv * (q / (2.0 * N) + np.pi**2 / (16.0 * N**2))
        return {"total": float(total), "q": float(q)} if detail else float(total)

    Dp, Dm = co["+"]["D"], co["-"]["D"]
    P = N * (1.0 / cp + 1.0 / abs(cm))
    reflection = np.log(abs(cm / cp)) / P
    diffusion = -(np.pi**2) * N * (Dp / cp**3 + Dm / abs(cm) ** 3) / P**3
    total = reflection + diffusion
    if detail:
        return {
            "total": float(total),
            "reflection_term": float(reflection),
            "diffusion_term": float(diffusion),
            "P": float(P),
            "q": float(q),
            "c_plus": cp,
            "c_minus": cm,
            "D_plus": Dp,
            "D_minus": Dm,
        }
    return float(total)


def velocity_asymmetry_family(gx: float, gv: float):
    """Nearest-neighbour stencils with symmetric position coupling, indexed by the velocity
    asymmetry q = 1 + 2*alpha_v1 (q = 0 is the fully symmetric flock; q > 0 weights the agent
    ahead, towards the leader, more heavily)."""
    from .stencil import nn_stencil

    def family(q):
        return nn_stencil(
            gx, gv, alpha_x1=-0.5, alpha_v1=(q - 1.0) / 2.0, name=f"q={q:+.4g}"
        )

    return family


def critical_velocity_asymmetry(
    gx: float, gv: float, N: int, method: str = "wave"
) -> float:
    """The zero q* predicted by ``least_stable_eigenvalue``.

    The estimate changes from growth to decay at q*. The linear form gives
    q* = -pi^2/(8N), independent of the gains: weighting the agent ahead helps, and any
    fixed backward asymmetry destabilises a long enough flock.
    """
    _positive_integer(N, "N")
    _transient_domain(velocity_asymmetry_family(gx, gv)(0.0))
    if method == "linear":
        return float(-(np.pi**2) / (8.0 * N))
    from scipy.optimize import brentq

    fam = velocity_asymmetry_family(gx, gv)

    def f(q):
        return least_stable_eigenvalue(fam(q), N, method=method)

    lo, hi = -0.5, 0.5
    if not (f(lo) > 0.0 > f(hi)):
        raise RuntimeError(
            "no sign change of the least-stable eigenvalue for q in [-0.5, 0.5]"
        )
    return float(brentq(f, lo, hi, xtol=1e-12))


@dataclass
class StabilityMap:
    """Max Re eigenvalue of the leaderless line flock over a (parameter, N) grid.

    Arrays are indexed [i_N, i_param].  ``kappa``/``reliable`` are None when the sweep ran
    without the eigenvalue sensitivity heuristic; ``theory`` is None when no theory was supplied.
    """

    params: np.ndarray
    Ns: np.ndarray
    max_re: np.ndarray
    kappa: object
    reliable: object
    theory: object
    param_name: str = "q"
    bc: str = "drop"
    label: str = ""


def stability_map(
    family,
    params,
    Ns,
    bc: str = "drop",
    theory=None,
    reliability: bool = True,
    param_name: str = "q",
    label: str = "",
) -> StabilityMap:
    """Sweep a one-parameter stencil family over N on the line.

    family      : callable param -> Stencil (e.g. ``velocity_asymmetry_family(gx, gv)``)
    theory      : optional callable (Stencil, N) -> predicted Re, e.g. least_stable_eigenvalue
    reliability : compute eigenvalue condition numbers and the sensitivity heuristic

    ``params`` must be a nonempty finite vector; ``Ns`` must be a nonempty
    vector of positive integers. Boolean and floating-point sizes are rejected.
    """
    from .matrices import system_matrix_line_leaderless

    params = np.asarray(params, dtype=float)
    if params.ndim != 1 or params.size == 0 or not np.isfinite(params).all():
        raise ValueError("params must be a nonempty finite vector")
    # Preserve scalar types until validation: a mixed [1, True] must not become
    # the integer array [1, 1], and a floating size must not be truncated.
    Ns = np.asarray(Ns, dtype=object)
    if Ns.ndim != 1 or Ns.size == 0:
        raise ValueError("Ns must be a nonempty vector of positive integers")
    for N in Ns:
        _positive_integer(N, "each entry of Ns")
    Ns = np.asarray(Ns, dtype=int)
    shape = (Ns.size, params.size)
    max_re = np.empty(shape)
    kappa = np.empty(shape) if reliability else None
    reliable = np.empty(shape, dtype=bool) if reliability else None
    th = np.empty(shape) if theory is not None else None
    for i, N in enumerate(Ns):
        for j, p in enumerate(params):
            st = family(p)
            M = system_matrix_line_leaderless(st, int(N), bc=bc)
            if reliability:
                r = eigen_reliability(M)
                max_re[i, j] = r["eigenvalue"].real
                kappa[i, j] = r["kappa"]
                reliable[i, j] = r["reliable"]
            else:
                max_re[i, j] = np.linalg.eigvals(M).real.max()
            if th is not None:
                th[i, j] = theory(st, int(N))
    return StabilityMap(params, Ns, max_re, kappa, reliable, th, param_name, bc, label)
