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

from .stencil import Stencil
from .dispersion import signal_velocities, coefficients

__all__ = ["Extremum", "extrema", "transient_summary", "theory_amplitudes",
           "corrected_amplitudes", "relative_error_law", "eigen_reliability",
           "least_stable_eigenvalue", "critical_velocity_asymmetry",
           "velocity_asymmetry_family", "StabilityMap", "stability_map"]


@dataclass
class Extremum:
    t: float
    value: float
    index: int


def extrema(t: np.ndarray, w: np.ndarray, nmax: int = 8,
            refine: bool = True) -> list[Extremum]:
    """Local extrema of w(t), in time order, with optional parabolic refinement."""
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    d = np.diff(w)
    s = np.sign(d)
    s[s == 0] = 1
    idx = np.where(np.diff(s) != 0)[0] + 1
    out: list[Extremum] = []
    for i in idx:
        tt, ww = float(t[i]), float(w[i])
        if refine and 1 <= i < len(w) - 1:
            y0, y1, y2 = w[i - 1], w[i], w[i + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                shift = 0.5 * (y0 - y2) / den
                dt = float(t[i + 1] - t[i])
                tt = float(t[i]) + shift * dt
                ww = float(y1 - 0.25 * (y0 - y2) * shift)
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


def theory_amplitudes(st: Stencil, N: int, v0: float = 1.0, kmax: int = 6) -> dict:
    """Leading-order predictions (Cantos-Hammond-Veerman Theorem 3)."""
    cp, cm = signal_velocities(st)
    r = cm / cp
    P = N * (1.0 / cp - 1.0 / cm)
    A = [-(r ** (k - 1)) * N * v0 / cp for k in range(1, kmax + 1)]
    T_k = [N / cp + (k - 1) * P for k in range(1, kmax + 1)]
    return {"c_plus": cp, "c_minus": cm, "r": r, "P": P,
            "A": A, "T_k": T_k, "T": 2 * P, "gamma": r ** 2}


def corrected_amplitudes(st: Stencil, N: int, v0: float = 1.0, kmax: int = 6) -> dict:
    """Transient amplitudes including the leading diffusive correction.

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

    cp, cm = signal_velocities(st)
    co = coefficients(st)
    Dp, Dm = co["+"]["D"], co["-"]["D"]
    r = cm / cp
    xs = norm.ppf(1.0 / (1.0 - r))
    phi = norm.pdf(xs)
    A, S = [], []
    for k in range(1, kmax + 1):
        s2 = 2.0 * N * (k * Dp / cp ** 3 + (k - 1) * Dm / abs(cm) ** 3)
        s = np.sqrt(s2)
        sharp = abs(r) ** (k - 1) * N * v0 / cp
        A.append(sharp - abs(r) ** (k - 1) * (1.0 - r) * v0 * phi * s)
        S.append(s)
    return {"A": A, "s_k": S, "r": r, "x_star": float(xs), "phi": float(phi),
            "c_plus": cp, "c_minus": cm, "D_plus": Dp, "D_minus": Dm}


def relative_error_law(st: Stencil, v0: float = 1.0) -> float:
    """Constant C in  (N v0/c_+ - |A_1|)/|A_1| ~ C N^(-1/2)  ;  C = (1-r) phi(x*) sqrt(2 D_+/c_+)."""
    from scipy.stats import norm

    cp, cm = signal_velocities(st)
    Dp = coefficients(st)["+"]["D"]
    r = cm / cp
    return float((1.0 - r) * norm.pdf(norm.ppf(1.0 / (1.0 - r))) * np.sqrt(2.0 * Dp / cp))


def eigen_reliability(M: np.ndarray) -> dict:
    """The largest-real-part eigenvalue, with its condition number and a round-off bound.

    Line flocks can be non-normal enough that double-precision eigenvalues are meaningless:
    a computed eigenvalue is only good to about kappa * eps * ||M||_2, where
    kappa = 1/|y^H x| for unit left and right eigenvectors.  ``reliable`` means that bound
    is below 10% of |Re lambda|.  Pass a leaderless matrix (``system_matrix_line_leaderless``):
    the leader's Jordan block at 0 is defective and would fail the check for the wrong reason.
    """
    from scipy.linalg import eig

    w, vl, vr = eig(M, left=True, right=True)
    i = int(np.argmax(w.real))
    kappa = 1.0 / abs(np.vdot(vl[:, i], vr[:, i]))
    tol = kappa * np.finfo(float).eps * np.linalg.norm(M, 2)
    return {"eigenvalue": complex(w[i]), "kappa": float(kappa), "tol": float(tol),
            "reliable": bool(tol < 0.1 * abs(w[i].real))}


# ------------------------------------------------------ least-stable eigenvalue on the line
def least_stable_eigenvalue(st: Stencil, N: int, method: str = "wave", detail: bool = False):
    """Real part of the least-stable eigenvalue of the leaderless line flock.

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

    Accuracy against the leaderless line matrix (gain pairs (-1,-3), (-1,-1), (-2,-1),
    q in [-0.6, 0.8], 'drop' and 'regular'): the wave form is within 0.3% at N = 400 and 2.6%
    at N = 50, its error shrinking like 1/N and smallest for 'regular'.  The leading-order
    behaviour does not depend on the boundary condition.  The linear form is good only for
    small |q g_v|: at gv = -3 its error is about 3% at |q| = 0.1 and 45% at |q| = 0.4.

    detail=True returns a dict with the separate terms.
    """
    if method not in ("wave", "linear"):
        raise ValueError("method must be 'wave' or 'linear'")
    if not st.is_nearest_neighbour:
        raise ValueError("least_stable_eigenvalue covers nearest-neighbour stencils only")
    st.require_decentralized()
    stn = st.normalised()
    if abs(stn.bx1) > 1e-12:
        raise ValueError("position coupling must be symmetric (alpha_x1 = alpha_x,-1); "
                         "asymmetric position coupling is not covered by this theory")
    if not (stn.gx < 0 and stn.gv < 0):
        raise ValueError("requires gx < 0 and gv < 0 after normalisation (stable on the circle)")

    q = stn.bv1
    if method == "linear":
        total = stn.gv * (q / (2.0 * N) + np.pi ** 2 / (16.0 * N ** 2))
        return {"total": float(total), "q": float(q)} if detail else float(total)

    cp, cm = signal_velocities(stn)
    co = coefficients(stn)
    Dp, Dm = co["+"]["D"], co["-"]["D"]
    P = N * (1.0 / cp + 1.0 / abs(cm))
    reflection = np.log(abs(cm / cp)) / P
    diffusion = -np.pi ** 2 * N * (Dp / cp ** 3 + Dm / abs(cm) ** 3) / P ** 3
    total = reflection + diffusion
    if detail:
        return {"total": float(total), "reflection_term": float(reflection),
                "diffusion_term": float(diffusion), "P": float(P), "q": float(q),
                "c_plus": cp, "c_minus": cm, "D_plus": Dp, "D_minus": Dm}
    return float(total)


def velocity_asymmetry_family(gx: float, gv: float):
    """Nearest-neighbour stencils with symmetric position coupling, indexed by the velocity
    asymmetry q = 1 + 2*alpha_v1 (q = 0 is the fully symmetric flock; q > 0 weights the agent
    ahead, towards the leader, more heavily)."""
    from .stencil import nn_stencil

    def family(q):
        return nn_stencil(gx, gv, alpha_x1=-0.5, alpha_v1=(q - 1.0) / 2.0, name=f"q={q:+.4g}")
    return family


def critical_velocity_asymmetry(gx: float, gv: float, N: int, method: str = "wave") -> float:
    """The velocity asymmetry q* where the least-stable eigenvalue crosses zero.

    The line flock is stable for q > q* and unstable for q < q*.  The linear form gives
    q* = -pi^2/(8N), independent of the gains: weighting the agent ahead helps, and any
    fixed backward asymmetry destabilises a long enough flock.
    """
    if method == "linear":
        return float(-np.pi ** 2 / (8.0 * N))
    from scipy.optimize import brentq

    fam = velocity_asymmetry_family(gx, gv)
    f = lambda q: least_stable_eigenvalue(fam(q), N, method=method)
    lo, hi = -0.5, 0.5
    if not (f(lo) > 0.0 > f(hi)):
        raise RuntimeError("no sign change of the least-stable eigenvalue for q in [-0.5, 0.5]")
    return float(brentq(f, lo, hi, xtol=1e-12))


@dataclass
class StabilityMap:
    """Max Re eigenvalue of the leaderless line flock over a (parameter, N) grid.

    Arrays are indexed [i_N, i_param].  ``kappa``/``reliable`` are None when the sweep ran
    without the round-off check; ``theory`` is None when no theory was supplied.
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


def stability_map(family, params, Ns, bc: str = "drop", theory=None, reliability: bool = True,
                  param_name: str = "q", label: str = "") -> StabilityMap:
    """Sweep a one-parameter stencil family over N on the line.

    family      : callable param -> Stencil (e.g. ``velocity_asymmetry_family(gx, gv)``)
    theory      : optional callable (Stencil, N) -> predicted Re, e.g. least_stable_eigenvalue
    reliability : also compute eigenvalue condition numbers (slower: needs left eigenvectors)
    """
    from .matrices import system_matrix_line_leaderless

    params = np.asarray(params, dtype=float)
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
