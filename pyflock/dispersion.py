"""
Symbols, eigenvalue branches and signal velocities.

For a circulant (circle) flock the Fourier mode exp(i k theta) diagonalises both
Laplacians, with eigenvalues

    nu_x(theta) = gx sum_j rx[j+2] exp(i j theta)
    nu_v(theta) = gv sum_j rv[j+2] exp(i j theta)

and the system matrix M = [[0, I], [gx Lx, gv Lv]] then has eigenvalues solving

    lambda^2 - nu_v(theta) lambda - nu_x(theta) = 0.

Near theta = 0 each branch expands as

    lambda_sigma(theta) = i mu_sigma theta - D_sigma theta^2 + i beta_sigma theta^3 + ...

whose first three coefficients are, respectively, the (negated) SIGNAL VELOCITY,
a DIFFUSION coefficient and a DISPERSION coefficient.  ``taylor`` computes these
for an arbitrary stencil; ``matlab_fq`` reproduces the closed forms used in
Cantos's ``flock45sim_function.m``, where ``f = mu`` and ``q = -D``.
"""
from __future__ import annotations

import numpy as np

from .stencil import Stencil

__all__ = ["nu_x", "nu_v", "branches", "signal_velocities", "phase_velocity",
           "taylor", "matlab_fq"]

_J = np.arange(-2, 3)


def nu_x(theta, st: Stencil):
    theta = np.asarray(theta, dtype=float)
    return st.gx * np.sum(st.rx * np.exp(1j * np.multiply.outer(theta, _J)), axis=-1)


def nu_v(theta, st: Stencil):
    theta = np.asarray(theta, dtype=float)
    return st.gv * np.sum(st.rv * np.exp(1j * np.multiply.outer(theta, _J)), axis=-1)


def branches(theta, st: Stencil, labelling: str = "continuous"):
    """Both roots of lambda^2 - nu_v lambda - nu_x = 0, shape (2,) + theta.shape.

    labelling:
      'continuous' -- analytic continuation in theta outward from theta = 0, so that
                      branch 0 carries the signal velocity c_+ and branch 1 carries c_-.
                      Hermitian symmetry lam_i(-theta) = conj(lam_i(theta)) is exact.
      'by_imag'    -- branch 0 has Im >= 0, branch 1 has Im < 0.  This is what
                      ``flock.m`` does (it splits on ``angle(nu)``), and is what you
                      want when reproducing that script's figures.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    shape = theta.shape
    tf = theta.ravel()
    nv = nu_v(tf, st)
    nxx = nu_x(tf, st)
    disc = np.sqrt((nv ** 2 / 4.0 + nxx).astype(complex))
    lo, hi = nv / 2.0 - disc, nv / 2.0 + disc

    if labelling == "by_imag":
        first = np.where(lo.imag >= hi.imag, lo, hi)
        second = np.where(lo.imag >= hi.imag, hi, lo)
        return np.stack([first, second]).reshape((2,) + shape)

    if labelling != "continuous":
        raise ValueError(f"unknown labelling {labelling!r}")

    # analytic continuation outward from theta = 0 along |theta|, then conjugate-reflect
    a = st.a
    at = np.abs(tf)
    order = np.argsort(at, kind="stable")
    out = np.empty((2, tf.size), dtype=complex)
    prev = np.array([0j, 0j])
    slope = np.array([1j * (st.moment_v(1) / 2 - np.sqrt(a + 0j)),
                      1j * (st.moment_v(1) / 2 + np.sqrt(a + 0j))])
    last = 0.0
    for idx in order:
        t = at[idx]
        roots = np.array([lo[idx], hi[idx]]) if tf[idx] >= 0 else \
                np.conj(np.array([lo[idx], hi[idx]]))
        # recompute on |theta| so the sweep is monotone
        nvv = nu_v(np.array([t]), st)[0]
        d = np.sqrt(complex(nvv ** 2 / 4.0 + nu_x(np.array([t]), st)[0]))
        roots = np.array([nvv / 2.0 - d, nvv / 2.0 + d])
        pred = prev + slope * (t - last)
        if (abs(roots[0] - pred[0]) + abs(roots[1] - pred[1])
                > abs(roots[1] - pred[0]) + abs(roots[0] - pred[1])):
            roots = roots[::-1]
        if t > last:
            slope = (roots - prev) / (t - last)
        prev, last = roots, t
        out[:, idx] = roots if tf[idx] >= 0 else np.conj(roots)
    return out.reshape((2,) + shape)


def signal_velocities(st: Stencil) -> tuple[float, float]:
    """(c_+, c_-) with c_+ > 0 > c_-; the theta -> 0 phase velocities."""
    mu = taylor(st, order=1)[:, 1].imag          # lambda ~ i mu theta
    c = -mu
    return float(max(c)), float(min(c))


def phase_velocity(theta, st: Stencil, labelling: str = "continuous"):
    """c_m = -Im(lambda)/theta for each branch (the quantity plotted by flock.m)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    lam = branches(theta, st, labelling)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -lam.imag / theta


def taylor(st: Stencil, order: int = 3, h: float = 1e-3, fit: int = 8) -> np.ndarray:
    """Taylor coefficients of both branches at theta = 0.

    Returns an array ``c`` of shape (2, order+1) with
    ``lambda_branch(theta) ~ sum_n c[branch, n] theta**n``.

    So ``c[:,1] = i mu``, ``c[:,2] = -D``, ``c[:,3] = i beta``.  Computed by a
    least-squares polynomial fit to the exact branches, hence valid for ANY stencil,
    not only nearest-neighbour ones.
    """
    if fit < order + 2:
        fit = order + 2
    th = np.arange(-fit, fit + 1) * h
    lam = branches(th, st, labelling="continuous")
    V = np.vander(th, fit + 1, increasing=True)
    out = np.empty((2, order + 1), dtype=complex)
    for i in (0, 1):
        coef, *_ = np.linalg.lstsq(V, lam[i], rcond=None)
        out[i] = coef[: order + 1]
    return out


def coefficients(st: Stencil) -> dict:
    """Convenience: signal velocity, diffusion and dispersion for both branches."""
    c = taylor(st, order=3)
    mu = c[:, 1].imag
    D = -c[:, 2].real
    beta = c[:, 3].imag
    idx = np.argsort(-(-mu))                      # branch with larger c = -mu first
    order_ = np.argsort(-(-mu))
    res = {}
    for lbl, i in (("+", int(np.argmax(-mu))), ("-", int(np.argmin(-mu)))):
        res[lbl] = {"c": float(-mu[i]), "D": float(D[i]), "beta": float(beta[i])}
    return res


def matlab_fq(st: Stencil) -> dict:
    """The closed forms used in ``flock45sim_function.m``.

    Returns ``{'fm','fp','qm','qp'}`` where (verified to machine precision against
    ``taylor``) ``f_sigma = mu_sigma`` and ``q_sigma = -D_sigma``.  Requires the
    normalisation r_0 = 1 that the MATLAB drivers assume.
    """
    gx, gv = st.gx, st.gv
    B = st.bv1 + 2 * st.bv2
    S = np.sqrt(gv ** 2 * B ** 2 - 2 * gx * (4 + 3 * st.ax1))
    fm = 0.5 * gv * B - 0.5 * S
    fp = 0.5 * gv * B + 0.5 * S
    num = gv ** 2 * B * (4 + 3 * st.av1) + 2 * gx * st.bx1
    qm = 0.25 * (gv * (4 + 3 * st.av1) - num / S)
    qp = 0.25 * (gv * (4 + 3 * st.av1) + num / S)
    return {"fm": float(fm), "fp": float(fp), "qm": float(qm), "qp": float(qp)}
