"""
Symbols, eigenvalue branches and signal velocities.

For a circulant (circle) flock the Fourier mode exp(i k theta) diagonalises both
Laplacians, with eigenvalues

    nu_x(theta) = gx sum_j rx[j+2] exp(i j theta)
    nu_v(theta) = gv sum_j rv[j+2] exp(i j theta)

and the system matrix M = [[0, I], [gx Lx, gv Lv]] then has eigenvalues solving

    lambda^2 - nu_v(theta) lambda - nu_x(theta) = 0.

Under the hypotheses of ``taylor``, with a > 0, each branch expands near theta = 0 as

    lambda_sigma(theta) = -i c_sigma theta - D_sigma theta^2 + i beta_sigma theta^3 + ...

Here c_sigma is signal velocity, D_sigma is diffusion, and beta_sigma is the
cubic dispersion coefficient. ``taylor`` computes the expansion from stencil
moments; ``coefficients`` returns these three quantities for each branch.
The signal-velocity formula is given in Herbrych et al., "Dynamics of locally
coupled agents with next nearest neighbor interaction", Theorem 5.1.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from .stencil import Stencil

__all__ = [
    "branches",
    "coefficients",
    "nu_v",
    "nu_x",
    "phase_velocity",
    "signal_velocities",
    "taylor",
]

_J = np.arange(-2, 3)


def _symbol(theta, weights, gain):
    """Evaluate the finite Fourier symbol without subtracting nearby cosines."""
    theta = np.asarray(theta, dtype=float)
    angles = theta[..., None] * np.array([1, 2])
    positive, negative = weights[3:], weights[1::-1]
    real = weights.sum() - 2 * np.sum(
        (positive + negative) * np.sin(angles / 2) ** 2, axis=-1
    )
    odd = positive - negative
    # sin(2 theta)-2 sin(theta)=-4 sin(theta) sin(theta/2)^2;
    # keep the cubic term when the first position moment vanishes.
    imaginary = np.sin(theta) * (
        odd[0] + 2 * odd[1] - 4 * odd[1] * np.sin(theta / 2) ** 2
    )
    return gain * (real + 1j * imaginary)


def nu_x(theta, st: Stencil):
    return _symbol(theta, st.rx, st.gx)


def nu_v(theta, st: Stencil):
    return _symbol(theta, st.rv, st.gv)


def branches(theta, st: Stencil, labelling: str = "continuous"):
    """Both roots of lambda^2 - nu_v lambda - nu_x = 0, shape (2,) + theta.shape.

    labelling:
      'continuous' -- track the roots outward along the supplied |theta| values and
                      conjugate-reflect to negative theta, swapping the two branches
                      when a < 0 to preserve the analytic labels of ``taylor``.
                      Near zero, branch 0 has
                      the larger signal velocity when the hypotheses of ``taylor``
                      hold and a > 0. Tracking through a root collision depends on
                      the sampling; this is not a test of analyticity.
      'by_imag'    -- branch 0 has the larger imaginary part. The two imaginary
                      parts need not have opposite signs.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    shape = theta.shape
    tf = theta.ravel()
    nv = nu_v(tf, st)
    nxx = nu_x(tf, st)
    disc = np.sqrt((nv**2 / 4.0 + nxx).astype(complex))
    lo, hi = nv / 2.0 - disc, nv / 2.0 + disc

    if labelling == "by_imag":
        first = np.where(lo.imag >= hi.imag, lo, hi)
        second = np.where(lo.imag >= hi.imag, hi, lo)
        return np.stack([first, second]).reshape((2, *shape))

    if labelling != "continuous":
        raise ValueError(f"unknown labelling {labelling!r}")

    # Track along |theta|; conjugation swaps the local analytic branches when a < 0.
    a = st.a
    at = np.abs(tf)
    order = np.argsort(at, kind="stable")
    out = np.empty((2, tf.size), dtype=complex)
    prev = np.array([0j, 0j])
    slope = np.array(
        [
            1j * (st.moment_v(1) / 2 - np.sqrt(a + 0j)),
            1j * (st.moment_v(1) / 2 + np.sqrt(a + 0j)),
        ]
    )
    last = 0.0
    for idx in order:
        t = at[idx]
        # recompute on |theta| so the sweep is monotone
        nvv = nu_v(np.array([t]), st)[0]
        d = np.sqrt(complex(nvv**2 / 4.0 + nu_x(np.array([t]), st)[0]))
        roots = np.array([nvv / 2.0 - d, nvv / 2.0 + d])
        pred = prev + slope * (t - last)
        if abs(roots[0] - pred[0]) + abs(roots[1] - pred[1]) > abs(
            roots[1] - pred[0]
        ) + abs(roots[0] - pred[1]):
            roots = roots[::-1]
        if t > last:
            slope = (roots - prev) / (t - last)
        prev, last = roots, t
        if tf[idx] >= 0:
            out[:, idx] = roots
        else:
            out[:, idx] = np.conj(roots[::-1] if a < 0 else roots)
    return out.reshape((2, *shape))


def signal_velocities(st: Stencil) -> tuple[float, float]:
    """The two real long-wave velocities, in decreasing order.

    They need not have opposite signs. Requires decentralized coupling, zero first
    position moment, and a > 0; these conditions do not establish stability.
    """
    if st.a <= 0:
        raise ValueError("real distinct signal velocities require a > 0")
    mu = taylor(st, order=1)[:, 1].imag  # lambda ~ i mu theta
    c = -mu
    return float(max(c)), float(min(c))


def phase_velocity(theta, st: Stencil, labelling: str = "continuous"):
    """Phase velocity -Im(lambda)/theta for each Fourier branch."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    lam = branches(theta, st, labelling)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -lam.imag / theta


def taylor(st: Stencil, order: int = 3) -> np.ndarray:
    """Taylor coefficients of both branches at theta = 0.

    Returns an array ``c`` of shape (2, order+1) with
    ``lambda_branch(theta) ~ sum_n c[branch, n] theta**n``.

    For a > 0, the coefficients are -i times signal velocity, minus diffusion,
    and i times cubic dispersion at orders one, two and three, respectively.
    Coefficients follow recursively from the symbol moments, without numerical
    differentiation. Requires decentralized coupling, I_x,1 = 0 and a != 0.
    Otherwise fractional powers or coincident first-order roots need separate
    treatment. Negative a is allowed here; then the slopes have nonzero real parts.
    """
    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or order < 0
    ):
        raise ValueError("order must be a nonnegative integer")
    for weights in (st.rx, st.rv):
        if abs(weights.sum()) > 1e-12 * max(
            np.abs(weights).sum(), np.finfo(float).tiny
        ):
            raise ValueError("Taylor expansion requires decentralized coupling")
    moment_scale = abs(st.gx) * np.sum(np.abs(st.rx * _J))
    if abs(st.moment_x(1)) > 1e-12 * max(moment_scale, np.finfo(float).tiny):
        raise ValueError(
            "Taylor expansion requires zero first position moment; otherwise roots may have fractional powers"
        )
    scale = st.moment_v(1) ** 2 / 4 + abs(st.moment_x(2)) / 2
    if abs(st.a) <= 64 * np.finfo(float).eps * scale or st.a == 0:
        raise ValueError(
            "Taylor expansion requires distinct first-order roots (a != 0)"
        )

    out = np.zeros((2, order + 1), dtype=complex)
    if order == 0:
        return out
    out[:, 1] = 1j * (st.moment_v(1) / 2 + np.array([-1, 1]) * np.sqrt(st.a + 0j))
    vx = [st.moment_v(k) * 1j**k / factorial(k) for k in range(order + 1)]
    xx = [st.moment_x(k) * 1j**k / factorial(k) for k in range(order + 2)]
    # Coefficient of theta**(n+1) in lambda**2 - nu_v*lambda - nu_x.
    for branch in range(2):
        c = out[branch]
        for n in range(2, order + 1):
            forcing = xx[n + 1] + sum(vx[k] * c[n + 1 - k] for k in range(2, n + 1))
            forcing -= sum(c[k] * c[n + 1 - k] for k in range(2, n))
            c[n] = forcing / (2 * c[1] - vx[1])
    return out


def coefficients(st: Stencil) -> dict:
    """Convenience: signal velocity, diffusion and dispersion for both branches."""
    if st.a <= 0:
        raise ValueError("real long-wave coefficients require a > 0")
    c = taylor(st, order=3)
    mu = c[:, 1].imag
    D = -c[:, 2].real
    beta = c[:, 3].imag
    res = {}
    for lbl, i in (("+", int(np.argmax(-mu))), ("-", int(np.argmin(-mu)))):
        res[lbl] = {"c": float(-mu[i]), "D": float(D[i]), "beta": float(beta[i])}
    return res
