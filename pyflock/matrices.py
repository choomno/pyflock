"""
Laplacians and system matrices.

Line ("path") geometry, following ``flock45.m``: agents are numbered 0..N with agent 0
the LEADER, whose row is identically zero so that it travels at whatever velocity it is
given.  The state is z = (z_0..z_N, zdot_0..zdot_N) and

    d/dt (z, zdot) = M (z, zdot),      M = [[0, I], [gx Lx, gv Lv]].

Boundary conventions for agents that are missing neighbours:

  'drop'    Discard out-of-range neighbours and set the diagonal to minus the sum of the
            weights that remain, so every row still sums to zero.  This is the "second
            version" in ``flock45.m`` -- the one actually used -- and for a
            nearest-neighbour stencil it coincides with the "variable mass" boundary
            condition of Cantos-Hammond-Veerman, Definition 1.1.
  'fold'    Add the weight of a missing neighbour onto the nearest in-range one, and
            rescale the last row by 1/|r_{-2}+r_{-1}|.  This is the commented-out "first
            version" in ``flock45.m``.
  'regular' Nearest-neighbour only: the last agent uses zddot_N = gx (z_N - z_{N-1}) +
            gv (zdot_N - zdot_{N-1}), i.e. chi = 1 in Definition 1.2 of the same paper.

Circle geometry: both Laplacians are circulant, built from the same stencil.
"""
from __future__ import annotations

import numpy as np

from .stencil import Stencil

__all__ = ["laplacian_line", "laplacian_circle", "system_matrix_line",
           "system_matrix_line_leaderless", "system_matrix_circle", "BOUNDARY_CONDITIONS"]

BOUNDARY_CONDITIONS = ("drop", "fold", "regular")
_OFFSETS = (-2, -1, 0, 1, 2)


def laplacian_line(r: np.ndarray, N: int, bc: str = "drop", leader: bool = True) -> np.ndarray:
    """(N+1) x (N+1) Laplacian for agents 0..N on the line.

    If ``leader`` is True, row 0 is left identically zero (agent 0 is driven externally).
    """
    r = np.asarray(r, dtype=float)
    n = N + 1
    L = np.zeros((n, n))
    if bc not in BOUNDARY_CONDITIONS:
        raise ValueError(f"bc must be one of {BOUNDARY_CONDITIONS}, got {bc!r}")

    first = 1 if leader else 0
    for k in range(first, n):
        if bc == "drop":
            tot = 0.0
            for j in _OFFSETS:
                if j == 0 or not (0 <= k + j < n):
                    continue
                L[k, k + j] += r[j + 2]
                tot += r[j + 2]
            L[k, k] -= tot
        elif bc == "fold":
            # fold an out-of-range weight onto the nearest in-range neighbour
            for j in _OFFSETS:
                if j == 0:
                    continue
                m = k + j
                if 0 <= m < n:
                    L[k, m] += r[j + 2]
                else:
                    step = 1 if j > 0 else -1
                    m2 = m
                    while not (0 <= m2 < n) and m2 != k:
                        m2 -= step
                    if m2 != k and 0 <= m2 < n:
                        L[k, m2] += r[j + 2]
                    else:
                        L[k, k] += r[j + 2]
            L[k, k] += r[2]
        elif bc == "regular":
            if abs(r[0]) > 1e-12 or abs(r[4]) > 1e-12:
                raise ValueError("bc='regular' is defined for nearest-neighbour stencils only")
            if k == n - 1:
                L[k, k - 1] += -1.0
                L[k, k] += 1.0
            else:
                for j in (-1, 1):
                    if 0 <= k + j < n:
                        L[k, k + j] += r[j + 2]
                L[k, k] += r[2]

    if bc == "fold" and n >= 3:
        # flock45.m rescales the final row:  sx = abs(1/sum(rx(1:2)))
        s = r[0] + r[1]
        if abs(s) > 1e-12:
            sx = abs(1.0 / s)
            L[n - 1, :] = 0.0
            L[n - 1, n - 2] = sx * r[1]
            if n >= 3:
                L[n - 1, n - 3] = sx * r[0]
            L[n - 1, n - 1] = r[2]
    return L


def laplacian_circle(r: np.ndarray, N: int) -> np.ndarray:
    """N x N circulant Laplacian: (L z)_k = sum_j r[j+2] z_{k+j mod N}."""
    r = np.asarray(r, dtype=float)
    L = np.zeros((N, N))
    for k in range(N):
        for j in _OFFSETS:
            L[k, (k + j) % N] += r[j + 2]
    return L


def system_matrix_line(st: Stencil, N: int, bc: str = "drop", leader: bool = True) -> np.ndarray:
    """2(N+1) x 2(N+1) matrix M = [[0, I],[gx Lx, gv Lv]] for agents 0..N."""
    Lx = laplacian_line(st.rx, N, bc, leader)
    Lv = laplacian_line(st.rv, N, bc, leader)
    n = N + 1
    M = np.zeros((2 * n, 2 * n))
    M[:n, n:] = np.eye(n)
    M[n:, :n] = st.gx * Lx
    M[n:, n:] = st.gv * Lv
    return M


def system_matrix_circle(st: Stencil, N: int) -> np.ndarray:
    """2N x 2N matrix for N agents on the circle."""
    Lx = laplacian_circle(st.rx, N)
    Lv = laplacian_circle(st.rv, N)
    M = np.zeros((2 * N, 2 * N))
    M[:N, N:] = np.eye(N)
    M[N:, :N] = st.gx * Lx
    M[N:, N:] = st.gv * Lv
    return M


def system_matrix_line_leaderless(st: Stencil, N: int, bc: str = "drop") -> np.ndarray:
    """2N x 2N matrix for agents 1..N, with the leader removed (its coupling becomes forcing).

    The full matrix from ``system_matrix_line`` is block lower-triangular with the leader's
    2x2 Jordan block at 0, so its spectrum is this matrix's spectrum plus a double zero.
    Use this one for stability margins: otherwise the leader's zero hides the true max Re,
    and its defectiveness corrupts eigenvalue condition numbers.
    """
    M = system_matrix_line(st, N, bc=bc, leader=True)
    keep = [i for i in range(2 * (N + 1)) if i not in (0, N + 1)]
    return M[np.ix_(keep, keep)]
