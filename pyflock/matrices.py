"""
Laplacians and system matrices.

Path geometry: agents are numbered 0..N with leader 0. Its acceleration row
is zero, so it retains its prescribed velocity. The state is
z = (z_0..z_N, zdot_0..zdot_N) and

    d/dt (z, zdot) = M (z, zdot),      M = [[0, I], [gx Lx, gv Lv]].

Boundary conventions for agents that are missing neighbours:

  'drop'    Discard out-of-range neighbours and set the diagonal to minus the sum of the
            weights that remain, so every row still sums to zero. For nearest
            neighbours this is the variable-mass boundary in Cantos-Hammond-Veerman,
            Transients, Definition 1.
  'fold'    Clamp missing neighbour indices to the endpoints. For N>=2, when
            r_{-2}+r_{-1} is nonzero, the last row instead has diagonal r_0 and
            predecessor weights r_{-j}/|r_{-2}+r_{-1}| for j=1,2.
  'regular' Nearest-neighbour only: the last agent uses zddot_N = gx (z_N - z_{N-1}) +
            gv (zdot_N - zdot_{N-1}), the unit terminal coefficient in Transients.

Circle geometry: both Laplacians are circulant, built from the same stencil.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from .stencil import Stencil

__all__ = [
    "BOUNDARY_CONDITIONS",
    "laplacian_circle",
    "laplacian_line",
    "system_matrix_circle",
    "system_matrix_line",
    "system_matrix_line_leaderless",
]

BOUNDARY_CONDITIONS = ("drop", "fold", "regular")
_OFFSETS = (-2, -1, 0, 1, 2)


def _check_size(N):
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be a positive integer")


def _weights(r):
    r = np.asarray(r, dtype=float)
    if r.shape != (5,) or not np.isfinite(r).all():
        raise ValueError("weights must be a finite vector of length 5")
    return r


def laplacian_line(
    r: np.ndarray,
    N: int,
    bc: str = "drop",
    leader: bool = True,
    *,
    sparse: bool = False,
):
    """(N+1) x (N+1) Laplacian for agents 0..N on the line.

    If ``leader`` is True, row 0 is left identically zero (agent 0 is driven externally).
    ``sparse=True`` returns a CSR matrix with O(N) stored entries.
    """
    _check_size(N)
    r = _weights(r)
    n = N + 1
    if bc not in BOUNDARY_CONDITIONS:
        raise ValueError(f"bc must be one of {BOUNDARY_CONDITIONS}, got {bc!r}")
    if bc == "regular" and (abs(r[0]) > 1e-12 or abs(r[4]) > 1e-12):
        raise ValueError("bc='regular' is defined for nearest-neighbour stencils only")
    rows, cols, data = [], [], []

    def add(k, m, value):
        rows.append(k)
        cols.append(m)
        data.append(value)

    first = 1 if leader else 0
    for k in range(first, n):
        if bc == "fold" and k == N and n >= 3 and abs(r[0] + r[1]) > 1e-12:
            sx = abs(1.0 / (r[0] + r[1]))
            add(k, N - 1, sx * r[1])
            add(k, N - 2, sx * r[0])
            add(k, N, r[2])
            continue
        if bc == "drop":
            tot = 0.0
            for j in _OFFSETS:
                if j == 0 or not (0 <= k + j < n):
                    continue
                add(k, k + j, r[j + 2])
                tot += r[j + 2]
            add(k, k, -tot)
        elif bc == "fold":
            for j in _OFFSETS:
                if j == 0:
                    continue
                add(k, min(max(k + j, 0), N), r[j + 2])
            add(k, k, r[2])
        elif bc == "regular":
            if k == n - 1:
                add(k, k - 1, -1.0)
                add(k, k, 1.0)
            else:
                for j in (-1, 1):
                    if 0 <= k + j < n:
                        add(k, k + j, r[j + 2])
                add(k, k, r[2])
    L = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    L.eliminate_zeros()
    return L if sparse else L.toarray()


def laplacian_circle(r: np.ndarray, N: int, *, sparse: bool = False):
    """N x N circulant Laplacian: (L z)_k = sum_j r[j+2] z_{k+j mod N}."""
    _check_size(N)
    r = _weights(r)
    rows = np.repeat(np.arange(N), 5)
    cols = (rows + np.tile(_OFFSETS, N)) % N
    L = sp.coo_matrix((np.tile(r, N), (rows, cols)), shape=(N, N)).tocsr()
    L.eliminate_zeros()
    return L if sparse else L.toarray()


def system_matrix_line(
    st: Stencil, N: int, bc: str = "drop", leader: bool = True, *, sparse: bool = False
):
    """2(N+1) x 2(N+1) matrix M = [[0, I],[gx Lx, gv Lv]] for agents 0..N."""
    Lx = laplacian_line(st.rx, N, bc, leader, sparse=sparse)
    Lv = laplacian_line(st.rv, N, bc, leader, sparse=sparse)
    n = N + 1
    if sparse:
        return sp.bmat([[None, sp.eye(n)], [st.gx * Lx, st.gv * Lv]], format="csr")
    M = np.zeros((2 * n, 2 * n))
    M[:n, n:] = np.eye(n)
    M[n:, :n] = st.gx * Lx
    M[n:, n:] = st.gv * Lv
    return M


def system_matrix_circle(st: Stencil, N: int, *, sparse: bool = False):
    """2N x 2N matrix for N agents on the circle."""
    Lx = laplacian_circle(st.rx, N, sparse=sparse)
    Lv = laplacian_circle(st.rv, N, sparse=sparse)
    if sparse:
        return sp.bmat([[None, sp.eye(N)], [st.gx * Lx, st.gv * Lv]], format="csr")
    M = np.zeros((2 * N, 2 * N))
    M[:N, N:] = np.eye(N)
    M[N:, :N] = st.gx * Lx
    M[N:, N:] = st.gv * Lv
    return M


def system_matrix_line_leaderless(
    st: Stencil, N: int, bc: str = "drop", *, sparse: bool = False
):
    """2N x 2N matrix for agents 1..N, with the leader removed (its coupling becomes forcing).

    The full matrix from ``system_matrix_line`` is block lower-triangular with the leader's
    2x2 Jordan block at 0, so its spectrum is this matrix's spectrum plus a double zero.
    Use this one for stability margins: otherwise the leader's zero hides the true max Re,
    and its defectiveness corrupts eigenvalue condition numbers.
    """
    M = system_matrix_line(st, N, bc=bc, leader=True, sparse=sparse)
    keep = [i for i in range(2 * (N + 1)) if i not in (0, N + 1)]
    if sparse:
        return M[keep][:, keep]
    return M[np.ix_(keep, keep)]
