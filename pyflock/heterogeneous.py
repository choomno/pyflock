"""Nearest-neighbour arrays with distinct positive gain magnitudes.

The convention is that of Lyons (2022), §§2.2 and 2.6: ``gx, gv > 0`` and
``zddot = -diag(gx) Lx z - diag(gv) Lv zdot``. Thus constant arrays ``gx=a,
gv=b`` agree with the existing homogeneous ``nn_stencil(-a, -b, -.5,
(beta_v-1)/2)``. Here N always counts *all* agents. A path has leader 0 and
followers 1, ..., N-1; a circle has no leader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from .solve import _initial, _times, propagate

__all__ = [
    "CircleResult",
    "HeterogeneousNN",
    "PathResult",
    "c1_slopes",
    "gain_moments",
    "laplacians",
    "simulate_circle",
    "simulate_path",
    "simulate_tail",
    "system_matrix",
]


def _gain_vector(values, name):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError(f"{name} must be a nonempty finite vector of positive gains")
    # Copies the input and prevents write access being re-enabled.
    return np.frombuffer(values.tobytes(), dtype=float)


@dataclass(frozen=True, eq=False)
class HeterogeneousNN:
    """Positive agent gains with symmetric position and uniform velocity weights.

    ``rho_x,- = rho_x,+ = -1/2`` and ``beta_v = rho_v,+ - rho_v,-``.
    The condition ``-1 <= beta_v <= 1`` makes both velocity neighbour weights
    nonpositive. It does not establish stability of a heterogeneous array.
    """

    gx: np.ndarray
    gv: np.ndarray
    beta_v: float = 0.0

    def __post_init__(self):
        for name in ("gx", "gv"):
            object.__setattr__(self, name, _gain_vector(getattr(self, name), name))
        if self.gx.shape != self.gv.shape:
            raise ValueError("gx and gv must have the same length")
        beta = np.asarray(self.beta_v, dtype=float)
        if beta.ndim != 0 or not np.isfinite(beta) or abs(beta) > 1:
            raise ValueError("beta_v must be a finite scalar in [-1, 1]")
        object.__setattr__(self, "beta_v", float(beta))

    @property
    def n_agents(self):
        return self.gx.size


def laplacians(model: HeterogeneousNN, *, geometry="circle", bc="regular"):
    """Return dimensionless CSR (Lx, Lv), with zero row sums.

    A path's leader row is zero. The regular tail uses ``z_tail-z_previous``
    in both Laplacians (Lyons Definition 2.6.5, gamma_x=gamma_v=1).
    The drop tail retains only the previous-neighbour weight and adjusts its
    diagonal. ``bc`` is relevant only to a path.
    """
    if geometry not in ("circle", "path"):
        raise ValueError("geometry must be 'circle' or 'path'")
    n = model.n_agents
    if geometry == "path":
        if n < 2:
            raise ValueError("a path requires a leader and at least one follower")
        if bc not in ("regular", "drop"):
            raise ValueError("bc must be 'regular' or 'drop'")

    def build(minus, plus):
        rows, cols, data = [], [], []

        def add(k, j, weight):
            rows.append(k)
            cols.append(j)
            data.append(weight)

        for k in range(0 if geometry == "circle" else 1, n):
            if geometry == "circle":
                add(k, (k - 1) % n, minus)
                add(k, k, 1.0)
                add(k, (k + 1) % n, plus)
            elif k == n - 1:
                tail = -1.0 if bc == "regular" else minus
                add(k, k - 1, tail)
                add(k, k, -tail)
            else:
                add(k, k - 1, minus)
                add(k, k, 1.0)
                add(k, k + 1, plus)
        matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        matrix.eliminate_zeros()
        return matrix

    return build(-0.5, -0.5), build(-(1 + model.beta_v) / 2, (model.beta_v - 1) / 2)


def system_matrix(
    model: HeterogeneousNN, *, geometry="circle", bc="regular", followers_only=False
):
    """CSR generator in state order (z_0,...,z_N-1, zdot_0,...,zdot_N-1).

    ``followers_only=True`` removes the path leader's two coordinates. This is
    the homogeneous follower operator; a prescribed leader supplies forcing in
    the laboratory frame. It is also the autonomous operator in the frame of
    a leader moving at constant velocity.
    """
    if followers_only and geometry != "path":
        raise ValueError("followers_only requires geometry='path'")
    lx, lv = laplacians(model, geometry=geometry, bc=bc)
    n = model.n_agents
    matrix = sparse.bmat(
        [
            [None, sparse.eye(n)],
            [-sparse.diags(model.gx) @ lx, -sparse.diags(model.gv) @ lv],
        ],
        format="csr",
    )
    if followers_only:
        keep = np.r_[np.arange(1, n), np.arange(n + 1, 2 * n)]
        matrix = matrix[keep][:, keep].tocsr()
    return matrix


@dataclass
class PathResult:
    """Path trajectories in the leader frame, including leader column 0.

    ``r = z-v0*t`` and ``rdot = zdot-v0``. Equilibrium spacing has already
    been subtracted from z. The stored arrays start with r=0, rdot=-v0 for
    every follower, representing agents initially at rest and a leader that
    acquires velocity v0 at t=0. This is the abrupt velocity-step limit of the
    smooth ramps in Lyons Definition 2.6.4.
    """

    t: np.ndarray
    r: np.ndarray
    rdot: np.ndarray
    model: HeterogeneousNN
    bc: str
    v0: float

    @property
    def z(self):
        return self.r + self.v0 * self.t[:, None]

    @property
    def zdot(self):
        return self.rdot + self.v0

    @property
    def last_relative(self):
        """Tail position relative to the leader, z_N-1-z_0."""
        return self.r[:, -1]

    @property
    def leader_gap(self):
        """Lyons's d(t)=z_0-z_N-1, positive during the initial lag."""
        return -self.r[:, -1]


def _path_initial(model, bc, v0):
    if np.ndim(v0) != 0 or not np.isfinite(v0):
        raise ValueError("v0 must be finite")
    matrix = system_matrix(model, geometry="path", bc=bc, followers_only=True)
    nf = model.n_agents - 1
    initial = np.r_[np.zeros(nf), np.full(nf, -float(v0))]
    return matrix, initial


def simulate_path(
    model: HeterogeneousNN,
    times,
    *,
    bc="regular",
    v0=1.0,
    method="expm",
    rtol=1e-10,
    atol=1e-12,
):
    """Propagate a leader velocity step; sparse exponential action is the default."""
    times = _times(times)
    matrix, initial = _path_initial(model, bc, v0)
    states = propagate(matrix, initial, times, method=method, rtol=rtol, atol=atol)
    nf = model.n_agents - 1
    r = np.column_stack((np.zeros(times.size), states[:, :nf]))
    rdot = np.column_stack((np.zeros(times.size), states[:, nf:]))
    return PathResult(times.copy(), r, rdot, model, bc, float(v0))


def simulate_tail(
    model: HeterogeneousNN, times, *, bc="regular", v0=1.0, chunk_size=128
):
    """Return (times, z_tail-z_leader) with O(N*chunk_size + len(times)) memory.

    Uniform grids are evaluated in chunks using sparse exponential action.
    Arbitrary strictly increasing grids advance one state at a time.
    """
    times = _times(times)
    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    matrix, state = _path_initial(model, bc, v0)
    trace = float(matrix.diagonal().sum())
    tail_index = model.n_agents - 2
    tail = np.empty(times.size)
    if times[0] != 0:
        state = expm_multiply(matrix * times[0], state, traceA=trace * times[0])
    tail[0] = state[tail_index]
    uniform = times.size > 1 and np.allclose(
        np.diff(times), times[1] - times[0], rtol=1e-10, atol=1e-14
    )
    if uniform:
        dt = float(times[1] - times[0])
        for start in range(0, times.size - 1, chunk_size):
            count = min(chunk_size, times.size - 1 - start)
            block = expm_multiply(
                matrix,
                state,
                start=0.0,
                stop=dt * count,
                num=count + 1,
                endpoint=True,
                traceA=trace,
            )
            tail[start + 1 : start + count + 1] = block[1:, tail_index]
            state = block[-1].copy()
    else:
        for j, dt in enumerate(np.diff(times), start=1):
            state = expm_multiply(matrix * dt, state, traceA=trace * dt)
            tail[j] = state[tail_index]
    return times.copy(), tail


@dataclass
class CircleResult:
    t: np.ndarray
    z: np.ndarray
    zdot: np.ndarray
    model: HeterogeneousNN


def simulate_circle(
    model: HeterogeneousNN, times, z0, zdot0, *, method="expm", rtol=1e-10, atol=1e-12
):
    """Propagate a circle from explicit displacement and velocity vectors."""
    times = _times(times)
    n = model.n_agents
    initial = np.r_[_initial(z0, n, "z0"), _initial(zdot0, n, "zdot0")]
    matrix = system_matrix(model, geometry="circle")
    states = propagate(matrix, initial, times, method=method, rtol=rtol, atol=atol)
    return CircleResult(times.copy(), states[:, :n], states[:, n:], model)


def gain_moments(model: HeterogeneousNN):
    """Means in Lyons (125)-(126), over exactly the supplied agent sequence."""
    return {
        "mean_inverse_gx": float(np.mean(1 / model.gx)),
        "mean_gv_over_gx": float(np.mean(model.gv / model.gx)),
    }


def c1_slopes(model: HeterogeneousNN):
    """Return (c1_plus, c1_minus) from Lyons Theorem 2.5.1, equation (107).

    These are the first-order imaginary eigenvalue coefficients for a repeated
    periodic cell. Their physical velocities point toward decreasing agent
    number; the corresponding velocities along increasing indices are their
    negatives. Applying the formula to a single sampled array or to a path is
    an asymptotic prediction with additional hypotheses (§§2.5.4 and 2.6).
    It neither diagnoses finite-array stability nor gives a rigorous path error
    bound. The moments include every entry in model, including a path leader.
    """
    moments = gain_moments(model)
    m, v = moments["mean_inverse_gx"], moments["mean_gv_over_gx"]
    drift = model.beta_v * v
    radical = np.hypot(drift, np.sqrt(2 * m))
    # Rationalisation avoids cancellation for very large gain ratios.
    if drift >= 0:
        plus, minus = 1 / (radical + drift), -(radical + drift) / (2 * m)
    else:
        plus, minus = (radical - drift) / (2 * m), -1 / (radical - drift)
    return float(plus), float(minus)
