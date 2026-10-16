"""
Time integration.

The flock equations are linear and autonomous, ``zdot = M z`` with M constant.
The default propagators evaluate the matrix exponential in floating-point arithmetic:

  * line   : a dense matrix exponential or its sparse action on the initial state;
  * circle : the block-circulant structure is diagonalised by the FFT, so each Fourier
             mode is solved in closed form -- O(N log N) per output time.

``method='rk'`` keeps a Runge-Kutta path available for cross-checking.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp

from .stencil import Stencil
from .matrices import system_matrix_line, system_matrix_circle
from .dispersion import nu_x, nu_v

__all__ = [
    "LineResult",
    "CircleResult",
    "simulate_line",
    "simulate_tail",
    "simulate_circle",
    "propagate",
]


def _times(times):
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if (
        times.ndim != 1
        or times.size == 0
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "times must be a nonempty, finite, strictly increasing vector with times >= 0"
        )
    return times


def _initial(values, n, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (n,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite vector of length {n}")
    return values.copy()


def propagate(
    M: np.ndarray,
    y0: np.ndarray,
    times: np.ndarray,
    method: str = "expm",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> np.ndarray:
    """Evaluate exp(t M) y0 for y(0)=y0. Dense and sparse matrices are supported."""
    times = _times(times)
    M = M if issparse(M) else np.asarray(M)
    y0 = np.asarray(y0)
    entries = M.data if issparse(M) else M
    if (
        M.ndim != 2
        or M.shape[0] != M.shape[1]
        or M.shape[0] == 0
        or not np.isfinite(entries).all()
    ):
        raise ValueError("M must be a finite nonempty square matrix")
    if y0.shape != (M.shape[0],) or not np.isfinite(y0).all():
        raise ValueError("y0 must be a finite vector matching M")
    dtype = np.result_type(M.dtype, y0.dtype, float)
    y0 = y0.astype(dtype, copy=True)
    if method == "rk":
        if times[-1] == 0:
            return y0[None, :]
        sol = solve_ivp(
            lambda t, y: M @ y,
            (0.0, float(times[-1])),
            y0,
            t_eval=times,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Runge-Kutta propagation failed: {sol.message}")
        return sol.y.T
    if method != "expm":
        raise ValueError("method must be 'expm' or 'rk'")

    n = M.shape[0]
    uniform = times.size > 1 and np.allclose(
        np.diff(times), times[1] - times[0], rtol=1e-10, atol=1e-14
    )
    if uniform and n <= 900 and not issparse(M):
        # one dense exponential, then repeated application
        dt = float(times[1] - times[0])
        out = np.empty((times.size, n), dtype=dtype)
        y = expm(M * float(times[0])) @ y0 if times[0] != 0 else y0.copy()
        E = expm(M * dt)
        out[0] = y
        for i in range(1, times.size):
            y = E @ y
            out[i] = y
        return out
    if issparse(M) or (uniform and n > 900):
        sparse_M = csr_matrix(M)
        trace = sparse_M.diagonal().sum()
        if uniform:
            return np.asarray(
                expm_multiply(
                    sparse_M,
                    y0,
                    start=float(times[0]),
                    stop=float(times[-1]),
                    num=times.size,
                    endpoint=True,
                    traceA=trace,
                )
            )
        return np.stack(
            [expm_multiply(sparse_M * t, y0, traceA=trace * t) for t in times]
        )
    return np.stack([expm(M * float(t)) @ y0 for t in times])


@dataclass
class LineResult:
    t: np.ndarray
    z: np.ndarray  # (nt, N+1) deviation from equilibrium
    zdot: np.ndarray  # (nt, N+1)
    x: np.ndarray  # (nt, N+1) absolute position, x_k = z_k - k*delta
    xrel: np.ndarray  # (nt, N+1) position relative to the leader, x_k - x_0
    stencil: Stencil
    N: int
    delta: float
    v0: float
    bc: str

    @property
    def last_relative(self) -> np.ndarray:
        """z_N(t) - z_0(t); its local extrema are the A_k of the transient theory."""
        return self.z[:, self.N] - self.z[:, 0]

    def collisions(self) -> str:
        """First time two neighbouring agents cross, or a clean bill of health."""
        d = np.diff(self.xrel, axis=1)
        bad = np.argwhere(d >= 0.0)
        if bad.size == 0:
            return "No collisions"
        i, k = bad[0]
        return f"Collision at t={self.t[i]:g}, between agents {k} and {k + 1}"


@dataclass
class CircleResult:
    t: np.ndarray
    z: np.ndarray
    zdot: np.ndarray
    stencil: Stencil
    N: int


def simulate_line(
    st: Stencil,
    N: int,
    tmax: float,
    nt: int = 4000,
    *,
    bc: str = "drop",
    v0: float = 1.0,
    delta: float = 1.0,
    method: str = "expm",
    z0=None,
    zdot0=None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> LineResult:
    """Path response to a prescribed leader velocity.

    By default, agents start at equilibrium spacing with followers at rest,
    and at t=0 the leader acquires velocity ``v0``.
    """
    if not np.isfinite(tmax) or tmax <= 0:
        raise ValueError("tmax must be positive and finite")
    if (
        isinstance(nt, (bool, np.bool_))
        or not isinstance(nt, (int, np.integer))
        or nt < 2
    ):
        raise ValueError("nt must be an integer >= 2")
    if not np.isfinite(v0) or not np.isfinite(delta):
        raise ValueError("v0 and delta must be finite")
    M = system_matrix_line(st, N, bc=bc, leader=True, sparse=N >= 200)
    n = N + 1
    z_init = np.zeros(n) if z0 is None else _initial(z0, n, "z0")
    zd_init = np.zeros(n) if zdot0 is None else _initial(zdot0, n, "zdot0")
    if zdot0 is None:
        zd_init[0] = v0
    y0 = np.concatenate([z_init, zd_init])
    times = np.linspace(0.0, float(tmax), nt)
    Y = propagate(M, y0, times, method=method, rtol=rtol, atol=atol)
    z, zdot = Y[:, :n], Y[:, n:]
    k = np.arange(n)
    x = z - k * delta
    xrel = x - x[:, [0]]
    return LineResult(times, z, zdot, x, xrel, st, N, delta, v0, bc)


def simulate_tail(
    st: Stencil,
    N: int,
    tmax: float,
    nt: int = 4000,
    *,
    bc: str = "regular",
    v0: float = 1.0,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, z_N-z_0) for the standard leader velocity step.

    Initial positions and follower velocities vanish; the leader starts at v0.
    Sparse propagation retains at most chunk_size + 1 states, using O(N chunk_size
    + nt) memory. For N < 200, a dense step matrix advances one state at a time.
    """
    if not np.isfinite(tmax) or tmax <= 0:
        raise ValueError("tmax must be positive and finite")
    if (
        isinstance(nt, (bool, np.bool_))
        or not isinstance(nt, (int, np.integer))
        or nt < 2
    ):
        raise ValueError("nt must be an integer >= 2")
    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(v0):
        raise ValueError("v0 must be finite")
    M = system_matrix_line(st, N, bc=bc, leader=True, sparse=N >= 200)
    times = np.linspace(0.0, float(tmax), nt)
    dt = float(times[1])
    state = np.zeros(2 * (N + 1))
    state[N + 1] = v0
    tail = np.empty(nt)
    tail[0] = 0.0
    if not issparse(M):
        step = expm(M * dt)
        for i in range(1, nt):
            state = step @ state
            tail[i] = state[N] - state[0]
        return times, tail

    trace = M.diagonal().sum()
    for start in range(0, nt - 1, chunk_size):
        count = min(chunk_size, nt - 1 - start)
        block = expm_multiply(
            M,
            state,
            start=0.0,
            stop=dt * count,
            num=count + 1,
            endpoint=True,
            traceA=trace,
        )
        tail[start + 1 : start + count + 1] = block[1:, N] - block[1:, 0]
        state = block[-1].copy()
        del block
    return times, tail


def simulate_circle(
    st: Stencil, N: int, times, z0, zdot0, method: str = "fft"
) -> CircleResult:
    """Flock on the circle. ``method='fft'`` uses Fourier modes; 'expm'/'rk' also work."""
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be a positive integer")
    times = _times(times)
    z0 = _initial(z0, N, "z0")
    zdot0 = _initial(zdot0, N, "zdot0")
    if method != "fft":
        M = system_matrix_circle(st, N, sparse=N >= 200)
        Y = propagate(M, np.concatenate([z0, zdot0]), times, method=method)
        return CircleResult(times, Y[:, :N], Y[:, N:], st, N)

    m = np.fft.fftfreq(N, d=1.0 / N)
    th = 2 * np.pi * m / N
    nv = nu_v(th, st)
    nxx = nu_x(th, st)
    disc = np.sqrt((nv**2 / 4.0 + nxx).astype(complex))
    center = nv / 2.0
    lo, hi = center - disc, center + disc
    Z0, Zd0 = np.fft.fft(z0), np.fft.fft(zdot0)
    zt = np.empty((times.size, N))
    zdt = np.empty((times.size, N))
    for i, t in enumerate(times):
        ehi, elo = np.exp(hi * t), np.exp(lo * t)
        C = (ehi + elo) / 2
        S = np.empty(N, dtype=complex)
        small = np.abs(disc * t) < 1e-4
        # exp(center*t) sinh(disc*t)/disc, with its continuous double-root limit.
        u = disc[small] * t
        S[small] = np.exp(center[small] * t) * t * (1 + u**2 / 6 + u**4 / 120)
        S[~small] = (ehi[~small] - elo[~small]) / (2 * disc[~small])
        Zt = C * Z0 + S * (Zd0 - center * Z0)
        Zdt = C * Zd0 + S * (nxx * Z0 + center * Zd0)
        zt[i] = np.real(np.fft.ifft(Zt))
        zdt[i] = np.real(np.fft.ifft(Zdt))
    return CircleResult(times, zt, zdt, st, N)
