"""
Time integration.

The flock equations are LINEAR and AUTONOMOUS, ``zdot = M z`` with M constant, so an
explicit Runge-Kutta method (``ode45`` in the original MATLAB, odeint in the C++) is the
wrong tool: it costs many steps and carries a tolerance.  Here the default is an exact
propagator instead --

  * line   : a matrix exponential, applied over a uniform time grid;
  * circle : the block-circulant structure is diagonalised by the FFT, so each Fourier
             mode is solved in closed form -- O(N log N) per output time, and exact.

``method='rk'`` keeps a Runge-Kutta path available for cross-checking.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp

from .stencil import Stencil
from .matrices import system_matrix_line, system_matrix_circle
from .dispersion import nu_x, nu_v

__all__ = ["LineResult", "CircleResult", "simulate_line", "simulate_circle", "propagate"]


def propagate(M: np.ndarray, y0: np.ndarray, times: np.ndarray,
              method: str = "expm", rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
    """Solve ydot = M y at the given times. Returns shape (len(times), len(y0))."""
    times = np.asarray(times, dtype=float)
    if method == "rk":
        sol = solve_ivp(lambda t, y: M @ y, (float(times[0]), float(times[-1])), y0,
                        t_eval=times, method="DOP853", rtol=rtol, atol=atol)
        return sol.y.T
    if method != "expm":
        raise ValueError("method must be 'expm' or 'rk'")

    n = M.shape[0]
    uniform = times.size > 1 and np.allclose(np.diff(times), times[1] - times[0])
    if uniform and n <= 900:
        # one dense exponential, then repeated application
        dt = float(times[1] - times[0])
        out = np.empty((times.size, n))
        y = expm(M * float(times[0])) @ y0 if times[0] != 0 else y0.copy()
        E = expm(M * dt)
        out[0] = y
        for i in range(1, times.size):
            y = E @ y
            out[i] = y
        return out
    if uniform:
        return np.asarray(expm_multiply(csr_matrix(M), y0, start=float(times[0]),
                                        stop=float(times[-1]), num=times.size, endpoint=True))
    return np.stack([expm(M * float(t)) @ y0 for t in times])


@dataclass
class LineResult:
    t: np.ndarray
    z: np.ndarray           # (nt, N+1) deviation from equilibrium
    zdot: np.ndarray        # (nt, N+1)
    x: np.ndarray           # (nt, N+1) absolute position, x_k = z_k - k*delta
    xrel: np.ndarray        # (nt, N+1) position relative to the leader, x_k - x_0
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


def simulate_line(st: Stencil, N: int, tmax: float, nt: int = 4000, *, bc: str = "drop",
                  v0: float = 1.0, delta: float = 1.0, method: str = "expm",
                  z0=None, zdot0=None, rtol: float = 1e-10, atol: float = 1e-12) -> LineResult:
    """Flock on the line with a leader, replicating ``flock45.m``.

    Default initial condition (as in ``flock45.m``): every agent sits at its equilibrium
    spacing and is at rest, and at t = 0 the leader acquires velocity ``v0``.
    """
    M = system_matrix_line(st, N, bc=bc, leader=True)
    n = N + 1
    z_init = np.zeros(n) if z0 is None else np.asarray(z0, dtype=float).copy()
    zd_init = np.zeros(n) if zdot0 is None else np.asarray(zdot0, dtype=float).copy()
    if zdot0 is None:
        zd_init[0] = v0
    y0 = np.concatenate([z_init, zd_init])
    times = np.linspace(0.0, float(tmax), int(nt))
    Y = propagate(M, y0, times, method=method, rtol=rtol, atol=atol)
    z, zdot = Y[:, :n], Y[:, n:]
    k = np.arange(n)
    x = z - k * delta
    xrel = x - x[:, [0]]
    return LineResult(times, z, zdot, x, xrel, st, N, delta, v0, bc)


def simulate_circle(st: Stencil, N: int, times, z0, zdot0, method: str = "fft") -> CircleResult:
    """Flock on the circle. ``method='fft'`` diagonalises exactly; 'expm'/'rk' also work."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    z0 = np.asarray(z0, dtype=float)
    zdot0 = np.asarray(zdot0, dtype=float)
    if method != "fft":
        M = system_matrix_circle(st, N)
        Y = propagate(M, np.concatenate([z0, zdot0]), times, method=method)
        return CircleResult(times, Y[:, :N], Y[:, N:], st, N)

    m = np.fft.fftfreq(N, d=1.0 / N)
    th = 2 * np.pi * m / N
    nv = nu_v(th, st)
    nxx = nu_x(th, st)
    disc = np.sqrt((nv ** 2 / 4.0 + nxx).astype(complex))
    lo, hi = nv / 2.0 - disc, nv / 2.0 + disc
    Z0, Zd0 = np.fft.fft(z0), np.fft.fft(zdot0)
    d = hi - lo
    ok = np.abs(d) > 1e-13
    B = np.zeros(N, complex)
    B[ok] = (Zd0[ok] - lo[ok] * Z0[ok]) / d[ok]
    A = Z0 - B
    zt = np.empty((times.size, N))
    zdt = np.empty((times.size, N))
    for i, t in enumerate(times):
        Zt = np.empty(N, complex)
        Zdt = np.empty(N, complex)
        Zt[ok] = A[ok] * np.exp(lo[ok] * t) + B[ok] * np.exp(hi[ok] * t)
        Zdt[ok] = A[ok] * lo[ok] * np.exp(lo[ok] * t) + B[ok] * hi[ok] * np.exp(hi[ok] * t)
        if (~ok).any():                       # double root: (Z0 + (Zd0 - l Z0) t) e^{l t}
            l = lo[~ok]
            s = Zd0[~ok] - l * Z0[~ok]
            e = np.exp(l * t)
            Zt[~ok] = (Z0[~ok] + s * t) * e
            Zdt[~ok] = (s + l * (Z0[~ok] + s * t)) * e
        zt[i] = np.real(np.fft.ifft(Zt))
        zdt[i] = np.real(np.fft.ifft(Zdt))
    return CircleResult(times, zt, zdt, st, N)
