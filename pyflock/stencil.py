"""
Interaction stencils for one-dimensional flocks.

A stencil lists the position or velocity weights in offset order:

    rx = [r_{-2}, r_{-1}, r_0, r_{+1}, r_{+2}]

indexed by the offset j = -2,-1,0,1,2, so that agent k feels

    zddot_k = gx * sum_j rx[j+2] z_{k+j}  +  gv * sum_j rv[j+2] zdot_{k+j}.

Decentralisation forces the row sums to vanish: sum_j rx[j+2] = sum_j rv[j+2] = 0.
Nearest-neighbour stencils are the special case r_{-2} = r_{+2} = 0.

The even and odd Fourier coefficients are

    alpha_0 = r_0,  alpha_j = r_{+j}+r_{-j},  beta_j = r_{+j}-r_{-j},  j=1,2.

which are exposed here as ``ax0, ax1, ax2, bx1, bx2`` (and ``av*``, ``bv*``).
These conventions follow Herbrych et al., "Dynamics of locally coupled agents
with next nearest neighbor interaction", Definition 2.1 and Section 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Stencil", "nn_stencil"]

_TOL = 1e-12


@dataclass(frozen=True)
class Stencil:
    """A nearest / next-nearest neighbour flock: weights rx, rv and gains gx, gv."""

    rx: np.ndarray
    rv: np.ndarray
    gx: float
    gv: float
    name: str = ""
    notes: str = ""

    def __post_init__(self):
        for name in ("rx", "rv"):
            weights = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if weights.size != 5:
                raise ValueError("rx and rv must have length 5")
            if not np.isfinite(weights).all():
                raise ValueError("stencil weights must be finite")
            # An immutable backing buffer also prevents write access being re-enabled.
            object.__setattr__(
                self, name, np.frombuffer(weights.tobytes(), dtype=float)
            )
        for name in ("gx", "gv"):
            gain = float(getattr(self, name))
            if not np.isfinite(gain):
                raise ValueError("stencil gains must be finite")
            object.__setattr__(self, name, gain)

    # ---------------------------------------------------------------- checks
    @property
    def row_sums(self) -> tuple[float, float]:
        return float(self.rx.sum()), float(self.rv.sum())

    def is_decentralized(self, tol: float = 1e-9) -> bool:
        sx, sv = self.row_sums
        return abs(sx) < tol and abs(sv) < tol

    def require_decentralized(self, tol: float = 1e-9) -> Stencil:
        if not self.is_decentralized(tol):
            sx, sv = self.row_sums
            raise ValueError(
                f"stencil is not decentralized: sum(rx)={sx:g}, sum(rv)={sv:g}"
            )
        return self

    @property
    def is_nearest_neighbour(self) -> bool:
        return (
            abs(self.rx[0]) < _TOL
            and abs(self.rx[4]) < _TOL
            and abs(self.rv[0]) < _TOL
            and abs(self.rv[4]) < _TOL
        )

    # ------------------------------------------------- cosine / sine parts
    @property
    def ax0(self):
        return float(self.rx[2])

    @property
    def ax1(self):
        return float(self.rx[3] + self.rx[1])

    @property
    def ax2(self):
        return float(self.rx[4] + self.rx[0])

    @property
    def bx1(self):
        return float(self.rx[3] - self.rx[1])

    @property
    def bx2(self):
        return float(self.rx[4] - self.rx[0])

    @property
    def av0(self):
        return float(self.rv[2])

    @property
    def av1(self):
        return float(self.rv[3] + self.rv[1])

    @property
    def av2(self):
        return float(self.rv[4] + self.rv[0])

    @property
    def bv1(self):
        return float(self.rv[3] - self.rv[1])

    @property
    def bv2(self):
        return float(self.rv[4] - self.rv[0])

    # ------------------------------------------------------------- moments
    def moment_x(self, n: int) -> float:
        """I_{x,n} = gx * sum_j rx[j+2] j^n."""
        j = np.arange(-2, 3, dtype=float)
        return float(self.gx * np.sum(self.rx * j**n))

    def moment_v(self, n: int) -> float:
        """I_{v,n} = gv * sum_j rv[j+2] j^n."""
        j = np.arange(-2, 3, dtype=float)
        return float(self.gv * np.sum(self.rv * j**n))

    @property
    def a(self) -> float:
        """a = I_{v,1}^2/4 + I_{x,2}/2; real speeds satisfy (c_+ - c_-)^2 = 4a."""
        return self.moment_v(1) ** 2 / 4.0 + self.moment_x(2) / 2.0

    @property
    def q_asym(self) -> float:
        """Velocity asymmetry q = 1 + 2*rho_{v,1} for a normalised NN stencil."""
        if not self.is_nearest_neighbour:
            raise ValueError("q_asym is defined for nearest-neighbour stencils only")
        return self.bv1 / self.av0

    def normalised(self) -> Stencil:
        """Rescale so that r_0 = 1 in both stencils (absorbing the factor into gx, gv)."""
        sx, sv = self.ax0, self.av0
        if abs(sx) < _TOL or abs(sv) < _TOL:
            raise ValueError("cannot normalise a stencil with zero diagonal")
        return Stencil(
            self.rx / sx,
            self.rv / sv,
            self.gx * sx,
            self.gv * sv,
            self.name,
            self.notes,
        )

    def __repr__(self):
        nm = f" {self.name!r}" if self.name else ""
        return (
            f"<Stencil{nm} gx={self.gx:g} gv={self.gv:g}\n"
            f"   rx={np.array2string(self.rx, precision=6)}\n"
            f"   rv={np.array2string(self.rv, precision=6)}>"
        )


def nn_stencil(
    gx: float, gv: float, alpha_x1: float = -0.5, alpha_v1: float = -0.5, name: str = ""
) -> Stencil:
    """Nearest-neighbour stencil with diagonal weights equal to one.

    ``alpha_x1`` and ``alpha_v1`` specify rho_{x,1} and rho_{v,1}; the weights
    at offset -1 are then -1-rho_{x,1} and -1-rho_{v,1}, respectively.
    The default alpha_x1 = alpha_v1 = -1/2 is the fully symmetric flock.
    """
    rx = np.array([0.0, -(1.0 + alpha_x1), 1.0, alpha_x1, 0.0])
    rv = np.array([0.0, -(1.0 + alpha_v1), 1.0, alpha_v1, 0.0])
    return Stencil(rx, rv, gx, gv, name=name)
