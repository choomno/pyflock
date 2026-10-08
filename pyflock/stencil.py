"""
Interaction stencils for one-dimensional flocks.

Convention (matching Cantos's MATLAB code exactly).  A stencil is a 5-vector

    rx = [r_{-2}, r_{-1}, r_0, r_{+1}, r_{+2}]

indexed by the offset j = -2,-1,0,1,2, so that agent k feels

    zddot_k = gx * sum_j rx[j+2] z_{k+j}  +  gv * sum_j rv[j+2] zdot_{k+j}.

Decentralisation forces the row sums to vanish: sum_j rx[j+2] = sum_j rv[j+2] = 0.
Nearest-neighbour stencils are the special case r_{-2} = r_{+2} = 0.

The MATLAB drivers use the derived combinations

    a?0 = r_0,   a?1 = r_{+1}+r_{-1},   a?2 = r_{+2}+r_{-2}     (cosine part)
                 b?1 = r_{+1}-r_{-1},   b?2 = r_{+2}-r_{-2}     (sine part)

which are exposed here as ``ax0, ax1, ax2, bx1, bx2`` (and ``av*``, ``bv*``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    _checked: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "rx", np.asarray(self.rx, dtype=float).reshape(-1))
        object.__setattr__(self, "rv", np.asarray(self.rv, dtype=float).reshape(-1))
        if self.rx.size != 5 or self.rv.size != 5:
            raise ValueError("Nearest & next nearest neighbours only: rx, rv must have length 5")

    # ---------------------------------------------------------------- checks
    @property
    def row_sums(self) -> tuple[float, float]:
        return float(self.rx.sum()), float(self.rv.sum())

    def is_decentralized(self, tol: float = 1e-9) -> bool:
        sx, sv = self.row_sums
        return abs(sx) < tol and abs(sv) < tol

    def require_decentralized(self, tol: float = 1e-9) -> "Stencil":
        if not self.is_decentralized(tol):
            sx, sv = self.row_sums
            raise ValueError(f"stencil is not decentralized: sum(rx)={sx:g}, sum(rv)={sv:g}")
        return self

    @property
    def is_nearest_neighbour(self) -> bool:
        return abs(self.rx[0]) < _TOL and abs(self.rx[4]) < _TOL \
            and abs(self.rv[0]) < _TOL and abs(self.rv[4]) < _TOL

    # ------------------------------------------------- cosine / sine parts
    @property
    def ax0(self): return float(self.rx[2])
    @property
    def ax1(self): return float(self.rx[3] + self.rx[1])
    @property
    def ax2(self): return float(self.rx[4] + self.rx[0])
    @property
    def bx1(self): return float(self.rx[3] - self.rx[1])
    @property
    def bx2(self): return float(self.rx[4] - self.rx[0])

    @property
    def av0(self): return float(self.rv[2])
    @property
    def av1(self): return float(self.rv[3] + self.rv[1])
    @property
    def av2(self): return float(self.rv[4] + self.rv[0])
    @property
    def bv1(self): return float(self.rv[3] - self.rv[1])
    @property
    def bv2(self): return float(self.rv[4] - self.rv[0])

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
        """a = I_{v,1}^2/4 + I_{x,2}/2 ; positive for asymptotically stable systems.

        Matches the constant in ``flock.m``'s commented block
        ``a = 0.25*Iv1^2 + 0.5*Ix2``.
        """
        return self.moment_v(1) ** 2 / 4.0 + self.moment_x(2) / 2.0

    @property
    def q_asym(self) -> float:
        """Velocity asymmetry q = 1 + 2*alpha_{v,1} for a normalised NN stencil."""
        if not self.is_nearest_neighbour:
            raise ValueError("q_asym is defined for nearest-neighbour stencils only")
        return self.bv1 / self.av0

    def normalised(self) -> "Stencil":
        """Rescale so that r_0 = 1 in both stencils (absorbing the factor into gx, gv)."""
        sx, sv = self.ax0, self.av0
        if abs(sx) < _TOL or abs(sv) < _TOL:
            raise ValueError("cannot normalise a stencil with zero diagonal")
        return Stencil(self.rx / sx, self.rv / sv, self.gx * sx, self.gv * sv,
                       self.name, self.notes)

    def __repr__(self):  # pragma: no cover - cosmetic
        nm = f" {self.name!r}" if self.name else ""
        return (f"<Stencil{nm} gx={self.gx:g} gv={self.gv:g}\n"
                f"   rx={np.array2string(self.rx, precision=6)}\n"
                f"   rv={np.array2string(self.rv, precision=6)}>")


def nn_stencil(gx: float, gv: float, alpha_x1: float = -0.5, alpha_v1: float = -0.5,
               name: str = "") -> Stencil:
    """Nearest-neighbour stencil in the Cantos-Veerman-Hammond normalisation.

    alpha_{?,0} = 1, alpha_{?,1} = given, alpha_{?,-1} = -(1 + alpha_{?,1}).
    The default alpha_x1 = alpha_v1 = -1/2 is the fully symmetric flock.
    """
    rx = np.array([0.0, -(1.0 + alpha_x1), 1.0, alpha_x1, 0.0])
    rv = np.array([0.0, -(1.0 + alpha_v1), 1.0, alpha_v1, 0.0])
    return Stencil(rx, rv, gx, gv, name=name)
