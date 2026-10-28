"""
Named stencils for nearest- and next-nearest-neighbour calculations.

Paper-specific experiments record their source figures and complete parameter
choices separately. These stencils also provide examples for numerical checks.

Access them by name via ``CASES``, or call ``case(name)``.
"""

from __future__ import annotations

import numpy as np

from .stencil import Stencil

__all__ = ["CASES", "case", "case_names"]


def _neg_pos_equal_velocities(
    bb: float,
    sign: float,
    *,
    aa: float = 0.3,
    gx: float = -2.0,
    gv: float = -2.0,
    bv: float = -1.5,
    cv: float = 0.5,
    eps: float = 0.0,
):
    """Construct positive- or negative-velocity next-nearest-neighbour stencils."""
    rx = np.array([1.0 + bb, -2.0 - aa, 1.0, 3.0 * aa - 4.0 * bb, -2.0 * aa + 3.0 * bb])
    Ax1 = 4.0 + 3.0 * (3.0 * aa - 4.0 * bb - 2.0 - aa)
    root = 0.5 * np.sqrt(2.0 * gx * Ax1 / gv**2)
    av = 0.5 * (-1.0 - 1.5 * bv - 0.5 * cv + sign * root)
    dv = 0.5 * (-1.0 - 0.5 * bv - 1.5 * cv - sign * root)
    rv = np.array([av, bv, 1.0, cv - eps, dv + eps])
    return rx, rv, gx, gv


def _different_reflections(kk: int):
    rx = np.array([-0.25, -0.25, 1.0, -0.25, -0.25])
    rv = np.array([0.0, -0.5 + kk * 0.25, 1.0, -0.5 - kk * 0.25, 0.0])
    return rx, rv, -2.0, -2.0


def _workflow(i: int = 3):
    """A position-asymmetry family with fixed velocity weights and gains -2."""
    a = 15.0 / 4.0 + 1.0 * i
    b = -21.0 / 4.0 - 1.0 * i
    c = -0.75 * a - 0.25 * b - 0.5
    d = -0.75 * b - 0.25 * a - 0.5
    rx = np.array([c, a, 1.0, b, d])
    rv = np.array([-1.0, 4.0, 1.0, -5.0, 1.0])
    return rx, rv, -2.0, -2.0


_RAW = {
    "type_I": {
        "rx": [4 / 27, -289 / 432, 1.0, -253 / 432, 23 / 216],
        "rv": [47 / 216, -29 / 108, 1.0, -79 / 108, -47 / 216],
        "gx": -2.0,
        "gv": -2.0,
        "tlim": 3000.0,
        "notes": "Next-nearest-neighbour stencil for circle/path comparisons.",
    },
    "type_III": {
        "rx": [-2.0, 15 / 4, 1.0, -21 / 4, 5 / 2],
        "rv": [-1.0, 4.0, 1.0, -5.0, 1.0],
        "gx": -2.0,
        "gv": -2.0,
        "tlim": 200.0,
        "notes": "Type III parameters from NNN-agents, Figure 7, with the position-weight sign correction.",
    },
    "random_example": {
        "rx": [0.0, -0.5, 1.0, -0.5, 0.0],
        "rv": [0.0, -0.5, 1.0, -0.5, 0.0],
        "gx": -3.0,
        "gv": -3.0,
        "tlim": 1500.0,
        "notes": "Symmetric nearest-neighbour position and velocity coupling, with gains -3.",
    },
    "zero_diagonal": {
        "rx": [1.3, -2.3, 1.0, -0.3, 0.3],
        "rv": [0.0, -1.8, 1.0, 0.8, 0.0],
        "gx": -2.0,
        "gv": -2.0,
        "tlim": 1500.0,
        "notes": "Next-nearest-neighbour position coupling with gains -2.",
    },
    "two_positive_velocities": {
        "rx": [
            1.0 + 0.2,
            -2.0 - 0.3,
            1.0,
            3.0 * 0.3 - 4.0 * 0.2,
            -2.0 * 0.3 + 3.0 * 0.2,
        ],
        "rv": [0.0, -1.5, 1.0, 0.5, 0.0],
        "gx": -2.0,
        "gv": -2.0,
        "tlim": 500.0,
        "notes": "Next-nearest-neighbour stencil with two positive signal velocities and gains -2.",
    },
    "nonsymmetric_att1": {
        "rx": [-2 / 8, 0.0, 1.0, -1.0, 2 / 8],
        "rv": [-2 / 8, 0.0, 1.0, -1.0, 2 / 8],
        "gx": -2.0,
        "gv": -2.0,
        "tlim": 1500.0,
        "notes": "Identical asymmetric position and velocity weights, with gains -2.",
    },
    "small_positive_loop": {
        "rx": [0.25, -0.75, 1.0, -0.75, 0.25],
        "rv": [-0.4, -1.3, 1.0, 0.5, 0.2],
        "gx": -2.0,
        "gv": -2.0,
        "tlim": 1500.0,
        "notes": "Next-nearest-neighbour stencil for complex-spectrum comparisons, with gains -2.",
    },
}

_BUILDERS = {
    "different_reflections_m1": (
        lambda: _different_reflections(-1),
        3000.0,
        "Symmetric position stencil; nearest-neighbour velocity weights -3/4 and -1/4.",
    ),
    "different_reflections_0": (
        lambda: _different_reflections(0),
        3000.0,
        "Symmetric position stencil; nearest-neighbour velocity weights -1/2 and -1/2.",
    ),
    "different_reflections_p1": (
        lambda: _different_reflections(+1),
        3000.0,
        "Symmetric position stencil; nearest-neighbour velocity weights -1/4 and -3/4.",
    ),
    "negative_equal_velocities": (
        lambda: _neg_pos_equal_velocities(0.4163, +1.0),
        1500.0,
        "A near-coincident negative-velocity branch pair, with gains -2.",
    ),
    "positive_equal_velocities": (
        lambda: _neg_pos_equal_velocities(0.315, -1.0),
        1500.0,
        "A near-coincident positive-velocity branch pair, with gains -2.",
    ),
    "workflow_i3": (
        lambda: _workflow(3),
        1500.0,
        "A position-asymmetry example with fixed velocity weights and gains -2.",
    ),
}


def _build() -> dict:
    out = {}
    for name, d in _RAW.items():
        st = Stencil(d["rx"], d["rv"], d["gx"], d["gv"], name=name, notes=d["notes"])
        out[name] = {"stencil": st, "tlim": d["tlim"]}
    for name, (fn, tlim, notes) in _BUILDERS.items():
        rx, rv, gx, gv = fn()
        out[name] = {
            "stencil": Stencil(rx, rv, gx, gv, name=name, notes=notes),
            "tlim": tlim,
        }
    return out


CASES = _build()


def case_names() -> list[str]:
    return sorted(CASES)


def case(name: str) -> Stencil:
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; available: {', '.join(case_names())}")
    return CASES[name]["stencil"]
