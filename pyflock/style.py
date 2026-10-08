"""
Figure styling: one validated palette, applied by the job each colour does.

  categorical (identity)  slots 1-3 below; validated all-pairs on the light surface
                          (worst CVD dE 9.2, worst normal-vision dE 24.0)
  sequential (magnitude)  a single blue hue, light -> dark, for agent index
  diverging (polarity)    blue <-> red with a neutral grey midpoint, for signed fields

Identity is never carried by colour alone: every categorical series also gets a
distinct linestyle or marker, and a legend is always present for two or more series.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["CAT", "SEQ", "DIVERGING", "SURFACE", "INK", "GRID", "apply_style",
           "agent_colors", "MARKERS", "DASHES"]

# categorical slots (light surface) -- fixed order, never cycled
CAT = {
    1: "#2a78d6",   # blue
    2: "#eb6834",   # orange
    3: "#1baf7a",   # aqua  (contrast 2.74 vs surface: always direct-labelled)
}
MARKERS = {1: "o", 2: "s", 3: "^"}
DASHES = {1: (), 2: (5, 2), 3: (1.5, 1.5)}   # () == solid

# sequential blue ramp; ordinal floor on light is step 250
_SEQ_STEPS = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("pyflock_seq", _SEQ_STEPS)

# diverging blue <-> red, neutral grey midpoint
DIVERGING = LinearSegmentedColormap.from_list(
    "pyflock_div",
    ["#0d366b", "#256abf", "#2a78d6", "#9ec5f4", "#f0efec",
     "#f6b3b2", "#e34948", "#c02d2c", "#7d1a1a"],
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d3"


def apply_style(rc: dict | None = None) -> None:
    """Install recessive axes, a light surface and readable defaults."""
    import matplotlib as mpl

    base = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "medium",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "font.size": 9,
        "figure.dpi": 130,
    }
    if rc:
        base.update(rc)
    mpl.rcParams.update(base)


def agent_colors(n: int) -> np.ndarray:
    """Sequential colours for agents 0..n-1 (leader lightest, tail darkest)."""
    return SEQ(np.linspace(0.0, 1.0, max(n, 2)))
