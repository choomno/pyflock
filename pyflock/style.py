"""Quiet Matplotlib defaults; use ``plot_style`` to avoid changing caller state."""

from contextlib import contextmanager

import numpy as np
from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap

CAT = {1: "#2467a2", 2: "#bd542c", 3: "#568265"}
MARKERS = {1: "o", 2: "s", 3: "^"}
DASHES = {1: (), 2: (5, 2), 3: (1.5, 2)}
SURFACE, INK, INK_2, GRID = "white", "#222222", "#505050", "#dddddd"
SEQ = LinearSegmentedColormap.from_list("flock_blue", ["#edf3f8", "#79a9c9", "#174c74"])
DIVERGING = LinearSegmentedColormap.from_list(
    "flock_signed", ["#255b83", "#f8f8f6", "#b54b36"]
)
RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.65,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "pyflock",
}


@contextmanager
def plot_style(rc=None):
    with rc_context({**RC, **(rc or {})}):
        yield


def apply_style(rc=None):
    """Explicit global opt-in for interactive sessions; figure functions use a context."""
    import matplotlib as mpl

    mpl.rcParams.update({**RC, **(rc or {})})


def agent_colors(n):
    return SEQ(np.linspace(0.3, 1, max(n, 2)))
