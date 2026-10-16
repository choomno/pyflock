"""
pyflock -- one-dimensional flocks of coupled linear oscillators.

A reconstruction of experiments in the signal-velocity and transients papers by
Cantos, Hammond, Veerman and collaborators. See docs/sources.md for provenance.

Quick start::

    import pyflock as pf

    st = pf.nn_stencil(gx=-1.0, gv=-1.0, alpha_x1=-0.5, alpha_v1=-0.3)
    print(pf.signal_velocities(st))            # (c_+, c_-)
    print(pf.coefficients(st))                 # c, D (diffusion), beta (dispersion)

    res = pf.simulate_line(st, N=200, tmax=800)
    print(pf.transient_summary(res.t, res.last_relative))

The five-point stencil convention is ``r = [r_-2, r_-1, r_0, r_+1, r_+2]``; see
``pyflock.stencil``. Named parameter sets for spectral and transient examples
live in ``pyflock.cases``.
"""

from . import plotting, style
from .analysis import (
    Extremum,
    StabilityMap,
    corrected_amplitudes,
    critical_velocity_asymmetry,
    eigen_reliability,
    extrema,
    least_stable_eigenvalue,
    relative_error_law,
    stability_map,
    theory_amplitudes,
    transient_summary,
    velocity_asymmetry_family,
)
from .cases import CASES, case, case_names
from .dispersion import (
    branches,
    coefficients,
    nu_v,
    nu_x,
    phase_velocity,
    signal_velocities,
    taylor,
)
from .matrices import (
    BOUNDARY_CONDITIONS,
    laplacian_circle,
    laplacian_line,
    system_matrix_circle,
    system_matrix_line,
    system_matrix_line_leaderless,
)
from .solve import CircleResult, LineResult, propagate, simulate_circle, simulate_line
from .stencil import Stencil, nn_stencil

__version__ = "0.2.0"

__all__ = [
    "BOUNDARY_CONDITIONS",
    "CASES",
    "CircleResult",
    "Extremum",
    "LineResult",
    "StabilityMap",
    "Stencil",
    "__version__",
    "branches",
    "case",
    "case_names",
    "coefficients",
    "corrected_amplitudes",
    "critical_velocity_asymmetry",
    "eigen_reliability",
    "extrema",
    "laplacian_circle",
    "laplacian_line",
    "least_stable_eigenvalue",
    "nn_stencil",
    "nu_v",
    "nu_x",
    "phase_velocity",
    "plotting",
    "propagate",
    "relative_error_law",
    "signal_velocities",
    "simulate_circle",
    "simulate_line",
    "stability_map",
    "style",
    "system_matrix_circle",
    "system_matrix_line",
    "system_matrix_line_leaderless",
    "taylor",
    "theory_amplitudes",
    "transient_summary",
    "velocity_asymmetry_family",
]
