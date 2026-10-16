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
``pyflock.stencil``.  Curated parameter sets from the original code live in
``pyflock.cases``.
"""

from .stencil import Stencil, nn_stencil
from .dispersion import (
    nu_x,
    nu_v,
    branches,
    signal_velocities,
    phase_velocity,
    taylor,
    coefficients,
    matlab_fq,
)
from .matrices import (
    laplacian_line,
    laplacian_circle,
    system_matrix_line,
    system_matrix_line_leaderless,
    system_matrix_circle,
    BOUNDARY_CONDITIONS,
)
from .solve import LineResult, CircleResult, simulate_line, simulate_circle, propagate
from .analysis import (
    Extremum,
    extrema,
    transient_summary,
    theory_amplitudes,
    corrected_amplitudes,
    relative_error_law,
    eigen_reliability,
    least_stable_eigenvalue,
    critical_velocity_asymmetry,
    velocity_asymmetry_family,
    StabilityMap,
    stability_map,
)
from .cases import CASES, case, case_names
from . import style, plotting

__version__ = "0.2.0"

__all__ = [
    "Stencil",
    "nn_stencil",
    "nu_x",
    "nu_v",
    "branches",
    "signal_velocities",
    "phase_velocity",
    "taylor",
    "coefficients",
    "matlab_fq",
    "laplacian_line",
    "laplacian_circle",
    "system_matrix_line",
    "system_matrix_circle",
    "BOUNDARY_CONDITIONS",
    "system_matrix_line_leaderless",
    "LineResult",
    "CircleResult",
    "simulate_line",
    "simulate_circle",
    "propagate",
    "Extremum",
    "extrema",
    "transient_summary",
    "theory_amplitudes",
    "corrected_amplitudes",
    "relative_error_law",
    "eigen_reliability",
    "least_stable_eigenvalue",
    "critical_velocity_asymmetry",
    "velocity_asymmetry_family",
    "StabilityMap",
    "stability_map",
    "CASES",
    "case",
    "case_names",
    "style",
    "plotting",
    "__version__",
]
