"""Finite-system spectra for the historical cases; sensitivity is not a certificate."""

import numpy as np
import pyflock as pf
from pyflock.plotting import circle_eigenvalues

if __name__ == "__main__":
    N = 60
    print(
        f"{'case':30s} {'circle max Re':>14s} {'path max Re':>14s} {'sensitivity':>14s}"
    )
    for name in pf.case_names():
        st = pf.case(name)
        path = pf.system_matrix_line_leaderless(st, N, bc="drop")
        result = pf.eigen_reliability(path)
        print(
            f"{name:30s} {circle_eigenvalues(st, N).real.max():+14.5g} "
            f"{np.linalg.eigvals(path).real.max():+14.5g} {result['tol']:14.3g}"
        )
