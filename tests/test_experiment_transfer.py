"""Algebraic and independent finite-matrix checks of transfer illustrations."""

import numpy as np
from scipy.linalg import eigvals

from pyflock.experiment_transfer import (
    _smallest_high_precision,
    path_modes,
    transfer_roots,
)
from pyflock.matrices import system_matrix_line_leaderless
from pyflock.stencil import nn_stencil


def test_transfer_roots_satisfy_characteristic_polynomial_and_dc_limits():
    omega = np.r_[0.0, np.geomspace(1e-5, 100, 60)]
    nu = 1j * omega
    gamma = (-1 - nu - nu**2) / (-1 - nu)
    for rho in (0.4, 0.5, 0.6):
        roots = transfer_roots(omega, rho)
        np.testing.assert_allclose(
            rho * roots**2 - gamma * roots + 1 - rho, 0, atol=1e-11
        )
        np.testing.assert_allclose(roots[0] * roots[1], (1 - rho) / rho, rtol=1e-14)
        np.testing.assert_allclose(
            roots[:, 0], sorted([1.0, (1 - rho) / rho], reverse=True)
        )
        assert (abs(roots[1, 1:]) < abs(roots[0, 1:])).all()


def test_similarity_reduction_matches_original_follower_matrix():
    for rho in (0.4, 0.5, 0.6):
        _, modes = path_modes(10, rho)
        matrix = system_matrix_line_leaderless(
            nn_stencil(-1.0, -2.0, -rho, -rho), 10, bc="regular"
        )
        direct = eigvals(matrix)
        distances = abs(modes.ravel()[:, None] - direct[None, :])
        assert distances.min(axis=1).max() < 1e-10


def test_symmetric_path_laplacian_exact_spectrum():
    N = 100
    laplacian, _ = path_modes(N, 0.5)
    expected = 1 - np.cos((2 * np.arange(N) + 1) * np.pi / (2 * N))
    np.testing.assert_allclose(laplacian, expected, atol=3e-15)


def test_smallest_mode_precision_check_and_nonzero_path_mode():
    low = path_modes(100, 0.55)[0][0]
    kappa = 0.45 / 0.55
    leading = (2 * 0.55 - 1) ** 2 / (2 * 0.55**2) * kappa**99
    assert low > 0
    np.testing.assert_allclose(low, leading, rtol=1e-7)
    np.testing.assert_allclose(
        _smallest_high_precision(12, 0.6), path_modes(12, 0.6)[0][0], rtol=1e-14
    )
