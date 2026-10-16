"""Independent checks of the reconstructed NN experiments."""

import numpy as np
from scipy.linalg import solve

from pyflock.experiment_nn import (
    compute_instability,
    compute_phase,
    compute_pulse,
    compute_stable,
    tail_transfer,
)
from pyflock.matrices import laplacian_line
from pyflock.stencil import nn_stencil


def test_transfer_matches_full_linear_system():
    for rho in (0.45, 0.5, 0.55):
        st = nn_stencil(-1.0, -2.0, -rho, -rho)
        L = laplacian_line(st.rx, 9, bc="regular")
        for w in (0.0001, 0.03, 0.1):
            h = st.gx + 1j * w * st.gv
            matrix = -w * w * np.eye(9) - h * L[1:, 1:]
            rhs = h * L[1:, 0]
            expected = solve(matrix, rhs)[-1]
            assert abs(tail_transfer(9, rho, np.array([w]))[0] - expected) < 1e-10
        assert abs(tail_transfer(9, rho, np.array([0.0]))[0] - 1) < 1e-10


def test_regular_single_follower_transfer():
    w = np.array([0.0, 0.1, 1.0])
    h = -1.0 - 2j * w
    np.testing.assert_allclose(tail_transfer(1, 0.5, w), -h / (-w * w - h))


def test_source_stable_and_unstable_spectra():
    stable, meta = compute_stable()
    assert stable["roots"].real.max() < 1e-7
    assert meta["stencil"]["gx"] == -3
    unstable, _ = compute_instability()
    for i in (0, 1):
        assert unstable[f"{i}_roots"].real.max() > 0.01


def test_phase_limit_and_decay():
    data, _meta = compute_phase()
    zero = data["theta"] == 0
    np.testing.assert_allclose(
        data["phase_speed"][:, zero].ravel(),
        [0.45 + np.sqrt(1.7025), 0.45 - np.sqrt(1.7025)],
        rtol=1e-12,
    )
    assert data["damping"].max() < 1e-7


def test_impulse_and_smooth_have_equal_total_velocity_and_conserve_mean():
    data, _ = compute_pulse()
    for i in (0, 1):
        assert np.isclose(data[f"{i}_initial_velocity"].sum(), 20.0)
        np.testing.assert_allclose(data[f"{i}_velocity"].sum(axis=1), 20.0, atol=1e-10)


def test_resonance_enrichment_recovers_narrow_peak():
    from pyflock.experiment_nn import harmonic_samples

    w, h = harmonic_samples(100, 0.55, 1200)
    assert 6e-6 < w[abs(h).argmax()] < 6.5e-6
    assert abs(h).max() > 5e4


def test_high_precision_transfer_matches_well_conditioned_dense_solve():
    from pyflock.experiment_nn import high_precision_transfer

    omega = np.array([0.0, 0.01, 0.1])
    np.testing.assert_allclose(
        high_precision_transfer(8, 0.55, omega),
        tail_transfer(8, 0.55, omega),
        rtol=1e-11,
        atol=1e-12,
    )
