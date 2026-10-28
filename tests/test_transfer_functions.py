"""Finite recurrence, Fourier limits, and plotted transfer data."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyflock import experiment_transfer as transfer


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": 0},
        {"rho": 1},
        {"rho": 0.5, "f": 0},
        {"rho": 0.5, "f": np.nan},
        {"rho": 0.5, "f": -np.inf},
        {"rho": 0.5, "g": 1},
        {"rho": 0.5, "g": np.nan},
        {"rho": 0.5, "g": -np.inf},
        {"rho": 0.5, "omega": -1},
        {"rho": 0.5, "omega": np.nan},
        {"rho": 0.5, "omega": np.inf},
    ],
)
def test_transfer_domain_rejects_nonphysical_inputs(kwargs):
    values = {"omega": np.array([0.0, 0.1]), **kwargs}
    with pytest.raises(ValueError, match="Require"):
        transfer.transfer_roots(**values)


@pytest.mark.parametrize("profile, samples", [("quick", 601), ("paper", 1601)])
def test_root_experiment_has_source_parameters_and_plots_actual_roots(profile, samples):
    data, meta = transfer.compute_roots(profile=profile)
    assert meta["profile"] == profile
    assert meta["f"] == meta["g"] == -1
    assert data["omega"].shape == (samples,)
    assert_allclose(data["rhos"], [0.4, 0.5, 0.6])
    assert data["omega"][0] == 0 and data["omega"][-1] == 1
    for i, rho in enumerate(data["rhos"]):
        gamma = 1 + data["omega"] ** 2 / (-1 - 1j * data["omega"])
        plus, minus = data["roots"][i]
        assert_allclose(plus + minus, gamma / rho, atol=2e-15)
        assert_allclose(plus * minus, (1 - rho) / rho, atol=2e-15)
    fig = transfer.draw_roots(data, meta)
    try:
        assert len(fig.axes) == 6
        for i in range(3):
            # Top row: reference circles, then each computed root and its DC dot.
            root_line = fig.axes[i].lines[2]
            assert_allclose(root_line.get_xdata(), data["roots"][i, 0].real)
            assert_allclose(root_line.get_ydata(), data["roots"][i, 0].imag)
            ratio_line = fig.axes[3 + i].lines[0]
            assert_allclose(ratio_line.get_xdata(), data["omega"])
            assert_allclose(
                ratio_line.get_ydata(), abs(data["roots"][i, 1] / data["roots"][i, 0])
            )
            assert fig.axes[i].get_aspect() == 1
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("n,rho", [(1, 0.5), (True, 0.5), (3.2, 0.5), (3, 0), (3, 1)])
def test_path_modes_rejects_missing_followers_and_degenerate_couplings(n, rho):
    with pytest.raises(ValueError, match="N>=2"):
        transfer.path_modes(n, rho)


@pytest.mark.parametrize(
    "kwargs", [{"f": np.nan}, {"f": -np.inf}, {"g": np.nan}, {"g": -np.inf}]
)
def test_path_modes_rejects_nonfinite_gains(kwargs):
    with pytest.raises(ValueError, match="finite"):
        transfer.path_modes(3, 0.5, **kwargs)


@pytest.mark.parametrize("n", [2, 3, 4, 8])
def test_sturm_count_handles_an_exact_zero_pivot(n):
    # For N=3, rho=1/2, the first bisection value 1/2 makes the second
    # elimination pivot exactly zero. The Dirichlet/free spectrum is explicit.
    low = transfer._smallest_high_precision(n, 0.5)
    # Use the half-angle form to avoid cancellation in the independent value.
    expected = 2 * np.sin(np.pi / (4 * n)) ** 2
    assert_allclose(low, expected, rtol=2e-15)


def test_sturm_precision_resolves_an_exponentially_small_eigenvalue():
    import mpmath as mp

    # Diagonalization of the literal symmetric matrix is independent of the
    # scalar Sturm-count and bisection implementation. The old fixed iteration
    # count stopped at 3e-64 instead of this eigenvalue near 7e-79.
    n, rho = 40, 0.99
    with mp.workdps(160):
        r = mp.mpf("0.99")
        matrix = mp.eye(n)
        for k in range(n - 1):
            off = -mp.sqrt(r if k == n - 2 else r * (1 - r))
            matrix[k, k + 1] = matrix[k + 1, k] = off
        expected = float(mp.eigsy(matrix, eigvals_only=True)[0])
    low = transfer._smallest_high_precision(n, rho)
    assert 7e-79 < expected < 8e-79
    assert_allclose(low, expected, rtol=2e-15, atol=0)


def test_path_modes_reports_a_smallest_eigenvalue_below_binary64_range():
    with pytest.raises(FloatingPointError, match="below the binary64 range"):
        transfer.path_modes(180, 0.99)


def test_lost_positive_laplacian_mode_is_reported(monkeypatch):
    def lost_mode(diagonal, off_diagonal):
        assert len(off_diagonal) == len(diagonal) - 1
        return np.zeros_like(diagonal)

    monkeypatch.setattr(transfer, "eigvalsh_tridiagonal", lost_mode)
    with pytest.raises(RuntimeError, match="Positive path Laplacian lost"):
        transfer.path_modes(3, 0.4)


def test_spectral_experiment_retains_every_follower_mode_and_zoom_subset():
    data, meta = transfer.compute_spectra()
    assert meta["N"] == 100
    assert meta["f"] == -1 and meta["g"] == -2
    fig = transfer.draw_spectra(data, meta)
    try:
        assert len(fig.axes) == 6
        for i in range(3):
            values = data[f"{i}_modes"]
            laplacian = data[f"{i}_laplacian"]
            assert values.shape == (2, meta["N"])
            assert np.all(laplacian > 0)
            assert_allclose(values[0] + values[1], -2 * laplacian, atol=1e-14)
            assert_allclose(values[0] * values[1], laplacian, atol=1e-14)
            assert_allclose(fig.axes[i].lines[0].get_xdata(), values.real.ravel())
            assert_allclose(fig.axes[i].lines[0].get_ydata(), values.imag.ravel())
            assert_allclose(
                fig.axes[i + 3].lines[0].get_xdata(), values[:, :8].real.ravel()
            )
            assert_allclose(
                fig.axes[i + 3].lines[0].get_ydata(), values[:, :8].imag.ravel()
            )
        fig.canvas.draw()
    finally:
        plt.close(fig)
