"""Source profiles, figure data and independent NN transfer identities."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyflock import experiment_nn as nn


@pytest.mark.parametrize("profile,size", [("quick", 160), ("paper", 500)])
@pytest.mark.parametrize(
    "compute,draw,root_keys,axes_count",
    [
        (nn.compute_stable, nn.draw_stable, ["roots"], 2),
        (nn.compute_instability, nn.draw_instability, ["0_roots", "1_roots"], 4),
        (nn.compute_phase, nn.draw_phase, [], 2),
    ],
)
def test_source_spectral_profiles_and_plotted_data(
    profile, size, compute, draw, root_keys, axes_count
):
    data, meta = compute(profile)
    assert meta["N"] == size
    assert meta["profile"] == profile
    assert data["theta"].size in (size, size + 1)
    for key in root_keys:
        prefix = key.removesuffix("roots")
        roots = data[key]
        residual = roots**2 - data[prefix + "velocity"] * roots
        residual -= data[prefix + "position"]
        np.testing.assert_allclose(residual, 0, atol=2e-14)
        np.testing.assert_allclose(roots[:, 0], 0, atol=1e-14)
    fig = draw(data, meta)
    try:
        assert len(fig.axes) == axes_count
        fig.canvas.draw()
        if compute is nn.compute_stable:
            np.testing.assert_array_equal(
                fig.axes[0].lines[0].get_xdata(), data["position"].real
            )
            np.testing.assert_array_equal(
                fig.axes[1].lines[0].get_ydata(), data["roots"].imag.ravel()
            )
        elif compute is nn.compute_instability:
            for i in range(2):
                np.testing.assert_array_equal(
                    fig.axes[2 * i + 1].lines[0].get_xdata(),
                    data[f"{i}_roots"].real.ravel(),
                )
        else:
            for i in range(2):
                np.testing.assert_array_equal(
                    fig.axes[0].lines[i].get_ydata(), data["phase_speed"][i]
                )
                np.testing.assert_array_equal(
                    fig.axes[1].lines[i].get_ydata(), data["damping"][i]
                )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("profile,nt", [("quick", 301), ("paper", 601)])
def test_pulse_profiles_and_figures_preserve_velocity_data(profile, nt):
    data, meta = nn.compute_pulse(profile)
    assert data["t"].shape == (nt,)
    assert meta["N"] == 200
    for i in range(2):
        velocity = data[f"{i}_velocity"]
        assert velocity.shape == (nt, 200)
        np.testing.assert_allclose(velocity.sum(axis=1), 20, atol=1e-11)
        np.testing.assert_allclose(
            velocity[0], data[f"{i}_initial_velocity"], atol=1e-14
        )
    fig = nn.draw_pulse(data, meta)
    try:
        assert len(fig.axes) == 3  # Two fields and their shared colorbar.
        fig.canvas.draw()
        for i in range(2):
            np.testing.assert_allclose(
                fig.axes[i].collections[0].get_array(), data[f"{i}_velocity"]
            )
            for line in fig.axes[i].lines:
                path = np.asarray(line.get_xdata())
                assert np.nanmin(path) >= 0
                assert np.nanmax(path) < 200
                assert np.nanmax(np.abs(np.diff(path))) <= 100
    finally:
        plt.close(fig)


@pytest.mark.parametrize("N,rho", [(0, 0.5), (1, 0), (1, 1), (2, -0.1)])
def test_transfer_rejects_invalid_geometry_and_weights(N, rho):
    with pytest.raises(ValueError, match="N>=1"):
        nn.tail_transfer(N, rho, [0.1])


@pytest.mark.parametrize("omega", [[-0.1], [np.nan], [np.inf]])
def test_transfer_rejects_invalid_frequencies(omega):
    with pytest.raises(ValueError, match="finite nonnegative"):
        nn.tail_transfer(2, 0.5, omega)


def test_high_precision_single_follower_and_array_shape():
    omega = np.array([[0, 0.01], [0.1, 1]])
    h = -1 - 2j * omega
    np.testing.assert_allclose(
        nn.high_precision_transfer(1, 0.7, omega), -h / (-(omega**2) - h), atol=1e-15
    )
    assert nn.high_precision_transfer(4, 0.7, np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("N,rho", [(1, 0.45), (1, 0.55), (10, 0.45), (10, 0.55)])
def test_harmonic_sampling_is_ordered_and_matches_independent_recurrence(N, rho):
    omega, transfer = nn.harmonic_samples(N, rho, count=120)
    assert omega[0] == 1e-8
    assert omega[-1] == 0.5
    assert np.all(np.diff(omega) > 0)
    # Small paths have well-conditioned frequencies throughout this interval.
    np.testing.assert_allclose(
        transfer, nn.high_precision_transfer(N, rho, omega), rtol=1e-10, atol=1e-11
    )


@pytest.mark.parametrize("profile,nsizes", [("quick", 3), ("paper", 4)])
def test_matched_profiles_and_figures_report_the_saved_sampled_maxima(profile, nsizes):
    data, meta = nn.compute_matched(profile)
    assert data["sizes"].shape == (nsizes,)
    assert data["sampled_peak"].shape == (3, nsizes)
    np.testing.assert_allclose(data["sizes"][[0, -1]], [25, 100])
    for i in range(3):
        assert data[f"{i}_t"].size == (2001 if profile == "paper" else 1001)
        np.testing.assert_allclose(data[f"{i}_xrel"][:, 0], 0, atol=0)
        np.testing.assert_allclose(
            data[f"{i}_tail"], data[f"{i}_xrel"][:, -1] + meta["N"]
        )
        for j, N in enumerate(data["sizes"]):
            omega = data["peak_frequency"][i, j]
            expected = abs(nn.high_precision_transfer(int(N), meta["rhos"][i], omega))
            np.testing.assert_allclose(data["sampled_peak"][i, j], expected, rtol=2e-9)
    fig = nn.draw_matched(data, meta)
    try:
        assert len(fig.axes) == 9
        fig.canvas.draw()
        for i in range(3):
            np.testing.assert_array_equal(
                fig.axes[3 + i].lines[0].get_ydata(), abs(data[f"{i}_transfer"])
            )
            np.testing.assert_array_equal(
                fig.axes[6 + i].lines[0].get_ydata(), data["sampled_peak"][i]
            )
    finally:
        plt.close(fig)
