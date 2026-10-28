"""Check that public figure helpers represent the supplied numerical data."""

import io
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyflock as pf
from pyflock import plotting, style
from pyflock.analysis import StabilityMap

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def stencil():
    return pf.nn_stencil(-1.0, -1.0, -0.5, -0.3)


@pytest.fixture
def response(stencil):
    t = np.linspace(0, 24, 97)
    n = 4
    z = t[:, None] - np.arange(n + 1)[None, :] * np.sin(t[:, None] / 2)
    z[:, 0] = t
    return SimpleNamespace(
        t=t,
        z=z,
        zdot=1 - np.arange(n + 1)[None, :] * np.cos(t[:, None] / 2) / 2,
        xrel=z - z[:, [0]] - np.arange(n + 1)[None, :],
        N=n,
        stencil=stencil,
        v0=1.0,
        last_relative=z[:, -1] - z[:, 0],
    )


def assert_renderable(fig):
    data = io.BytesIO()
    fig.savefig(data, format="png", dpi=45)
    assert data.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_circle_spectrum_excludes_only_known_neutral_modes(stencil):
    n = 7
    actual = plotting.circle_eigenvalues(stencil, n)
    full = plotting.circle_eigenvalues(stencil, n, include_neutral=True)
    assert actual.size == 2 * (n - 1)
    assert full.size == 2 * n
    np.testing.assert_allclose(full.reshape(2, n)[:, 1:], actual.reshape(2, n - 1))
    np.testing.assert_allclose(full.reshape(2, n)[:, 0], 0, atol=1e-14)


def test_symbol_curve_and_spectrum_axes_use_full_symbol(stencil):
    theta, expected = plotting.symbol_curve(stencil, n=33)
    assert theta[[0, -1]] == pytest.approx([-np.pi, np.pi])
    fig, ax = plt.subplots()
    assert plotting.spectrum_figure(stencil, ax=ax, n=33) is ax
    assert len(ax.lines) == 3  # both branches and the imaginary axis
    for curve, branch in zip(ax.lines[:2], expected):
        np.testing.assert_allclose(curve.get_xdata(), branch.real)
        np.testing.assert_allclose(curve.get_ydata(), branch.imag)
    assert_renderable(fig)


def test_spectrum_local_errors_are_independently_evaluated(stencil):
    fig = plotting.spectrum_figure(stencil, orders=(1, 2, 3), n=41)
    ax = fig.axes[1]
    theta = ax.lines[0].get_xdata()
    cp, cm = pf.signal_velocities(stencil)
    first_order = -1j * np.array([cp, cm])[:, None] * theta
    expected = np.max(abs(pf.branches(theta, stencil) - first_order), axis=0)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
    assert [curve.get_label() for curve in ax.lines] == [
        "Order 1",
        "Order 2",
        "Order 3",
    ]
    assert ax.get_xscale() == ax.get_yscale() == "log"
    assert_renderable(fig)


def test_spectrum_explains_inapplicable_expansion():
    stencil = pf.nn_stencil(-1.0, -1.0, -0.3, -0.3)
    fig = plotting.spectrum_figure(stencil, n=21)
    assert not fig.axes[1].axison
    assert "No regular two-branch expansion" in fig.axes[1].texts[0].get_text()
    assert_renderable(fig)


@pytest.mark.parametrize("sensitive", [False, True])
def test_circle_path_plot_preserves_eigenvalue_coordinates(
    stencil, monkeypatch, sensitive
):
    monkeypatch.setattr(
        plotting, "eigen_reliability", lambda matrix: {"reliable": not sensitive}
    )
    fig = plotting.circle_vs_line_figure(stencil, N=6, bc="regular")
    ax = fig.axes[0]
    circle = plotting.circle_eigenvalues(stencil, 6)
    path = np.linalg.eigvals(pf.system_matrix_line_leaderless(stencil, 6, bc="regular"))
    for collection, values in zip(ax.collections, (circle, path)):
        np.testing.assert_allclose(
            collection.get_offsets(), np.c_[values.real, values.imag]
        )
    assert ("Sensitive path spectrum" in ax.texts[0].get_text()) is sensitive
    assert_renderable(fig)
    _, supplied = plt.subplots()
    assert plotting.circle_vs_line_figure(stencil, N=6, ax=supplied) is supplied


@pytest.mark.parametrize("field", ["spacing", "velocity", "displacement"])
def test_spacetime_fields_keep_values_and_coordinates(response, field):
    fig, ax = plt.subplots()
    assert plotting.spacetime_figure(response, field=field, ax=ax, rays=False) is ax
    expected = {
        "spacing": np.diff(response.z, axis=1),
        "velocity": response.zdot,
        "displacement": response.z - response.z[:, [0]],
    }[field]
    np.testing.assert_allclose(ax.collections[0].get_array(), expected)
    assert ax.collections[0].get_clim() == pytest.approx(
        (-abs(expected).max(), abs(expected).max())
    )
    assert_renderable(fig)


def test_spacetime_unsigned_robust_scale_and_annotation(response):
    response.zdot = np.zeros_like(response.z)
    response.zdot[-1, -1] = 10
    fig = plotting.spacetime_figure(
        response, field="velocity", rays=False, robust=True, pct=90
    )
    ax = fig.axes[0]
    assert ax.collections[0].get_clim() == (0.0, 1e-12)
    np.testing.assert_array_equal(ax.collections[0].get_array(), response.zdot)
    assert "Colour clipped" in ax.texts[0].get_text()
    assert_renderable(fig)


def test_spacetime_rays_follow_signal_velocities_and_stop_at_window(response):
    cp, cm = pf.signal_velocities(response.stencil)
    fig = plotting.spacetime_figure(response, max_rays=2)
    rays = fig.axes[0].lines
    assert len(rays) == 2
    assert rays[0].get_xdata() == pytest.approx([0, response.N])
    assert rays[0].get_ydata() == pytest.approx([0, 1 / cp])
    assert rays[1].get_ydata() == pytest.approx([1 / cp, 1 / cp - 1 / cm])
    response.t = response.t / 100
    short = plotting.spacetime_figure(response, max_rays=10)
    assert len(short.axes[0].lines) == 1
    assert short.axes[0].lines[0].get_ydata()[-1] == pytest.approx(
        response.t[-1] / response.N
    )
    assert short.axes[0].lines[0].get_xdata()[-1] == pytest.approx(cp * response.t[-1])
    assert_renderable(fig)
    assert_renderable(short)


@pytest.mark.parametrize(
    "stencil", [pf.nn_stencil(1.0, -1.0), pf.case("two_positive_velocities")]
)
def test_spacetime_omits_inapplicable_reflected_rays(response, stencil):
    response.stencil = stencil
    fig = plotting.spacetime_figure(response)
    assert not fig.axes[0].lines
    assert_renderable(fig)


def test_spacetime_rejects_unknown_observable(response):
    with pytest.raises(ValueError, match="field must be"):
        plotting.spacetime_figure(response, field="acceleration")


@pytest.mark.parametrize("stride,highlight", [(None, True), (2, False), (1, True)])
def test_trajectory_selection_and_last_follower(response, stride, highlight):
    fig = plotting.trajectory_figure(response, stride=stride, highlight_last=highlight)
    ax = fig.axes[0]
    selected = np.arange(5) if stride is None else np.arange(0, 5, stride)
    if highlight:
        selected = np.r_[selected[selected != 4], 4]
    assert len(ax.lines) == len(selected)
    for line, index in zip(ax.lines, selected):
        np.testing.assert_allclose(
            line.get_xdata(), response.xrel[:, index] / response.N
        )
        np.testing.assert_allclose(line.get_ydata(), response.t / response.N)
    assert (ax.lines[-1].get_label() == "Last follower") is highlight
    assert_renderable(fig)
    _, supplied = plt.subplots()
    assert plotting.trajectory_figure(response, ax=supplied) is supplied


def test_transient_markers_have_measured_and_predicted_coordinates(response):
    fig = plotting.transient_figure(response, nmax=3)
    ax = fig.axes[0]
    measured = pf.extrema(response.t, response.last_relative, nmax=3)
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(),
        [[point.t / response.N, point.value / response.N] for point in measured],
    )
    prediction = pf.theory_amplitudes(response.stencil, response.N, kmax=3)
    keep = np.asarray(prediction["T_k"]) <= response.t[-1]
    np.testing.assert_allclose(
        ax.collections[1].get_offsets(),
        np.c_[np.asarray(prediction["T_k"])[keep], np.asarray(prediction["A"])[keep]]
        / response.N,
    )
    assert ax.collections[1].get_facecolors().size == 0
    assert_renderable(fig)


def test_transient_handles_monotone_response_and_inapplicable_prediction(response):
    response.last_relative = response.t.copy()
    fig, ax = plt.subplots()
    assert plotting.transient_figure(response, show_theory=False, ax=ax) is ax
    assert not ax.collections
    response.stencil = pf.nn_stencil(-1.0, -1.0, -0.3)
    fig = plotting.transient_figure(response)
    assert not fig.axes[0].collections
    assert_renderable(fig)


def test_packet_curve_matches_mode_solution(stencil):
    n, times = 512, (0.0, 1.0, 2.0)
    fig = plotting.packet_figure(stencil, N=n, times=times)
    ax = fig.axes[0]
    index = np.arange(n)
    initial = np.exp(-0.5 * ((index - n // 4) / 10) ** 2)
    expected = pf.simulate_circle(stencil, n, times, np.zeros(n), initial)
    for line, values in zip(ax.lines, expected.zdot):
        np.testing.assert_allclose(line.get_xdata(), index - n // 4)
        np.testing.assert_allclose(line.get_ydata(), values)
    assert_renderable(fig)
    _, supplied = plt.subplots()
    assert plotting.packet_figure(stencil, N=n, times=times, ax=supplied) is supplied


@pytest.mark.parametrize("n,times", [(64, (0.0, 1.0)), (256, (0.0, 300.0))])
def test_packet_rejects_periodic_seam(stencil, n, times):
    with pytest.raises(ValueError, match="periodic seam"):
        plotting.packet_figure(stencil, N=n, times=times)


@pytest.mark.parametrize("density", [None, 2.0])
def test_convergence_curves_retain_independent_amplitude_errors(stencil, density):
    sizes = (5, 8)
    fig = plotting.convergence_figure(stencil, Ns=sizes, nt_per_unit=density)
    cp, _ = pf.signal_velocities(stencil)
    expected, corrected = [], []
    for n in sizes:
        response = pf.simulate_line(
            stencil,
            n,
            tmax=1.6 * n / cp,
            nt=1601 if density is None else max(3, int(density * 1.6 * n / cp)),
            bc="regular",
        )
        amplitude = abs(pf.extrema(response.t, response.last_relative, nmax=1)[0].value)
        scale = n / cp
        expected.append(abs(scale - amplitude) / scale)
        corrected.append(
            abs(pf.corrected_amplitudes(stencil, n, kmax=1)["A"][0] - amplitude) / scale
        )
    for line, values in zip(fig.axes[0].lines, (expected, corrected)):
        np.testing.assert_allclose(line.get_xdata(), sizes)
        np.testing.assert_allclose(line.get_ydata(), values)
    assert_renderable(fig)
    _, supplied = plt.subplots()
    assert plotting.convergence_figure(stencil, Ns=(5,), ax=supplied) is supplied


@pytest.mark.parametrize(
    "stencil",
    [pf.nn_stencil(-1, -1), pf.nn_stencil(1, -1), pf.case("two_positive_velocities")],
)
def test_overview_handles_distinct_wave_regimes(stencil):
    fig = plotting.overview_figure(stencil, N=4, bc="drop")
    assert len(fig.axes) == 3  # two data axes and the field colour bar
    assert fig.axes[0].get_title(loc="left") == "Propagation and reflection"
    assert fig.axes[1].get_title(loc="left") == "Response of the last follower"
    assert_renderable(fig)


@pytest.mark.parametrize(
    "theory,reliable", [(False, None), (True, np.array([[True, False], [True, True]]))]
)
def test_stability_map_uses_scaled_abscissae_and_identifies_sensitivity(
    theory, reliable
):
    params = np.array([-0.1, 0.1])
    sizes = np.array([4, 8])
    values = np.array([[-0.1, 0.2], [-0.05, 0.1]])
    comparison = values / 2 if theory else None
    smap = StabilityMap(params, sizes, values, None, reliable, comparison, "asymmetry")
    fig = plotting.stability_map_figure(smap, figsize=(9, 3))
    data_axes = [axis for axis in fig.axes if axis.get_xlabel() == "asymmetry"]
    assert len(data_axes) == 1 + theory
    np.testing.assert_allclose(
        data_axes[0].collections[0].get_array(), sizes[:, None] * values
    )
    if theory:
        np.testing.assert_allclose(
            data_axes[1].collections[0].get_array(),
            sizes[:, None] * (values - comparison),
        )
        np.testing.assert_allclose(
            data_axes[0].collections[-1].get_offsets(), [[0.1, 4]]
        )
    assert_renderable(fig)


def test_stability_map_without_sign_change_or_sensitive_points():
    smap = StabilityMap(
        np.array([0.0, 0.1]),
        np.array([4, 8]),
        -np.ones((2, 2)),
        None,
        np.ones((2, 2), dtype=bool),
        None,
    )
    fig = plotting.stability_map_figure(smap)
    assert len(fig.axes[0].collections) == 1
    assert_renderable(fig)


def test_style_context_restores_callers_settings(stencil):
    before = matplotlib.rcParams.copy()
    with style.plot_style({"axes.titlesize": 17}):
        assert matplotlib.rcParams["axes.titlesize"] == 17
        assert matplotlib.rcParams["axes.spines.top"] is False
    assert dict(matplotlib.rcParams) == dict(before)
    plotting.spectrum_figure(stencil, n=11)
    assert dict(matplotlib.rcParams) == dict(before)


def test_explicit_global_style_is_optional_and_colours_follow_requested_count():
    with matplotlib.rc_context():
        style.apply_style({"axes.titlesize": 19})
        assert matplotlib.rcParams["axes.titlesize"] == 19
        assert matplotlib.rcParams["axes.facecolor"] == "white"
    assert style.agent_colors(0).shape == (0, 4)
    assert style.agent_colors(1).shape == (1, 4)
    assert style.agent_colors(5).shape == (5, 4)
    np.testing.assert_allclose(style.agent_colors(5)[[0, -1]], style.SEQ([0.3, 1.0]))


@pytest.mark.parametrize("n", [-1, 1.5, True, np.bool_(False)])
def test_agent_colors_rejects_invalid_counts(n):
    with pytest.raises(ValueError, match="nonnegative integer"):
        style.agent_colors(n)
