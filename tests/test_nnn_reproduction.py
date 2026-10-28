"""Boundary, estimator and orchestration checks for the NNN studies."""

from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.linalg import solve

from pyflock import experiment_nnn as nnn
from pyflock.dispersion import signal_velocities
from pyflock.stencil import Stencil


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("boundary,size", [("bad", 8), ("drop", 3)])
def test_path_matrix_rejects_undefined_geometry(boundary, size):
    with pytest.raises(ValueError, match="N>=4"):
        nnn._path_matrix(nnn._configuration(0, 0), size, boundary)


@pytest.mark.parametrize("count", [0, 501])
def test_configuration_count_is_bounded(count):
    with pytest.raises(ValueError, match="between 1 and 500"):
        nnn.select_configurations(count)


def test_source_examples_and_plotted_coordinates():
    data, meta = nnn.compute_types(profile="paper")
    assert len(meta["examples"]) == 4
    assert "+15/4" in meta["source_corrections"][0]
    fig = nnn.draw_types(data, meta)
    assert len(fig.axes) == 4
    for i, (axis, specification) in enumerate(zip(fig.axes, meta["examples"])):
        n = specification["N"]
        assert data[f"{i}_z"].shape == (3001, n + 1)
        assert data[f"{i}_velocity"][0, 0] == pytest.approx(1)
        np.testing.assert_allclose(data[f"{i}_z"][:, 0], data[f"{i}_t"], atol=1e-10)
        tail = next(line for line in axis.lines if line.get_label() == "Last follower")
        np.testing.assert_array_equal(tail.get_xdata(), data[f"{i}_t"] / n)
        np.testing.assert_allclose(
            tail.get_ydata(), (data[f"{i}_z"][:, n] - data[f"{i}_t"]) / n
        )
        assert sum(line.get_label() == "Arrival $N/c_+$" for line in axis.lines) == (
            i < 2
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bc": "fold"},
        {"N": 3},
        {"N": True},
        {"N": 4.5},
        {"s": [[1, 2]]},
        {"s": [np.inf]},
        {"s": [0]},
        {"s": [-1 + 2j]},
    ],
)
def test_spatial_transfer_rejects_invalid_parameters(kwargs):
    arguments = {"N": 6, "s": np.array([1 + 2j]), "bc": "drop"} | kwargs
    with pytest.raises(ValueError):
        nnn.spatial_transfer(nnn._configuration(0, 0), **arguments)


@pytest.mark.parametrize("position", [[0, 0, 1, 0, 0], [0, -0.4, 1, -0.6, 0]])
def test_spatial_transfer_requires_zero_position_moments(position):
    st = Stencil(position, [0, -0.5, 1, -0.5, 0], -1, -1)
    with pytest.raises(ValueError, match="zero position first moment"):
        nnn.spatial_transfer(st, 6, [1 + 2j])


def test_transfer_of_one_sided_velocity_chain_is_exact_product():
    st = Stencil([0] * 5, [0, -1, 1, 0, 0], 0, -1)
    s = np.array([0.2 + 0.1j, 1 + 2j])
    np.testing.assert_allclose(
        nnn.spatial_transfer(st, 9, s), (1 / (1 + s)) ** 9, rtol=2e-13
    )


def test_transfer_rejects_a_chain_with_no_leader_coupling():
    st = Stencil([0] * 5, [0, 0, 1, -1, 0], 0, -1)
    with pytest.raises(ValueError, match="towards the leader"):
        nnn.spatial_transfer(st, 6, [1])


@pytest.mark.parametrize("bc", nnn.BOUNDARIES)
def test_transfer_groups_frequencies_where_spatial_degree_drops(bc):
    st = Stencil([1, -2, 1, 0, 0], [-1, 0, 1, 0, 0], -1, -1)
    s = np.array([1, 2 + 0.3j])
    matrix = nnn._path_matrix(st, 7, bc).toarray()
    initial = np.zeros(len(matrix))
    initial[8] = 1
    expected = [solve(z * np.eye(len(matrix)) - matrix, initial)[7] * z**2 for z in s]
    np.testing.assert_allclose(
        nnn.spatial_transfer(st, 7, s, bc), expected, rtol=2e-13, atol=1e-14
    )


def test_nonfinite_boundary_solution_is_rejected(monkeypatch):
    def nonfinite(matrix, rhs):
        return np.full_like(rhs, np.nan)

    monkeypatch.setattr(np.linalg, "solve", nonfinite)
    with pytest.raises(FloatingPointError, match="Nonfinite spatial transfer"):
        nnn.spatial_transfer(nnn._configuration(0, 0), 8, [1 + 2j])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tmax": 0},
        {"tmax": np.nan},
        {"v0": np.inf},
        {"nfft": 31},
        {"nfft": 32.5},
        {"nfft": 34},
        {"period_factor": 0},
        {"period_factor": False},
        {"period_factor": 1},
        {"period_factor": 2.5},
        {"damping": 0},
        {"damping": np.inf},
    ],
)
def test_inversion_rejects_invalid_settings(kwargs):
    arguments = {"tmax": 10.0, "nfft": 64} | kwargs
    with pytest.raises(ValueError):
        nnn.tail_trace(nnn._configuration(0, 0), 8, **arguments)


def test_inversion_is_linear_in_the_leader_velocity():
    st = nnn._configuration(1, 0)
    positive = nnn.tail_trace(st, 8, 20, nfft=2048)
    negative = nnn.tail_trace(st, 8, 20, nfft=2048, v0=-3)
    np.testing.assert_allclose(negative.relative, -3 * positive.relative, atol=2e-12)


def test_resolution_exhaustion_reports_failure():
    with pytest.raises(RuntimeError, match="No trace convergence"):
        nnn.converged_tail(
            nnn._configuration(0, 0), 8, 50, start_nfft=32, max_nfft=64, tolerance=1e-16
        )


def test_peak_windows_reject_missing_or_endpoint_extrema():
    t = np.arange(5.0)
    assert nnn._peak(t, t, 2, 0.1) == (0, 0, False)
    assert nnn._peak(t, t, 2, 2) == (0, 0, False)
    assert nnn._peak(t, t, 2, 2, positive=True) == (0, 0, False)
    assert nnn._peak(t, np.ones(5), 2, 2) == (0, 0, False)
    assert nnn._peak(t, np.array([2, 1, 1, 1, 2]), 2, 2) == (0, 0, False)


def test_peak_exact_parabola_and_continuous_linear_fit():
    t = np.linspace(0, 5, 51)
    peak_t, peak_value, valid = nnn._peak(t, (t - 2.34) ** 2 - 7, 2.5, 2)
    assert valid and peak_t == pytest.approx(2.34) and peak_value == pytest.approx(-7)
    assert nnn._linear_zero(t, 3 * t - 6, 0.3, 4.6) == pytest.approx((2, True))
    assert nnn._linear_zero(t, -t, 0.3, 4.6) == (0, False)
    assert nnn._linear_zero(t, t, 2, 2.1) == (0, False)


def test_type_one_estimator_tracks_wave_extrema_and_zeros():
    st, n = nnn._configuration(0, 0), 25
    cp, cm = signal_velocities(st)
    first = n / cp
    period_half = n * (1 / cp - 1 / cm)
    t = np.linspace(0, first + 3 * period_half, 5001)
    # This analytic sinusoid has minima at first+2jP, maxima at first+(2j+1)P.
    w = -first * np.cos(np.pi * (t - first) / period_half)
    measured, prediction, valid = nnn._measure(0, st, n, t, w)
    assert valid.all()
    assert measured[0] == pytest.approx(first, rel=1e-7)
    assert measured[2] == pytest.approx(1, rel=1e-7)
    np.testing.assert_allclose(prediction, [first, 2 * period_half, abs(cm / cp)])
    measured, _, valid = nnn._measure(0, st, n, t, np.ones_like(t))
    assert not valid.any()
    np.testing.assert_array_equal(measured, [0, 0, 0])


def test_type_one_zero_plateau_is_not_a_period_measurement():
    st, n = nnn._configuration(0, 0), 25
    cp, cm = signal_velocities(st)
    half_period = n * (1 / cp - 1 / cm)
    t = np.linspace(0, 4 * half_period, 501)
    measured, _, valid = nnn._measure(0, st, n, t, np.zeros_like(t))
    assert not valid[1]
    assert measured[1] == 0


def _audit_row(spec):
    kind, index, _record, boundary, n, _checkpoint = spec
    row = np.zeros(23)
    row[:4] = kind, nnn.BOUNDARIES.index(boundary), n, index
    row[7:10] = 1
    row[10:13] = [1e-3, 2e-3, 3e-3]
    row[16:19] = 1
    row[21] = 2048
    return row


@pytest.mark.parametrize("profile,jobs", [("quick", 1), ("paper", 1), ("paper", 2)])
def test_ensemble_enumerates_every_declared_case(
    profile, jobs, monkeypatch, tmp_path, capsys
):
    # Test the large-run coordinator independently of numerical convergence, which
    # is checked against matrix exponentials in test_experiment_nnn.py.
    monkeypatch.setattr(nnn, "_ensemble_case", _audit_row)
    monkeypatch.setattr(nnn, "ProcessPoolExecutor", ThreadPoolExecutor)
    arrays, meta = nnn.compute_ensemble(profile, jobs=jobs, checkpoint=tmp_path)
    count, sizes = (500, 12) if profile == "paper" else (6, 4)
    rows = arrays["measurements"]
    assert rows.shape == (2 * count * 2 * sizes, 23)
    assert meta["completed_cases"] == meta["required_cases"] == len(rows)
    assert meta["cases_with_all_valid_observables"] == len(rows)
    assert meta["cases_passing_numerical_diagnostics"] == len(rows)
    assert (tmp_path / "configurations.npz").exists()
    assert len({tuple(row[:4]) for row in rows}) == len(rows)
    if profile == "paper":
        assert "250/24000 cases" in capsys.readouterr().out


@pytest.mark.parametrize("kwargs", [{"profile": "unknown"}, {"jobs": 0}])
def test_ensemble_rejects_invalid_run_settings(kwargs):
    with pytest.raises(ValueError, match="profile must be"):
        nnn.compute_ensemble(**kwargs)


def test_ensemble_graph_preserves_resolved_and_unresolved_values():
    rows = []
    for kind in (0, 1):
        for boundary in nnn.BOUNDARIES:
            for n in (25, 50):
                for index in (0, 1):
                    row = _audit_row((kind, index, None, boundary, n, None))
                    row[10:13] = [1e-8] * 3 if n == 25 else [1e-3, 2e-3, 3e-3]
                    rows.append(row)
    meta = {"Ns": [25, 50, 100], "configurations_per_class": 2}
    fig = nnn.draw_ensemble({"measurements": np.array(rows)}, meta)
    assert len(fig.axes) == 4
    aggregates = meta["rendered_error_sensitivity"]["unresolved_aggregates"]
    assert len(aggregates) == 20  # 3+3+2+2 observables, each mean and maximum.
    assert {item["N"] for item in aggregates} == {25}
    assert all(item["raw_error"] == 1e-8 for item in aggregates)
    assert all(item["sensitivity"] == 1e-7 for item in aggregates)
    first = fig.axes[0].lines[0]
    np.testing.assert_array_equal(first.get_xdata(), [25, 50])
    assert np.isnan(first.get_ydata()[0])
    assert first.get_ydata()[1] == pytest.approx(1e-3)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"start_nfft": False}, "start_nfft"),
        ({"start_nfft": 31}, "start_nfft"),
        ({"max_nfft": False}, "max_nfft"),
        ({"start_nfft": 64, "max_nfft": 64}, "max_nfft"),
        ({"tolerance": 0}, "tolerance"),
        ({"tolerance": np.nan}, "tolerance"),
    ],
)
def test_convergence_rejects_impossible_refinement(kwargs, message):
    with pytest.raises(ValueError, match=message):
        nnn.converged_tail(nnn._configuration(0, 0), 8, 50, **kwargs)


def test_uncached_type_one_case_has_finite_valid_observations():
    st = nnn._configuration(0, 249)
    record = np.r_[st.gx, st.gv, st.rx, st.rv]
    row = nnn._ensemble_case((0, 249, record, "drop", 25, None))
    assert np.isfinite(row).all()
    np.testing.assert_array_equal(row[:4], [0, 0, 25, 249])
    np.testing.assert_array_equal(row[16:19], [1, 1, 1])
    assert row[22] == 0


def test_nonfinite_estimator_cannot_be_archived(monkeypatch):
    def bad_measure(*args):
        return np.array([np.inf, 1, 1]), np.ones(3), np.zeros(3, dtype=bool)

    monkeypatch.setattr(nnn, "_measure", bad_measure)
    st = nnn._configuration(1, 0)
    record = np.r_[st.gx, st.gv, st.rx, st.rv]
    with (
        np.errstate(invalid="ignore"),
        pytest.raises(RuntimeError, match="Nonfinite NNN measurement"),
    ):
        nnn._ensemble_case((1, 0, record, "drop", 25, None))


def test_uncheckpointed_ensemble_and_fully_resolved_graph(monkeypatch):
    monkeypatch.setattr(nnn, "_ensemble_case", _audit_row)
    arrays, meta = nnn.compute_ensemble()
    assert arrays["measurements"].shape == (96, 23)
    fig = nnn.draw_ensemble(arrays, meta)
    assert len(fig.axes) == 4
    assert meta["rendered_error_sensitivity"]["unresolved_aggregates"] == []


@pytest.mark.parametrize("status", [0, 1, 2, 4, 8, 15])
def test_ensemble_status_retains_each_independent_diagnostic(status, monkeypatch):
    """A failed numerical diagnostic must not become an asymptotic error."""
    n = 25
    t = np.arange(9.0)
    trace = nnn.TailTrace(
        t,
        -t,
        262144,
        32,
        4,
        convergence_error=3e-8 if status & 8 else 0,
    )
    contour = nnn.TailTrace(t, -t + (n * 2e-7 if status & 2 else 0), 262144, 40, 4)
    monkeypatch.setattr(nnn, "converged_tail", lambda *args, **kwargs: trace)
    monkeypatch.setattr(nnn, "tail_trace", lambda *args, **kwargs: contour)

    def measurement(_kind, _stencil, _n, times, _values):
        measured = np.ones(3)
        if status & 4 and len(times) == len(t):
            measured[0] += 1e-5
        valid = np.ones(3, dtype=bool)
        if status & 1:
            valid[1] = False
        return measured, np.ones(3), valid

    monkeypatch.setattr(nnn, "_measure", measurement)
    st = nnn._configuration(0, 0)
    row = nnn._ensemble_case((0, 0, np.r_[st.gx, st.gv, st.rx, st.rv], "drop", n, None))
    assert row[22] == status
    assert (row[17] == 0) == bool(status & 1)
    assert (row[20] > 1e-7) == bool(status & 2)
    assert (row[13] > 2e-6) == bool(status & 4)
    assert (row[19] > 2e-8) == bool(status & 8)
    assert row[11] == 0  # Missing second observable has no invented relative error.
