"""Independent observables and controlled orchestration for Lyons experiments."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence

from pyflock import experiment_lyons as E
from pyflock.heterogeneous import HeterogeneousNN


def test_refinement_is_local_and_constant_or_unbracketed_data_is_retained():
    t = np.array([0.0, 1.0, 2.0])
    assert E._refine(t, np.ones(3), 1) == (1.0, 1.0)
    assert E._refine(t, np.array([0.0, 2.0, 5.0]), 1) == (1.0, 2.0)
    assert E._refine(t, -((t - 0.8) ** 2), 1) == pytest.approx((0.8, 0.0))


@pytest.mark.parametrize("profile", [None, "unknown", "Paper"])
def test_profile_must_select_a_declared_experiment(profile):
    with pytest.raises(ValueError, match="profile"):
        E._base("lyons_response", profile)


@pytest.mark.parametrize(
    "times, values",
    [
        ([0, 1, 2, 3], [0, -1, 0, 1]),
        ([0, 1, 2, 3, 4], [0, -1]),
        ([0, 1, 2, 3, 4], [0, -1, np.nan, 1, 0]),
        ([0, 1, 2, np.inf, 4], [0, -1, 0, 1, 0]),
        ([0, 1, 2, 2, 4], [0, -1, 0, 1, 0]),
        ([0, 1, 2, 4, 6], [0, -1, 0, 1, 0]),
        ([0, 1e-16, 3e-16, 4e-16, 5e-16], [0, -1, 0, 1, 0]),
        ([[0, 1, 2, 3, 4]], [[0, -1, 0, 1, 0]]),
    ],
)
def test_peak_measurement_rejects_incompatible_sampling(times, values):
    with pytest.raises(ValueError, match="equally spaced"):
        E.measure_response(times, values)


def test_first_negative_minimum_without_following_positive_maximum_is_masked():
    times = np.linspace(0, np.pi, 101)
    result = E.measure_response(times, -np.sin(times))
    assert result["arrival_valid"] and not result["ratio_valid"]
    assert result["status"] == "no_following_positive_maximum"
    assert result["T1"] == pytest.approx(np.pi / 2)
    assert result["A1"] == pytest.approx(-1)
    assert result["A2"] == result["ratio"] == 0


def test_empty_partial_eigensolve_returns_explicit_empty_result(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ArpackNoConvergence("none", None, None)

    monkeypatch.setattr(E, "eigs", fail)
    model = HeterogeneousNN(np.ones(6), np.full(6, 3.0), 0.3)
    values, residual, record = E._spectrum(model, "path")
    assert values.size == residual.size == 0
    assert np.iscomplexobj(values)
    assert record["status"] == "no_eigenpairs"
    assert record["neutral_modes_removed"] == 0


def test_process_mapping_preserves_order_and_results():
    assert E._map_cases(abs, [-3, 0, 5], jobs=1) == [3, 0, 5]
    assert E._map_cases(abs, [-3, 0, 5], jobs=3) == [3, 0, 5]


@pytest.mark.parametrize(
    "geometry,beta", [("circle", 0.3), ("path", 0.3), ("path", -0.3)]
)
def test_small_boundary_case_has_finite_normalized_residuals_and_direct_initial_data(
    geometry, beta
):
    gx = np.linspace(1.0, 2.0, 12)
    gv = np.linspace(3.0, 5.0, 12)
    arrays, meta = E._boundary_case((2, gx, gv, beta, geometry, 31))
    assert arrays["selected_displacements"].shape == (31, 12)
    assert np.all(arrays["selected_displacements"][0] == 0)
    assert np.all(arrays["max_displacement"] >= abs(arrays["tail"]))
    assert np.max(arrays["eigen_residuals"]) < 1e-9
    assert meta["family"] == 2 and meta["geometry"] == geometry
    if geometry == "path" and beta > 0:
        assert meta["solver_check"]["max_displacement_difference"] < 1e-10
        assert meta["solver_check"]["max_velocity_difference"] < 1e-10
    else:
        assert meta["solver_check"] is None


@pytest.fixture
def analytic_tail(monkeypatch):
    """An exact smooth trace isolates sampling from the array ODE solver."""

    def tail(model, times):
        arrival = E._theory(model)["T1"]
        return times, -np.sin(np.pi * times / (2 * arrival))

    monkeypatch.setattr(E, "simulate_tail", tail)
    return tail


@pytest.mark.parametrize("profile", ["quick", "paper"])
def test_boundary_profile_records_seeded_paired_gain_samples(profile, monkeypatch):
    captured = []

    def sample(spec):
        family, _gx, _gv, beta, geometry, _nt = spec
        captured.append(spec)
        return {"t": np.array([0.0, 1.0]), "tail": np.zeros(2)}, {
            "family": family,
            "beta_v": beta,
            "geometry": geometry,
        }

    monkeypatch.setattr(E, "_boundary_case", sample)
    arrays, meta = E._compute_boundary(profile, 1)
    sizes = [400, 1600] if profile == "paper" else [80, 160]
    rng = np.random.default_rng(E.SEED)
    for family, (n, upper_v) in enumerate(zip(sizes, [13, 18])):
        expected_gx = rng.uniform(1, 5, n)
        expected_gv = rng.uniform(3 if family == 0 else 2, upper_v, n)
        np.testing.assert_array_equal(arrays[f"family_{family}_gx"], expected_gx)
        np.testing.assert_array_equal(arrays[f"family_{family}_gv"], expected_gv)
        group = [spec for spec in captured if spec[0] == family]
        assert len(group) == 4
        for _, gx, gv, _beta, _geometry, nt in group:
            np.testing.assert_array_equal(gx, expected_gx)
            np.testing.assert_array_equal(gv, expected_gv)
            assert nt == (1001 if profile == "paper" else 401)
    assert bool(meta["quick_profile"]) == (profile == "quick")


@pytest.mark.parametrize("profile", ["quick", "paper"])
def test_response_orchestration_retains_every_sample_and_measured_trace(
    profile, analytic_tail
):
    arrays, meta = E._compute_response(profile, 1)
    samples = 3 if profile == "paper" else 2
    n = 400 if profile == "paper" else 80
    betas = np.linspace(-0.6, 0.6, 13) if profile == "paper" else [-0.3, 0, 0.3, 0.6]
    assert arrays["gx"].shape == arrays["gv"].shape == (2, samples, n)
    rows = arrays["measurements"]
    assert rows.shape == (2 * samples * len(betas), 19)
    assert np.all(np.isfinite(rows))
    assert arrays["offsets"][0] == 0
    assert (
        arrays["offsets"][-1]
        == len(arrays["tail_times"])
        == len(arrays["tail_responses"])
    )
    for index, row in enumerate(rows):
        family, sample = row[:2].astype(int)
        model = HeterogeneousNN(
            arrays["gx"][family, sample], arrays["gv"][family, sample], row[2]
        )
        prediction = E._theory(model)
        assert row[8] == pytest.approx(prediction["T1"])
        assert row[9] == pytest.approx(prediction["ratio"])
        start, stop = arrays["offsets"][index : index + 2]
        times = arrays["tail_times"][start:stop]
        _, expected = analytic_tail(model, times)
        np.testing.assert_array_equal(arrays["tail_responses"][start:stop], expected)
        assert row[10] == 1
    assert meta["ensemble_size"] == samples
    checks = [case["solver_check"] for case in meta["cases"] if case["solver_check"]]
    assert len(checks) == 2
    assert all(check["max_displacement_difference"] < 1e-9 for check in checks)


@pytest.mark.parametrize("profile", ["quick", "paper"])
def test_ordering_construction_preserves_gain_pairs_and_effective_moments(
    profile, analytic_tail
):
    arrays, meta = E._compute_ordering(profile, 1)
    n = 240 if profile == "paper" else 80
    assert arrays["gx"].shape == arrays["gv"].shape == (12, n)
    for amplitude in range(3):
        original, shuffled = amplitude, 3 + amplitude
        np.testing.assert_array_equal(
            arrays["gx"][original][arrays["shuffle_permutation"]],
            arrays["gx"][shuffled],
        )
        np.testing.assert_array_equal(
            arrays["gv"][original][arrays["shuffle_permutation"]],
            arrays["gv"][shuffled],
        )
        distinct, ordered = 6 + amplitude, 9 + amplitude
        np.testing.assert_array_equal(
            arrays["gx"][distinct][arrays["sorted_indices"]], arrays["gx"][ordered]
        )
        np.testing.assert_array_equal(
            arrays["gv"][distinct][arrays["sorted_indices"]], arrays["gv"][ordered]
        )
    np.testing.assert_allclose(
        arrays["measurements"][:, 8:10],
        arrays["homogeneous_measurements"][:, 8:10],
        rtol=1e-14,
    )
    assert meta["ensemble_size"] == 1


@pytest.mark.parametrize("jobs", [False, 0, -1, 1.5])
def test_cache_rejects_invalid_worker_counts(jobs):
    with pytest.raises(ValueError, match="positive integer"):
        E._cached("lyons_response", "quick", jobs, None, lambda *_: None)


def test_cache_reuses_valid_content_and_recomputes_corrupt_data(tmp_path):
    calls = []

    def compute(profile, jobs):
        calls.append((profile, jobs))
        return {"trace": np.array([1.0, 2.0])}, {"profile": profile}

    first, record = E._cached("lyons_response", "quick", 1, tmp_path, compute)
    second, cached = E._cached("lyons_response", "quick", 1, tmp_path, compute)
    np.testing.assert_array_equal(first["trace"], second["trace"])
    assert cached == record and len(calls) == 1
    path = next(tmp_path.glob("*.npz"))
    path.write_bytes(b"corrupt")
    third, _ = E._cached("lyons_response", "quick", 1, tmp_path, compute)
    np.testing.assert_array_equal(third["trace"], first["trace"])
    assert len(calls) == 2


@pytest.mark.parametrize("value", [np.array([np.inf]), np.array(["nonnumeric"])])
def test_cache_rejects_nonfinite_or_nonnumeric_arrays(value):
    with pytest.raises(ValueError, match="finite and numeric"):
        E._cached("lyons_response", "quick", 1, None, lambda *_: ({"bad": value}, {}))


@pytest.mark.parametrize("name", ["boundary", "response", "ordering"])
def test_public_compute_wrappers_select_their_own_orchestrators(name, monkeypatch):
    calls = []

    def compute(profile, jobs):
        calls.append((profile, jobs))
        return {"one": np.array([1.0])}, {"experiment": "lyons_" + name}

    monkeypatch.setattr(E, "_compute_" + name, compute)
    data, record = getattr(E, "compute_" + name)(profile="paper", jobs=1)
    assert data["one"].tolist() == [1.0]
    assert record["experiment"] == "lyons_" + name and calls == [("paper", 1)]


def test_boundary_plot_represents_saved_eigenpairs_and_scaled_tail():
    cases, data = [], {}
    for family in (0, 1):
        for beta in (0.3, -0.3):
            for geometry in ("circle", "path"):
                index = len(cases)
                cases.append({"family": family, "beta_v": beta, "geometry": geometry})
                data[f"case_{index}_eigenvalues"] = np.array([-0.2 + 0.4j, -0.2 - 0.4j])
                data[f"case_{index}_t"] = np.array([0.0, 1.0, 2.0])
                data[f"case_{index}_tail"] = np.array([0.0, -1.0, 0.0])
    figure = E.draw_boundary(
        data, {"families": [{"N_total": 12}, {"N_total": 24}], "cases": cases}
    )
    figure.canvas.draw()
    np.testing.assert_allclose(
        figure.axes[0].collections[0].get_offsets(), [[-2.4, 4.8], [-2.4, -4.8]]
    )
    np.testing.assert_allclose(figure.axes[2].lines[0].get_ydata(), [0, -1 / 11, 0])
    plt.close(figure)


def test_response_plot_masks_all_unavailable_observables():
    rows = np.zeros((4, 19))
    rows[:, 0] = [0, 0, 1, 1]
    rows[:, 2] = [0, 0.3, 0, 0.3]
    meta = {
        "families": [{"gx_uniform": [1, 3], "gv_uniform": [4, 12]}] * 2,
        "N_total": 80,
        "beta_values": [0, 0.3],
        "ensemble_size": 2,
    }
    figure = E.draw_response({"measurements": rows}, meta)
    figure.canvas.draw()
    assert all(not axis.lines for axis in figure.axes)
    assert all(
        any("Valid samples" in text.get_text() for text in axis.texts)
        for axis in figure.axes
    )
    plt.close(figure)


def test_ordering_plot_uses_saved_tail_and_masks_unresolved_arrivals(analytic_tail):
    data, meta = E._compute_ordering("quick", 1)
    figure = E.draw_ordering(data, meta)
    figure.canvas.draw()
    assert len(figure.axes[2].lines) == 5
    plt.close(figure)
    rows = data["measurements"]
    rows[2, 10] = 0
    rows[5, 4] = 0
    rows[8, 10] = 0
    figure = E.draw_ordering(data, meta)
    figure.canvas.draw()
    assert len(figure.axes[2].lines) == 1
    plt.close(figure)


def test_small_homogeneous_response_matches_literal_follower_equations():
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize_scalar

    n, gx, gv, beta = 16, 2.0, 3.0, 0.3
    row, times, tail, record = E._response_case(
        (0, 0, np.full(n, gx), np.full(n, gv), beta, 301, True)
    )

    def derivative(_time, state):
        position = np.r_[0.0, state[: n - 1]]
        velocity = np.r_[0.0, state[n - 1 :]]
        acceleration = np.empty(n - 1)
        for k in range(1, n - 1):
            acceleration[k - 1] = -gx * (
                position[k] - (position[k - 1] + position[k + 1]) / 2
            ) - gv * (
                velocity[k]
                - (1 + beta) * velocity[k - 1] / 2
                - (1 - beta) * velocity[k + 1] / 2
            )
        acceleration[-1] = -gx * (position[-1] - position[-2]) - gv * (
            velocity[-1] - velocity[-2]
        )
        return np.r_[velocity[1:], acceleration]

    initial = np.r_[np.zeros(n - 1), -np.ones(n - 1)]
    independent = solve_ivp(
        derivative,
        (0, times[-1]),
        initial,
        method="DOP853",
        t_eval=times,
        dense_output=True,
        rtol=1e-12,
        atol=1e-14,
    )
    assert independent.success
    np.testing.assert_allclose(tail, independent.y[n - 2], atol=2e-10)
    first = int(np.argmin(tail))
    optimum = minimize_scalar(
        lambda time: independent.sol(time)[n - 2],
        bounds=(times[first - 1], times[first + 1]),
        method="bounded",
    )
    assert row[3] == pytest.approx(optimum.x, abs=0.002)
    assert row[4] == pytest.approx(optimum.fun, abs=2e-5)
    assert row[10] == row[11] == 1
    assert record["solver_check"]["max_velocity_difference"] < 1e-10
