"""Observables and independent spectral checks for the Lyons experiments."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence

from pyflock import experiment_lyons as E
from pyflock.heterogeneous import HeterogeneousNN, system_matrix


def test_peak_estimator_recovers_signed_extrema_without_theory():
    t = np.linspace(0, 3 * np.pi, 302)
    result = E.measure_response(t, -np.sin(t))
    assert result["arrival_valid"] and result["ratio_valid"]
    assert result["T1"] == pytest.approx(np.pi / 2, abs=1e-5)
    assert result["A1"] == pytest.approx(-1, abs=1e-6)
    assert result["T2"] == pytest.approx(3 * np.pi / 2, abs=1e-5)
    assert result["A2"] == pytest.approx(1, abs=1e-6)
    assert result["ratio"] == pytest.approx(1, abs=1e-6)


def test_missing_peaks_are_masked_and_excluded_from_statistics():
    t = np.linspace(0, 3, 61)
    result = E.measure_response(t, -t)
    assert not result["arrival_valid"] and not result["ratio_valid"]
    assert result["T1"] == result["ratio"] == 0
    rows = np.zeros((3, 16))
    rows[:, 2] = 0.3
    rows[:2, 3] = [4, 6]
    rows[:2, 10] = 1
    assert E._sample_summary(rows, 0.3, 3, 10) == (5, 4, 6)
    assert E._sample_summary(rows, 0.3, 7, 11) is None


def test_response_sampling_refines_until_the_claim_is_resolved(monkeypatch):
    gx, gv = np.ones(8), np.ones(8)
    prediction = E._theory(HeterogeneousNN(gx, gv, 0))["T1"]
    grids = []

    def sinusoidal_tail(model, t):
        grids.append(len(t))
        return t, -np.sin(np.pi * t / (2 * prediction))

    monkeypatch.setattr(E, "simulate_tail", sinusoidal_tail)
    row, t, _tail, case = E._response_case((0, 0, gx, gv, 0, 17, False))
    assert len(grids) > 1
    assert grids[-1] == len(t) == case["sampling"]["final_samples"]
    assert row[10] == row[11] == row[16] == row[17] == 1
    assert row[12] <= max(1e-4 * abs(row[8]), 0.05 * abs(row[3] - row[8]))
    assert row[13] <= max(1e-4 * abs(row[9]), 0.05 * abs(row[7] - row[9]))
    assert row[3] == pytest.approx(prediction, rel=1e-4)


def test_sampled_spectra_match_dense_eigenvalues_and_have_small_residuals():
    rng = np.random.default_rng(71)
    model = HeterogeneousNN(rng.uniform(1, 4, 17), rng.uniform(2, 6, 17), 0.3)
    for geometry in ("circle", "path"):
        values, residual, meta = E._spectrum(model, geometry)
        dense = np.linalg.eigvals(
            system_matrix(
                model, geometry=geometry, followers_only=geometry == "path"
            ).toarray()
        )
        i, j = linear_sum_assignment(abs(values[:, None] - dense[None, :]))
        assert np.max(abs(values[i] - dense[j])) < 2e-7
        assert np.max(residual) < 1e-8
        assert len(values) < len(dense)
        assert meta["neutral_modes_removed"] == 0


def test_homogeneous_reduction_preserves_first_order_wave_speeds():
    gx = np.asarray([1, 2, 4, 3, 2.5, 5])
    gv = np.asarray([3, 4, 1, 6, 2, 3])
    first = HeterogeneousNN(gx, gv, 0.4)
    m, v = np.mean(1 / gx), np.mean(gv / gx)
    reference = HeterogeneousNN(np.full(len(gx), 1 / m), np.full(len(gx), v / m), 0.4)
    assert E._theory(first) == pytest.approx(E._theory(reference))


def test_partial_eigen_solve_keeps_physical_modes_when_neutral_pair_is_missing(
    monkeypatch,
):
    model = HeterogeneousNN(np.linspace(1, 3, 12), np.linspace(2, 5, 12), 0.2)
    matrix = system_matrix(model, geometry="circle").toarray()
    values, vectors = np.linalg.eig(matrix)
    physical = np.argsort(abs(values))[2:5]

    def partial(*args, **kwargs):
        raise ArpackNoConvergence("partial", values[physical], vectors[:, physical])

    monkeypatch.setattr(E, "eigs", partial)
    sampled, residual, record = E._spectrum(model, "circle")
    assert len(sampled) == 3
    assert np.sort_complex(sampled) == pytest.approx(np.sort_complex(values[physical]))
    assert record["status"] == "partial_ARPACK_convergence"
    assert record["neutral_modes_removed"] == 0
    assert max(residual) < 1e-12


def test_boundary_specifications_reuse_identical_saved_agent_arrays(monkeypatch):
    seen = []

    def capture(spec):
        family, gx, gv, beta, geometry, _nt = spec
        seen.append((family, gx.copy(), gv.copy(), beta, geometry))
        return {"t": np.asarray([0.0, 1.0]), "tail": np.zeros(2)}, {
            "family": family,
            "beta_v": beta,
            "geometry": geometry,
        }

    monkeypatch.setattr(E, "_boundary_case", capture)
    arrays, meta = E._compute_boundary("paper", 1)
    assert [f["N_total"] for f in meta["families"]] == [400, 1600]
    assert len(seen) == 8
    for family, gx, gv, beta, geometry in seen:
        assert np.array_equal(gx, arrays[f"family_{family}_gx"])
        assert np.array_equal(gv, arrays[f"family_{family}_gv"])
    json.dumps(meta, allow_nan=False)


def test_response_draw_honours_validity_masks_and_preserves_style():
    rows = []
    for family in (0, 1):
        for sample in (0, 1):
            for beta in (0.0, 0.3):
                row = np.zeros(16)
                row[:3] = [family, sample, beta]
                row[3], row[7], row[8], row[9] = 40 + sample, 0.6, 42, 0.65
                row[10:12] = 1
                rows.append(row)
    rows = np.asarray(rows)
    rows[0, 7] = 0  # Placeholder for an unavailable second peak.
    rows[0, 11] = 0
    meta = {
        "families": [{"gx_uniform": [1, 3], "gv_uniform": [4, 12]}] * 2,
        "N_total": 40,
        "beta_values": [0.0, 0.3],
        "ensemble_size": 2,
    }
    before = dict(matplotlib.rcParams)
    fig = E.draw_response({"measurements": rows}, meta)
    fig.canvas.draw()
    measured = [
        line
        for ax in fig.axes
        for line in ax.lines
        if line.get_label() == "Measured mean"
    ]
    assert all(np.all(line.get_ydata() > 0) for line in measured)
    plt.close(fig)
    assert dict(matplotlib.rcParams) == before
