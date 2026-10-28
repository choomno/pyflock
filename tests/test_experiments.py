"""Scientific and reproducibility regressions for the reconstructed gallery."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.linalg import block_diag, null_space
from scipy.optimize import linear_sum_assignment

import pyflock as pf
from pyflock import experiments as E
from pyflock import plotting as P


def test_circle_mode_removal_preserves_unstable_eigenvalues():
    st = pf.nn_stencil(-1.0, -1.0, -0.2, -0.5)
    N = 12
    values = P.circle_eigenvalues(st, N)
    matrix = pf.system_matrix_circle(st, N)
    # Restrict to mean-zero position and velocity vectors. The exact coherent
    # Jordan block otherwise acquires spurious O(sqrt(eps)) eigenvalues.
    basis = block_diag(*[null_space(np.ones((1, N)))] * 2)
    restricted = basis.T @ matrix @ basis
    np.testing.assert_allclose(matrix @ basis, basis @ restricted, atol=2e-14, rtol=0)
    expected = np.linalg.eigvals(restricted)
    assert values.size == expected.size == 2 * (N - 1)
    assert values.real.max() > 0.01
    rows, cols = linear_sum_assignment(abs(values[:, None] - expected[None, :]))
    np.testing.assert_allclose(values[rows], expected[cols], atol=2e-12, rtol=0)
    position = np.r_[np.ones(N), np.zeros(N)]
    velocity = np.r_[np.zeros(N), np.ones(N)]
    np.testing.assert_allclose(matrix @ position, 0, atol=2e-14, rtol=0)
    np.testing.assert_allclose(matrix @ velocity, position, atol=2e-14, rtol=0)


def test_plotting_preserves_global_style_and_invalid_theory_is_absent():
    before = dict(matplotlib.rcParams)
    st = pf.case("two_positive_velocities")
    res = pf.simulate_line(st, 12, 12, nt=61)
    fig = P.transient_figure(res)
    assert all("asymptotic" not in str(a.get_label()) for a in fig.axes[0].collections)
    fig.canvas.draw()
    plt.close(fig)
    assert dict(matplotlib.rcParams) == before


def test_map_without_reliability_and_generic_parameter_label():
    sm = pf.stability_map(
        pf.velocity_asymmetry_family(-1, -1),
        [-0.1, 0, 0.1],
        [8, 12],
        reliability=False,
        param_name="Custom coupling",
    )
    fig = P.stability_map_figure(sm)
    fig.canvas.draw()
    assert fig.axes[0].get_xlabel() == "Custom coupling"
    plt.close(fig)


def test_redraw_uses_saved_data_and_checks_integrity(tmp_path, monkeypatch):
    E.run("dispersion", tmp_path)
    meta = json.loads((tmp_path / "dispersion/experiment.json").read_text())
    assert (
        meta["data_sha256"]
        and meta["source_sha256"]
        and meta["artifacts"]["figure.pdf"]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("redraw tried to compute")

    monkeypatch.setattr(E, "compute", forbidden)
    E.run("dispersion", tmp_path, redraw=True)
    redrawn = json.loads((tmp_path / "dispersion/experiment.json").read_text())
    assert redrawn["artifacts"] == meta["artifacts"]
    data = tmp_path / "dispersion/data.npz"
    data.write_bytes(data.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        E.run("dispersion", tmp_path, redraw=True)


def test_local_expansion_error_improves_at_expected_orders():
    data, _ = E.compute("dispersion")
    x = data["local_theta"]
    co = data["coefficients"]
    exact = data["local_branches"]
    for order in [1, 2, 3]:
        approximation = sum(co[:, j, None] * x**j for j in range(order + 1))
        error = abs(exact - approximation).max(axis=0)
        selected = (x > 0.015) & (x < 0.08)
        slope = np.polyfit(np.log(x[selected]), np.log(error[selected]), 1)[0]
        assert slope == pytest.approx(order + 1, abs=0.05)


def test_smooth_pulse_stays_inside_displayed_window():
    data, meta = E.compute("pulse")
    centre = meta["centre"]
    N = meta["N"]
    x = (np.arange(N) - centre) / N
    visible = (x >= -0.32) & (x <= 0.6)
    # Every omitted part is below 1/1000 of the initial peak.
    assert abs(data["velocity"][:, ~visible]).max() < 1e-3


def test_convergence_trace_can_be_reanalysed_and_checkpoint_checked(
    tmp_path, monkeypatch
):
    spec = (1, "regular", 40, -2.0, 0.0, str(tmp_path), "test-source")
    row, t, w = E._convergence_case(spec)
    measured = pf.transient_summary(t, w, nmax=3)
    assert row[4:7] == pytest.approx(
        [abs(measured["A"][0]), measured["T"], measured["gamma"]]
    )
    assert np.all(
        np.asarray(row[13:16]) <= np.maximum(1e-4, 0.05 * np.asarray(row[10:13]))
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("cache missed")

    monkeypatch.setattr(E, "simulate_tail", forbidden)
    cached = E._convergence_case(spec)
    assert np.array_equal(cached[1], t) and np.array_equal(cached[2], w)
    saved = next(tmp_path.glob("*.npz"))
    saved.write_bytes(saved.read_bytes() + b"bad")
    with pytest.raises(AssertionError, match="cache missed"):
        E._convergence_case(spec)


def test_failed_recomputation_preserves_previous_source_record(tmp_path, monkeypatch):
    E.run("dispersion", tmp_path)
    dest = tmp_path / "dispersion"
    source = dest / "source/dispersion.py"
    source.write_bytes(source.read_bytes() + b"\n# Earlier archived source.\n")
    meta = json.loads((dest / "experiment.json").read_text())
    meta["source_sha256"] = E._source_digest(dest / "source")
    for entry in meta["files"]:
        if entry["path"] == "source/dispersion.py":
            entry.update(E._file_record(source, entry["path"]))
    (dest / "experiment.json").write_text(json.dumps(meta))

    def failed(*args, **kwargs):
        raise RuntimeError("Failed experiment")

    monkeypatch.setattr(E, "compute", failed)
    with pytest.raises(RuntimeError, match="Failed experiment"):
        E.run("dispersion", tmp_path)
    # The previous result can still be drawn, with its own matching source record.
    E.run("dispersion", tmp_path, redraw=True)
