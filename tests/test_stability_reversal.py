"""Failure handling and profile checks for the finite stability witness."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from pyflock import experiment_lyons_reversal as E


def test_paper_profile_retains_the_witness_and_refines_only_time_sampling():
    data, record = E.compute(profile="paper")
    assert record["profile"] == "paper"
    assert record["N_total"] == 64
    assert data["times"] == pytest.approx(np.linspace(0, 3000, 1201))
    assert np.array_equal(data["gx"], np.ones(64))
    assert np.array_equal(data["gv"], np.tile([2, 18], 32))
    assert data["circle_response"][0] == pytest.approx(1, abs=1e-12)
    assert data["path_response"][0] == pytest.approx(1, abs=1e-12)
    assert record["classification"]["circle_physical_abscissa"] > 0
    assert record["classification"]["path_follower_abscissa"] < 0
    assert record["validation"]["path_dense_exponential_max_state_error"] < 1e-9


def test_unknown_profile_is_rejected_before_searching(monkeypatch):
    def no_search():
        pytest.fail("profile validation must precede the witness search")

    monkeypatch.setattr(E, "_search_witness", no_search)
    with pytest.raises(ValueError, match="profile"):
        E.compute(profile="unrecorded")


def test_search_rejects_a_grid_without_negative_curvature(monkeypatch):
    monkeypatch.setattr(E, "_curvatures", lambda *args: np.ones(2))
    with pytest.raises(ArithmeticError, match="No reversal"):
        E._search_witness()


def test_negative_curvature_screen_alone_does_not_establish_a_reversal(monkeypatch):
    # A controlled screen nominates a homogeneous cell. Its actual finite
    # Bloch spectra remain stable at every tested size, so it is rejected.
    monkeypatch.setattr(E, "product", lambda *args, **kwargs: [(1.0, 1.0)])
    monkeypatch.setattr(E, "_curvatures", lambda *args: -np.ones(2))
    with pytest.raises(ArithmeticError, match="No reversal"):
        E._search_witness()


def test_bloch_factor_check_rejects_unresolved_coherent_cancellation():
    # The coherent factor is algebraically zero; extremely scaled arithmetic
    # can no longer cancel its linear coefficient to the recorded tolerance.
    with pytest.raises(ArithmeticError, match="coherent factor"):
        E._bloch_spectrum(
            np.array([1.0, 5.0]) * 1e50, np.array([2.0, 18.0]) * 1e50, 0.3, 2
        )


def test_independent_ode_failure_aborts_the_witness_record(monkeypatch):
    monkeypatch.setattr(
        E,
        "solve_ivp",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, message="independent integration did not converge"
        ),
    )
    with pytest.raises(ArithmeticError, match="independent integration"):
        E.compute()


def test_broken_axis_rejects_an_unseparated_real_cluster():
    data = {
        "bloch_eigenvalues": np.array([0j, 0.001j, -0.001j]),
        "path_eigenvalues": np.array([0.002j, -0.002j]),
    }
    with pytest.raises(ValueError, match="no separated real cluster"):
        E.draw(data, {})
