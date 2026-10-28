"""Energy-operator identities, independent Gramian integration, and diagnostics."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from scipy.linalg import expm

from pyflock import experiment_energy as energy


@pytest.mark.parametrize(
    "n,hp,r",
    [
        (0, 0.5, 1),
        (True, 0.5, 1),
        (1.5, 0.5, 1),
        (2, -1, 1),
        (2, np.nan, 1),
        (2, np.inf, 1),
        (2, 0, 0),
        (2, 0, np.inf),
        (2, 0, np.nan),
    ],
)
def test_damping_requires_positive_size_and_physical_gains(n, hp, r):
    with pytest.raises(ValueError, match="Require"):
        energy.damping(n, hp, r)


@pytest.mark.parametrize("kwargs", [{"duration": 0}, {"nt": 1}, {"end": 2}])
def test_pulse_response_requires_a_resolved_nonempty_pulse(kwargs):
    with pytest.raises(ValueError):
        energy.force_response(3, 0.5, **kwargs)


def test_energy_counterexample_sign_is_canonical(monkeypatch):
    eigh = energy.eigh

    def negative_first_component(*args, **kwargs):
        values, vectors = eigh(*args, **kwargs)
        vectors[:, 0] *= -1 if vectors[0, 0] > 0 else 1
        return values, vectors

    monkeypatch.setattr(energy, "eigh", negative_first_component)
    counter = energy.energy_counterexample(5, 0.5)
    y = np.asarray(counter["vector"])
    assert y[0] > 0
    assert_allclose(y @ y, 1, atol=2e-15)
    d = energy.damping(5, 0.5)
    assert_allclose(y @ d @ y, np.linalg.eigvalsh((d + d.T) / 2)[0], atol=1e-15)


@pytest.mark.parametrize("sign", [-1, 1])
def test_finite_witness_matches_independently_integrated_gramian(monkeypatch, sign):
    eigh = energy.eigh

    def prescribed_eigenvector_sign(*args, **kwargs):
        values, vectors = eigh(*args, **kwargs)
        vectors[:, 0] *= sign * (1 if vectors[0, 0] >= 0 else -1)
        return values, vectors

    monkeypatch.setattr(energy, "eigh", prescribed_eigenvector_sign)
    n, horizon = 2, 1.5
    data, meta = energy.finite_energy_witness(n=n, horizon=horizon, nt=33)
    # Assemble the n=2 component equations explicitly, independently of the
    # production block assembly and stationary Lyapunov subtraction.
    a = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-2, 1, -2, 0.5], [1, -1, 1.5, -1.5]])
    f = np.r_[np.zeros((n, n)), np.eye(n)]
    expected, quadrature_error = quad_vec(
        lambda t: expm(a * t) @ f @ f.T @ expm(a.T * t), 0, horizon, epsabs=1e-12
    )
    assert quadrature_error < 1e-11
    assert_allclose(data["gramian_WT"], expected, atol=2e-15)
    assert_allclose(data["q"] @ data["q"], 1, atol=1e-15)
    assert data["q"][0] >= 0
    assert_allclose(meta["independently_integrated_force_L2_squared"], 1, atol=5e-10)
    assert_allclose(
        meta["independently_integrated_final_energy"], meta["mu_T"] / 2, atol=2e-10
    )
    spacing = np.diff(np.c_[np.zeros(len(data["t"])), data["x"]], axis=1)
    assert_allclose(data["energy"], np.sum(spacing**2 + data["v"] ** 2, axis=1) / 2)


@pytest.mark.parametrize("failed_call", [1, 2])
def test_failed_adjoint_or_forward_integration_does_not_produce_a_witness(
    monkeypatch, failed_call
):
    solve_ivp = energy.solve_ivp
    calls = 0

    def failed_integrator(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            return SimpleNamespace(
                success=False, message="controlled integration failure"
            )
        return solve_ivp(*args, **kwargs)

    monkeypatch.setattr(energy, "solve_ivp", failed_integrator)
    with pytest.raises(RuntimeError, match="controlled integration failure"):
        energy.finite_energy_witness(n=2, horizon=1, nt=17)


@pytest.mark.parametrize("compute", [energy.compute_energy, energy.compute_continuum])
def test_experiments_reject_unknown_profiles(compute):
    with pytest.raises(ValueError, match="profile"):
        compute(profile="unknown")


@pytest.mark.parametrize(
    "profile, count, response_n", [("quick", 5, 64), ("paper", 6, 128)]
)
def test_energy_experiment_records_distinct_forcings_and_draws_saved_energy(
    monkeypatch, profile, count, response_n
):
    witness = energy.finite_energy_witness

    def compact_witness():
        return witness(n=4, horizon=12, nt=65)

    monkeypatch.setattr(energy, "finite_energy_witness", compact_witness)
    data, meta = energy.compute_energy(profile=profile)
    assert len(data["N"]) == count
    assert meta["parameters"]["response_N"] == response_n
    assert meta["profile"] == profile
    assert data["sigma_min"].shape == data["coercivity"].shape == (2, count)
    assert_allclose(data["sigma_min"][0], data["coercivity"][0], atol=2e-15)
    assert np.all(data["sigma_min"][1] > data["coercivity"][1])
    for i in range(2):
        assert_allclose(data[f"{i}_input_energy"][-1], 1)
        assert_allclose(data["peaks"][i], max(data[f"{i}_energy"]))
    fig = energy.draw_energy(data, meta)
    try:
        assert len(fig.axes) == 4
        assert_allclose(fig.axes[0].lines[0].get_ydata(), data["sigma_min"][0])
        assert_allclose(fig.axes[1].lines[0].get_ydata(), data["0_energy"])
        assert_allclose(
            fig.axes[3].lines[0].get_ydata(),
            data["witness_energy"] / meta["witness"]["source_bound_coefficient"],
        )
        assert_allclose(fig.axes[3].lines[1].get_ydata(), data["witness_input_energy"])
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("profile, count", [("quick", 5), ("paper", 6)])
def test_continuum_profiles_preserve_physical_mode_and_second_order_limit(
    profile, count
):
    data, meta = energy.compute_continuum(profile=profile)
    assert len(data["N"]) == count
    assert meta["profile"] == profile
    assert meta["parameters"]["mode"] == meta["parameters"]["k"] == 1
    assert_allclose(data["discrete"][:, 0], 1)
    assert 1.99 < meta["fitted_order"] < 2.01
    fig = energy.draw_continuum(data, meta)
    try:
        assert len(fig.axes) == 3
        assert_allclose(fig.axes[0].lines[0].get_ydata(), data["continuum"])
        assert_allclose(fig.axes[0].lines[1].get_ydata(), data["discrete"][0])
        assert_allclose(fig.axes[2].lines[0].get_ydata(), data["error"])
        fig.canvas.draw()
    finally:
        plt.close(fig)
