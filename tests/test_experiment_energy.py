"""Independent algebra, integration and modal checks for energy experiments."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pyflock.experiment_energy import (
    compute_continuum,
    damping,
    energy_counterexample,
    force_response,
)


def test_paper_endpoint_and_asymmetry_entries():
    assert_allclose(
        damping(4, 0.5),
        [[2, -0.5, 0, 0], [-1.5, 2, -0.5, 0], [0, -1.5, 2, -0.5], [0, 0, -1.5, 1.5]],
    )
    assert_allclose(damping(1, 0.5), [[1.5]])


def test_symmetric_singular_value_exact_formula():
    # Eigenvalues of the Dirichlet/free tridiagonal path Laplacian.
    for n in (2, 7, 20):
        expected = 2 - 2 * np.cos(np.pi / (2 * n + 1))
        assert_allclose(
            np.linalg.svd(damping(n, 0), compute_uv=False)[-1], expected, rtol=1e-12
        )


def test_source_quadratic_form_counterexample_is_reproducible():
    c = energy_counterexample()
    y = np.array(c["vector"])
    assert_allclose(y @ y, 1, atol=1e-14)
    assert_allclose(y @ damping(32, 0.5) @ y, c["y_D_y"], rtol=1e-13)
    assert c["lower_bound_residual"] < -0.04
    assert c["claimed_Hdot_violation"] > 0.04


def test_force_response_against_componentwise_ode():
    n = 5
    hp = 0.5
    duration = 2
    result, j = force_response(n, hp, end=9, nt=151)

    def rhs(t, state, amplitude):
        x, v = state[:n], state[n:]
        acc = np.empty(n)
        for i in range(n):
            left_x = 0 if i == 0 else x[i - 1]
            left_v = 0 if i == 0 else v[i - 1]
            acc[i] = (left_x - x[i]) + (1 + hp) * (left_v - v[i])
            if i < n - 1:
                acc[i] += (x[i + 1] - x[i]) + (1 - hp) * (v[i + 1] - v[i])
        acc[j - 1] += amplitude
        return np.r_[v, acc]

    first = solve_ivp(
        lambda t, y: rhs(t, y, 1 / np.sqrt(duration)),
        [0, duration],
        np.zeros(2 * n),
        method="DOP853",
        rtol=2e-12,
        atol=1e-13,
        dense_output=True,
    )
    second = solve_ivp(
        lambda t, y: rhs(t, y, 0),
        [duration, 9],
        first.y[:, -1],
        method="DOP853",
        rtol=2e-12,
        atol=1e-13,
        dense_output=True,
    )
    ts = result["t"]
    mask = ts <= duration
    expected = np.empty((len(ts), 2 * n))
    expected[mask] = first.sol(ts[mask]).T
    expected[~mask] = second.sol(ts[~mask]).T
    assert_allclose(np.c_[result["x"], result["v"]], expected, atol=2e-11)
    spacing = np.diff(np.c_[np.zeros(len(ts)), result["x"]], axis=1)
    assert_allclose(
        result["energy"],
        0.5 * np.sum(result["v"] ** 2 + spacing**2, axis=1),
        atol=1e-14,
    )
    assert_allclose(result["input_energy"][-1], 1)


def test_energy_balance_and_valid_bound():
    n = 8
    hp = 0.5
    result, j = force_response(n, hp, nt=301)
    d = damping(n, hp)
    k = damping(n, 0)
    x, v = result["x"], result["v"]
    f = np.zeros_like(v)
    f[result["t"] < 2, j - 1] = 1 / np.sqrt(2)
    acceleration = -x @ k.T - v @ d.T + f
    hdot = np.sum(v * acceleration + (x @ k.T) * v, axis=1)
    supply = np.sum(v * f, axis=1)
    dissipation = np.einsum("ij,jk,ik->i", v, d, v)
    assert_allclose(hdot, supply - dissipation, atol=1e-14)
    alpha = np.linalg.eigvalsh((d + d.T) / 2)[0]
    # Pointwise completion of squares proves the cumulative bound, independently
    # of the source's questionable use of the nonsymmetric singular value.
    assert np.all(hdot <= np.sum(f * f, axis=1) / (2 * alpha) + 1e-14)
    assert np.all(result["energy"] <= result["input_energy"] / (2 * alpha) + 1e-14)


def test_periodic_mode_against_full_matrix_exponential():
    data, _meta = compute_continuum()
    n = int(data["N"][0])
    h = data["h"][0]
    lap = (
        np.roll(np.eye(n), 1, axis=0) + np.roll(np.eye(n), -1, axis=0) - 2 * np.eye(n)
    ) / h**2
    matrix = np.block([[np.zeros((n, n)), np.eye(n)], [lap, 0.1 * lap]])
    sine = np.sin(2 * np.pi * np.arange(n) / n)
    initial = np.r_[sine, np.zeros(n)]
    for idx in (200, 700, 1600):
        direct = expm(data["t"][idx] * matrix) @ initial
        assert_allclose(direct[:n], data["discrete"][0, idx] * sine, atol=2e-13)


def test_continuum_convergence_second_order():
    data, meta = compute_continuum()
    assert np.all((data["rates"] > 1.98) & (data["rates"] < 2.03))
    assert 1.99 < meta["fitted_order"] < 2.01
    assert np.all(np.diff(data["error"]) < 0)
    # The gain product h^2 remains fixed; mode index and physical length do too.
    assert meta["parameters"]["k"] == 1
    assert_allclose(data["k2_discrete"], (2 * np.sin(data["h"] / 2) / data["h"]) ** 2)


def test_fixed_velocity_asymmetry_singular_scaling():
    ns = np.array([32, 64, 128])
    sigma = np.array(
        [np.linalg.svd(damping(int(n), 0.5), compute_uv=False)[-1] for n in ns]
    )
    orders = -np.log(sigma[1:] / sigma[:-1]) / np.log(2)
    assert np.all((orders > 0.98) & (orders < 1.02))


def test_finite_time_distributed_force_violates_source_bound():
    # finite_energy_witness builds its input from a Gramian, then propagates
    # two independent ODEs (adjoint and forward), including the input integral.
    # Compare the actual forward result with both the claimed bound and the
    # Gramian optimizer; a failed Lyapunov solve cannot pass on its own.
    from scipy.integrate import simpson

    from pyflock.experiment_energy import finite_energy_witness

    arrays, meta = finite_energy_witness()
    n = meta["N"]
    assert n == 128 and meta["T"] == 512
    assert arrays["force"].shape == (2049, n)
    assert_allclose(arrays["q"] @ arrays["q"], 1, atol=1e-13)
    assert abs(meta["independently_integrated_force_L2_squared"] - 1) < 2e-8
    assert meta["relative_final_state_error"] < 2e-8
    assert meta["adjoint_endpoint_relative_error"] < 2e-7
    assert meta["lyapunov_relative_residual"] < 2e-7
    assert meta["finite_gramian_relative_residual"] < 2e-7
    # Independent sampled quadrature verifies the recorded distributed force.
    sampled_norm = simpson(np.sum(arrays["force"] ** 2, axis=1), x=arrays["t"])
    assert abs(sampled_norm - 1) < 2e-5
    spacing = np.diff(np.c_[np.zeros(len(arrays["t"])), arrays["x"]], axis=1)
    energy = np.sum(spacing**2 + arrays["v"] ** 2, axis=1) / 2
    assert_allclose(energy, arrays["energy"], rtol=2e-13, atol=1e-13)
    assert_allclose(energy[-1], meta["predicted_final_energy"], rtol=2e-8)
    assert 1.07 < meta["violation_ratio"] < 1.08
    assert meta["violation"] > 2.9
    # A second verification uses the paper's (momentum, spacing-error)
    # coordinates and cubic interpolation of only the saved force samples.
    # Assemble component equations directly, without the state matrix, C, W,
    # or the continuous adjoint used by finite_energy_witness.
    from scipy.interpolate import CubicSpline

    force_interpolator = CubicSpline(arrays["t"], arrays["force"], axis=0)

    def momentum_spacing_rhs(t, state):
        momentum, delta = state[:n], state[n : 2 * n]
        momentum_dot = -2 * momentum.copy()
        momentum_dot[1:] += 1.5 * momentum[:-1]
        momentum_dot[:-1] += 0.5 * momentum[1:]
        momentum_dot[-1] += 0.5 * momentum[-1]
        momentum_dot += delta
        momentum_dot[:-1] -= delta[1:]
        force = force_interpolator(t)
        momentum_dot += force
        delta_dot = np.r_[-momentum[0], momentum[:-1] - momentum[1:]]
        return np.r_[momentum_dot, delta_dot, force @ force]

    independent = solve_ivp(
        momentum_spacing_rhs,
        (0, meta["T"]),
        np.zeros(2 * n + 1),
        method="DOP853",
        rtol=2e-10,
        atol=2e-12,
    )
    assert independent.success
    momentum, delta = independent.y[:n, -1], independent.y[n : 2 * n, -1]
    independent_energy = (momentum @ momentum + delta @ delta) / 2
    independent_input_norm = independent.y[-1, -1]
    assert abs(independent_input_norm - 1) < 2e-7
    assert_allclose(independent_energy, energy[-1], rtol=2e-7)
    assert_allclose(momentum, arrays["v"][-1], atol=2e-6)
    assert_allclose(delta, -spacing[-1], atol=2e-6)
    assert (
        independent_energy
        > 1.07 * meta["source_bound_coefficient"] * independent_input_norm
    )
