"""Independent finite spectra and trajectory checks for the boundary reversal."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, eigvalsh, helmert
from scipy.optimize import linear_sum_assignment

from pyflock import experiment_lyons_reversal as E


@pytest.fixture(scope="module")
def result():
    return E.compute(profile="quick", jobs=1)


def _independent_matrices(gx, gv, beta):
    """Construct each row directly, using N total agents in both geometries."""
    n = len(gx)
    circle, path = np.zeros((2 * n, 2 * n)), np.zeros((2 * n, 2 * n))
    circle[:n, n:] = path[:n, n:] = np.eye(n)
    for k in range(n):
        circle[n + k, k] = -gx[k]
        circle[n + k, (k - 1) % n] = gx[k] / 2
        circle[n + k, (k + 1) % n] = gx[k] / 2
        circle[n + k, n + k] = -gv[k]
        circle[n + k, n + (k - 1) % n] = gv[k] * (1 + beta) / 2
        circle[n + k, n + (k + 1) % n] = gv[k] * (1 - beta) / 2
        if 1 <= k < n - 1:
            path[n + k] = circle[n + k]
    path[-1, n - 2], path[-1, n - 1] = gx[-1], -gx[-1]
    path[-1, 2 * n - 2], path[-1, 2 * n - 1] = gv[-1], -gv[-1]
    keep = np.r_[np.arange(1, n), np.arange(n + 1, 2 * n)]
    return circle, path, path[np.ix_(keep, keep)]


def _match_eigenvalues(actual, expected, tolerance):
    assert len(actual) == len(expected)
    rows, cols = linear_sum_assignment(abs(actual[:, None] - expected[None, :]))
    assert np.max(abs(actual[rows] - expected[cols])) < tolerance


def _bloch_spectrum(beta, q):
    """Exact 2-agent cells, with both neighbour edges added separately."""
    values = []
    for m in range(q):
        phase = 2 * np.pi * m / q
        lx, lv = np.eye(2, dtype=complex), np.eye(2, dtype=complex)
        for k in range(2):
            for j in (-1, 1):
                twist = np.exp(1j * phase * ((k + j) // 2))
                lx[k, (k + j) % 2] -= twist / 2
                lv[k, (k + j) % 2] -= (1 - j * beta) * twist / 2
        generator = np.block(
            [[np.zeros((2, 2)), np.eye(2)], [-lx, -np.array([2, 18])[:, None] * lv]]
        )
        values.extend(np.linalg.eigvals(generator))
    return np.asarray(values)


def test_same_agents_and_regular_tail_have_independent_direct_assembly(result):
    data, _ = result
    gx, gv, beta = data["gx"], data["gv"], float(data["beta"])
    assert gx == pytest.approx(np.ones(64), abs=0)
    assert gv == pytest.approx(np.tile([2, 18], 32), abs=0)
    assert beta == pytest.approx(0.1, abs=0)
    circle, path_full, path = _independent_matrices(gx, gv, beta)
    assert data["circle_matrix"] == pytest.approx(circle, abs=0)
    assert data["path_matrix"] == pytest.approx(path, abs=0)
    assert np.count_nonzero(path_full[64]) == 0
    assert path_full[-1, [62, 63, 126, 127]] == pytest.approx([1, -1, 18, -18])


def test_coherent_subspace_is_invariant_and_quotient_is_well_defined(result):
    data, _ = result
    n = len(data["gx"])
    circle = data["circle_matrix"]
    coherent = data["coherent_basis"]
    quotient = data["quotient_basis"]
    expected_coherent = block_diag(
        np.ones((n, 1)) / np.sqrt(n), np.ones((n, 1)) / np.sqrt(n)
    )
    assert coherent == pytest.approx(expected_coherent, abs=1e-15)
    assert coherent.T @ coherent == pytest.approx(np.eye(2), abs=2e-15)
    assert quotient.T @ quotient == pytest.approx(np.eye(2 * n - 2), abs=2e-15)
    assert coherent.T @ quotient == pytest.approx(np.zeros((2, 2 * n - 2)), abs=2e-15)
    assert circle @ coherent == pytest.approx(
        coherent @ np.array([[0, 1], [0, 0]]), abs=2e-15
    )
    assert data["circle_quotient_matrix"] == pytest.approx(
        quotient.T @ circle @ quotient, abs=2e-14
    )
    # A separate orthonormal basis gives the same quotient spectrum.
    other = block_diag(helmert(n, full=False).T, helmert(n, full=False).T)
    independent = np.linalg.eigvals(other.T @ circle @ other)
    _match_eigenvalues(data["circle_quotient_eigenvalues"], independent, 3e-10)


def test_complete_finite_spectra_and_exact_bloch_union_confirm_reversal(result):
    data, meta = result
    circle = data["circle_eigenvalues"]
    quotient = data["circle_quotient_eigenvalues"]
    path = data["path_eigenvalues"]
    assert len(circle) == 128 and len(quotient) == len(path) == 126
    physical = circle[abs(circle) > 1e-6]
    assert len(physical) == 126
    _match_eigenvalues(physical, quotient, 3e-10)
    expected_bloch = _bloch_spectrum(float(data["beta"]), 32)
    _match_eigenvalues(data["bloch_eigenvalues"], expected_bloch, 3e-7)
    _match_eigenvalues(physical, expected_bloch[abs(expected_bloch) > 1e-6], 3e-10)
    assert np.count_nonzero(physical.real > 1e-8) == 2
    assert np.max(quotient.real) == pytest.approx(0.00056231273447, abs=3e-11)
    assert np.max(path.real) == pytest.approx(-0.0081290378529, abs=3e-11)
    assert np.max(data["circle_residuals"]) < 1e-12
    assert np.max(data["path_residuals"]) < 1e-12
    classification = meta["classification"]
    assert classification["circle_physical_abscissa"] == pytest.approx(
        np.max(quotient.real)
    )
    assert classification["path_follower_abscissa"] == pytest.approx(np.max(path.real))


def test_numerical_lyapunov_matrix_has_positive_and_negative_definiteness_clearance(
    result,
):
    data, meta = result
    matrix, p = data["path_matrix"], data["lyapunov_matrix"]
    assert p == pytest.approx(p.T, abs=1e-7)
    eigenvalues = eigvalsh((p + p.T) / 2)
    assert eigenvalues[0] > 0.01
    derivative = matrix.T @ p + p @ matrix
    residual = derivative + np.eye(len(p))
    assert np.linalg.norm(residual, 2) < 1e-4
    assert eigvalsh((derivative + derivative.T) / 2)[-1] < -0.99
    assert data["lyapunov_residual_matrix"] == pytest.approx(residual, abs=1e-9)
    assert meta["classification"]["lyapunov_min_eigenvalue"] == pytest.approx(
        eigenvalues[0], abs=1e-8
    )


def test_matched_initial_states_and_quotient_norms_agree_with_independent_short_ode(
    result,
):
    data, _ = result
    gx, gv, beta = data["gx"], data["gv"], float(data["beta"])
    n = len(gx)
    initial = np.r_[data["initial_displacement"], data["initial_velocity"]]
    assert abs(initial[0]) < 1e-14 and abs(initial[n]) < 1e-14
    circle_states, path_states = data["circle_states"], data["path_states"]
    assert circle_states[0] == pytest.approx(initial, abs=1e-13)
    assert path_states[0] == pytest.approx(initial, abs=1e-13)
    times = data["times"]
    assert times[0] == 0 and np.all(np.diff(times) > 0)
    sample = np.flatnonzero(times <= 10)
    assert len(sample) >= 2

    def rhs(geometry):
        def derivative(t, state):
            z, v = state[:n], state[n:]
            acceleration = -gx * (z - (np.roll(z, 1) + np.roll(z, -1)) / 2) - gv * (
                v - (1 + beta) / 2 * np.roll(v, 1) - (1 - beta) / 2 * np.roll(v, -1)
            )
            if geometry == "path":
                acceleration[0] = 0
                acceleration[-1] = -gx[-1] * (z[-1] - z[-2]) - gv[-1] * (v[-1] - v[-2])
            return np.r_[v, acceleration]

        return derivative

    for geometry, states in (("circle", circle_states), ("path", path_states)):
        independent = solve_ivp(
            rhs(geometry),
            (0, float(times[sample[-1]])),
            initial,
            t_eval=times[sample],
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert independent.success
        assert states[sample] == pytest.approx(independent.y.T, abs=2e-9)
        # Translation-free state norm without using the stored quotient basis.
        z, v = states[:, :n], states[:, n:]
        norms = np.sqrt(
            np.sum(
                abs(z - z.mean(axis=1, keepdims=True)) ** 2
                + abs(v - v.mean(axis=1, keepdims=True)) ** 2,
                axis=1,
            )
        )
        assert data[f"{geometry}_response"] == pytest.approx(
            norms / norms[0], abs=1e-10
        )
    assert data["circle_response"][-1] > 4
    assert data["path_response"][-1] < 0.01


def test_draw_preserves_style_and_produces_finite_plotted_curves(result):
    data, meta = result
    before = dict(matplotlib.rcParams)
    figure = E.draw(data, meta)
    figure.canvas.draw()
    assert len(figure.axes) >= 2
    for axis in figure.axes:
        for line in axis.lines:
            assert np.isfinite(np.asarray(line.get_ydata())).all()
    plt.close(figure)
    assert dict(matplotlib.rcParams) == before
