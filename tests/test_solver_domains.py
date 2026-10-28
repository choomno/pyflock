"""Domains and independent exact solutions for matrix construction and dynamics."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pyflock as pf
from pyflock import heterogeneous as ht
from pyflock.solve import simulate_tail


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
@pytest.mark.parametrize("matrix", [pf.laplacian_circle, pf.laplacian_line])
def test_matrix_construction_requires_positive_integer_agent_counts(matrix, size):
    with pytest.raises(ValueError, match="positive integer"):
        matrix([0, -0.5, 1, -0.5, 0], size)


@pytest.mark.parametrize("weights", [[1, -1], [0, -0.5, np.nan, -0.5, 0]])
def test_matrix_construction_requires_a_finite_five_point_stencil(weights):
    with pytest.raises(ValueError, match="weights"):
        pf.laplacian_line(weights, 3)


def test_unknown_boundary_is_rejected_before_matrix_assembly():
    with pytest.raises(ValueError, match="bc"):
        pf.laplacian_line([0, -0.5, 1, -0.5, 0], 3, bc="clamped")


def test_regular_boundary_without_a_driven_leader_retains_the_left_equation():
    weights = [0, -0.75, 1, -0.25, 0]
    matrix = pf.laplacian_line(weights, 2, bc="regular", leader=False)
    assert matrix == pytest.approx(
        np.array([[1, -0.25, 0], [-0.75, 1, -0.25], [0, -1, 1]])
    )


def test_sparse_follower_matrix_matches_the_two_agent_equation():
    st = pf.nn_stencil(-2, -3)
    reduced = pf.system_matrix_line_leaderless(st, 1, bc="regular", sparse=True)
    assert reduced.toarray() == pytest.approx(np.array([[0, 1], [-2, -3]]))


@pytest.mark.parametrize("size", [1, 2, 7])
def test_sparse_circle_operator_accumulates_coincident_offsets(size):
    st = pf.nn_stencil(-2, -3, -0.5, -0.25)
    actual = pf.system_matrix_circle(st, size, sparse=True).toarray()
    expected = np.zeros((2 * size, 2 * size))
    for k in range(size):
        expected[k, size + k] = 1
        for offset, rx, rv in zip(range(-2, 3), st.rx, st.rv):
            expected[size + k, (k + offset) % size] += -2 * rx
            expected[size + k, size + (k + offset) % size] += -3 * rv
    assert actual == pytest.approx(expected, abs=0)


@pytest.mark.parametrize(
    "matrix", [np.zeros(0), np.zeros((1, 2)), np.zeros((0, 0)), [[np.inf]]]
)
def test_propagation_requires_a_finite_nonempty_square_generator(matrix):
    with pytest.raises(ValueError, match="M must"):
        pf.propagate(matrix, [1], [0, 1])


@pytest.mark.parametrize("initial", [[1, 2], [np.nan]])
def test_propagation_requires_a_finite_initial_state_of_matching_dimension(initial):
    with pytest.raises(ValueError, match="y0 must"):
        pf.propagate([[-1]], initial, [0, 1])


def test_unknown_propagator_is_rejected():
    with pytest.raises(ValueError, match="method"):
        pf.propagate([[-1]], [1], [0, 1], method="euler")


def test_large_dense_generator_matches_independent_scalar_exponentials():
    rates = np.linspace(-1, 0, 901)
    actual = pf.propagate(np.diag(rates), np.ones(901), [0.3, 0.7])
    assert actual == pytest.approx(np.exp(np.outer([0.3, 0.7], rates)), abs=1e-14)


@pytest.mark.parametrize("function", [pf.simulate_line, simulate_tail])
@pytest.mark.parametrize("tmax", [0, -1, np.inf, np.nan])
def test_path_solvers_reject_invalid_final_times(function, tmax):
    with pytest.raises(ValueError, match="tmax"):
        function(pf.nn_stencil(-1, -1), 3, tmax)


@pytest.mark.parametrize("function", [pf.simulate_line, simulate_tail])
@pytest.mark.parametrize("nt", [1, 1.5, True])
def test_path_solvers_require_at_least_two_output_times(function, nt):
    with pytest.raises(ValueError, match="nt must"):
        function(pf.nn_stencil(-1, -1), 3, 1, nt=nt)


@pytest.mark.parametrize("function", [pf.simulate_line, simulate_tail])
def test_path_solvers_reject_nonfinite_leader_velocity(function):
    with pytest.raises(ValueError, match="v0"):
        function(pf.nn_stencil(-1, -1), 3, 1, v0=np.inf)


def test_path_solver_rejects_nonfinite_formation_spacing():
    with pytest.raises(ValueError, match="delta"):
        pf.simulate_line(pf.nn_stencil(-1, -1), 3, 1, delta=np.nan)


@pytest.mark.parametrize("initial", [[0], [0, 0, np.inf]])
def test_custom_path_initial_conditions_are_validated(initial):
    with pytest.raises(ValueError, match="z0"):
        pf.simulate_line(pf.nn_stencil(-1, -1), 2, 1, z0=initial)


def test_custom_path_state_and_formation_coordinates_follow_free_motion():
    st = pf.nn_stencil(0, 0)
    initial, velocity = np.array([2.0, 1.0, -1.0]), np.array([0.3, 0.7, 1.2])
    result = pf.simulate_line(st, 2, 3, nt=7, z0=initial, zdot0=velocity, delta=4)
    expected = initial + np.outer(result.t, velocity)
    assert result.z == pytest.approx(expected, abs=2e-15)
    assert result.zdot == pytest.approx(np.broadcast_to(velocity, expected.shape))
    assert result.x == pytest.approx(expected - [0, 4, 8])
    assert result.last_relative == pytest.approx(expected[:, -1] - expected[:, 0])


def test_collision_report_identifies_the_first_sampled_crossing():
    st = pf.nn_stencil(0, 0)
    safe = pf.simulate_line(st, 2, 1, nt=3, v0=1)
    assert safe.collisions() == "No collisions"
    crossing = pf.simulate_line(st, 2, 1, nt=3, v0=0, zdot0=[0, 2, 5])
    assert crossing.collisions() == "Collision at t=0.5, between agents 0 and 1"


@pytest.mark.parametrize("size", [0, 1.5, True])
def test_circle_solver_rejects_invalid_agent_counts(size):
    with pytest.raises(ValueError, match="N must"):
        pf.simulate_circle(pf.nn_stencil(-1, -1), size, [0, 1], [0], [0])


@pytest.mark.parametrize("method", ["expm", "rk"])
def test_circle_ode_methods_reproduce_uniform_free_motion(method):
    initial, velocity, times = np.full(4, 2.0), np.full(4, 0.3), np.array([0, 0.5, 1])
    result = pf.simulate_circle(
        pf.nn_stencil(-1, -1), 4, times, initial, velocity, method=method
    )
    assert result.z == pytest.approx(initial + np.outer(times, velocity), abs=2e-14)
    assert result.zdot == pytest.approx(np.broadcast_to(velocity, result.z.shape))


@pytest.mark.parametrize("velocity", [[1], np.nan])
def test_heterogeneous_path_rejects_nonscalar_or_nonfinite_leader_velocity(velocity):
    with pytest.raises(ValueError, match="v0"):
        ht.simulate_path(ht.HeterogeneousNN([1, 2], [2, 3]), [0, 1], v0=velocity)


def test_sparse_complex_generator_preserves_the_unitary_phase():
    times = np.array([0.0, 0.3, 1.7])
    result = pf.propagate(csr_matrix([[1j]]), [1], times)
    assert result[:, 0] == pytest.approx(np.exp(1j * times), abs=2e-15)


def test_phase_velocity_is_minus_the_fourier_eigenvalue_slope():
    theta = np.array([0.1, 0.4, -0.8])
    st = pf.nn_stencil(-1, -0.7)
    # Symmetric NN modes have Im(lambda)=+/-sqrt(-nx-nv**2/4).
    nx = -2 * np.sin(theta / 2) ** 2
    nv = -1.4 * np.sin(theta / 2) ** 2
    magnitude = np.sqrt(-nx - nv**2 / 4)
    expected = np.stack([-magnitude / theta, magnitude / theta])
    assert pf.phase_velocity(theta, st, labelling="by_imag") == pytest.approx(expected)


def test_unknown_branch_labelling_is_rejected():
    with pytest.raises(ValueError, match="labelling"):
        pf.branches([0, 1], pf.nn_stencil(-1, -1), labelling="arbitrary")


def test_zeroth_order_taylor_polynomial_records_the_double_zero_eigenvalue():
    assert pf.taylor(pf.nn_stencil(-1, -1), order=0) == pytest.approx(np.zeros((2, 1)))
