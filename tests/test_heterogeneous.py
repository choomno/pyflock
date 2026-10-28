"""Independent checks for distinct-agent nearest-neighbour arrays."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse

import pyflock as pf
from pyflock import heterogeneous as ht


def test_gain_arrays_are_owned_and_immutable():
    gains = np.array([1.0, 2.0, 3.0])
    model = ht.HeterogeneousNN(gains, gains, 0.3)
    gains[0] = 5
    assert model.gx[0] == model.gv[0] == 1
    with pytest.raises(ValueError):
        model.gx.setflags(write=True)
    with pytest.raises(ValueError):
        model.gv[0] = 5


@pytest.mark.parametrize(
    "gx,gv",
    [
        ([], []),
        ([0], [1]),
        ([-1], [1]),
        ([1], [np.inf]),
        ([np.nan], [1]),
        ([[1]], [1]),
        ([1, 2], [1]),
    ],
)
def test_invalid_gains_are_rejected(gx, gv):
    with pytest.raises(ValueError):
        ht.HeterogeneousNN(gx, gv)


@pytest.mark.parametrize("beta", [-1.01, 1.01, np.inf, np.nan, [0]])
def test_invalid_velocity_asymmetry_is_rejected(beta):
    with pytest.raises(ValueError, match="beta_v"):
        ht.HeterogeneousNN([1, 2], [2, 3], beta)


@pytest.mark.parametrize("geometry", ["path", "circle"])
@pytest.mark.parametrize("bc", ["regular", "drop"])
def test_sparse_matrices_have_zero_laplacian_rows_and_gain_scaled_blocks(geometry, bc):
    model = ht.HeterogeneousNN([1, 2, 3, 4], [4, 3, 2, 1], 0.4)
    lx, lv = ht.laplacians(model, geometry=geometry, bc=bc)
    matrix = ht.system_matrix(model, geometry=geometry, bc=bc)
    assert issparse(lx) and issparse(lv) and issparse(matrix)
    assert lx.sum(axis=1).A1 == pytest.approx(np.zeros(4), abs=1e-15)
    assert lv.sum(axis=1).A1 == pytest.approx(np.zeros(4), abs=1e-15)
    assert lx.toarray()[1] == pytest.approx([-0.5, 1, -0.5, 0])
    assert lv.toarray()[1] == pytest.approx([-0.7, 1, -0.3, 0])
    dense = matrix.toarray()
    assert dense[5, :4] == pytest.approx([1, -2, 1, 0])
    assert dense[5, 4:] == pytest.approx([2.1, -3, 0.9, 0])
    if geometry == "path":
        assert np.all(dense[4] == 0)
        assert lx.toarray()[0] == pytest.approx(np.zeros(4), abs=0)
        if bc == "regular":
            assert lx.toarray()[-1] == pytest.approx([0, 0, -1, 1])
            assert lv.toarray()[-1] == pytest.approx([0, 0, -1, 1])
        else:
            assert lx.toarray()[-1] == pytest.approx([0, 0, -0.5, 0.5])
            assert lv.toarray()[-1] == pytest.approx([0, 0, -0.7, 0.7])
    else:
        assert lx.toarray()[0] == pytest.approx([1, -0.5, 0, -0.5])
        assert lv.toarray()[0] == pytest.approx([1, -0.3, 0, -0.7])


@pytest.mark.parametrize("n", [1, 2])
def test_small_circles_sum_coincident_neighbour_entries(n):
    model = ht.HeterogeneousNN(np.ones(n), np.ones(n), 0.7)
    lx, lv = ht.laplacians(model)
    expected = np.zeros((1, 1)) if n == 1 else np.array([[1, -1], [-1, 1]])
    assert lx.toarray() == pytest.approx(expected, abs=1e-15)
    assert lv.toarray() == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("geometry", ["path", "circle"])
@pytest.mark.parametrize("bc", ["regular", "drop"])
@pytest.mark.parametrize("beta", [-1.0, -0.4, 0, 0.6, 1.0])
def test_homogeneous_reduction_matches_existing_signed_gain_convention(
    geometry, bc, beta
):
    n = 11
    model = ht.HeterogeneousNN(np.full(n, 1.3), np.full(n, 0.7), beta)
    st = pf.nn_stencil(-1.3, -0.7, -0.5, (beta - 1) / 2)
    actual = ht.system_matrix(model, geometry=geometry, bc=bc).toarray()
    if geometry == "circle":
        expected = pf.system_matrix_circle(st, n)
    else:
        expected = pf.system_matrix_line(st, n - 1, bc=bc)
    assert actual == pytest.approx(expected, abs=5e-16)


def test_follower_operator_removes_exactly_the_leader_jordan_block():
    model = ht.HeterogeneousNN([1, 2, 3, 4], [1, 2, 3, 4])
    full = ht.system_matrix(model, geometry="path").toarray()
    follower = ht.system_matrix(model, geometry="path", followers_only=True).toarray()
    keep = np.r_[1:4, 5:8]
    assert follower == pytest.approx(full[np.ix_(keep, keep)], abs=0)
    eigenvalues = np.linalg.eigvals(full)
    nonzero = eigenvalues[np.abs(eigenvalues) > 1e-9]
    assert len(nonzero) == 6
    assert nonzero.real.max() < 0
    reduced = np.linalg.eigvals(follower)
    rows, cols = linear_sum_assignment(abs(nonzero[:, None] - reduced[None, :]))
    assert nonzero[rows] == pytest.approx(reduced[cols], abs=2e-14)


@pytest.mark.parametrize("bc", ["regular", "drop"])
def test_path_exponential_matches_independent_laboratory_frame_ode(bc):
    rng = np.random.default_rng(508)
    n, beta, v0 = 13, -0.35, 1.7
    gx, gv = rng.uniform(0.4, 3, size=(2, n))
    model = ht.HeterogeneousNN(gx, gv, beta)
    times = np.array([0.0, 0.13, 0.7, 1.8, 4.3, 8.0])

    def rhs(t, state):
        z, vel = state[:n], state[n:]
        acceleration = np.zeros(n)
        for k in range(1, n - 1):
            acceleration[k] = -gx[k] * (z[k] - (z[k - 1] + z[k + 1]) / 2) - gv[k] * (
                vel[k] - (1 + beta) / 2 * vel[k - 1] + (beta - 1) / 2 * vel[k + 1]
            )
        tx = 1 if bc == "regular" else 0.5
        tv = 1 if bc == "regular" else (1 + beta) / 2
        acceleration[-1] = -gx[-1] * tx * (z[-1] - z[-2]) - gv[-1] * tv * (
            vel[-1] - vel[-2]
        )
        return np.r_[vel, acceleration]

    initial = np.zeros(2 * n)
    initial[n] = v0
    independent = solve_ivp(
        rhs,
        (0, times[-1]),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert independent.success
    actual = ht.simulate_path(model, times, bc=bc, v0=v0)
    assert actual.z == pytest.approx(independent.y[:n].T, abs=2e-11)
    assert actual.zdot == pytest.approx(independent.y[n:].T, abs=3e-11)
    assert actual.r[0] == pytest.approx(np.zeros(n), abs=0)
    assert actual.rdot[0] == pytest.approx(np.r_[0, np.full(n - 1, -v0)], abs=0)
    assert actual.r[:, 0] == pytest.approx(np.zeros(times.size), abs=0)
    assert actual.leader_gap == pytest.approx(-actual.last_relative, abs=0)
    assert actual.leader_gap[1] > 0


def test_circle_exponential_matches_independent_homogeneous_fourier_solution():
    n, beta = 29, 0.42
    rng = np.random.default_rng(42)
    z0, zdot0 = rng.normal(size=(2, n))
    model = ht.HeterogeneousNN(np.full(n, 1.6), np.full(n, 0.85), beta)
    times = np.array([0.0, 0.13, 0.8, 2, 7])
    actual = ht.simulate_circle(model, times, z0, zdot0)
    st = pf.nn_stencil(-1.6, -0.85, -0.5, (beta - 1) / 2)
    independent = pf.simulate_circle(st, n, times, z0, zdot0, method="fft")
    assert actual.z == pytest.approx(independent.z, abs=2e-13)
    assert actual.zdot == pytest.approx(independent.zdot, abs=2e-13)


@pytest.mark.parametrize("geometry", ["path", "circle"])
def test_heterogeneous_arrays_preserve_uniform_free_motion(geometry):
    rng = np.random.default_rng(47)
    n = 400
    model = ht.HeterogeneousNN(rng.uniform(0.5, 5, n), rng.uniform(0.5, 5, n), -0.3)
    matrix = ht.system_matrix(model, geometry=geometry)
    assert matrix.nnz < 8 * n
    times = np.array([0.0, 0.3, 2])
    result = pf.propagate(matrix, np.r_[np.full(n, 2), np.full(n, 0.3)], times)
    assert result[:, :n] == pytest.approx(
        np.broadcast_to(2 + 0.3 * times[:, None], (3, n)), abs=4e-13
    )
    assert result[:, n:] == pytest.approx(np.full((3, n), 0.3), abs=4e-13)


@pytest.mark.parametrize(
    "times",
    [
        np.linspace(0, 12, 35),
        np.linspace(0.4, 12, 35),
        [0.3, 0.9, 1.1, 3.0, 7.0],
        [0.3],
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 7, 128])
def test_tail_only_matches_full_trajectory_on_uniform_and_irregular_grids(
    times, chunk_size
):
    model = ht.HeterogeneousNN(np.linspace(1, 3, 17), np.linspace(0.5, 2, 17), 0.25)
    full = ht.simulate_path(model, times, v0=1.7)
    actual_t, tail = ht.simulate_tail(model, times, v0=1.7, chunk_size=chunk_size)
    assert actual_t == pytest.approx(full.t, abs=0)
    assert tail == pytest.approx(full.last_relative, abs=8e-13)


@pytest.mark.parametrize("beta", [-0.8, 0, 0.7])
def test_c1_coefficients_match_homogeneous_theory_with_explicit_sign(beta):
    model = ht.HeterogeneousNN(np.full(12, 1.3), np.full(12, 0.7), beta)
    cplus, cminus = ht.c1_slopes(model)
    st = pf.nn_stencil(-1.3, -0.7, -0.5, (beta - 1) / 2)
    increasing_plus, increasing_minus = pf.signal_velocities(st)
    assert (-cminus, -cplus) == pytest.approx((increasing_plus, increasing_minus))
    assert ht.gain_moments(model) == pytest.approx(
        {"mean_inverse_gx": 1 / 1.3, "mean_gv_over_gx": 0.7 / 1.3}
    )


def test_c1_coefficients_use_gain_moments_and_have_the_required_symmetries():
    gx, gv = np.array([0.8, 1.7, 2.4]), np.array([1.1, 2.1, 0.7])
    m, v = np.mean(1 / gx), np.mean(gv / gx)
    beta = 0.37
    actual = ht.c1_slopes(ht.HeterogeneousNN(gx, gv, beta))
    expected = (
        (-beta * v + np.sqrt(2 * m + beta**2 * v**2)) / (2 * m),
        (-beta * v - np.sqrt(2 * m + beta**2 * v**2)) / (2 * m),
    )
    assert actual == pytest.approx(expected, abs=2e-16)
    opposite = ht.c1_slopes(ht.HeterogeneousNN(gx, gv, -beta))
    assert opposite == pytest.approx((-actual[1], -actual[0]))
    symmetric = ht.c1_slopes(ht.HeterogeneousNN(gx, gv))
    assert symmetric == pytest.approx((1 / np.sqrt(2 * m), -1 / np.sqrt(2 * m)))


def test_invalid_geometries_and_path_boundaries_are_rejected():
    model = ht.HeterogeneousNN([1, 2], [2, 3])
    with pytest.raises(ValueError, match="geometry"):
        ht.system_matrix(model, geometry="triangle")
    with pytest.raises(ValueError, match="bc"):
        ht.system_matrix(model, geometry="path", bc="fold")
    with pytest.raises(ValueError, match="followers_only"):
        ht.system_matrix(model, followers_only=True)
    with pytest.raises(ValueError, match="at least one follower"):
        ht.system_matrix(ht.HeterogeneousNN([1], [2]), geometry="path")


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_tail_only_rejects_invalid_chunk_size(chunk_size):
    model = ht.HeterogeneousNN([1, 2], [2, 3])
    with pytest.raises(ValueError, match="chunk_size"):
        ht.simulate_tail(model, [0, 1], chunk_size=chunk_size)


def test_small_nonuniform_time_grid_matches_a_single_follower_closed_form():
    scale = 1e-16
    model = ht.HeterogeneousNN([1 / scale**2] * 2, [1 / scale] * 2)
    times = scale * np.array([0.0, 1.0, 3.0])
    tau = times / scale
    expected = -2 / np.sqrt(3) * np.exp(-tau / 2) * np.sin(np.sqrt(3) * tau / 2)
    _, tail = ht.simulate_tail(model, times, v0=1 / scale)
    assert tail == pytest.approx(expected, abs=2e-15)
