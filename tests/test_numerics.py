"""Independent numerical checks and regressions for unsupported mathematical regimes."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix, issparse

import pyflock as pf
from pyflock.solve import simulate_tail


def test_stencil_owns_immutable_weights():
    weights = np.array([0.0, -0.5, 1.0, -0.5, 0.0])
    st = pf.Stencil(weights, weights, -1, -1)
    weights[2] = 7
    assert st.rx[2] == 1
    with pytest.raises(ValueError):
        st.rx[2] = 3
    with pytest.raises(ValueError):
        st.rx.setflags(write=True)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_stencil_rejects_nonfinite_inputs(bad):
    with pytest.raises(ValueError):
        pf.nn_stencil(bad, -1)
    with pytest.raises(ValueError):
        pf.Stencil([0, -0.5, 1, -0.5, bad], [0, -0.5, 1, -0.5, 0], -1, -1)


def test_moment_expansion_has_seventh_order_quadratic_residual():
    """A fifth-order lambda expansion must leave O(theta**7) in the quadratic."""
    st = pf.nn_stencil(-1.3, -0.9, -0.5, -0.3)
    theta = np.array([0.16, 0.08, 0.04])
    coefficients = pf.taylor(st, order=5)
    lam = np.array(
        [np.polynomial.polynomial.polyval(theta, row) for row in coefficients]
    )
    # Independent trigonometric symbols, avoiding branch tracking and fitted values.
    nx = st.gx * (1 - np.cos(theta))
    nv = st.gv * ((1 - np.cos(theta)) + 0.4j * np.sin(theta))
    residual = np.abs(lam**2 - nv * lam - nx)
    assert np.all((residual[:, :-1] / residual[:, 1:]) > 120)
    assert np.all((residual[:, :-1] / residual[:, 1:]) < 135)
    assert pf.taylor(st, 5, h=1e-8, fit=2) == pytest.approx(coefficients, abs=0)


def test_taylor_rejects_fractional_and_coincident_roots():
    with pytest.raises(ValueError, match="first position moment"):
        pf.taylor(pf.nn_stencil(-1, -1, -0.3, -0.5))
    with pytest.raises(ValueError, match="distinct first-order"):
        pf.taylor(pf.nn_stencil(0, -1))
    with pytest.raises(ValueError, match="decentralized"):
        pf.taylor(pf.Stencil([0, -0.5, 2, -0.5, 0], [0, -0.5, 1, -0.5, 0], -1, -1))
    with pytest.raises(ValueError, match="nonnegative integer"):
        pf.taylor(pf.nn_stencil(-1, -1), order=-1)


def test_unstable_analytic_branches_have_complex_velocities():
    st = pf.nn_stencil(1, -1)
    c = pf.taylor(st, order=3)
    assert sorted(c[:, 1].real) == pytest.approx([-1 / np.sqrt(2), 1 / np.sqrt(2)])
    with pytest.raises(ValueError, match="real distinct"):
        pf.signal_velocities(st)
    with pytest.raises(ValueError, match="real long-wave"):
        pf.coefficients(st)


def test_negative_a_branch_labels_match_the_analytic_expansion_on_both_sides():
    st = pf.nn_stencil(1, -0.7, -0.5, -0.3)
    assert st.a < 0
    theta = np.array([0.16, 0.08, 0.04])
    coefficients = pf.taylor(st, order=3)
    for sign in (-1, 1):
        exact = pf.branches(sign * theta, st)
        approximation = np.array(
            [
                np.polynomial.polynomial.polyval(sign * theta, row)
                for row in coefficients
            ]
        )
        error = np.abs(exact - approximation)
        # A cubic expansion has O(theta**4) error for each analytic branch.
        ratio = error[:, :-1] / error[:, 1:]
        assert np.all((ratio > 15) & (ratio < 17))
    assert pf.branches(-theta, st) == pytest.approx(
        np.conj(pf.branches(theta, st)[::-1]), abs=0
    )


def test_signal_velocities_may_have_the_same_sign():
    cp, cm = pf.signal_velocities(pf.case("two_positive_velocities"))
    assert cp > cm > 0


@pytest.mark.parametrize("formula", [pf.theory_amplitudes, pf.corrected_amplitudes])
@pytest.mark.parametrize(
    "st",
    [
        pf.case("two_positive_velocities"),
        pf.nn_stencil(-1, -1, -0.3, -0.5),
        pf.nn_stencil(1, -1),
    ],
)
def test_path_reflection_formulas_reject_unsupported_stencils(formula, st):
    with pytest.raises(ValueError):
        formula(st, 100)


@pytest.mark.parametrize(
    "formula",
    [pf.theory_amplitudes, pf.corrected_amplitudes, pf.least_stable_eigenvalue],
)
def test_asymptotic_formulas_reject_zero_size(formula):
    with pytest.raises(ValueError, match="positive integer"):
        formula(pf.nn_stencil(-1, -1), 0)


def test_rk_initial_state_is_at_zero_even_when_first_output_time_is_positive():
    M = np.array([[0.0, 1.0], [-2.0, -0.7]])
    y0 = np.array([1.2, -0.4])
    times = np.array([0.7, 1.1, 2.0])
    expected = np.stack([expm(t * M) @ y0 for t in times])
    for matrix in (M, csr_matrix(M)):
        assert pf.propagate(matrix, y0, times, method="rk") == pytest.approx(
            expected, abs=2e-10
        )
        assert pf.propagate(matrix, y0, times) == pytest.approx(expected, abs=2e-12)
        assert pf.propagate(matrix, y0, [0], method="rk") == pytest.approx(y0[None, :])


def test_propagate_preserves_complex_states():
    M = np.array([[1j]])
    assert pf.propagate(M, [1j], [0.0, 0.5, 1.0])[:, 0] == pytest.approx(
        1j * np.exp(1j * np.array([0.0, 0.5, 1.0]))
    )


@pytest.mark.parametrize("times", [[], [-1], [0, 0], [1, 0], [np.nan], [[0, 1]]])
def test_propagate_rejects_invalid_times(times):
    with pytest.raises(ValueError, match="times"):
        pf.propagate(np.eye(1), np.ones(1), times)


def test_rk_failure_is_reported(monkeypatch):
    import pyflock.solve as solver

    monkeypatch.setattr(
        solver,
        "solve_ivp",
        lambda *a, **k: SimpleNamespace(success=False, message="step failed"),
    )
    with pytest.raises(RuntimeError, match="step failed"):
        solver.propagate(np.eye(1), np.ones(1), [0, 1], method="rk")


@pytest.mark.parametrize("perturbation", [0.0, 1e-14])
def test_fft_handles_repeated_and_near_repeated_nonzero_roots(perturbation):
    st = pf.nn_stencil(-1 + perturbation, -np.sqrt(2))
    N = 8
    z0 = (-1.0) ** np.arange(N)
    zdot0 = 0.3 * z0
    times = np.array([0.0, 0.1, 0.7, 3.0])
    result = pf.simulate_circle(st, N, times, z0, zdot0)
    M = pf.system_matrix_circle(st, N)
    for j, t in enumerate(times):
        expected = expm(M * t) @ np.r_[z0, zdot0]
        assert result.z[j] == pytest.approx(expected[:N], abs=1e-13)
        assert result.zdot[j] == pytest.approx(expected[N:], abs=1e-13)


def test_fft_double_zero_limit_is_free_motion():
    rng = np.random.default_rng(27)
    z0, velocity = rng.normal(size=(2, 9))
    times = np.array([0.0, 0.7, 10.0])
    result = pf.simulate_circle(pf.nn_stencil(0, 0), 9, times, z0, velocity)
    assert result.z == pytest.approx(z0 + times[:, None] * velocity, abs=1e-13)
    assert result.zdot == pytest.approx(
        np.broadcast_to(velocity, result.z.shape), abs=1e-13
    )


@pytest.mark.parametrize("bc", ["drop", "fold", "regular"])
def test_sparse_path_assembly_and_propagation_agree_with_dense(bc):
    st = pf.nn_stencil(-1.3, -0.9, -0.5, -0.3)
    dense = pf.system_matrix_line(st, 8, bc=bc)
    sparse = pf.system_matrix_line(st, 8, bc=bc, sparse=True)
    assert sparse.toarray() == pytest.approx(dense, abs=0)
    rng = np.random.default_rng(19)
    y0 = rng.normal(size=dense.shape[0])
    times = np.linspace(0.3, 2.3, 11)
    expected = np.stack([expm(t * dense) @ y0 for t in times])
    assert pf.propagate(sparse, y0, times) == pytest.approx(expected, abs=1e-12)


def test_large_sparse_path_preserves_uniform_free_motion():
    N = 3200
    M = pf.system_matrix_line(pf.nn_stencil(-1, -1), N, sparse=True)
    assert issparse(M) and M.nnz < 10 * N
    y0 = np.r_[np.full(N + 1, 2.0), np.full(N + 1, 0.3)]
    times = np.array([0.0, 0.1])
    actual = pf.propagate(M, y0, times)
    assert actual[:, : N + 1] == pytest.approx(
        np.broadcast_to((2 + 0.3 * times)[:, None], (2, N + 1)), abs=1e-12
    )
    assert actual[:, N + 1 :] == pytest.approx(np.full((2, N + 1), 0.3), abs=1e-12)


def test_extrema_refinement_handles_nonuniform_times():
    times = np.array([0.0, 0.7, 1.1, 2.0])
    values = 3 - (times - 1) ** 2
    (extremum,) = pf.extrema(times, values)
    assert (extremum.t, extremum.value) == pytest.approx((1.0, 3.0))


def test_tail_only_dense_trace_matches_full_and_independent_exponentials():
    st = pf.nn_stencil(-1.3, -0.9, -0.5, -0.3)
    N, tmax, nt, v0 = 12, 45.0, 39, 1.7
    full = pf.simulate_line(st, N, tmax, nt=nt, bc="drop", v0=v0)
    times, tail = simulate_tail(st, N, tmax, nt, bc="drop", v0=v0, chunk_size=7)
    assert times == pytest.approx(full.t, abs=0)
    assert tail == pytest.approx(full.last_relative, abs=1e-12)
    M = pf.system_matrix_line(st, N, bc="drop")
    initial = np.zeros(2 * (N + 1))
    initial[N + 1] = v0
    for i in (0, nt // 2, nt - 1):
        state = expm(M * times[i]) @ initial
        assert tail[i] == pytest.approx(state[N] - state[0], abs=3e-11)


def test_tail_only_sparse_trace_preserves_state_across_chunk_boundaries():
    st = pf.nn_stencil(-1.0, -1.0, -0.5, -0.3)
    N, tmax, nt = 200, 480.0, 67
    full = pf.simulate_line(st, N, tmax, nt=nt, bc="regular")
    for chunk_size in (1, 8, 128):
        times, tail = simulate_tail(st, N, tmax, nt, chunk_size=chunk_size)
        assert times == pytest.approx(full.t, abs=0)
        assert tail == pytest.approx(full.last_relative, rel=3e-11, abs=3e-10)
        assert tail[0] == 0


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_tail_only_rejects_invalid_chunk_sizes(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        simulate_tail(pf.nn_stencil(-1, -1), 8, 10, 20, chunk_size=chunk_size)
