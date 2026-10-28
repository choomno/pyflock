"""Checks against independent physical matrices and the restricted-family proofs."""

import numpy as np
import pytest
from scipy.linalg import eigvals, solve
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammainc

from pyflock.dispersion import branches, signal_velocities
from pyflock.experiment_nnn import (
    _configuration,
    _ensemble_case,
    _path_matrix,
    converged_tail,
    select_configurations,
    spatial_transfer,
    tail_trace,
)
from pyflock.stencil import Stencil


def test_paper_fixed_mass_boundary_rows_are_endpoint_clamping():
    st = Stencil([-0.5, 0.25, 1, -0.75, 0], [-1, 0.75, 1, -1, 0.25], -2, -2)
    N = 8
    matrix = _path_matrix(st, N, "fixed_mass").toarray()
    for offset, weights, gain in ((0, st.rx, st.gx), (N + 1, st.rv, st.gv)):
        block = matrix[N + 1 :, offset : offset + N + 1] / gain
        assert block[1, :4] == pytest.approx(
            [weights[0] + weights[1], weights[2], weights[3], weights[4]]
        )
        assert block[N - 1, N - 3 :] == pytest.approx(
            [weights[0], weights[1], weights[2], weights[3] + weights[4]]
        )
        assert block[N, N - 2 :] == pytest.approx(
            [weights[0], weights[1], weights[2] + weights[3] + weights[4]]
        )
        assert block.sum(axis=1) == pytest.approx(np.zeros(N + 1), abs=1e-15)


@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("bc", ["drop", "fixed_mass"])
def test_spatial_transfer_matches_independent_matrix_resolvent(kind, bc):
    st = _configuration(kind, 249)
    N = 25
    matrix = _path_matrix(st, N, bc).toarray()
    initial = np.zeros(2 * (N + 1))
    initial[N + 1] = 1
    s = np.array([0.01 + 0.02j, 0.2 + 0.7j, 1 + 3j])
    expected = np.array(
        [(solve(z * np.eye(len(matrix)) - matrix, initial)[N] * z**2) for z in s]
    )
    assert spatial_transfer(st, N, s, bc) == pytest.approx(
        expected, rel=2e-10, abs=3e-12
    )


@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("bc", ["drop", "fixed_mass"])
def test_fft_inversion_matches_independent_time_evolution(kind, bc):
    st = _configuration(kind, 249)
    N, end = 25, 180.0
    result = tail_trace(st, N, end, bc=bc, nfft=8192)
    matrix = _path_matrix(st, N, bc)
    initial = np.zeros(2 * (N + 1))
    initial[N + 1] = 1
    direct = expm_multiply(
        matrix,
        initial,
        start=0,
        stop=end,
        num=len(result.t),
        traceA=matrix.diagonal().sum(),
    )
    expected = direct[:, N] - direct[:, 0]
    assert np.max(abs(result.relative - expected)) / N < 2e-10


def test_full_configuration_counts_and_exact_type_two_factorization():
    records, classes, indices = select_configurations()
    assert records.shape == (1000, 12)
    assert np.array_equal(np.bincount(classes), [500, 500])
    assert len(np.unique(records, axis=0)) == 1000
    assert np.array_equal(indices[:500], np.arange(500))
    theta = np.array([0.07, 0.4, 1.2])
    for index in (0, 249, 499):
        st = _configuration(1, index)
        a, b = signal_velocities(st)
        actual = branches(theta, st, "by_imag")
        expected = np.array(
            [-b * (1 - np.exp(-1j * theta)), -a * (1 - np.exp(-1j * theta))]
        )
        assert actual == pytest.approx(expected, abs=2e-13)


@pytest.mark.parametrize("index", [0, 249, 499])
@pytest.mark.parametrize("bc", ["drop", "fixed_mass"])
def test_type_one_energy_matrices_and_type_two_diagonal_blocks_are_stable(index, bc):
    N = 10
    for kind in (0, 1):
        st = _configuration(kind, index)
        M = _path_matrix(st, N, bc).toarray()
        keep = np.r_[1 : N + 1, N + 2 : 2 * (N + 1)]
        assert eigvals(M[np.ix_(keep, keep)]).real.max() < 0
        if kind == 0:
            K = M[N + 2 :, 1 : N + 1] / st.gx
            L = M[N + 2 :, N + 2 :] / st.gv
            assert K == pytest.approx(K.T, abs=0)
            assert np.linalg.eigvalsh(K).min() > 0
            assert np.linalg.eigvalsh((L + L.T) / 2).min() > 0


def test_large_path_refinement_and_contour_diagnostics():
    st = _configuration(1, 499)
    N = 51200
    trace = converged_tail(st, N, 1.25 * N, tolerance=2e-8)
    moved = tail_trace(st, N, 1.25 * N, nfft=trace.nfft, damping=40)
    assert trace.convergence_error < 2e-8
    assert np.max(abs(moved.relative - trace.relative)) / N < 1e-7


def test_large_type_two_response_matches_independent_incomplete_gamma_formula():
    a, b, N = 2.0, 0.4, 51200
    st = _configuration(1, 0)
    trace = tail_trace(st, N, 1.25 * N / b, bc="fixed_mass", nfft=65536)
    t = trace.t

    # Inverse Laplace of (c/(s+c))**N / s**2, from the gamma distribution.
    def integrated_gamma(c):
        return t * gammainc(N, c * t) - N / c * gammainc(N + 1, c * t)

    exact = (a * integrated_gamma(a) - b * integrated_gamma(b)) / (a - b) - t
    assert np.max(abs(exact - trace.relative)) / N < 1e-8


def test_checkpointed_case_keeps_validity_and_numerical_error_separate(tmp_path):
    record = select_configurations(1)[0][1]
    spec = (1, 1, record, "fixed_mass", 25, str(tmp_path))
    row = _ensemble_case(spec)
    assert np.isfinite(row).all()
    assert np.array_equal(row[16:19], [1, 1, 1])
    assert row[22] == 0
    assert row[10] == row[11]  # T1=A/v0 is disclosed as the same estimator.
    assert row[19] < 2e-8 and row[20] < 1e-7
    assert _ensemble_case(spec) == pytest.approx(row, abs=0)
