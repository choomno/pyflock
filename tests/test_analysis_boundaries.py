"""Independent limiting formulas and boundary cases for transient analysis."""

import numpy as np
import pytest

import pyflock as pf


@pytest.mark.parametrize(
    "t,w",
    [
        ([[0, 1, 2]], [0, 1, 0]),
        ([0, 1, 2], [0, 1]),
        ([0, 1], [0, 1]),
        ([0, np.inf, 2], [0, 1, 0]),
        ([0, 1, 2], [0, np.nan, 0]),
        ([0, 0, 1], [0, 1, 0]),
    ],
)
def test_extrema_require_finite_ordered_matching_vectors(t, w):
    with pytest.raises(ValueError, match="finite vectors"):
        pf.extrema(t, w)


@pytest.mark.parametrize("nmax", [0, -1, True, 2.5])
def test_extrema_require_a_positive_integer_limit(nmax):
    with pytest.raises(ValueError, match="nmax"):
        pf.extrema([0, 1, 2], [0, 1, 0], nmax=nmax)


def test_uninterpolated_extrema_retain_sample_values_and_indices():
    result = pf.extrema([0, 1, 2, 3, 4], [0, -2, 1, -1, 0], refine=False, nmax=2)
    assert [(e.t, e.value, e.index) for e in result] == [(1, -2, 1), (2, 1, 2)]


def test_summary_uses_signed_first_to_third_extrema():
    # Exact parabolic vertices at every other sample, so interpolation is inert.
    t = np.arange(7)
    w = np.array([-1, 0, -1, -2, -1, 0, -1])
    summary = pf.transient_summary(t, w)
    assert summary["A"] == [0, -2, 0]
    assert summary["T_k"] == [1, 3, 5]
    assert summary["n_extrema"] == 3
    assert np.isnan(summary["gamma"])
    assert summary["T"] == 4 and summary["half_period"] == 2
    assert pf.transient_summary([0, 1, 2], [0, 1, 2]) == {
        "A": [],
        "T_k": [],
        "n_extrema": 0,
    }


@pytest.mark.parametrize("v0", [0, np.inf, -np.inf, np.nan])
def test_relative_error_requires_nonzero_finite_velocity(v0):
    with pytest.raises(ValueError, match="finite nonzero"):
        pf.relative_error_law(pf.nn_stencil(-1, -1), v0=v0)


@pytest.mark.parametrize("formula", [pf.theory_amplitudes, pf.corrected_amplitudes])
@pytest.mark.parametrize("parameter", ["v0", "kmax"])
def test_amplitude_formulas_reject_invalid_scalars(formula, parameter):
    with pytest.raises(ValueError, match=parameter):
        formula(
            pf.nn_stencil(-1, -1), 10, **{parameter: np.nan if parameter == "v0" else 0}
        )


def test_symmetric_gaussian_ansatz_reduces_to_a_normal_density_at_zero():
    # For symmetric NN arrays c_+=-c_-=sqrt(-gx/2), D_±=-gv/4,
    # r=-1 and the normal quantile is exactly zero.
    gx, gv, N, v0 = -8.0, -12.0, 700, -2.3
    speed, diffusion = 2.0, 3.0
    phi = 1 / np.sqrt(2 * np.pi)
    st = pf.nn_stencil(gx, gv)
    result = pf.corrected_amplitudes(st, N, v0=v0, kmax=4)
    variances = 2 * N * (2 * np.arange(1, 5) - 1) * diffusion / speed**3
    expected = N * abs(v0) / speed - 2 * abs(v0) * phi * np.sqrt(variances)
    assert result["A"] == pytest.approx(expected, abs=1e-12)
    assert result["s_k"] == pytest.approx(np.sqrt(variances), abs=1e-14)
    assert result["x_star"] == 0 and result["r"] == -1
    assert result["phi"] == pytest.approx(phi)
    assert pf.relative_error_law(st) == pytest.approx(
        2 * phi * np.sqrt(2 * diffusion / speed)
    )
    assert pf.relative_error_law(st, v0=-3) == pf.relative_error_law(st)
    assert pf.corrected_amplitudes(st, N, v0=0)["A"] == [0] * 6


def test_linear_stability_approximation_has_its_stated_exact_root():
    N, gx, gv, q = 17, -2.0, -3.0, 0.3
    st = pf.velocity_asymmetry_family(gx, gv)(q)
    result = pf.least_stable_eigenvalue(st, N, method="linear", detail=True)
    assert result == pytest.approx(
        {"total": gv * (q / (2 * N) + np.pi**2 / (16 * N**2)), "q": q}
    )
    root = pf.critical_velocity_asymmetry(gx, gv, N, method="linear")
    assert root == pytest.approx(-(np.pi**2) / (8 * N))
    zero = pf.velocity_asymmetry_family(gx, gv)(root)
    assert pf.least_stable_eigenvalue(zero, N, method="linear") == pytest.approx(
        0, abs=1e-16
    )


def test_wave_critical_asymmetry_reports_absence_of_a_bracket():
    with pytest.raises(RuntimeError, match="no sign change"):
        pf.critical_velocity_asymmetry(-1, -1, 1)


def test_negative_normalized_velocity_gain_is_required():
    with pytest.raises(ValueError, match="negative normalized gains"):
        pf.theory_amplitudes(pf.nn_stencil(-1, 1), 10)


def test_unresolved_small_speed_is_rejected_by_reflection_formulas():
    with pytest.raises(ValueError, match="opposite nonzero"):
        pf.theory_amplitudes(pf.nn_stencil(-1e-20, -1, -0.5, 0), 10)


def test_near_symmetry_tolerance_does_not_admit_negative_diffusion():
    # Approximate moment/symmetry checks can pass when an enormous gain ratio
    # makes the residual first position moment significant in the expansion.
    st = pf.nn_stencil(-1e30, -1, -0.5 + 1e-13, (1e15 - 1) / 2)
    assert abs(st.bx1) < 1e-12
    assert pf.coefficients(st)["-"]["D"] < 0
    with pytest.raises(ValueError, match="nonnegative diffusion"):
        pf.theory_amplitudes(st, 10)


@pytest.mark.parametrize("times", [[-1e308, 0, 1e308], [0, 1e-320, 2e-320]])
def test_parabolic_refinement_retains_sample_when_float_time_scale_is_unresolved(times):
    # Underflowing curvature or overflowing slopes cannot improve the sample.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        (result,) = pf.extrema(times, [0, 1, 0])
    assert (result.t, result.value, result.index) == (times[1], 1, 1)


def test_spectral_sensitivity_flags_a_defective_zero_eigenvalue():
    result = pf.eigen_reliability(np.array([[0.0, 1e200], [0.0, 0.0]]))
    assert result["eigenvalue"] == 0
    assert np.isinf(result["kappa"]) and np.isinf(result["tol"])
    assert not result["reliable"] and result["heuristic"]


def test_stability_map_without_sensitivity_or_theory_agrees_with_analytic_one_follower():
    # A regular path with one follower has lambda²+2lambda+3=0, independent
    # of velocity asymmetry, and spectral abscissa -1.
    family = pf.velocity_asymmetry_family(-3, -2)
    result = pf.stability_map(
        family, [-0.4, 0, 0.7], [1], bc="regular", reliability=False
    )
    assert result.max_re == pytest.approx(np.full((1, 3), -1.0), abs=1e-14)
    assert result.kappa is None and result.reliable is None and result.theory is None


@pytest.mark.parametrize(
    "sizes",
    [[], 1, [[1, 2]], [0], [-1], [1.9], [1.0], [True], [1, True], [np.nan], [np.inf]],
)
def test_stability_sweep_rejects_invalid_sizes_before_calling_the_family(sizes):
    def family(parameter):
        pytest.fail(
            "invalid sweep domains must be rejected before constructing a model"
        )

    with pytest.raises(ValueError, match="positive integer"):
        pf.stability_map(family, [0.0], sizes)


@pytest.mark.parametrize("params", [[], 0.0, [[0, 1]], [np.nan], [np.inf]])
def test_stability_sweep_requires_nonempty_finite_parameter_vectors(params):
    def family(parameter):
        pytest.fail(
            "invalid sweep domains must be rejected before constructing a model"
        )

    with pytest.raises(ValueError, match="params"):
        pf.stability_map(family, params, [1])


def test_stability_sweep_accepts_numpy_integer_sizes_with_exact_symmetric_spectra():
    # For the regular symmetric NN path, the two-follower Laplacian has
    # eigenvalues 1 +/- 1/sqrt(2). With gx=-3, gv=-2, each temporal quadratic
    # lambda²+2*mu*lambda+3*mu has real part -mu for these two mu values.
    result = pf.stability_map(
        pf.velocity_asymmetry_family(-3, -2),
        np.array([0.0]),
        [np.int64(1), np.int32(2)],
        bc="regular",
        reliability=False,
    )
    assert result.Ns.dtype.kind in "iu"
    assert result.Ns.tolist() == [1, 2]
    assert result.params.tolist() == [0.0]
    assert result.max_re.shape == (2, 1)
    assert result.max_re[:, 0] == pytest.approx([-1, -1 + 1 / np.sqrt(2)], abs=2e-14)
