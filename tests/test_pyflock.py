"""Validation of flock matrices, Fourier dynamics and paper asymptotics."""

from itertools import pairwise

import numpy as np
import pytest
from scipy.linalg import expm

import pyflock as pf
from pyflock.cases import CASES


# --------------------------------------------------------------------- stencils
def test_all_cases_are_decentralized():
    for name, d in CASES.items():
        st = d["stencil"]
        sx, sv = st.row_sums
        assert abs(sx) < 1e-9, f"{name}: sum(rx) = {sx}"
        assert abs(sv) < 1e-9, f"{name}: sum(rv) = {sv}"


def test_moment_identities():
    """I_{x,1} = gx(bx1 + 2 bx2) and I_{x,2} = -gx(4 + 3 ax1) when r_0 = 1."""
    for d in CASES.values():
        st = d["stencil"]
        assert st.moment_x(1) == pytest.approx(st.gx * (st.bx1 + 2 * st.bx2))
        assert st.moment_v(1) == pytest.approx(st.gv * (st.bv1 + 2 * st.bv2))
        if abs(st.ax0 - 1.0) < 1e-12:
            assert st.moment_x(2) == pytest.approx(-st.gx * (4 + 3 * st.ax1))


def test_signal_speed_discriminant_matches_moments():
    """The squared half-difference of the two speeds is Iv1^2/4 + Ix2/2."""
    st = pf.case("type_I")
    assert st.a == pytest.approx(0.25 * st.moment_v(1) ** 2 + 0.5 * st.moment_x(2))


# ------------------------------------------------------------------- dispersion
def test_branches_solve_the_quadratic():
    st = pf.case("type_III")
    th = np.linspace(-np.pi, np.pi, 401)
    for lab in ("continuous", "by_imag"):
        lam = pf.branches(th, st, labelling=lab)
        res = lam**2 - pf.nu_v(th, st) * lam - pf.nu_x(th, st)
        assert np.max(np.abs(res)) < 1e-10


def test_continuous_labelling_is_conjugate_symmetric():
    st = pf.case("small_positive_loop")
    th = np.linspace(0, np.pi, 257)
    a = pf.branches(th, st, "continuous")
    b = pf.branches(-th, st, "continuous")
    assert np.max(np.abs(a - np.conj(b))) < 1e-10


def test_branches_match_circle_eigenvalues():
    st = pf.case("random_example")
    N = 48
    M = pf.system_matrix_circle(st, N)
    ev = np.linalg.eigvals(M)
    th = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / N) / N
    lam = pf.branches(th, st, "by_imag").ravel()
    rem = list(lam)
    worst = 0.0
    for e in ev:
        j = int(np.argmin([abs(e - r) for r in rem]))
        worst = max(worst, abs(e - rem[j]))
        rem.pop(j)
    assert worst < 1e-8


def test_branch_coefficients_match_moment_closed_forms():
    """Pair c_± with D_± from the quadratic through cubic order in theta."""
    for name in (
        "random_example",
        "two_positive_velocities",
        "small_positive_loop",
        "nonsymmetric_att1",
        "different_reflections_p1",
    ):
        st = pf.case(name)
        if st.a <= 0:
            continue
        for weights in ((st.rx, st.rv), (st.rx[::-1], st.rv[::-1])):
            reflected = pf.Stencil(*weights, st.gx, st.gv)
            offsets = np.arange(-2, 3)
            Iv1 = reflected.gv * np.dot(reflected.rv, offsets)
            Iv2 = reflected.gv * np.dot(reflected.rv, offsets**2)
            Ix2 = reflected.gx * np.dot(reflected.rx, offsets**2)
            Ix3 = reflected.gx * np.dot(reflected.rx, offsets**3)
            splitting = np.sqrt(Iv1**2 / 4 + Ix2 / 2)
            diffusion_split = (3 * Iv1 * Iv2 + 2 * Ix3) / (24 * splitting)
            expected = (
                (-Iv1 / 2 + splitting, Iv2 / 4 - diffusion_split),
                (-Iv1 / 2 - splitting, Iv2 / 4 + diffusion_split),
            )
            result = pf.coefficients(reflected)
            for branch, (speed, diffusion) in zip(("+", "-"), expected):
                assert result[branch]["c"] == pytest.approx(speed, abs=1e-12), name
                assert result[branch]["D"] == pytest.approx(diffusion, abs=1e-12), name


def test_nn_closed_forms():
    """NN coupling: c_±=-gv*q/2±sqrt(a), D_±=-gv/4±gv^2*q/(8*sqrt(a))."""
    for q in (-0.6, 0.0, 0.35, 0.8):
        gx, gv = -1.3, -0.9
        st = pf.nn_stencil(gx, gv, -0.5, (q - 1) / 2)
        a = gv**2 * q**2 / 4 - gx / 2
        assert st.a == pytest.approx(a)
        cp, cm = pf.signal_velocities(st)
        assert cp == pytest.approx(-gv * q / 2 + np.sqrt(a))
        assert cm == pytest.approx(-gv * q / 2 - np.sqrt(a))
        co = pf.coefficients(st)
        assert co["+"]["D"] == pytest.approx(
            -gv / 4 + gv**2 * q / (8 * np.sqrt(a)), abs=1e-8
        )
        assert co["-"]["D"] == pytest.approx(
            -gv / 4 - gv**2 * q / (8 * np.sqrt(a)), abs=1e-8
        )


def test_diffusion_is_positive_for_stable_systems():
    for q in np.linspace(-0.95, 0.95, 25):
        st = pf.nn_stencil(-1.0, -1.0, -0.5, (q - 1) / 2)
        co = pf.coefficients(st)
        assert co["+"]["D"] > 0 and co["-"]["D"] > 0


# --------------------------------------------------------------------- matrices
def test_line_matrix_drop_discards_missing_neighbours_and_preserves_row_sums():
    """Incomplete rows retain existing interactions and adjust their diagonal."""
    r = np.array([0.11, -0.37, 1.0, -0.53, 0.79])
    r = r - r.sum() / 5.0  # make it decentralized
    N = 9
    L = pf.laplacian_line(r, N, bc="drop")
    assert np.allclose(L[0], 0.0)  # leader row
    assert L[1, 0] == pytest.approx(r[1])
    assert L[1, 1] == pytest.approx(-(r[1] + r[3] + r[4]))
    assert L[1, 2] == pytest.approx(r[3])
    assert L[1, 3] == pytest.approx(r[4])
    assert L[N - 1, N - 3] == pytest.approx(r[0])
    assert L[N - 1, N - 2] == pytest.approx(r[1])
    assert L[N - 1, N - 1] == pytest.approx(-(r[0] + r[1] + r[3]))
    assert L[N - 1, N] == pytest.approx(r[3])
    assert L[N, N - 2] == pytest.approx(r[0])
    assert L[N, N - 1] == pytest.approx(r[1])
    assert L[N, N] == pytest.approx(-(r[0] + r[1]))
    for k in range(3, N - 1):  # interior
        assert np.allclose(L[k, k - 2 : k + 3], r)
    assert np.allclose(L[1:].sum(axis=1), 0.0)  # zero row sums


def test_line_matrix_fold_clamps_neighbours_and_normalises_terminal_row():
    r = np.array([0.11, -0.37, 1.0, -0.53, -0.21])
    r = r - r.sum() / 5.0
    N = 9
    L = pf.laplacian_line(r, N, bc="fold")
    assert L[1, 0] == pytest.approx(r[0] + r[1])
    assert L[1, 1] == pytest.approx(r[2])
    assert L[N - 1, N] == pytest.approx(r[3] + r[4])
    assert L[N - 1, N - 1] == pytest.approx(r[2])
    sx = abs(1.0 / (r[0] + r[1]))
    assert L[N, N - 2] == pytest.approx(sx * r[0])
    assert L[N, N - 1] == pytest.approx(sx * r[1])
    assert L[N, N] == pytest.approx(r[2])


def test_drop_equals_variable_mass_for_nearest_neighbour():
    """For NN coupling, 'drop' sets chi=-rho_{-1}, the variable-mass boundary."""
    st = pf.nn_stencil(-1.0, -1.0, -0.5, -0.3)
    N = 7
    L = pf.laplacian_line(st.rx, N, bc="drop")
    assert L[N, N - 1] == pytest.approx(st.rx[1])
    assert L[N, N] == pytest.approx(-st.rx[1])
    Lr = pf.laplacian_line(st.rx, N, bc="regular")
    assert Lr[N, N - 1] == pytest.approx(-1.0)
    assert Lr[N, N] == pytest.approx(1.0)


def test_regular_rejects_next_nearest_neighbour():
    with pytest.raises(ValueError):
        pf.laplacian_line(np.array([0.1, -0.3, 1.0, -0.4, -0.4]), 8, bc="regular")


def test_circle_laplacian_is_circulant():
    r = np.array([0.2, -0.5, 1.0, -0.9, 0.2])
    L = pf.laplacian_circle(r, 11)
    for k in range(11):
        assert np.allclose(np.roll(L[k], -k), np.roll(L[0], 0))
    assert np.allclose(L.sum(axis=1), r.sum())


# ----------------------------------------------------------------------- solve
def test_circle_fft_matches_expm():
    st = pf.case("random_example")
    N = 40
    rng = np.random.default_rng(0)
    z0, zd0 = rng.standard_normal(N), rng.standard_normal(N)
    times = np.array([0.0, 0.7, 3.0, 11.0])
    a = pf.simulate_circle(st, N, times, z0, zd0, method="fft")
    M = pf.system_matrix_circle(st, N)
    for i, t in enumerate(times):
        y = expm(M * t) @ np.concatenate([z0, zd0])
        assert np.max(np.abs(a.z[i] - y[:N])) < 1e-9
        assert np.max(np.abs(a.zdot[i] - y[N:])) < 1e-9


def test_line_expm_matches_runge_kutta():
    st = pf.nn_stencil(-1.0, -1.0, -0.5, -0.3)
    a = pf.simulate_line(st, N=30, tmax=40.0, nt=200, method="expm")
    b = pf.simulate_line(st, N=30, tmax=40.0, nt=200, method="rk")
    assert np.max(np.abs(a.z - b.z)) < 1e-7


def test_leader_moves_at_v0_and_others_start_at_rest():
    st = pf.nn_stencil(-1.0, -1.0, -0.5, -0.3)
    res = pf.simulate_line(st, N=20, tmax=10.0, nt=101, v0=2.0)
    assert np.allclose(res.z[:, 0], 2.0 * res.t, atol=1e-8)
    assert np.allclose(res.xrel[:, 0], 0.0)
    assert np.allclose(res.z[0], 0.0, atol=1e-12)


# -------------------------------------------------------------------- transients
def test_symmetric_flock_matches_theorem_3_to_leading_order():
    """|A_1| -> N v0/c_+ and gamma -> 1 for the symmetric nearest-neighbour flock."""
    st = pf.nn_stencil(-1.0, -1.0, -0.5, -0.5)
    cp, _ = pf.signal_velocities(st)
    for N, tol in ((100, 0.11), (400, 0.04)):
        res = pf.simulate_line(
            st, N, tmax=1.6 * N / cp, nt=int(40 * 1.6 * N / cp), bc="regular"
        )
        s = pf.transient_summary(res.t, res.last_relative, nmax=1)
        A1 = abs(s["A"][0])
        assert abs(A1 - N / cp) / A1 < tol
        assert A1 < N / cp  # diffusion always reduces the overshoot


def test_theory_amplitudes_are_self_consistent():
    st = pf.nn_stencil(-1.0, -1.0, -0.5, -0.3)
    th = pf.theory_amplitudes(st, N=100)
    assert th["c_plus"] > 0 > th["c_minus"]
    assert th["gamma"] == pytest.approx((th["c_minus"] / th["c_plus"]) ** 2)


# ------------------------------------------------------------------------ cases
def test_type_I_is_stable_on_the_circle():
    """'STABLE on CIRCLE' -- every eigenvalue bar the double zero is in the open left half."""
    st = pf.case("type_I")
    for N in (60, 120, 240):
        ev = np.linalg.eigvals(pf.system_matrix_circle(st, N))
        re = np.sort(ev.real)
        assert abs(re[-1]) < 1e-6 and abs(re[-2]) < 1e-6, f"N={N}: zero pair {re[-2:]}"
        assert re[-3] < 0.0, f"N={N}: third largest Re = {re[-3]}"


def test_type_I_is_asymptotically_unstable_on_the_line():
    """'UNSTABLE on LINE' means genuinely asymptotically unstable: max Re > 0.

    This is the sharpest circle/line discrepancy in the curated set -- the periodic and
    path problems disagree about ASYMPTOTIC stability, not merely about transient size.
    The unstable real part decays roughly like 1/N but stays strictly positive.
    """
    st = pf.case("type_I")
    worst = []
    for N in (20, 40, 60, 80):
        ev = np.linalg.eigvals(pf.system_matrix_line(st, N, bc="drop"))
        worst.append(ev.real.max())
    assert all(w > 0 for w in worst), worst
    assert all(a > b for a, b in pairwise(worst)), worst  # decreasing
    scaled = [w * N for w, N in zip(worst, (20, 40, 60, 80))]
    assert max(scaled) / min(scaled) < 2.0, scaled  # roughly ~ 1/N


# --------------------------------------------------------- eigenvalue reliability
def test_leaderless_spectrum_is_full_spectrum_minus_leader_double_zero():
    st = pf.nn_stencil(-1.0, -1.0, -0.5, -0.3)
    N = 12
    full = list(np.linalg.eigvals(pf.system_matrix_line(st, N)))
    red = np.linalg.eigvals(pf.system_matrix_line_leaderless(st, N))
    assert red.size == 2 * N
    for e in red:
        j = int(np.argmin([abs(e - f) for f in full]))
        assert abs(e - full[j]) < 1e-8
        full.pop(j)
    assert len(full) == 2 and max(abs(f) for f in full) < 1e-6


def test_eigen_reliability_separates_well_and_ill_conditioned_cases():
    good = pf.eigen_reliability(pf.system_matrix_line_leaderless(pf.case("type_I"), 60))
    assert good["reliable"] and good["kappa"] < 1e3 and good["eigenvalue"].real > 0
    bad = pf.eigen_reliability(
        pf.system_matrix_line_leaderless(pf.case("two_positive_velocities"), 60)
    )
    assert bad["kappa"] > 1e10 and not bad["reliable"]


# ------------------------------------------------------ least-stable eigenvalue on the line
def _line_max_re(st, N, bc):
    return np.linalg.eigvals(pf.system_matrix_line_leaderless(st, N, bc=bc)).real.max()


def test_least_stable_eigenvalue_wave_form_matches_numerics():
    N = 200
    for gx, gv in ((-1.0, -3.0), (-2.0, -1.0)):
        for bc in ("regular", "drop"):
            for q in (-0.4, 0.0, 0.3, 0.8):
                st = pf.nn_stencil(gx, gv, -0.5, (q - 1) / 2)
                assert pf.least_stable_eigenvalue(st, N) == pytest.approx(
                    _line_max_re(st, N, bc), rel=0.01
                ), (gx, gv, bc, q)


def test_least_stable_linear_form_is_the_small_asymmetry_limit():
    N = 300
    exact0 = -2.0 * np.pi**2 / (16 * N**2)
    st0 = pf.nn_stencil(-1.0, -2.0, -0.5, -0.5)
    assert pf.least_stable_eigenvalue(st0, N) == pytest.approx(exact0, rel=1e-5)
    assert pf.least_stable_eigenvalue(st0, N, method="linear") == pytest.approx(
        exact0, rel=1e-12
    )
    for q in (-0.01, 0.01):
        st = pf.nn_stencil(-1.0, -2.0, -0.5, (q - 1) / 2)
        assert pf.least_stable_eigenvalue(st, N) == pytest.approx(
            pf.least_stable_eigenvalue(st, N, method="linear"), rel=5e-3
        )
    d = pf.least_stable_eigenvalue(
        pf.nn_stencil(-1.0, -2.0, -0.5, -0.4), N, detail=True
    )
    assert d["total"] == pytest.approx(d["reflection_term"] + d["diffusion_term"])
    assert d["diffusion_term"] < 0.0


def test_least_stable_eigenvalue_rejects_stencils_it_does_not_cover():
    with pytest.raises(ValueError):  # asymmetric position coupling
        pf.least_stable_eigenvalue(pf.nn_stencil(-1.0, -1.0, -0.3, -0.3), 100)
    with pytest.raises(ValueError):  # next-nearest neighbour
        pf.least_stable_eigenvalue(pf.case("type_I"), 100)
    with pytest.raises(ValueError):  # unstable gains
        pf.least_stable_eigenvalue(pf.nn_stencil(1.0, -1.0, -0.5, -0.5), 100)
    with pytest.raises(ValueError):
        pf.least_stable_eigenvalue(
            pf.nn_stencil(-1.0, -1.0, -0.5, -0.5), 100, method="bogus"
        )


def test_critical_velocity_asymmetry_separates_stable_from_unstable():
    fam = pf.velocity_asymmetry_family(-1.0, -3.0)
    for N in (80, 240):
        qc = pf.critical_velocity_asymmetry(-1.0, -3.0, N)
        assert qc == pytest.approx(-(np.pi**2) / (8 * N), rel=0.05)
        assert _line_max_re(fam(qc - 0.005), N, "regular") > 0.0
        assert _line_max_re(fam(qc + 0.005), N, "regular") < 0.0


def test_stability_map_matches_theory_and_renders():
    fam = pf.velocity_asymmetry_family(-1.0, -1.0)
    qs = np.linspace(-0.1, 0.1, 9)
    sm = pf.stability_map(
        fam, qs, (40, 80), bc="regular", theory=pf.least_stable_eigenvalue
    )
    assert sm.max_re.shape == (2, 9) and sm.theory.shape == (2, 9)
    assert sm.reliable.all()
    assert np.allclose(sm.theory, sm.max_re, rtol=0.05, atol=1e-6)
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import io

    from pyflock import plotting as P

    P.stability_map_figure(sm).savefig(io.BytesIO(), format="png")
