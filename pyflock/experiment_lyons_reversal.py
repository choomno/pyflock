"""A constructed finite circle-unstable/path-stable heterogeneous flock.

This supplies the qualitative phenomenon of Lyons (2022), Figures 12--14,
using an explicitly searched repeated cell. It does not reconstruct the
unrecorded random arrays used in those figures. All finite spectra are complete.
"""

from __future__ import annotations

from itertools import product
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import (
    block_diag,
    eig,
    eigvalsh,
    expm,
    helmert,
    solve_continuous_lyapunov,
)
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .heterogeneous import HeterogeneousNN, system_matrix
from .style import CAT, plot_style


def _curvatures(gx, gv, beta):
    """Lyons (108), restricted to a two-agent cell (the ordering term is zero)."""
    a, b = 1 / np.asarray(gx), np.asarray(gv) / np.asarray(gx)
    m, v = np.mean(a), np.mean(b)
    variance, covariance = np.var(b), np.mean(a * b) - m * v
    radical = np.sqrt(2 * m + v * v * beta * beta)
    odd = (
        2 * variance * m * v**2 * beta**3
        - 2 * covariance * v**3 * beta**3
        - m * m * v * v * beta
        + 2 * variance * m * m * beta
        - 3 * m * v * covariance * beta
    )
    even = (
        -2 * m * variance * v * beta**2
        + 2 * v * v * covariance * beta**2
        + m * m * v
        + covariance * m
    )
    return np.array([even + odd / radical, even - odd / radical]) / (4 * m**3)


def _bloch_spectrum(gx, gv, beta, q):
    """Complete p=2 spectrum by quartic determinants, with exact coherent zeros.

    At phase zero det(lambda^2 I+lambda Gv Lv+Gx Lx) has the known factor
    lambda^2. Divide that factor algebraically, rather than deleting small
    computed roots. The other phases are ordinary quartics.
    """
    full, physical, phases = [], [], []
    for j in range(q):
        phase = 2 * np.pi * j / q
        em, ep = np.exp(-1j * phase), np.exp(1j * phase)
        diagonal = [np.array([gx[k], gv[k], 1], complex) for k in range(2)]
        off = [
            np.array([gx[0] * (1 + em) / 2, gv[0] * ((1 + beta) * em + 1 - beta) / 2]),
            np.array([gx[1] * (1 + ep) / 2, gv[1] * (1 + beta + (1 - beta) * ep) / 2]),
        ]
        coefficients = np.polynomial.polynomial.polymul(*diagonal)
        interaction = np.polynomial.polynomial.polymul(*off)
        coefficients[: len(interaction)] -= interaction
        if j == 0:
            if not np.allclose(coefficients[:2], 0, atol=1e-12):
                raise ArithmeticError("The coherent factor lambda^2 is missing")
            nonzero = np.roots(coefficients[2:][::-1])
            roots = np.r_[np.zeros(2, complex), nonzero]
            physical.extend(nonzero)
        else:
            roots = np.roots(coefficients[::-1])
            physical.extend(roots)
        full.extend(roots)
        phases.extend([phase] * 4)
    return np.array(full), np.array(physical), np.array(phases)


def _direct_operators(gx, gv, beta):
    """Literal component equations, independent of the sparse model builder."""
    n = len(gx)
    circle = np.zeros((2 * n, 2 * n))
    circle[:n, n:] = np.eye(n)
    nf = n - 1
    path = np.zeros((2 * nf, 2 * nf))
    path[:nf, nf:] = np.eye(nf)
    interior = [(-1, -0.5, -(1 + beta) / 2), (0, 1.0, 1.0), (1, -0.5, (beta - 1) / 2)]
    for k in range(n):
        for j, rx, rv in interior:
            circle[n + k, (k + j) % n] -= gx[k] * rx
            circle[n + k, n + (k + j) % n] -= gv[k] * rv
    for k in range(1, n):
        stencil = [(-1, -1.0, -1.0), (0, 1.0, 1.0)] if k == n - 1 else interior
        for j, rx, rv in stencil:
            if k + j == 0:
                continue
            path[nf + k - 1, k + j - 1] -= gx[k] * rx
            path[nf + k - 1, nf + k + j - 1] -= gv[k] * rv
    return circle, path


def _search_witness():
    """Screen the entire declared p=2 grid; stop at its first finite reversal."""
    grid = []
    for gx in product((1.0, 5.0), repeat=2):
        for gv in product((2.0, 18.0), repeat=2):
            for beta in (0.1, 0.3, 0.5, 0.7):
                grid.append([*gx, *gv, beta, *_curvatures(gx, gv, beta)])
    grid = np.asarray(grid)
    candidates = np.flatnonzero(np.min(grid[:, -2:], axis=1) < 0)
    trials = []
    for candidate in candidates:
        gx, gv, beta = grid[candidate, :2], grid[candidate, 2:4], grid[candidate, 4]
        for q in (2, 4, 8, 16, 32, 64, 128):
            _, physical, _ = _bloch_spectrum(gx, gv, beta, q)
            alpha = float(np.max(physical.real))
            path_alpha, path_computed = 0.0, False
            if alpha > 1e-8:
                model = HeterogeneousNN(np.tile(gx, q), np.tile(gv, q), beta)
                path = system_matrix(
                    model, geometry="path", followers_only=True
                ).toarray()
                path_alpha = float(np.max(np.linalg.eigvals(path).real))
                path_computed = True
            trials.append([candidate, q, 2 * q, alpha, path_alpha, path_computed])
            if path_computed and path_alpha < -1e-8:
                return grid, np.asarray(trials), int(candidate), q
    raise ArithmeticError("No reversal found in the declared finite grid")


def _eigenpairs(matrix):
    values, left, right = eig(matrix, left=True, right=True)
    residuals = np.linalg.norm(matrix @ right - right * values, axis=0)
    residuals /= (np.linalg.norm(matrix, np.inf) + abs(values)) * np.linalg.norm(
        right, axis=0
    )
    conditions = 1 / np.abs(np.sum(left.conj() * right, axis=0))
    return values, right, residuals, conditions


def compute(profile="quick", jobs=1, checkpoint=None):
    """Return the fixed N=64 witness, its full spectra and matched responses.

    ``jobs`` and ``checkpoint`` follow the shared exporter interface. This small
    experiment runs serially, without a cache. Both profiles use the same gains;
    paper merely doubles the response sampling density.
    """
    if profile not in ("quick", "paper"):
        raise ValueError("profile must be 'quick' or 'paper'")
    started = perf_counter()
    grid, trials, candidate, q = _search_witness()
    row = grid[candidate]
    gx, gv, beta = np.tile(row[:2], q), np.tile(row[2:4], q), float(row[4])
    n, nf = len(gx), len(gx) - 1
    model = HeterogeneousNN(gx, gv, beta)
    circle = system_matrix(model, geometry="circle").toarray()
    path = system_matrix(model, geometry="path", followers_only=True).toarray()
    direct_circle, direct_path = _direct_operators(gx, gv, beta)

    # The coherent two-dimensional space is invariant, with the Jordan block J.
    # Therefore Q.T A Q represents the quotient, although its orthogonal
    # complement need not itself be invariant.
    ones = np.ones((n, 1)) / np.sqrt(n)
    coherent = block_diag(ones, ones)
    q_agents = helmert(n, full=False).T
    quotient = block_diag(q_agents, q_agents)
    reduced_circle = quotient.T @ circle @ quotient
    cv, cr, ce, cc = _eigenpairs(circle)
    qv, qr, qe, qc = _eigenpairs(reduced_circle)
    pv, pr, pe, pc = _eigenpairs(path)
    bloch, physical_bloch, phases = _bloch_spectrum(row[:2], row[2:4], beta, q)
    bi, qi = linear_sum_assignment(abs(physical_bloch[:, None] - qv[None, :]))

    unstable = np.flatnonzero((cv.real > 1e-8) & (cv.imag > 0))
    lead = unstable[np.argmax(cv[unstable].real)]
    lam, mode = cv[lead], cr[:, lead]
    # Fix a reproducible phase of the two real quadratures, then choose the
    # representative with z0=v0=0. The same representative starts both systems.
    pivot = np.argmax(abs(mode))
    mode = mode * np.exp(-1j * np.angle(mode[pivot]))
    coherent_initial = np.r_[np.full(n, mode[0]), np.full(n, mode[n])]
    initial = mode - coherent_initial
    normalization = np.linalg.norm(quotient.T @ initial)
    mode, initial = mode / normalization, initial / normalization
    times = np.linspace(0, 3000.0, 1201 if profile == "paper" else 601)
    circle_states = np.exp(times[:, None] * lam) * mode
    circle_states[:, :n] -= mode[0] + times[:, None] * mode[n]
    circle_states[:, n:] -= mode[n]
    follower_initial = np.r_[initial[1:n], initial[n + 1 :]]
    path_followers = expm_multiply(
        csr_matrix(path),
        follower_initial,
        start=0.0,
        stop=times[-1],
        num=len(times),
        traceA=float(np.trace(path)),
    )
    path_states = np.zeros((len(times), 2 * n), complex)
    path_states[:, 1:n], path_states[:, n + 1 :] = (
        path_followers[:, :nf],
        path_followers[:, nf:],
    )
    circle_response = np.linalg.norm(circle_states @ quotient, axis=1)
    path_response = np.linalg.norm(path_states @ quotient, axis=1)

    lyapunov = solve_continuous_lyapunov(path.T, -np.eye(2 * nf))
    lyapunov = (lyapunov + lyapunov.T) / 2
    lyapunov_residual = path.T @ lyapunov + lyapunov @ path + np.eye(2 * nf)
    lyapunov_eigenvalues = eigvalsh(lyapunov)
    dissipation_eigenvalues = eigvalsh(np.eye(2 * nf) - lyapunov_residual)

    check_times = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 3000.0])
    circle_check = np.array([expm(direct_circle * t) @ initial for t in check_times])
    circle_formula = np.exp(check_times[:, None] * lam) * mode
    circle_formula[:, :n] -= mode[0] + check_times[:, None] * mode[n]
    circle_formula[:, n:] -= mode[n]
    quotient_check = np.array(
        [expm(reduced_circle * t) @ (quotient.T @ initial) for t in check_times]
    )
    path_check = np.array(
        [expm(direct_path * t) @ follower_initial for t in check_times]
    )
    sample_indices = np.searchsorted(times, check_times)
    aligned = np.isin(check_times, times)
    path_check_error = float(
        np.max(
            np.linalg.norm(
                path_check[aligned] - path_followers[sample_indices[aligned]], axis=1
            )
        )
    )
    short_times = np.linspace(0.0, 10.0, 21)
    ode_errors = []
    ode_evaluations = []
    for matrix, state in [(direct_circle, initial), (direct_path, follower_initial)]:
        solve = solve_ivp(
            lambda t, x, matrix=matrix: matrix @ x,
            (0.0, 10.0),
            state,
            method="DOP853",
            t_eval=short_times,
            rtol=2e-12,
            atol=2e-14,
        )
        if not solve.success:
            raise ArithmeticError(solve.message)
        exact = np.array([expm(matrix * t) @ state for t in short_times])
        ode_errors.append(float(np.max(np.linalg.norm(solve.y.T - exact, axis=1))))
        ode_evaluations.append(solve.nfev)

    alpha_circle, alpha_path = float(max(qv.real)), float(max(pv.real))
    classification = {
        "circle_physical_abscissa": alpha_circle,
        "path_follower_abscissa": alpha_path,
        "circle_unstable_physical_count": int(np.count_nonzero(qv.real > 1e-8)),
        "coherent_modes": 2,
        "circle_leading_eigenvalue": [float(lam.real), float(lam.imag)],
        "circle_max_eigenpair_backward_residual": float(max(ce)),
        "circle_quotient_max_eigenpair_backward_residual": float(max(qe)),
        "path_max_eigenpair_backward_residual": float(max(pe)),
        "bloch_vs_quotient_max_matching_error": float(
            max(abs(physical_bloch[bi] - qv[qi]))
        ),
        "lyapunov_min_eigenvalue": float(lyapunov_eigenvalues[0]),
        "lyapunov_residual_spectral_norm": float(np.linalg.norm(lyapunov_residual, 2)),
        "lyapunov_dissipation_min_eigenvalue": float(dissipation_eigenvalues[0]),
    }
    data = {
        "gx": gx,
        "gv": gv,
        "beta": np.array(beta),
        "circle_matrix": circle,
        "path_matrix": path,
        "coherent_basis": coherent,
        "quotient_basis": quotient,
        "circle_quotient_matrix": reduced_circle,
        "circle_eigenvalues": cv,
        "circle_eigenvectors": cr,
        "circle_residuals": ce,
        "circle_eigenvalue_conditions": cc,
        "circle_quotient_eigenvalues": qv,
        "circle_quotient_eigenvectors": qr,
        "circle_quotient_residuals": qe,
        "circle_quotient_eigenvalue_conditions": qc,
        "path_eigenvalues": pv,
        "path_eigenvectors": pr,
        "path_residuals": pe,
        "path_eigenvalue_conditions": pc,
        "bloch_eigenvalues": bloch,
        "bloch_physical_eigenvalues": physical_bloch,
        "bloch_phases": phases,
        "initial_displacement": initial[:n],
        "initial_velocity": initial[n:],
        "times": times,
        "circle_states": circle_states,
        "path_states": path_states,
        "circle_response": circle_response,
        "path_response": path_response,
        "lyapunov_matrix": lyapunov,
        "lyapunov_residual_matrix": lyapunov_residual,
        "lyapunov_eigenvalues": lyapunov_eigenvalues,
        "lyapunov_dissipation_eigenvalues": dissipation_eigenvalues,
        "search_grid": grid,
        "search_trials": trials,
        "check_times": check_times,
        "circle_check_states": circle_check,
        "circle_quotient_check_states": quotient_check,
        "path_check_states": path_check,
    }
    meta = {
        "experiment": "lyons_stability_reversal",
        "profile": profile,
        "N_total": n,
        "cell_size": 2,
        "repetitions": q,
        "beta_v": beta,
        "gx_cell": row[:2].tolist(),
        "gv_cell": row[2:4].tolist(),
        "source": {
            "author": "Robert G. Lyons",
            "year": 2022,
            "title": "Linear Nearest Neighbor Flocks With All Distinct Agents",
            "doi": "10.15760/etd.7774",
            "figures": [12, 13, 14],
            "sections": "2.8; equation (108) provides the candidate screening criterion",
        },
        "relation_to_source": "New constructed finite example of the circle-unstable/path-stable phenomenon. Original random arrays and seeds are unavailable. The saved fixed-seed thesis-family comparisons are separate experiments.",
        "boundary": "Circle: indices modulo N. Path: agent0 fixed; regular last row uses z_tail-z_previous in both position and velocity interactions. N counts all agents.",
        "gain_convention": "Positive gain magnitudes; zddot=-diag(gx)Lx z-diag(gv)Lv zdot. Position weights (-1/2,1,-1/2); velocity weights (-(1+beta)/2,1,(beta-1)/2).",
        "search": {
            "purpose": "Existence witness; deterministic screening, not an ensemble statistic.",
            "grid": {
                "cell_size": 2,
                "gx_choices": [1, 5],
                "gv_choices": [2, 18],
                "beta_values": [0.1, 0.3, 0.5, 0.7],
                "repetitions": [2, 4, 8, 16, 32, 64, 128],
            },
            "order": "Lexicographic gx cell, then gv cell, then beta. Screen all 64 parameter cases by min(c2_plus,c2_minus)<0. For each retained candidate in order, test q in the stated increasing sequence; compute the full follower path spectrum only if the full Bloch circle spectrum has Re(lambda)>1e-8. Stop at the first path abscissa<-1e-8.",
            "parameter_cases_screened": len(grid),
            "negative_curvature_candidates": int(
                np.count_nonzero(np.min(grid[:, -2:], axis=1) < 0)
            ),
            "selected_grid_row": candidate,
            "complete_circle_spectra_evaluated": len(trials),
            "complete_path_spectra_evaluated": int(sum(trials[:, 5])),
            "grid_columns": ["gx0", "gx1", "gv0", "gv1", "beta", "c2_plus", "c2_minus"],
            "trial_columns": [
                "grid_row",
                "q",
                "N_total",
                "circle_physical_abscissa",
                "path_abscissa",
                "path_computed",
            ],
            "missing_path_values": "Finite zero placeholders where path_computed=0; these are not observations.",
            "curvature": "Screening predicts only the sign near Bloch phase0. The N64 unstable root is verified by the exact finite Bloch spectrum; its growth rate is not approximated by the quadratic curvature.",
        },
        "spectrum": "Complete dense circle, coherent quotient and follower path spectra; independent exact p2 Bloch quartics. The invariant coherent position/velocity space is removed by an orthogonal quotient, and the phase0 Bloch determinant is divided by its algebraic lambda^2 factor. No small numerical eigenvalues are discarded for classification.",
        "classification": classification,
        "initial_data": "Take the positive-imaginary member of the leading unstable circle eigenpair; rotate its largest component to real positive, subtract its agent0 position and velocity uniformly, then normalize its orthogonal coherent quotient to norm1. Save that same displacement/velocity on every agent in both geometries. The path leader is fixed at z0=v0=0.",
        "observable": "||Q.T [z(t),zdot(t)]||_2 / ||Q.T [z(0),zdot(0)]||_2, where Q spans the orthogonal complement of uniform position and velocity. Equivalently a normalized translation-free RMS state norm, with one model time unit weighting velocity. Complex states encode two real initial conditions; their squared norms add. This is a response to that saved state, not an operator norm.",
        "propagation": "Circle: exact unstable eigenmode plus the subtracted coherent affine solution. Path: sparse exponential action. Independent literal component matrices, dense exponentials at saved checkpoints and short DOP853 solves check both propagations.",
        "validation": {
            "direct_circle_matrix_max_error": float(
                np.max(abs(circle - direct_circle))
            ),
            "direct_path_matrix_max_error": float(np.max(abs(path - direct_path))),
            "coherent_invariance_residual": float(
                np.linalg.norm(
                    circle @ coherent - coherent @ np.array([[0.0, 1.0], [0.0, 0.0]])
                )
            ),
            "quotient_orthogonality_residual": float(
                np.linalg.norm(quotient.T @ quotient - np.eye(2 * nf))
            ),
            "circle_dense_exponential_max_state_error": float(
                np.max(np.linalg.norm(circle_check - circle_formula, axis=1))
            ),
            "circle_quotient_dense_exponential_max_state_error": float(
                np.max(
                    np.linalg.norm(quotient_check - circle_formula @ quotient, axis=1)
                )
            ),
            "path_dense_exponential_max_state_error": path_check_error,
            "short_ode_max_state_errors_circle_path": ode_errors,
            "short_ode_function_evaluations_circle_path": ode_evaluations,
            "short_ode_rtol": 2e-12,
            "short_ode_atol": 2e-14,
            "lyapunov_interpretation": "Numerical positive-definite P with positive-definite -(A.T P+P A); residual and all eigenvalues are saved. This is an independent numerical stability check, without interval arithmetic.",
        },
        "ensemble_size": 1,
        "random_seed": None,
        "computation_seconds": perf_counter() - started,
    }
    return data, meta


def draw(data, meta):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    circle, path = data["bloch_eigenvalues"], data["path_eigenvalues"]
    # Locate the largest empty real interval in the saved complete spectra.
    real_parts = np.sort(np.r_[circle.real, path.real])
    gap_index = int(np.argmax(np.diff(real_parts)))
    left_edge, right_edge = real_parts[gap_index : gap_index + 2]
    left_span = max(left_edge - real_parts[0], 0.1)
    right_span = max(real_parts[-1] - right_edge, 0.1)
    left_limits = (real_parts[0] - 0.06 * left_span, left_edge + 0.06 * left_span)
    right_limits = (right_edge - 0.07 * right_span, real_parts[-1] + 0.07 * right_span)
    if left_limits[1] >= right_limits[0]:
        raise ValueError("The saved spectrum has no separated real cluster to split")
    omitted = (left_limits[1], right_limits[0])

    with plot_style():
        fig = plt.figure(figsize=(11.4, 8.2), layout="constrained")
        outer = fig.add_gridspec(3, 1, height_ratios=(0.12, 1, 1))
        legend_ax = fig.add_subplot(outer[0])
        legend_ax.set_axis_off()
        top = outer[1].subgridspec(1, 2, width_ratios=(1, 3.1), wspace=0.035)
        left = fig.add_subplot(top[0, 0])
        bulk = fig.add_subplot(top[0, 1], sharey=left)
        bottom = outer[2].subgridspec(1, 2, width_ratios=(1, 1), wspace=0.12)
        detail = fig.add_subplot(bottom[0, 0])
        response = fig.add_subplot(bottom[0, 1])

        def spectrum(ax):
            ax.axvline(0, color="0.5", lw=0.65, zorder=0)
            ax.axhline(0, color="0.75", lw=0.5, zorder=0)
            ax.scatter(
                circle.real,
                circle.imag,
                s=14,
                color=CAT[2],
                marker="o",
                label=f"Circle ({len(circle)} roots)",
                alpha=0.8,
            )
            ax.scatter(
                path.real,
                path.imag,
                s=17,
                facecolors="none",
                edgecolors=CAT[1],
                marker="s",
                lw=0.75,
                label=f"Path followers ({len(path)} roots)",
            )
            ax.plot(0, 0, "+", color="0.2", ms=7, mew=1.0, label="2 coherent zeros")
            ax.set(xlabel=r"$\operatorname{Re}\lambda$")
            ax.grid(True, alpha=0.25)

        for ax in (left, bulk, detail):
            spectrum(ax)
        left.set(
            xlim=left_limits,
            title="Distant real cluster",
            ylabel=r"$\operatorname{Im}\lambda$",
        )
        bulk.set(xlim=right_limits, title="Complete spectra: near-origin bulk")
        extent = 1.12 * max(abs(circle.imag).max(), abs(path.imag).max())
        left.set_ylim(-extent, extent)
        left.xaxis.set_major_locator(MaxNLocator(3))
        bulk.xaxis.set_major_locator(MaxNLocator(6))
        bulk.tick_params(axis="y", left=False, labelleft=False)
        bulk.spines["left"].set_visible(False)
        # Paired slashes mark the discontinuous top real axis at both edges.
        for ax, x in ((left, 1), (bulk, 0)):
            for y in (0, 1):
                ax.plot(
                    [x - 0.012, x + 0.012],
                    [y - 0.022, y + 0.022],
                    transform=ax.transAxes,
                    color=".3",
                    lw=0.8,
                    clip_on=False,
                )
        handles, labels = bulk.get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center", ncol=3, fontsize=9)

        detail.set(
            title="Near the imaginary axis",
            xlim=(-0.016, 0.0025),
            ylim=(-0.17, 0.17),
            ylabel=r"$\operatorname{Im}\lambda$",
        )
        detail.ticklabel_format(
            axis="x", style="sci", scilimits=(-2, -2), useMathText=True
        )
        detail.text(
            0.03,
            0.97,
            "Unstable circle pair",
            transform=detail.transAxes,
            va="top",
            color=CAT[2],
            fontsize=8.5,
        )
        response.semilogy(
            data["times"], data["circle_response"], color=CAT[2], label="Circle"
        )
        response.semilogy(
            data["times"],
            data["path_response"],
            color=CAT[1],
            label="Path, same initial state",
        )
        response.set(
            title="Matched mode response",
            xlabel="Time",
            ylabel="Normalized translation-free state norm",
        )
        response.legend(fontsize=8)
        response.grid(True, alpha=0.3)
        fig.suptitle(
            r"A constructed circle instability with a stable path: $N=64$, $\beta_v=0.1$",
            fontsize=12,
        )
        fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
        fig.text(
            0.5,
            0.103,
            f"Top real axis omits the empty interval ({omitted[0]:.2f}, {omitted[1]:.2f}); every eigenvalue is retained.",
            ha="center",
            fontsize=8.5,
            color=".3",
        )
        fig.text(
            0.5,
            0.062,
            r"Cell $g_x=(1,1)$, $g_v=(2,18)$ repeated 32 times; fixed path leader and regular tail.",
            ha="center",
            fontsize=8.5,
            color=".3",
        )
        fig.text(
            0.5,
            0.022,
            "Two real quadratures of one circle mode; uniform position and velocity removed from the norm. New constructed example.",
            ha="center",
            fontsize=8,
            color=".35",
        )
        return fig


EXPERIMENTS = {"lyons_stability_reversal": {"compute": compute, "draw": draw}}
