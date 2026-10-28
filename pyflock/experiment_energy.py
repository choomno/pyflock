"""Force-input energy and a controlled periodic viscous-wave limit.

Herman–Knorn–Ahlén (2017), Automatica 82, equations (3)–(9), Lemma 6.
Lyons (2022), Appendix H, equations (191)–(200), symmetric specialization.
These experiments use explicitly stated observables and do not establish string
stability or the validity of the separate leader-step extrema approximation.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm, solve_continuous_lyapunov, svdvals
from scipy.sparse import bmat, csr_matrix, eye
from scipy.sparse.linalg import expm_multiply

from .style import CAT, plot_style


def damping(n, hp, r=1.0):
    """Paper (3),(4): leader fixed, last follower has forward error only."""
    if n < 1 or int(n) != n or hp < 0 or r <= 0:
        raise ValueError("Require integer n>=1, hp>=0, r>0")
    b = np.eye(n) - np.eye(n, k=1)
    return r * (b + hp * np.abs(b)) @ b.T


def energy_counterexample(n=32, hp=0.5):
    """A unit vector violating the lower bound used in source equation (8)."""
    d = damping(n, hp)
    sigma = float(svdvals(d)[-1])
    ev, vec = eigh((d + d.T) / 2, subset_by_index=[0, 0])
    y = vec[:, 0]
    if y[0] < 0:
        y = -y
    rayleigh = float(y @ d @ y)
    return {
        "N": n,
        "hp": hp,
        "r": 1.0,
        "vector": y.tolist(),
        "norm": float(np.linalg.norm(y)),
        "sigma_min": sigma,
        "symmetric_part_lambda_min": float(ev[0]),
        "y_D_y": rayleigh,
        "lower_bound_residual": rayleigh - sigma,
        "Hdot_at_d_zero": -rayleigh,
        "claimed_Hdot_upper_bound": -sigma,
        "claimed_Hdot_violation": sigma - rayleigh,
        "D_definition": "B=I-superdiag(1); D=(B+hp*abs(B)) @ B.T; M=A=R=I",
        "state": "p=y, Delta=0, v0=0, external force d=0; Hdot=-y.T D y",
        "construction": "Unit eigenvector of (D+D.T)/2 for its smallest eigenvalue",
        "source": "Herman, Knorn & Ahlen (2017), equations (3),(4),(5),(8)",
        "conclusion": "Equation (8)'s quadratic-form inequality fails. This does not itself falsify the integrated bound (7) or Lemma 6.",
    }


def force_response(n, hp, *, duration=2.0, end=None, nt=1001):
    """Zero-state response to e_j / sqrt(duration) on [0,duration).

    The spatial Euclidean norm and time L2 norm of the force together give
    integral ||d(t)||_2^2 dt = 1. States are x and v, leader x0=v0=0.
    No rescaling of the last row is performed.
    """
    if duration <= 0 or nt < 2:
        raise ValueError("Require duration>0 and nt>=2")
    end = 3.0 * n if end is None else float(end)
    if end <= duration:
        raise ValueError("end must exceed pulse duration")
    b = np.eye(n) - np.eye(n, k=1)
    k = b @ b.T
    d = damping(n, hp)
    a = bmat([[None, eye(n)], [-csr_matrix(k), -csr_matrix(d)]], format="csr")
    j = n // 2
    f = np.zeros(2 * n)
    f[n + j] = 1 / np.sqrt(duration)
    # An augmented constant state integrates the constant input exactly. Switch
    # at the exact pulse endpoint, then propagate the homogeneous system.
    aug = bmat([[a, csr_matrix(f[:, None])], [None, csr_matrix((1, 1))]], format="csr")
    init = np.zeros(2 * n + 1)
    init[-1] = 1
    t_on = np.linspace(0, duration, max(65, nt // 8))
    on = expm_multiply(aug, init, start=0, stop=duration, num=len(t_on))[:, :-1]
    t_off = np.linspace(duration, end, nt)
    off = expm_multiply(a, on[-1], start=0, stop=end - duration, num=nt)
    t = np.r_[t_on, t_off[1:]]
    state = np.vstack([on, off[1:]])
    x, v = state[:, :n], state[:, n:]
    spacing = -x @ b
    energy = (np.sum(v * v, axis=1) + np.sum(spacing * spacing, axis=1)) / 2
    return {
        "t": t,
        "x": x,
        "v": v,
        "energy": energy,
        "input_energy": np.minimum(t / duration, 1.0),
    }, j + 1


def finite_energy_witness(n=128, hp=0.5, horizon=512.0, nt=2049):
    """Construct a distributed unit-L2 force and independently integrate it.

    For Q=C.T C, the finite controllability Gramian WT gives the exact
    zero-state L2-to-final-energy optimum lambda_max(C WT C.T)/2. This is
    distinct from an H-infinity gain. The final response is checked by two
    DOP853 solves, without using the Gramian to propagate the forced state.
    """
    b = np.eye(n) - np.eye(n, k=1)
    k = b @ b.T
    d = damping(n, hp)
    a = np.block([[np.zeros((n, n)), np.eye(n)], [-k, -d]])
    f = np.r_[np.zeros((n, n)), np.eye(n)]
    c = np.block([[b.T, np.zeros((n, n))], [np.zeros((n, n)), np.eye(n)]])
    w = solve_continuous_lyapunov(a, -f @ f.T)
    w = (w + w.T) / 2
    propagator = expm(a * horizon)
    wt = w - propagator @ w @ propagator.T
    wt = (wt + wt.T) / 2
    eigenvalues, vectors = eigh(c @ wt @ c.T, subset_by_index=[2 * n - 1, 2 * n - 1])
    mu = float(eigenvalues[0])
    q = vectors[:, 0]
    if q[0] < 0:
        q = -q
    phi = c.T @ q
    expected_final = wt @ phi / np.sqrt(mu)
    source_coefficient = float(1 / (2 * svdvals(d)[-1]))
    sparse_a = csr_matrix(a)
    rtol, atol = 2e-11, 2e-13
    adjoint = solve_ivp(
        lambda t, p: sparse_a.T @ p,
        (0, horizon),
        phi,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not adjoint.success:
        raise RuntimeError(adjoint.message)

    def rhs(t, state):
        force = adjoint.sol(horizon - t)[n:] / np.sqrt(mu)
        return np.r_[
            sparse_a @ state[: 2 * n] + np.r_[np.zeros(n), force], force @ force
        ]

    response = solve_ivp(
        rhs,
        (0, horizon),
        np.zeros(2 * n + 1),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not response.success:
        raise RuntimeError(response.message)
    t = np.linspace(0, horizon, nt)
    saved = response.sol(t).T
    state = saved[:, : 2 * n]
    forces = adjoint.sol(horizon - t)[n:].T / np.sqrt(mu)
    energy = np.sum((state @ c.T) ** 2, axis=1) / 2
    input_energy = saved[:, -1]
    final = state[-1]
    rhs_norm = np.linalg.norm(f @ f.T)
    metrics = {
        "N": n,
        "hp": hp,
        "T": horizon,
        "samples": nt,
        "mu_T": mu,
        "source_bound_coefficient": source_coefficient,
        "predicted_final_energy": mu / 2,
        "independently_integrated_final_energy": float(energy[-1]),
        "independently_integrated_force_L2_squared": float(input_energy[-1]),
        "source_bound_with_measured_input": float(
            source_coefficient * input_energy[-1]
        ),
        "violation": float(energy[-1] - source_coefficient * input_energy[-1]),
        "violation_ratio": float(energy[-1] / (source_coefficient * input_energy[-1])),
        "relative_final_state_error": float(
            np.linalg.norm(final - expected_final) / np.linalg.norm(expected_final)
        ),
        "lyapunov_relative_residual": float(
            np.linalg.norm(a @ w + w @ a.T + f @ f.T) / rhs_norm
        ),
        "finite_gramian_relative_residual": float(
            np.linalg.norm(
                a @ wt + wt @ a.T + f @ f.T - propagator @ f @ f.T @ propagator.T
            )
            / rhs_norm
        ),
        "adjoint_endpoint_relative_error": float(
            np.linalg.norm(adjoint.y[:, -1] - propagator.T @ phi)
            / np.linalg.norm(propagator.T @ phi)
        ),
        "adjoint_function_evaluations": adjoint.nfev,
        "forward_function_evaluations": response.nfev,
        "ode_rtol": rtol,
        "ode_atol": atol,
        "parameters": "M=A_stiffness=R=I; h_delta=0; stationary leader; forward-only final follower.",
        "matrices": "B=I-superdiag(1); K=B B.T; D=(B+hp abs(B)) B.T; A=[[0,I],[-K,-D]]; F=[0;I]; C=diag(B.T,I); Q=C.T C.",
        "construction": "W solves A W+W A.T+F F.T=0; WT=W-exp(T A) W exp(T A.T); q is the unit top eigenvector of C WT C.T, eigenvalue mu_T; phi=C.T q. The distributed force is d(t)=F.T exp((T-t)A.T) phi/sqrt(mu_T) for 0<=t<=T and zero otherwise. Its exact squared L2 norm is one; H(T)=mu_T/2.",
        "independent_check": "DOP853 first integrates p_prime=A.T p, p(0)=phi, then the forward forced state with d(t)=F.T p(T-t)/sqrt(mu_T), plus J_prime=||d(t)||^2. Gramian matrices are not used to propagate either ODE.",
        "source_assumptions": "Automatica 2017 equations (1)-(5), Theorem 4 and equation (7). Unit positive homogeneous m,a,r satisfy Assumption 2 and r_i>=r_(i+1) in Assumption 3. Zero initial state. Arbitrary distributed L2 external forces are allowed by (1).",
        "interpretation": "Finite-time numerical counterexample to Eq. (7) under the stated homogeneous assumptions. This is a distributed force, distinct from the single-site pulse. It does not refute Lemma 6 singular-value scaling or asymptotic stability. This is not an H-infinity calculation.",
    }
    arrays = {
        "t": t,
        "force": forces,
        "x": state[:, :n],
        "v": state[:, n:],
        "energy": energy,
        "input_energy": input_energy,
        "q": q,
        "phi": phi,
        "gramian_W": w,
        "gramian_WT": wt,
        "expected_final_state": expected_final,
    }
    return arrays, metrics


def compute_energy(profile="quick", jobs=1, checkpoint=None):
    if profile not in ("quick", "paper"):
        raise ValueError("profile must be quick or paper")
    ns = np.array(
        [16, 32, 64, 128, 256] if profile == "quick" else [16, 32, 64, 128, 256, 512]
    )
    hps = np.array([0.0, 0.5])
    sigmas = np.array([[svdvals(damping(int(n), hp))[-1] for n in ns] for hp in hps])
    coercivity = np.array(
        [
            [
                eigh(
                    (damping(int(n), hp) + damping(int(n), hp).T) / 2,
                    subset_by_index=[0, 0],
                    eigvals_only=True,
                )[0]
                for n in ns
            ]
            for hp in hps
        ]
    )
    n = 64 if profile == "quick" else 128
    arrays = {"N": ns, "hp": hps, "sigma_min": sigmas, "coercivity": coercivity}
    bounds, peaks = [], []
    for i, hp in enumerate(hps):
        res, j = force_response(n, hp, nt=1601 if profile == "quick" else 3201)
        arrays.update({f"{i}_{key}": value for key, value in res.items()})
        d = damping(n, hp)
        alpha = eigh((d + d.T) / 2, subset_by_index=[0, 0], eigvals_only=True)[0]
        bounds.append([1 / (2 * svdvals(d)[-1]), 1 / (2 * alpha)])
        peaks.append(float(res["energy"].max()))
    arrays["bounds"] = np.array(bounds)
    arrays["peaks"] = np.array(peaks)
    counter = energy_counterexample()
    arrays["counterexample_vector"] = np.array(counter["vector"])
    witness_arrays, witness_meta = finite_energy_witness()
    arrays.update({f"witness_{key}": value for key, value in witness_arrays.items()})
    meta = {
        "experiment": "energy",
        "schema": 1,
        "profile": profile,
        "question": "How do damping singular values and force-driven total energy scale?",
        "source": "Herman, Knorn & Ahlen (2017), Automatica 82, 13–20; equations (1)–(9), Lemma 6 (11)–(12).",
        "source_doi": "10.1016/j.automatica.2017.04.010",
        "parameters": {
            "m": 1.0,
            "a": 1.0,
            "r": 1.0,
            "h_delta": 0.0,
            "hp": hps.tolist(),
            "response_N": n,
        },
        "geometry": "N followers; stationary prescribed leader; last follower considers forward error only; no row renormalization.",
        "initial_data": "x=v=0 in equilibrium displacement coordinates",
        "input": f"Single-site pulse: external force e_{j}/sqrt(2) for 0<=t<2; zero thereafter; integral_0^infinity ||d(t)||_2^2 dt=1.",
        "output": "H(t)=(||v(t)||_2^2+||B.T x(t)||_2^2)/2, total kinetic plus spacing energy",
        "solver": "Single-site pulse: sparse matrix exponential action with exact split at t=2. Distributed witness: finite controllability Gramian plus independent adjoint/forward DOP853 integration.",
        "bounds": "Source (7): 1/(2 sigma_min(D)); independently justified: 1/(2 lambda_min((D+D.T)/2)). Both multiply cumulative input energy.",
        "mathematical_caveat": "Equation (8) has a quadratic-form counterexample; the separate finite-time distributed-force witness numerically violates Eq. (7). Lemma 6 is not refuted.",
        "witness": witness_meta,
        "counterexample": {k: v for k, v in counter.items() if k != "vector"},
        "singular_value_fitted_slopes": [
            float(np.polyfit(np.log(ns[-3:]), np.log(row[-3:]), 1)[0]) for row in sigmas
        ],
        "interpretation": "The singular-value scaling and fixed single-site pulse are separate from the finite-time optimized distributed-force witness. Neither forcing is a leader-velocity step.",
        "parallelism": "jobs and checkpoint accepted for dispatcher compatibility; deterministic small serial computation.",
    }
    return arrays, meta


def draw_energy(data, meta):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    with plot_style():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), layout="constrained")
        for i, hp in enumerate(data["hp"]):
            label = rf"$h_p={hp:g}$"
            axes[0, 0].loglog(
                data["N"], data["sigma_min"][i], "o-", color=CAT[i + 1], label=label
            )
            axes[0, 1].plot(
                data[f"{i}_t"] / meta["parameters"]["response_N"],
                data[f"{i}_energy"],
                color=CAT[i + 1],
                label=label,
            )
            axes[1, 0].loglog(
                data["N"],
                data["sigma_min"][i],
                "o-",
                color=CAT[i + 1],
                label=label + r", $\sigma_{\min}(D)$",
            )
            axes[1, 0].loglog(
                data["N"],
                data["coercivity"][i],
                "s--",
                color=CAT[i + 1],
                label=label + r", $\lambda_{\min}(S)$",
            )
        ns = data["N"].astype(float)
        for power, row in [(-2, 0), (-1, 1)]:
            axes[0, 0].loglog(
                ns,
                data["sigma_min"][row, -1] * (ns / ns[-1]) ** power,
                ":",
                color="0.45",
                label=rf"$N^{{{power}}}$ reference",
            )
        axes[0, 0].set(
            title="Damping singular-value scaling",
            xlabel="Followers N",
            ylabel=r"$\sigma_{\min}(D)$",
        )
        axes[0, 1].set(
            title="Unit L² interior force pulse",
            xlabel="Time / N",
            ylabel="Total energy H(t)",
        )
        axes[1, 0].set(
            title="Singular value and quadratic-form coercivity",
            xlabel="Followers N",
            ylabel="Smallest value",
        )
        witness = meta["witness"]
        normalized_t = data["witness_t"] / witness["T"]
        normalized_energy = data["witness_energy"] / witness["source_bound_coefficient"]
        input_energy = data["witness_input_energy"]
        axes[1, 1].plot(
            normalized_t,
            normalized_energy,
            color=CAT[2],
            label=r"$H(t)/C_{\rm source}$",
        )
        axes[1, 1].plot(
            normalized_t,
            input_energy,
            "--",
            color=CAT[1],
            label=r"$\int_0^t \|d(s)\|_2^2\,ds$",
        )
        axes[1, 1].fill_between(
            normalized_t,
            input_energy,
            normalized_energy,
            where=normalized_energy > input_energy,
            color=CAT[2],
            alpha=0.15,
        )
        excess = 100 * (witness["violation_ratio"] - 1)
        axes[1, 1].annotate(
            f"Final excess: {excess:.2f}%",
            xy=(1, normalized_energy[-1]),
            xytext=(0.43, 1.12),
            arrowprops={"arrowstyle": "->", "color": "0.4"},
            fontsize=9,
        )
        axes[1, 1].set(
            xlabel="Time / T (T=512)",
            ylabel="Normalized energy",
            title=r"Distributed force: $N=128,\ h_p=0.5$",
            ylim=(0, 1.25),
        )
        for ax in axes[:, 0]:
            ax.set_xticks(data["N"], [str(n) for n in data["N"]])
            ax.xaxis.set_minor_formatter(NullFormatter())
        for ax in axes.flat:
            ax.legend(fontsize=8)
        fig.suptitle("Force disturbances: symmetric position coupling", fontsize=14)
        fig.supxlabel(
            r"$S=(D+D^T)/2$, $C_{\rm source}=1/(2\sigma_{\min}(D))$. Eq. (7) requires normalized energy ≤ accumulated input energy."
            "\nTop right: single-site pulse. Bottom right: constructed distributed force; final excess 7.26%.",
            fontsize=9,
        )
        return fig


def mode_amplitude(k2, t, c2=1.0, nu=0.1):
    """Underdamped scalar mode with displacement 1, velocity 0."""
    decay = nu * k2 / 2
    omega = np.sqrt(c2 * k2 - decay**2)
    return np.exp(-decay * t) * (np.cos(omega * t) + decay / omega * np.sin(omega * t))


def compute_continuum(profile="quick", jobs=1, checkpoint=None):
    if profile not in ("quick", "paper"):
        raise ValueError("profile must be quick or paper")
    ns = np.array(
        [16, 32, 64, 128, 256] if profile == "quick" else [16, 32, 64, 128, 256, 512]
    )
    length = 2 * np.pi
    mode = 1
    k = 2 * np.pi * mode / length
    h = length / ns
    k2h = (2 * np.sin(k * h / 2) / h) ** 2
    t = np.linspace(0, 8, 1601)
    continuum = mode_amplitude(k * k, t)
    discrete = np.array([mode_amplitude(q, t) for q in k2h])
    # Weighted spatial RMS is exactly |amplitude difference|/sqrt(2) for
    # a resolved non-Nyquist sine mode on each periodic grid.
    error = np.max(np.abs(discrete - continuum), axis=1) / np.sqrt(2)

    def roots(q):
        return np.array(
            [
                -0.1 * q / 2 + 1j * np.sqrt(q - (0.1 * q / 2) ** 2),
                -0.1 * q / 2 - 1j * np.sqrt(q - (0.1 * q / 2) ** 2),
            ]
        )

    lambdas = np.array([roots(q) for q in k2h])
    exact = roots(k * k)
    arrays = {
        "N": ns,
        "h": h,
        "t": t,
        "continuum": continuum,
        "discrete": discrete,
        "error": error,
        "k2_discrete": k2h,
        "lambda_discrete": lambdas,
        "lambda_continuum": exact,
        "eigen_error": np.max(abs(lambdas - exact), axis=1),
        "rates": np.log(error[:-1] / error[1:]) / np.log(2),
    }
    meta = {
        "experiment": "continuum",
        "schema": 1,
        "profile": profile,
        "question": "Does a symmetric periodic lattice converge to a viscous wave at fixed physical wavelength?",
        "source": "Lyons (2022), Linear and Nearest Neighbor Flocks, Appendix H, equations (191)–(200), especially (193),(194),(198).",
        "geometry": "Periodic interval [0,L); N nodes, h=L/N; no boundary approximation.",
        "parameters": {
            "L": length,
            "mode": mode,
            "k": k,
            "Gx": 2.0,
            "Gv": 0.2,
            "c_squared": 1.0,
            "viscosity": 0.1,
        },
        "stencil": "rho_x,+/-=rho_v,+/-=-1/2; package gains gx=-2/h^2, gv=-0.2/h^2. Lyons uses the opposite overall gain sign.",
        "initial_data": "u(x,0)=sin(kx), u_t(x,0)=0; same physical k at every resolution.",
        "pde": "u_tt = u_xx + 0.1 u_xxt",
        "solver": "Exact scalar Fourier-mode solution for continuum and discrete symbols; no time integration error.",
        "error_norm": "max over 1601 saved times on [0,8] of sqrt(h/L * sum_j |u_h(x_j,t)-u(x_j,t)|^2).",
        "approximation_status": "Controlled symmetric specialization. Appendix H calls the general derivation heuristic; asymmetric drift scaling is excluded.",
        "fitted_order": float(np.polyfit(np.log(h), np.log(error), 1)[0]),
        "parallelism": "jobs and checkpoint accepted; small deterministic serial calculation.",
    }
    return arrays, meta


def draw_continuum(data, meta):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    with plot_style():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        axes[0].plot(
            data["t"], data["continuum"], color="0.2", label="Viscous wave PDE"
        )
        axes[0].plot(
            data["t"],
            data["discrete"][0],
            "--",
            color=CAT[1],
            label=f"Lattice N={data['N'][0]}",
        )
        axes[0].set(
            xlabel="Time",
            ylabel="Sine-mode amplitude",
            title="Same physical wavelength",
        )
        for idx in (0, 1, 2):
            axes[1].plot(
                data["t"],
                (data["discrete"][idx] - data["continuum"]) / data["h"][idx] ** 2,
                label=f"N={data['N'][idx]}",
            )
        axes[1].set(
            xlabel="Time", ylabel=r"$(a_h-a)/h^2$", title="Leading error collapses"
        )
        axes[2].loglog(
            data["h"],
            data["error"],
            "o-",
            color=CAT[1],
            label="Spatial RMS, max in time",
        )
        axes[2].loglog(
            data["h"],
            data["eigen_error"],
            "s-",
            color=CAT[2],
            label="Mode eigenvalue error",
        )
        axes[2].loglog(
            data["h"],
            data["error"][-1] * (data["h"] / data["h"][-1]) ** 2,
            ":",
            color="0.4",
            label=r"$h^2$ reference",
        )
        axes[2].set(
            xlabel="Spacing h",
            ylabel="Absolute error",
            title="Second-order convergence",
        )
        axes[2].set_xticks(data["h"], [f"{h:.3f}" for h in data["h"]])
        axes[2].xaxis.set_minor_formatter(NullFormatter())
        for ax in axes:
            ax.legend(fontsize=8)
        fig.suptitle(
            r"Controlled continuum limit: $u_{tt}=u_{xx}+0.1u_{xxt}$", fontsize=14
        )
        return fig


EXPERIMENTS = {
    "energy": {"compute": compute_energy, "draw": draw_energy},
    "continuum": {"compute": compute_continuum, "draw": draw_continuum},
}
