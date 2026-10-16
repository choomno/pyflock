"""Reproducible flock experiments: compute once, redraw from saved arrays.

Run ``python -m pyflock.experiments --help``. The paper convergence profile is a
larger research run; the default profile demonstrates the same measurements.
"""

from __future__ import annotations
import argparse
from importlib import import_module
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np

from . import nn_stencil, case, simulate_line, simulate_circle
from .dispersion import branches, taylor, signal_velocities, coefficients
from .analysis import extrema, theory_amplitudes, transient_summary
from .matrices import system_matrix_line_leaderless
from .plotting import circle_eigenvalues
from .solve import simulate_tail
from .style import CAT, SEQ, DIVERGING, plot_style

ORIGINAL_NAMES = (
    "propagation",
    "reflection",
    "boundary",
    "dispersion",
    "pulse",
    "convergence",
)


def _extensions():
    """Discover the source-specific experiment modules shipped with the package."""
    registry = {}
    for path in sorted(Path(__file__).parent.glob("experiment_*.py")):
        module = import_module(f".{path.stem}", __package__)
        for name, handlers in module.EXPERIMENTS.items():
            if name in registry or name in ORIGINAL_NAMES:
                raise ValueError(f"Duplicate experiment: {name}")
            if not {"compute", "draw"} <= handlers.keys():
                raise ValueError(f"Incomplete experiment handlers: {name}")
            registry[name] = handlers
    return registry


EXTENSIONS = _extensions()
NAMES = (*ORIGINAL_NAMES, *EXTENSIONS)
SOURCES = {
    "signal": "Cantos, Veerman & Hammond (2016), Signal velocity in oscillator arrays, EPJ ST 225, 1115–1126; manuscript Theorems 1–2.",
    "transients": "Cantos, Hammond & Veerman (2016), Transients in the synchronization of asymmetrically coupled oscillator arrays, EPJ ST 225, 1199–1209; manuscript Theorem 3 and Section 4.",
    "nnn": "Herbrych et al., Dynamics of locally coupled agents with next nearest neighbor interaction, DOI 10.1007/s12591-017-0377-3; manuscript Figure 3.",
}


def stencil_record(st):
    return {"rx": st.rx.tolist(), "rv": st.rv.tolist(), "gx": st.gx, "gv": st.gv}


def line_arrays(res, prefix=""):
    return {prefix + "t": res.t, prefix + "z": res.z, prefix + "velocity": res.zdot}


def _base(name):
    return {
        "experiment": name,
        "schema": 1,
        "convention": "Path: N followers + leader 0; circle: N agents. z_k=x_k+k*delta.",
        "delta": 1.0,
        "leader_velocity": 1.0,
        "initial_data": "z=0; followers at rest; leader acquires velocity 1 at t=0.",
        "solver": "Matrix exponential; dense below N=200, sparse exponential action otherwise.",
        "approximation_status": "Path wave/reflection asymptotics depend on the conjectures in Transients, Section 3.",
    }


def compute(name, profile="quick", *, jobs=1, checkpoint=None):
    """Return numerical arrays and the complete experiment specification."""
    if name in EXTENSIONS:
        return EXTENSIONS[name]["compute"](
            profile=profile, jobs=jobs, checkpoint=checkpoint
        )
    meta = _base(name)
    if name == "propagation":
        st = nn_stencil(-1.0, -1.0, -0.5, -0.3)
        N = 160
        cp, cm = signal_velocities(st)
        end = N / cp + 2.15 * N * (1 / cp - 1 / cm)
        res = simulate_line(st, N, end, nt=1601, bc="regular")
        meta.update(
            N=N,
            boundary="regular",
            stencil=stencil_record(st),
            source=SOURCES["transients"],
            question="Where does a leader disturbance travel and reflect?",
            speeds=[cp, cm],
        )
        return line_arrays(res), meta
    if name == "reflection":
        N = 400 if profile == "paper" else 160
        arrays = {}
        specs = []
        for i, av in enumerate((-0.5, 0.0)):
            st = nn_stencil(-2.0, -2.0, -0.5, av)
            res = simulate_line(st, N, 7 * N, nt=1801, bc="regular")
            arrays.update(line_arrays(res, f"{i}_"))
            prediction = theory_amplitudes(st, N, kmax=4)
            arrays[f"{i}_pred_t"] = np.asarray(prediction["T_k"])
            arrays[f"{i}_pred_A"] = np.asarray(prediction["A"])
            specs.append(stencil_record(st))
        meta.update(
            N=N,
            profile=profile,
            boundary="regular",
            stencils=specs,
            source=SOURCES["transients"],
            question="How does velocity asymmetry alter the last-follower response?",
            relation_to_source=f"Figure 3 gains and weights; N={N} followers (source N=400); tail response normalized by N.",
        )
        return arrays, meta
    if name == "boundary":
        st = case("type_I")
        N = 200 if profile == "paper" else 80
        res = simulate_line(st, N, 14 * N, nt=2001, bc="drop")
        ec = circle_eigenvalues(st, N)
        matrix = system_matrix_line_leaderless(st, N, bc="drop")
        from .analysis import eigen_reliability

        reliability = eigen_reliability(matrix)
        arrays = line_arrays(res)
        arrays.update(circle=ec, path=np.linalg.eigvals(matrix))
        meta.update(
            N=N,
            profile=profile,
            boundary="drop",
            stencil=stencil_record(st),
            source=SOURCES["nnn"],
            question="Can periodic stability coexist with path instability?",
            early_window_t_over_N=[0, 3],
            spectral_sensitivity=float(reliability["tol"]),
            approximation_status="Direct finite-system calculation; no path amplitude-theory overlay.",
            relation_to_source=f"NNN Figure 3 stencil and fixed-interaction boundary; N={N} followers (source N=200), with new spectral and growth diagnostics.",
        )
        return arrays, meta
    if name == "dispersion":
        st = nn_stencil(-1.0, -1.0, -0.5, -0.3)
        th = np.linspace(-np.pi, np.pi, 1001)
        local = np.geomspace(0.003, 0.4, 200)
        co = taylor(st, order=3)
        meta.update(
            stencil=stencil_record(st),
            source=SOURCES["signal"],
            geometry="periodic symbol",
            question="Where is the small-wavenumber expansion accurate?",
            initial_data="Not applicable: spectral calculation.",
            solver="Quadratic roots and analytic moment recursion.",
            approximation_status="Analytic moment expansion with zero first position moment and distinct real signal speeds.",
        )
        return {
            "theta": th,
            "branches": branches(th, st),
            "local_theta": local,
            "local_branches": branches(local, st),
            "coefficients": co,
        }, meta
    if name == "pulse":
        N = 1024
        st = nn_stencil(-1.0, -1.0, -0.5, -0.3)
        k = np.arange(N)
        centre = N // 3
        width = N / 50
        # Smooth periodic Gaussian, including its periodic images.
        distance = (k - centre + N / 2) % N - N / 2
        velocity = np.exp(-0.5 * (distance / width) ** 2)
        times = N * np.asarray([0.0, 0.15, 0.3, 0.4])
        res = simulate_circle(st, N, times, np.zeros(N), velocity)
        meta.update(
            N=N,
            geometry="circle",
            solver="FFT reduction to closed-form 2-by-2 mode propagators.",
            stencil=stencil_record(st),
            centre=centre,
            width=width,
            speeds=list(signal_velocities(st)),
            source=SOURCES["signal"],
            question="Does a smooth velocity pulse split into two travelling profiles?",
            initial_data="Zero displacement; periodic Gaussian velocity centred at N/3, sigma=N/50.",
            approximation_status="Finite-N FFT evolution; dashed marks show low-frequency signal speeds, not the full pulse shape.",
        )
        return {"t": res.t, "velocity": res.zdot, "z": res.z}, meta
    if name != "convergence":
        raise ValueError(f"Unknown experiment: {name}")
    # The published parameter set and N sequence are available explicitly.
    Ns = (100, 200, 400, 800, 1600, 3200) if profile == "paper" else (50, 100, 200)
    avs = np.linspace(-0.5, 0.0, 6) if profile == "paper" else (-0.5, -0.25, 0.0)
    gains = (-0.25, -0.5, -1.0, -2.0, -4.0) if profile == "paper" else (-0.5, -2.0)
    from concurrent.futures import ProcessPoolExecutor

    specs = [
        (
            bi,
            bc,
            N,
            gv,
            float(av),
            str(checkpoint) if checkpoint else None,
            _source_digest(),
        )
        for bi, bc in enumerate(("drop", "regular"))
        for N in Ns
        for gv in gains
        for av in avs
    ]
    rows, tail_times, tail_responses = [], [], []

    def collect(results):
        for index, (row, t, w) in enumerate(results, 1):
            rows.append(row)
            tail_times.append(t)
            tail_responses.append(w)
            if profile == "paper" and index % 30 == 0:
                print(f"Convergence: {index}/{len(specs)} cases", flush=True)

    if jobs == 1:
        collect(map(_convergence_case, specs))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            collect(pool.map(_convergence_case, specs))
    meta.update(
        profile=profile,
        Ns=list(Ns),
        velocity_weights=list(avs),
        velocity_gains=list(gains),
        position_gain=-1.0,
        position_weights=[0, -0.5, 1, -0.5, 0],
        boundaries=["drop", "regular"],
        sampling="At least 2401 points, dt <= sqrt(2*N*D_plus/c_plus**3)/8; doubled intervals for the coarsening diagnostic.",
        tail_storage="Rows correspond to offsets[i]:offsets[i+1] in tail_times and tail_responses.",
        period_estimator="Time from first to third extremum (one same-sign interval). The source averages same-sign intervals.",
        horizon="N/c_plus + 2.3*P, where P=N*(1/c_plus-1/c_minus)",
        source=SOURCES["transients"],
        question="How do amplitude, period and attenuation errors vary with N and boundary?",
        error_denominator="Absolute predicted value, as in the source experiment.",
        columns=[
            "boundary_index",
            "N",
            "gv",
            "rv_plus1",
            "A1_measured",
            "T_measured",
            "gamma_measured",
            "A1_predicted",
            "T_predicted",
            "gamma_predicted",
            "A1_error",
            "T_error",
            "gamma_error",
            "A1_sampling_change",
            "T_sampling_change",
            "gamma_sampling_change",
        ],
        relation_to_source="Paper profile uses the published 30 parameter pairs and six sizes. Quick profile uses 6 pairs and 3 smaller sizes.",
    )
    return {
        "measurements": np.asarray(rows),
        "tail_times": np.concatenate(tail_times),
        "tail_responses": np.concatenate(tail_responses),
        "offsets": np.r_[0, np.cumsum([len(t) for t in tail_times])],
    }, meta


def _convergence_case(spec):
    """One parameter case, optionally resumed from a checksum-checked checkpoint."""
    bi, bc, N, gv, av, checkpoint, source_digest = spec
    key = sha256(json.dumps([bi, N, gv, av, source_digest]).encode()).hexdigest()
    path = Path(checkpoint) / f"{key}.npz" if checkpoint else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = path.with_suffix(".sha256")
        if (
            path.exists()
            and digest.exists()
            and sha256(path.read_bytes()).hexdigest() == digest.read_text().strip()
        ):
            with np.load(path, allow_pickle=False) as saved:
                return saved["row"].tolist(), saved["t"], saved["w"]
    st = nn_stencil(-1.0, gv, -0.5, av)
    prediction = theory_amplitudes(st, N, kmax=3)
    end = prediction["T_k"][2] + 0.3 * prediction["P"]
    cp = prediction["c_plus"]
    Dp = coefficients(st)["+"]["D"]
    width = np.sqrt(2 * N * Dp / cp**3)
    intervals = max(2400, int(np.ceil(end / (width / 8))))
    intervals += intervals % 2
    t, w = simulate_tail(st, N, end, nt=intervals + 1, bc=bc)
    measured = transient_summary(t, w, nmax=3)
    coarse = transient_summary(t[::2], w[::2], nmax=3)
    if measured["n_extrema"] != 3 or coarse["n_extrema"] != 3:
        raise RuntimeError(f"Insufficient extrema: {bc}, N={N}, gv={gv}, av={av}")
    observed = np.asarray([abs(measured["A"][0]), measured["T"], measured["gamma"]])
    predicted = np.asarray(
        [abs(prediction["A"][0]), prediction["T"], prediction["gamma"]]
    )
    coarse_values = np.asarray([abs(coarse["A"][0]), coarse["T"], coarse["gamma"]])
    error = abs(observed - predicted) / abs(predicted)
    sampling = abs(observed - coarse_values) / abs(predicted)
    # This is an empirical sampling criterion, not a bound on solver error.
    if np.any(sampling > np.maximum(1e-4, 0.05 * error)):
        raise RuntimeError(
            f"Sampling unresolved: {bc}, N={N}, gv={gv}, av={av}: {sampling}"
        )
    row = [bi, N, gv, av, *observed, *predicted, *error, *sampling]
    if path is not None:
        np.savez_compressed(path, row=row, t=t, w=w)
        path.with_suffix(".sha256").write_text(
            sha256(path.read_bytes()).hexdigest() + "\n"
        )
    return row, t, w


def _line_axes(fig, axs, data, meta):
    """Render the primary propagation experiment from arrays."""
    N = meta["N"]
    t = data["t"]
    z = data["z"]
    spacing = np.diff(z, axis=1)
    limit = abs(spacing).max()
    im = axs[0].pcolormesh(
        np.arange(1, N + 1),
        t / N,
        spacing,
        cmap=DIVERGING,
        vmin=-limit,
        vmax=limit,
        shading="nearest",
        rasterized=True,
    )
    fig.colorbar(im, ax=axs[0], label=r"$z_k-z_{k-1}$", shrink=0.92, pad=0.025)
    cp, cm = meta["speeds"]
    time = 0.0
    pos = 0.0
    direction = 1
    for i in range(5):
        speed = cp if direction == 1 else cm
        end = N if direction == 1 else 0
        new_time = time + (end - pos) / speed
        stop = min(new_time, t[-1])
        if time >= t[-1]:
            break
        axs[0].plot(
            [pos, pos + speed * (stop - time)],
            [time / N, stop / N],
            "--",
            color="#333333",
            lw=0.9,
            label="Leading wave prediction" if i == 0 else None,
        )
        time, pos, direction = new_time, end, -direction
    axs[0].set(
        xlabel="Follower index $k$",
        ylabel="Time $t/N$",
        title="Propagation and reflection",
    )
    axs[0].legend(
        loc="upper left",
        fontsize=8,
        facecolor="white",
        frameon=True,
        edgecolor="none",
        framealpha=0.9,
    )
    for k, col in zip(np.linspace(0, N, 6).astype(int), SEQ(np.linspace(0.4, 1, 6))):
        axs[1].plot(
            (z[:, k] - z[:, 0] - k) / N, t / N, color=col, lw=1, label=f"$k={k}$"
        )
    axs[1].set(
        xlabel=r"$(x_k-x_0)/N$", ylabel="Time $t/N$", title="Selected trajectories"
    )
    axs[1].legend(loc="upper left", fontsize=8)


def draw(name, data, meta):
    """Render using only recorded data and parameters; never run a simulation."""
    if name in EXTENSIONS:
        return EXTENSIONS[name]["draw"](data, meta)
    if name not in ORIGINAL_NAMES:
        raise ValueError(f"Unknown experiment: {name}")
    import matplotlib.pyplot as plt

    with plot_style():
        if name == "propagation":
            fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
            _line_axes(fig, axs, data, meta)
        elif name == "reflection":
            fig, axs = plt.subplots(
                1, 2, figsize=(10, 4), layout="constrained", sharey=True
            )
            for i, (ax, title) in enumerate(
                zip(
                    axs,
                    ("Symmetric velocity coupling", "Velocity weighted towards leader"),
                )
            ):
                t, z = data[f"{i}_t"], data[f"{i}_z"]
                N = meta["N"]
                ax.plot(
                    t / N,
                    (z[:, -1] - z[:, 0]) / N,
                    color=CAT[1],
                    label="Computed response",
                )
                ex = extrema(t, z[:, -1] - z[:, 0], nmax=4)
                ax.scatter(
                    [e.t / N for e in ex], [e.value / N for e in ex], s=16, color=CAT[1]
                )
                keep = data[f"{i}_pred_t"] <= t[-1]
                ax.scatter(
                    data[f"{i}_pred_t"][keep] / N,
                    data[f"{i}_pred_A"][keep] / N,
                    edgecolors=CAT[2],
                    facecolors="none",
                    linewidths=1.1,
                    marker="D",
                    s=28,
                    zorder=4,
                    label="Leading asymptotic extrema",
                )
                ax.axhline(0, lw=0.6, color="#999999")
                ax.set(xlabel="Time $t/N$", title=title)
            axs[0].set_ylabel(r"Last follower $(z_N-z_0)/N$")
            axs[1].legend(loc="lower right", fontsize=8)
        elif name == "boundary":
            fig, grid = plt.subplots(2, 2, figsize=(9.2, 7.2), layout="constrained")
            spectrum, zoom = grid[0]
            response, growth = grid[1]
            for key, col, label, marker in [
                ("circle", CAT[1], "Circle spectrum", "o"),
                ("path", CAT[2], "Path spectrum", "x"),
            ]:
                vals = data[key]
                spectrum.scatter(
                    vals.real, vals.imag, s=13, color=col, marker=marker, label=label
                )
                zoom.scatter(
                    1000 * vals.real, 1000 * vals.imag, s=22, color=col, marker=marker
                )
            for ax in (spectrum, zoom):
                ax.axvline(0, color="#777777", lw=0.7)
            spectrum.set(
                xlabel=r"$\operatorname{Re}\lambda$",
                ylabel=r"$\operatorname{Im}\lambda$",
                title="Full spectrum",
            )
            zoom.set(
                xlim=(-4, 5.5),
                ylim=(-60, 60),
                xticks=[-4, -2, 0, 2, 4],
                yticks=[-50, 0, 50],
                xlabel=r"$10^3\operatorname{Re}\lambda$",
                ylabel=r"$10^3\operatorname{Im}\lambda$",
                title="Spectrum near the origin",
            )
            handles, labels = spectrum.get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside upper center", ncol=2)
            N = meta["N"]
            t = data["t"]
            z = data["z"]
            early = t <= 3 * N
            field = np.diff(z[early], axis=1)
            limit = abs(field).max()
            im = response.pcolormesh(
                np.arange(1, N + 1),
                t[early] / N,
                field,
                cmap=DIVERGING,
                vmin=-limit,
                vmax=limit,
                shading="nearest",
                rasterized=True,
            )
            fig.colorbar(im, ax=response, label=r"$z_k-z_{k-1}$", shrink=0.9, pad=0.025)
            response.set(
                xlabel="Follower index $k$", ylabel="Time $t/N$", title="Early response"
            )
            amplitude = np.max(abs(z - z[:, [0]]), axis=1) / N
            keep = t >= 0.2 * N
            growth.semilogy(t[keep] / N, amplitude[keep], color=CAT[2])
            growth.set(
                xlabel="Time $t/N$",
                ylabel=r"$\max_k|z_k-z_0|/N$",
                title="Growth on the path",
            )
            growth.grid(True, alpha=0.4)
        elif name == "dispersion":
            fig, axs = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
            lam = data["branches"]
            axs[0].plot(lam.real.T, lam.imag.T, color=CAT[1], lw=1.2)
            axs[0].axvline(0, color="#777777", lw=0.7)
            axs[0].set(
                xlabel=r"$\operatorname{Re}\lambda$",
                ylabel=r"$\operatorname{Im}\lambda$",
                title="Full periodic symbol",
            )
            x = data["local_theta"]
            co = data["coefficients"]
            for order, col, ls in zip((1, 2, 3), CAT.values(), ("-", "--", ":")):
                approx = sum(co[:, j, None] * x**j for j in range(order + 1))
                error = np.max(abs(data["local_branches"] - approx), axis=0)
                axs[1].loglog(x, error, color=col, ls=ls, label=f"Order {order}")
            axs[1].set(
                xlabel=r"Wavenumber $\theta$",
                ylabel="Maximum branch error",
                title="Local expansion error",
            )
            axs[1].legend()
            axs[1].grid(True, alpha=0.4)
        elif name == "pulse":
            fig, axs = plt.subplots(
                2, 2, figsize=(9, 5.6), layout="constrained", sharex=True, sharey=True
            )
            N = meta["N"]
            k = (np.arange(N) - meta["centre"]) / N
            for ax, t, v in zip(axs.flat, data["t"], data["velocity"]):
                ax.plot(k, v, color=CAT[1])
                ax.set_title(f"$t/N={t / N:g}$")
                for j, speed in enumerate(meta["speeds"]):
                    ax.axvline(
                        speed * t / N,
                        color=CAT[2],
                        lw=0.8,
                        ls="--",
                        label="Leading wave prediction" if j == 0 else None,
                    )
                ax.set_ylim(-0.04, 0.6 if t else 1.05)
            # Same scales across every panel, including the initial pulse.
            for ax in axs.flat:
                ax.set_ylim(-0.04, 1.05)
                ax.set_xlim(-0.32, 0.6)
            for ax in axs[-1]:
                ax.set_xlabel("Agent offset / $N$")
            for ax in axs[:, 0]:
                ax.set_ylabel("Velocity")
            axs[0, 1].legend(loc="upper right", fontsize=8)
        else:
            fig, axs = plt.subplots(
                1, 3, figsize=(11, 3.6), layout="constrained", sharex=True
            )
            rows = data["measurements"]
            for j, (ax, label) in enumerate(
                zip(
                    axs,
                    (
                        r"First amplitude $|A_1|$",
                        r"Period $T$",
                        r"Attenuation $\gamma$",
                    ),
                )
            ):
                for bi, (bc, col) in enumerate(
                    zip(("Variable mass", "Regular"), (CAT[1], CAT[2]))
                ):
                    selected = rows[rows[:, 0] == bi]
                    Ns = np.unique(selected[:, 1])
                    means = [selected[selected[:, 1] == N, 10 + j].mean() for N in Ns]
                    worst = [selected[selected[:, 1] == N, 10 + j].max() for N in Ns]
                    ax.loglog(Ns, means, "o-", color=col, ms=4, label=f"{bc}: mean")
                    ax.loglog(
                        Ns,
                        worst,
                        "s--",
                        color=col,
                        ms=4,
                        mfc="white",
                        label=f"{bc}: maximum",
                    )
                ax.set(title=label, xlabel="Followers $N$")
                ax.grid(True, alpha=0.4)
            axs[0].set_ylabel("Relative error / prediction")
            axs[-1].legend(fontsize=8, loc="best")
            pairs = len(meta["velocity_weights"]) * len(meta["velocity_gains"])
            sizes = meta["Ns"]
            fig.suptitle(
                f"{meta['profile'].capitalize()} convergence study: {pairs} parameter pairs, N = {min(sizes)}–{max(sizes)}",
                fontsize=11,
            )
        return fig


def _software():
    from importlib.metadata import version

    return {
        "python": platform.python_version(),
        **{p: version(p) for p in ("numpy", "scipy", "matplotlib", "mpmath")},
    }


def _source_digest(directory=None):
    h = sha256()
    for path in sorted(
        (Path(directory) if directory else Path(__file__).parent).glob("*.py")
    ):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty():
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run(name, output, *, profile="quick", redraw=False, jobs=1):
    import matplotlib.pyplot as plt

    dest = Path(output) / name
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if redraw:
        meta = json.loads((dest / "experiment.json").read_text())
        if sha256((dest / "data.npz").read_bytes()).hexdigest() != meta["data_sha256"]:
            raise ValueError(f"Data checksum mismatch: {dest}")
        if (dest / "source").exists() and _source_digest(dest / "source") != meta[
            "source_sha256"
        ]:
            raise ValueError(f"Source snapshot checksum mismatch: {dest}")
        with np.load(dest / "data.npz", allow_pickle=False) as saved:
            data = dict(saved)
    else:
        computation_source = _source_digest()
        computation_commit = _git_commit()
        computation_dirty = _git_dirty()
        source_files = {
            module.name: module.read_bytes()
            for module in Path(__file__).parent.glob("*.py")
        }
        data, meta = compute(
            name,
            profile,
            jobs=jobs,
            checkpoint=dest / "cases",
        )
        if any(not np.isfinite(v).all() for v in data.values()):
            raise RuntimeError("Nonfinite experiment data")
        snapshot = dest / "source"
        snapshot.mkdir(exist_ok=True)
        for filename, content in source_files.items():
            (snapshot / filename).write_bytes(content)
        grids = {
            key: {
                "start": float(value[0]),
                "stop": float(value[-1]),
                "samples": len(value),
            }
            for key, value in data.items()
            if key == "t" or key.endswith("_t") and not key.endswith("pred_t")
        }
        if grids:
            meta["time_grids"] = grids
        np.savez_compressed(dest / "data.npz", **data)
        if "measurements" in data and "columns" in meta:
            np.savetxt(
                dest / "measurements.csv",
                data["measurements"],
                delimiter=",",
                header=",".join(meta["columns"]),
                comments="",
            )
        if name == "convergence":
            meta["maximum_sampling_change"] = dict(
                zip(
                    ("A1", "T", "gamma"),
                    data["measurements"][:, 13:16].max(axis=0).tolist(),
                )
            )
        meta.update(
            software=_software(),
            source_sha256=computation_source,
            git_commit=computation_commit,
            git_worktree_dirty=computation_dirty,
            data_sha256=sha256((dest / "data.npz").read_bytes()).hexdigest(),
        )
    with plot_style():
        fig = draw(name, data, meta)
        fig.savefig(dest / "figure.png", dpi=180)
        fig.savefig(
            dest / "figure.pdf", metadata={"CreationDate": None, "ModDate": None}
        )
        fig.savefig(dest / "figure.svg", metadata={"Date": None})
        plt.close(fig)
    meta["render"] = {"software": _software(), "source_sha256": _source_digest()}
    meta["artifacts"] = {
        f: sha256((dest / f).read_bytes()).hexdigest()
        for f in ("figure.png", "figure.pdf", "figure.svg")
    }
    (dest / "experiment.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n"
    )
    print(f"{name}: {time.perf_counter() - started:.1f}s -> {dest}", flush=True)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=(*NAMES, "all"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument(
        "--profile",
        choices=("quick", "paper"),
        default="quick",
        help="Small validation runs or published sizes and parameter families; consult each record for source departures.",
    )
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="Read recorded arrays; do not compute again.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for experiments that support them (default: 1).",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    import matplotlib

    matplotlib.use("Agg")
    names = NAMES if args.experiment == "all" else (args.experiment,)
    for name in names:
        run(name, args.output, profile=args.profile, redraw=args.redraw, jobs=args.jobs)


if __name__ == "__main__":
    main()
