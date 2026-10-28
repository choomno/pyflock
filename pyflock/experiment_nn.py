"""Source-specific nearest-neighbour experiments and matched-Laplacian responses.

Compute returns plain arrays and JSON-compatible specifications; draw uses no solver.
The historical figures provide experiments, not plotting or numerical conventions.
"""

import numpy as np
from scipy.linalg import solve_banded

from .dispersion import branches, nu_v, nu_x, signal_velocities
from .solve import simulate_circle, simulate_line
from .stencil import nn_stencil
from .style import CAT, SEQ, plot_style


def _record(st):
    return {"gx": st.gx, "gv": st.gv, "rx": st.rx.tolist(), "rv": st.rv.tolist()}


def _meta(name, profile, source, figure, question):
    return {
        "experiment": name,
        "profile": profile,
        "schema": 1,
        "source": source,
        "source_figure": figure,
        "question": question,
        "geometry": "circle",
        "neutral_mode": "m=0 included in spectral arrays",
        "initial_data": "Not applicable: spectral calculation",
        "solver": "Fourier symbols and quadratic roots",
        "observable": "Complex eigenvalues",
    }


def compute_stable(profile="quick", jobs=1, checkpoint=None):
    N = 500 if profile == "paper" else 160
    st = nn_stencil(-3.0, -1.5, -0.5, -0.2)
    theta = 2 * np.pi * np.arange(N) / N
    meta = _meta(
        "nn_stable_spectra",
        profile,
        "signalvelocity.pdf",
        "Fig.1, p.4",
        "How do the two coupling spectra determine the stable system spectrum?",
    )
    meta.update(
        N=N,
        stencil=_record(st),
        relation_to_source="Published parameters; paper profile uses published N=500. Distinct symbols replace the original colors.",
    )
    return {
        "theta": theta,
        "position": nu_x(theta, st),
        "velocity": nu_v(theta, st),
        "roots": branches(theta, st, "by_imag"),
    }, meta


def draw_stable(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
        for key, label, color, marker in [
            ("position", r"$\lambda_x$", CAT[1], "o"),
            ("velocity", r"$\lambda_v$", CAT[2], "."),
        ]:
            values = data[key]
            axes[0].plot(
                values.real, values.imag, marker, ms=2, label=label, color=color
            )
        values = data["roots"]
        axes[1].plot(values.real.ravel(), values.imag.ravel(), ".", ms=2, color=CAT[1])
        for ax, title in zip(axes, ("Coupling spectra", "First-order system spectrum")):
            ax.axvline(0, color=".6", lw=0.6)
            ax.axhline(0, color=".7", lw=0.5)
            ax.set(xlabel="Real part", ylabel="Imaginary part", title=title)
        axes[0].legend(loc="upper left")
        fig.suptitle(f"Published stable example · N={meta['N']}", fontsize=11)
        return fig


def compute_instability(profile="quick", jobs=1, checkpoint=None):
    N = 500 if profile == "paper" else 160
    specs = [nn_stencil(-3.0, -1.5, -0.4, -0.5), nn_stencil(0.5, -1.5, -0.5, -0.15)]
    theta = 2 * np.pi * np.arange(N) / N
    data = {"theta": theta}
    for i, st in enumerate(specs):
        data[f"{i}_position"] = nu_x(theta, st)
        data[f"{i}_velocity"] = nu_v(theta, st)
        data[f"{i}_roots"] = branches(theta, st, "by_imag")
    meta = _meta(
        "nn_instability",
        profile,
        "signalvelocity.pdf",
        "Fig.2, p.6",
        "How do position asymmetry and a wrong-sign gain destabilize the circle?",
    )
    meta.update(
        N=N,
        stencils=[_record(s) for s in specs],
        relation_to_source="Both published parameter sets; paper N=500. No unstable time integration.",
    )
    return data, meta


def draw_instability(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
        for i, label in enumerate(
            ("Asymmetric position coupling", "Positive position gain")
        ):
            for key, name, col in [
                ("position", r"$\lambda_x$", CAT[1]),
                ("velocity", r"$\lambda_v$", CAT[2]),
            ]:
                z = data[f"{i}_{key}"]
                axes[i, 0].plot(z.real, z.imag, ".", ms=2, color=col, label=name)
            z = data[f"{i}_roots"]
            axes[i, 1].plot(z.real.ravel(), z.imag.ravel(), ".", ms=2, color=CAT[1])
            axes[i, 0].set_title(label)
            axes[i, 1].set_title("System eigenvalues")
            axes[i, 0].legend(loc="upper left", fontsize=8)
        for ax in axes.flat:
            ax.axvline(0, color=".5", lw=0.7)
            ax.axhline(0, color=".75", lw=0.5)
            ax.set(xlabel="Real part", ylabel="Imaginary part")
        fig.suptitle(f"Two instability mechanisms · N={meta['N']}", fontsize=11)
        return fig


def compute_phase(profile="quick", jobs=1, checkpoint=None):
    N = 500 if profile == "paper" else 160
    st = nn_stencil(-3.0, -1.5, -0.5, -0.2)
    modes = np.arange(-N // 2, N // 2 + 1)
    theta = 2 * np.pi * modes / N
    roots = branches(theta, st)
    speeds = np.empty_like(roots.real)
    nonzero = theta != 0
    speeds[:, nonzero] = -roots[:, nonzero].imag / theta[nonzero]
    speeds[:, ~nonzero] = np.asarray(signal_velocities(st))[:, None]
    meta = _meta(
        "nn_phase_speeds",
        profile,
        "signalvelocity.pdf",
        "Fig.3(a), p.11",
        "Which phase speeds survive high-frequency damping?",
    )
    meta.update(
        N=N,
        stencil=_record(st),
        observable="Phase velocity -Im(lambda)/theta and damping Re(lambda)",
        zero_mode="Phase velocity at theta=0 uses its analytic limit",
        relation_to_source="Published parameters; separate axes avoid mixing speed and inverse-time units. Continuous branch labels may differ from original sign labels.",
    )
    return {
        "modes": modes,
        "theta": theta,
        "phase_speed": speeds,
        "damping": roots.real,
    }, meta


def draw_phase(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axes = plt.subplots(
            1, 2, figsize=(9, 3.6), layout="constrained", sharex=True
        )
        for i, (color, style) in enumerate(((CAT[1], "-"), (CAT[2], "--"))):
            axes[0].plot(
                data["modes"],
                data["phase_speed"][i],
                color=color,
                ls=style,
                label=f"Branch {i + 1}",
            )
            axes[1].plot(data["modes"], data["damping"][i], color=color, ls=style)
        axes[0].set(ylabel="Agents / unit time", title="Phase velocity")
        axes[0].legend()
        axes[1].set(ylabel="Real part of eigenvalue", title="Modal damping")
        for ax in axes:
            ax.set_xlabel("Fourier mode $m$")
            ax.axhline(0, color=".7", lw=0.5)
        fig.suptitle(f"Published phase-speed example · N={meta['N']}", fontsize=11)
        return fig


def compute_pulse(profile="quick", jobs=1, checkpoint=None):
    # Historical size retained in both profiles; quick reduces time samples only.
    N = 200
    st = nn_stencil(-1.0, -3.0, -0.5, -7 / 20)
    times = np.linspace(0, 150, 601 if profile == "paper" else 301)
    k = np.arange(N)
    distance = (k + N / 2) % N - N / 2
    impulse = np.zeros(N)
    impulse[0] = 20.0
    smooth = np.exp(-0.5 * (distance / 3.0) ** 2)
    smooth *= 20 / smooth.sum()
    data = {"t": times, "k": k}
    for i, velocity in enumerate((impulse, smooth)):
        result = simulate_circle(st, N, times, np.zeros(N), velocity)
        data[f"{i}_initial_velocity"] = velocity
        data[f"{i}_velocity"] = result.zdot
    meta = _meta(
        "nn_pulse_comparison",
        profile,
        "signalvelocity.pdf",
        "Fig.3(b), p.11",
        "How does smoothing change an impulse response under identical couplings?",
    )
    meta.update(
        N=N,
        delta=2.5,
        stencil=_record(st),
        speeds=list(signal_velocities(st)),
        initial_data="Zero displacement; (a) velocity 20 at agent0, (b) periodic Gaussian sigma3 normalized to sum20.",
        solver="FFT closed-form mode evolution",
        observable="Velocity field by agent index and time",
        relation_to_source="Historical impulse size, gains, weights, delta, and t<=150 retained. Agent coordinate replaces moving physical position; smooth initial data comparison is new.",
        normalization="Both inputs have equal total velocity20; shared symmetric-log color normalization (linear threshold.05) without clipping.",
        approximation_status="Impulse lacks theorem Fourier decay. Gaussian is finite smooth-data comparison; dashed lines show leading signal positions, with periodic wrapping.",
    )
    return data, meta


def draw_pulse(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        from matplotlib.colors import SymLogNorm

        fig, axes = plt.subplots(
            1, 2, figsize=(9, 4), layout="constrained", sharex=True, sharey=True
        )
        N = meta["N"]
        t = data["t"]
        limit = max(data["0_velocity"].max(), data["1_velocity"].max())
        low = min(data["0_velocity"].min(), data["1_velocity"].min(), 0.0)
        for i, ax in enumerate(axes):
            im = ax.pcolormesh(
                data["k"],
                t,
                data[f"{i}_velocity"],
                cmap=SEQ,
                norm=SymLogNorm(linthresh=0.05, vmin=low, vmax=limit),
                shading="nearest",
                rasterized=True,
            )
            # Split wrap discontinuities, so lines never cut diagonally across the plot.
            for speed in meta["speeds"]:
                path = (speed * t) % N
                path = path.astype(float)
                path[np.r_[False, abs(np.diff(path)) > N / 2]] = np.nan
                ax.plot(path, t, color="#b54b36", lw=0.8, ls="--")
            ax.set(
                xlabel="Agent index $k$",
                title=("Velocity impulse", "Smooth velocity pulse")[i],
            )
        axes[0].set_ylabel("Time $t$")
        fig.colorbar(im, ax=axes, label="Velocity (symmetric-log scale)", shrink=0.9)
        fig.suptitle(
            "Same couplings and total initial velocity; dashed: signal positions",
            fontsize=10,
        )
        return fig


def tail_transfer(N, rho, omega):
    """Complex leader-position to tail-position transfer, regular terminal boundary.

    Solve (s² I -(gx+s gv)L)Z=b with Z0=1; gx=-1, gv=-2.
    This is also the acceleration transfer at nonzero frequency.
    """
    if N < 1 or not 0 < rho < 1:
        raise ValueError("N>=1 and 0<rho<1 required")
    omega = np.asarray(omega, dtype=float)
    if np.any(omega < 0) or not np.isfinite(omega).all():
        raise ValueError("finite nonnegative frequencies required")
    values = np.empty(omega.shape, dtype=complex)
    for index, w in np.ndenumerate(omega):
        s = 1j * w
        h = -1.0 - 2 * s
        band = np.zeros((3, N), complex)
        band[1] = s * s - h
        band[0, 1:] = h * rho
        band[2, :-1] = h * (1 - rho)
        if N > 1:
            band[2, -2] = h
        rhs = np.zeros(N, complex)
        rhs[0] = -h * ((1 - rho) if N > 1 else 1.0)
        values[index] = solve_banded((1, 1), band, rhs, check_finite=False)[-1]
    return values


def high_precision_transfer(N, rho, omega):
    """70-digit independent scalar elimination for sensitive low-frequency resonances."""
    import mpmath as mp

    out = []
    with mp.workdps(70):
        r = mp.mpf(str(rho))
        for value in np.asarray(omega).ravel():
            w = mp.mpf(str(value))
            h = -1 - 2j * w
            diagonal = -w * w - h
            pivot = diagonal
            rhs = -h * ((1 - r) if N > 1 else 1)
            for row in range(1, N):
                lower = h * (1 - r) if row < N - 1 else h
                multiplier = lower / pivot
                pivot = diagonal - multiplier * h * r
                rhs = -multiplier * rhs
            out.append(complex(rhs / pivot))
    return np.asarray(out).reshape(np.shape(omega))


def harmonic_samples(N, rho, count=2000, low=1e-8):
    """A broad grid enriched at all detected local peaks, not a certified supremum."""
    from scipy.optimize import minimize_scalar

    omega = np.geomspace(low, 0.5, count)
    gain = abs(tail_transfer(N, rho, omega))
    maxima = (
        np.where(
            (gain[1:-1] > 1.001) & (gain[1:-1] > gain[:-2]) & (gain[1:-1] > gain[2:])
        )[0]
        + 1
    )
    added = []
    neighborhoods = []
    for index in maxima:
        left, right = np.log(omega[index - 1 : index + 2 : 2])
        neighborhoods.append((np.exp(left), np.exp(right)))
        result = minimize_scalar(
            lambda x: -abs(tail_transfer(N, rho, np.array([np.exp(x)]))[0]),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-14},
        )
        centre = np.exp(result.x)
        if rho > 0.5 and centre < 1e-3:
            refined = minimize_scalar(
                lambda x: -abs(high_precision_transfer(N, rho, np.array([x]))[0]),
                bounds=(np.exp(left), np.exp(right)),
                method="bounded",
                options={"xatol": 1e-18},
            )
            centre = refined.x
        # Include local shoulders and the optimized peak itself in the saved curve.
        added.extend(centre * np.exp(np.linspace(-1e-3, 1e-3, 101)))
        added.append(centre)
    omega = np.unique(np.r_[omega, added])
    omega = omega[(omega >= low) & (omega <= 0.5)]
    values = tail_transfer(N, rho, omega)
    if rho > 0.5:
        # All points in a detected sensitive resonance neighborhood use the same precision.
        sensitive = np.zeros(omega.shape, dtype=bool)
        for left, right in neighborhoods:
            sensitive |= (omega >= left) & (omega <= right)
        sensitive &= omega < 1e-3
        values[sensitive] = high_precision_transfer(N, rho, omega[sensitive])
    return omega, values


def compute_matched(profile="quick", jobs=1, checkpoint=None):
    N = 100
    rhos = (0.45, 0.5, 0.55)
    times_end = 1000.0
    data = {}
    specs = []
    for i, rho in enumerate(rhos):
        st = nn_stencil(-1.0, -2.0, -rho, -rho)
        res = simulate_line(
            st,
            N,
            times_end,
            nt=2001 if profile == "paper" else 1001,
            bc="regular",
            v0=0.1,
            delta=1.0,
        )
        data[f"{i}_t"] = res.t
        data[f"{i}_xrel"] = res.xrel
        data[f"{i}_tail"] = res.last_relative
        # Wide logarithmic sampling with explicit low-frequency coverage for rho>.5.
        omega, transfer = harmonic_samples(N, rho, 6000 if profile == "paper" else 2000)
        data[f"{i}_omega"] = omega
        data[f"{i}_transfer"] = transfer
        specs.append(_record(st))
    # Size dependence: sampled harmonic peaks, not certified H-infinity norms.
    sizes = np.array([25, 50, 75, 100] if profile == "paper" else [25, 50, 100])
    peaks = np.empty((3, len(sizes)))
    peakfreq = np.empty_like(peaks)
    for i, rho in enumerate(rhos):
        for j, n in enumerate(sizes):
            w, transfer = harmonic_samples(
                int(n), rho, 8000 if profile == "paper" else 2500, low=1e-9
            )
            amplitude = abs(transfer)
            p = amplitude.argmax()
            peaks[i, j] = amplitude[p]
            peakfreq[i, j] = w[p]
    data.update(sizes=sizes, sampled_peak=peaks, peak_frequency=peakfreq)
    meta = _meta(
        "matched_amplification",
        profile,
        "flocks_final.pdf; Symmetry and Stability of Homogeneous Flocks.pdf",
        "2012 Fig.2 p.2; position paper Figs.2–3 p.6",
        "How can stable matched-Laplacian paths have large impulse and harmonic responses?",
    )
    meta.update(
        N=N,
        rhos=list(rhos),
        stencils=specs,
        geometry="path",
        boundary="regular",
        delta=1.0,
        leader_velocity=0.1,
        initial_data="All agents at rest and spaced1; leader acquires velocity.1 at t0.",
        solver="Matrix exponential for time response; complex tridiagonal solve for transfer",
        observable="Physical leader-relative trajectories and tail displacement; harmonic magnitude |Z_N/Z_0|.",
        input_output="Time: step in leader velocity. Harmonic: unit leader-position sinusoid to tail position. Both position amplitudes measured in same length unit.",
        norm="Sampled absolute frequency-response magnitude. Peak on stated finite grid is a lower bound on H-infinity norm, not a certified supremum.",
        frequency_interval=[1e-8, 0.5],
        peak_frequency_interval=[1e-9, 0.5],
        verification_dependency="mpmath1.4.1; 70-digit recurrence replaces sensitive low-frequency resonance neighborhoods for rho>.5",
        resonance_refinement="Bounded minimization of negative magnitude in log-frequency around every detected coarse-grid local maximum, with inserted shoulders; no guarantee of finding every resonance.",
        relation_to_source="N100, gains, rho values, spacing1, v0.1 and t1000 follow 2012 figure caption. Surrounding text uses unit velocity. Modern frequency grids and size sweep are adaptations.",
        approximation_status="Finite-time responses cannot alone establish asymptotic stability or exponential size scaling. Harmonic maxima are sampled, and narrow resonances can be missed.",
    )
    return data, meta


def draw_matched(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axes = plt.subplots(
            3, 3, figsize=(11, 9), layout="constrained", sharex="row", sharey="row"
        )
        for i, rho in enumerate(meta["rhos"]):
            t = data[f"{i}_t"]
            x = data[f"{i}_xrel"]
            # Every trajectory is retained, as in the historical experiment.
            for k in range(x.shape[1]):
                axes[0, i].plot(
                    x[:, k], t, color=SEQ(0.35 + 0.6 * k / (x.shape[1] - 1)), lw=0.35
                )
            axes[0, i].plot(x[:, -1], t, color=CAT[2], lw=1.1, label="Tail")
            axes[0, i].plot(x[:, 0], t, color=".3", lw=0.8, label="Leader")
            axes[0, i].set(
                title=rf"$\rho={rho:g}$", xlabel="Position relative to leader"
            )
            axes[1, i].loglog(
                data[f"{i}_omega"], abs(data[f"{i}_transfer"]), color=CAT[1]
            )
            axes[1, i].set(xlabel=r"Angular frequency $\omega$")
            axes[2, i].semilogy(
                data["sizes"], data["sampled_peak"][i], "o-", color=CAT[2]
            )
            axes[2, i].set(xlabel="Followers $N$")
        axes[0, 0].set_ylabel("Time $t$")
        axes[0, 2].legend(loc="upper left", fontsize=8)
        axes[1, 0].set_ylabel(r"Harmonic gain $|Z_N/Z_0|$")
        axes[2, 0].set_ylabel("Largest sampled harmonic gain")
        for ax in axes[1:].flat:
            ax.grid(True, alpha=0.35)
        fig.suptitle(
            "Matched position and velocity coupling: fixed-time response and sampled harmonic gain",
            fontsize=11,
        )
        return fig


EXPERIMENTS = {
    "nn_stable_spectra": {"compute": compute_stable, "draw": draw_stable},
    "nn_instability": {"compute": compute_instability, "draw": draw_instability},
    "nn_phase_speeds": {"compute": compute_phase, "draw": draw_phase},
    "nn_pulse_comparison": {"compute": compute_pulse, "draw": draw_pulse},
    "matched_amplification": {"compute": compute_matched, "draw": draw_matched},
}
