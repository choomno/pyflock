"""Figures for individual scientific questions. Computations live in experiments.py.

Returned Matplotlib figures remain editable. None of these functions installs a
style globally. Figure metadata and full parameters belong in the companion record.
"""

from functools import wraps
import numpy as np
from .style import CAT, SEQ, DIVERGING, plot_style
from .dispersion import branches, taylor, signal_velocities
from .analysis import extrema, theory_amplitudes, eigen_reliability
from .matrices import system_matrix_line_leaderless


def _styled(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with plot_style():
            return fn(*args, **kwargs)

    return wrapped


def _axes(ax, size=(6.6, 4.1)):
    import matplotlib.pyplot as plt

    if ax is not None:
        return None, ax
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    return fig, ax


def circle_eigenvalues(st, N, *, include_neutral=False):
    """Remove the known zero Fourier mode, never the largest computed eigenvalues."""
    modes = np.arange(0 if include_neutral else 1, N)
    return branches(2 * np.pi * modes / N, st, "by_imag").ravel()


def symbol_curve(st, n=2001):
    th = np.linspace(-np.pi, np.pi, n)
    return th, branches(th, st, "by_imag")


@_styled
def spectrum_figure(st, orders=(1, 2, 3), ax=None, n=1001):
    """Full symbol, then a separate local error view for analytic branches."""
    import matplotlib.pyplot as plt

    if ax is not None:
        th, lam = symbol_curve(st, n)
        ax.plot(lam.real.T, lam.imag.T, color=CAT[1], lw=1)
        ax.axvline(0, color="#777777", lw=0.7)
        ax.set(
            xlabel=r"$\operatorname{Re}\lambda$",
            ylabel=r"$\operatorname{Im}\lambda$",
            title="Periodic symbol",
        )
        return ax
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    spectrum_figure(st, ax=axs[0], n=n)
    small = np.geomspace(0.002, 0.4, 180)
    try:
        co = taylor(st, order=max(orders or (3,)))
    except ValueError as exc:
        axs[1].text(
            0.05,
            0.5,
            "No regular two-branch expansion\n" + str(exc),
            transform=axs[1].transAxes,
            wrap=True,
        )
        axs[1].set_axis_off()
        return fig
    exact = branches(small, st)
    for o, color, ls in zip(orders, CAT.values(), ("-", "--", ":")):
        approx = sum(co[:, j, None] * small**j for j in range(o + 1))
        err = np.max(abs(exact - approx), axis=0)
        axs[1].loglog(
            small,
            np.maximum(err, np.finfo(float).eps),
            color=color,
            ls=ls,
            label=f"Order {o}",
        )
    axs[1].set(
        xlabel=r"Wavenumber $\theta$",
        ylabel="Maximum branch error",
        title="Local approximation error",
    )
    axs[1].legend()
    axs[1].grid(True, which="major", alpha=0.55)
    return fig


@_styled
def circle_vs_line_figure(st, N=80, bc="drop", ax=None):
    fig, ax = _axes(ax)
    ec = circle_eigenvalues(st, N)
    matrix = system_matrix_line_leaderless(st, N, bc=bc)
    el = np.linalg.eigvals(matrix)
    rel = eigen_reliability(matrix)
    ax.scatter(ec.real, ec.imag, s=10, color=CAT[1], label=f"Circle: {N} agents")
    ax.scatter(
        el.real,
        el.imag,
        s=14,
        facecolors="none",
        edgecolors=CAT[2],
        linewidths=0.7,
        label=f"Path: {N} followers",
    )
    ax.axvline(0, lw=0.8, color="#777777")
    ax.set(
        xlabel=r"$\operatorname{Re}\lambda$",
        ylabel=r"$\operatorname{Im}\lambda$",
        title="Boundary changes the spectrum",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0))
    note = f"Largest real part\ncircle {ec.real.max():+.3g}\npath {el.real.max():+.3g}"
    if not rel["reliable"]:
        note += "\nSensitive path spectrum"
    ax.text(
        0.97,
        0.04,
        note,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )
    return fig if fig else ax


@_styled
def spacetime_figure(
    res, field="spacing", ax=None, rays=True, max_rays=4, robust=False, pct=99.5
):
    fig, ax = _axes(ax, (6.7, 4.6))
    if field == "spacing":
        data, k, label = (
            np.diff(res.z, axis=1),
            np.arange(1, res.N + 1),
            r"$z_k-z_{k-1}$",
        )
    elif field == "velocity":
        data, k, label = res.zdot, np.arange(res.N + 1), r"$\dot z_k$"
    elif field == "displacement":
        data, k, label = res.z - res.z[:, [0]], np.arange(res.N + 1), r"$z_k-z_0$"
    else:
        raise ValueError("field must be spacing, velocity or displacement")
    full = float(abs(data).max())
    limit = float(np.percentile(abs(data), pct)) if robust else full
    limit = max(limit, 1e-12)
    signed = data.min() < -1e-12
    low = -limit if signed else 0
    im = ax.pcolormesh(
        k,
        res.t / res.N,
        data,
        cmap=DIVERGING if signed else SEQ,
        vmin=low,
        vmax=limit,
        shading="nearest",
        rasterized=True,
    )
    if rays:
        # A ray is an approximation of the interior wave, not an assertion of path stability.
        try:
            cp, cm = signal_velocities(res.stencil)
        except ValueError:
            cp = cm = 0
        if cp > 0 > cm:
            time, pos, direction = 0.0, 0.0, 1
            for i in range(max_rays):
                speed = cp if direction == 1 else cm
                end = res.N if direction == 1 else 0
                new_time = time + (end - pos) / speed
                if time >= res.t[-1]:
                    break
                stop = min(new_time, res.t[-1])
                ax.plot(
                    [pos, pos + speed * (stop - time)],
                    [time / res.N, stop / res.N],
                    color="#333333",
                    lw=1,
                    ls="--",
                    label="Interior wave prediction" if i == 0 else None,
                )
                time, pos, direction = new_time, end, -direction
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.025, shrink=0.94)
    cb.set_label(label)
    if limit < 0.99 * full:
        ax.text(
            0.99,
            0.02,
            f"Colour clipped; max |field| = {full:.3g}",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
        )
    ax.set(
        xlabel="Follower index $k$",
        ylabel="Time $t/N$",
        title="Propagation and reflection",
    )
    return fig if fig else ax


@_styled
def trajectory_figure(res, stride=None, ax=None, highlight_last=True):
    """Selected physical orbits relative to the leader, time on the vertical axis."""
    fig, ax = _axes(ax, (6.6, 4.6))
    ks = (
        np.unique(np.rint(np.linspace(0, res.N, 7)).astype(int))
        if stride is None
        else np.arange(0, res.N + 1, stride)
    )
    if highlight_last:
        ks = ks[ks != res.N]
    for k, col in zip(ks, SEQ(np.linspace(0.4, 1, len(ks)))):
        ax.plot(
            res.xrel[:, k] / res.N, res.t / res.N, color=col, lw=1, label=f"$k={k}$"
        )
    if highlight_last:
        ax.plot(
            res.xrel[:, -1] / res.N,
            res.t / res.N,
            color=CAT[2],
            lw=1.7,
            label="Last follower",
        )
    ax.set(
        xlabel=r"Position relative to leader $(x_k-x_0)/N$",
        ylabel="Time $t/N$",
        title="Selected follower trajectories",
    )
    ax.legend(ncol=2, fontsize=8)
    return fig if fig else ax


@_styled
def transient_figure(res, nmax=4, ax=None, show_theory=True):
    fig, ax = _axes(ax)
    ax.plot(
        res.t / res.N,
        res.last_relative / res.N,
        color=CAT[1],
        label="Computed response",
    )
    ex = extrema(res.t, res.last_relative, nmax=nmax)
    if ex:
        ax.scatter(
            [e.t / res.N for e in ex],
            [e.value / res.N for e in ex],
            s=16,
            color=CAT[1],
            zorder=4,
            label="Measured extrema",
        )
    if show_theory:
        try:
            th = theory_amplitudes(res.stencil, res.N, v0=res.v0, kmax=nmax)
        except ValueError:
            th = None
        if th is not None:
            keep = np.asarray(th["T_k"]) <= res.t[-1]
            ax.scatter(
                np.asarray(th["T_k"])[keep] / res.N,
                np.asarray(th["A"])[keep] / res.N,
                marker="D",
                s=28,
                linewidths=1.1,
                facecolors="none",
                edgecolors=CAT[2],
                zorder=5,
                label="Leading asymptotic extrema",
            )
    ax.axhline(0, lw=0.6, color="#999999")
    ax.set(
        xlabel="Time $t/N$",
        ylabel=r"Last follower $(z_N-z_0)/N$",
        title="Response of the last follower",
    )
    ax.legend(loc="upper right", fontsize=8)
    return fig if fig else ax


@_styled
def packet_figure(st, N=2048, times=(40, 80, 160), ax=None):
    """Smooth periodic velocity pulse in a common agent coordinate."""
    from .solve import simulate_circle

    k = np.arange(N)
    centre = N // 4
    pulse = np.exp(-0.5 * ((k - centre) / 10) ** 2)
    res = simulate_circle(st, N, np.asarray(times), np.zeros(N), pulse)
    fig, ax = _axes(ax)
    for i, time in enumerate(times):
        ax.plot(k - centre, res.zdot[i], lw=1.3, label=f"$t={time:g}$")
    cp, cm = signal_velocities(st)
    if (
        min(cm * max(times), cp * max(times)) - 50 < -centre
        or max(cm * max(times), cp * max(times)) + 50 > N - centre
    ):
        raise ValueError(
            "pulse reaches the plotted periodic seam; increase N or reduce times"
        )
    ax.set_xlim(min(0, cm * max(times)) - 50, max(0, cp * max(times)) + 50)
    ax.set(
        xlabel="Agent offset from pulse centre",
        ylabel="Velocity",
        title="A smooth pulse on the circle",
    )
    ax.legend()
    return fig if fig else ax


@_styled
def convergence_figure(
    st, Ns=(50, 100, 200, 400), bc="regular", v0=1.0, ax=None, nt_per_unit=None
):
    """Single-stencil error study; distinct from the paper's parameter-family study."""
    from .solve import simulate_line
    from .analysis import corrected_amplitudes

    fig, ax = _axes(ax)
    cp, _ = signal_velocities(st)
    leading, corrected = [], []
    for N in Ns:
        r = simulate_line(
            st,
            N,
            tmax=1.6 * N / cp,
            nt=1601 if nt_per_unit is None else max(3, int(nt_per_unit * 1.6 * N / cp)),
            bc=bc,
            v0=v0,
        )
        measured = abs(extrema(r.t, r.last_relative, nmax=1)[0].value)
        prediction = N * abs(v0) / cp
        leading.append(abs(prediction - measured) / prediction)
        correction = corrected_amplitudes(st, N, v0=v0, kmax=1)["A"][0]
        corrected.append(abs(correction - measured) / prediction)
    ax.loglog(Ns, leading, "o-", color=CAT[1], label="Leading prediction")
    ax.loglog(Ns, corrected, "s--", color=CAT[2], label="Formal diffusive correction")
    ax.set(
        xlabel="Followers $N$",
        ylabel="Absolute error / leading amplitude",
        title="First-amplitude approximation",
    )
    ax.legend()
    ax.grid(True, alpha=0.5)
    return fig if fig else ax


@_styled
def overview_figure(st, N=120, bc="regular", v0=1.0):
    import matplotlib.pyplot as plt
    from .solve import simulate_line

    try:
        cp, cm = signal_velocities(st)
        tmax = N / cp + 2.2 * N * (1 / cp - 1 / cm) if cp > 0 > cm else 5 * N
    except ValueError:
        tmax = 5 * N
    res = simulate_line(
        st,
        N,
        tmax=tmax,
        nt=1601,
        bc=bc,
        v0=v0,
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    spacetime_figure(res, ax=axs[0])
    transient_figure(res, ax=axs[1])
    return fig


@_styled
def stability_map_figure(smap, figsize=None):
    import matplotlib.pyplot as plt

    count = 2 if smap.theory is not None else 1
    fig, axs = plt.subplots(
        1,
        count,
        figsize=figsize or (6.2 * count, 4),
        layout="constrained",
        squeeze=False,
    )
    values = smap.Ns[:, None] * smap.max_re
    limit = max(float(abs(values).max()), 1e-12)
    for ax, z, title in [(axs[0, 0], values, "Path spectral abscissa")] + (
        [
            (
                axs[0, 1],
                smap.Ns[:, None] * (smap.max_re - smap.theory),
                "Formal approximation error",
            )
        ]
        if count == 2
        else []
    ):
        im = ax.pcolormesh(
            smap.params,
            smap.Ns,
            z,
            cmap=DIVERGING,
            vmin=-limit,
            vmax=limit,
            shading="nearest",
            rasterized=True,
        )
        if z.min() < 0 < z.max():
            ax.contour(
                smap.params, smap.Ns, z, levels=[0], colors="#333333", linewidths=1
            )
        yy, xx = np.where(~smap.reliable) if smap.reliable is not None else ([], [])
        if len(xx):
            ax.scatter(
                smap.params[xx],
                smap.Ns[yy],
                marker="x",
                s=20,
                color="black",
                label="Sensitive eigenvalue",
            )
            ax.legend(fontsize=8)
        ax.set(xlabel=smap.param_name, ylabel="Followers $N$", title=title)
        fig.colorbar(
            im,
            ax=ax,
            label=r"$N\,\operatorname{Re}\lambda$"
            if ax is axs[0, 0]
            else r"$N\,\Delta\operatorname{Re}\lambda$",
        )
    return fig
