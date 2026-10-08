"""
Figures for evaluating flock simulations.

Each function answers one question:

  spectrum_figure        Is it stable, and how good is the low-frequency approximation?
  circle_vs_line_figure  Does the boundary change the spectrum?
  spacetime_figure       Where does the disturbance go, and how fast?   <- start here
  trajectory_figure      What do the individual orbits look like?
  transient_figure       How big are A_k, T, gamma, and does theory match?
  packet_figure          Does the pulse spread diffusively?
  convergence_figure     How does the error in A_1 scale with N?
  overview_figure        All of the above at a glance, for one stencil.

Colour follows the job it does (see ``pyflock.style``): branch/geometry identity is
categorical with a distinct marker or dash per series, agent index is a single-hue
sequential ramp, and signed fields use a diverging map with a neutral midpoint.
"""
from __future__ import annotations

import numpy as np

from .stencil import Stencil
from .dispersion import nu_x, nu_v, branches, signal_velocities, coefficients, taylor
from .matrices import system_matrix_circle, system_matrix_line, system_matrix_line_leaderless
from .analysis import extrema, theory_amplitudes, eigen_reliability
from .style import (CAT, SEQ, DIVERGING, GRID, INK, INK_2, SURFACE, MARKERS, DASHES,
                    apply_style, agent_colors)

__all__ = ["spectrum_figure", "circle_vs_line_figure", "spacetime_figure",
           "trajectory_figure", "transient_figure", "packet_figure",
           "convergence_figure", "overview_figure", "stability_map_figure",
           "symbol_curve"]


def _legend(ax, loc="best", **kw):
    """Legend with a surface-coloured backing, for panels where marks run under it."""
    lg = ax.legend(loc=loc, frameon=True, framealpha=0.88, facecolor=SURFACE, edgecolor="none",
                   **kw)
    lg.get_frame().set_linewidth(0.0)
    return lg


def _subtitle(st: Stencil, extra: str = "") -> str:
    rx = ", ".join(f"{v:g}" for v in st.rx)
    rv = ", ".join(f"{v:g}" for v in st.rv)
    s = f"$g_x$={st.gx:g}, $g_v$={st.gv:g}   $r_x$=({rx})   $r_v$=({rv})"
    return s + (f"\n{extra}" if extra else "")


def symbol_curve(st: Stencil, n: int = 2001):
    """Both branches over theta in [0, 2 pi) -- the Toeplitz symbol of the line problem."""
    th = np.linspace(0.0, 2 * np.pi, n)
    return th, branches(th, st, labelling="by_imag")


# ------------------------------------------------------------------ spectrum
def spectrum_figure(st: Stencil, orders=(1, 2, 3), ax=None, n: int = 1201):
    """Eigenvalue branches in the complex plane, with low-frequency approximations.

    The exact branches are the Toeplitz symbol of the line problem; the dashed curves
    are the Taylor polynomials i mu th - D th^2 + i beta th^3 truncated at each order.
    """
    import matplotlib.pyplot as plt
    apply_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))

    th = np.linspace(-np.pi, np.pi, n)
    lam = branches(th, st, labelling="continuous")
    for i, lbl in ((0, r"$\lambda_+$"), (1, r"$\lambda_-$")):
        kw = {"dashes": DASHES[i + 1]} if DASHES[i + 1] else {}
        ax.plot(lam[i].real, lam[i].imag, color=CAT[i + 1], lw=1.6,
                label=f"{lbl} (exact)", zorder=3, **kw)

    c = taylor(st, order=max(orders) if orders else 3)
    small = np.linspace(-np.pi / 2, np.pi / 2, 400)
    styles = {1: (0.9, (4, 2)), 2: (1.1, (2, 2)), 3: (1.3, (1, 1.6))}
    for o in orders:
        for i in (0, 1):
            approx = sum(c[i, k] * small ** k for k in range(o + 1))
            ax.plot(approx.real, approx.imag, color=CAT[3], lw=styles[o][0],
                    dashes=styles[o][1], alpha=0.95, zorder=2,
                    label=f"order {o} approx." if i == 0 else None)

    ax.axhline(0, color=GRID, lw=0.9, zorder=1)
    ax.axvline(0, color=INK_2, lw=0.9, ls=":", zorder=1)
    mx = max(abs(lam.real).max(), 1e-3)
    ax.set_xlim(-1.08 * mx, 0.18 * mx)
    ax.set_xlabel(r"Re $\lambda$")
    ax.set_ylabel(r"Im $\lambda$")
    scale = float(np.abs(lam).max())
    mx_re = float(lam.real.max())
    # both branches vanish at theta = 0, where the square root has a branch point;
    # evaluating there costs ~1e-9, so the verdict uses a scale-relative tolerance
    # and the actual value is always printed.
    stable = mx_re < 1e-7 * max(scale, 1.0)
    ax.set_title("Eigenvalue branches  ·  " +
                 (rf"max Re $\lambda$ = {mx_re:+.1e}" if stable
                  else f"ENTERS Re > 0  (max {mx_re:+.1e})"))
    _legend(ax, loc="center")
    if fig:
        fig.text(0.5, -0.02, _subtitle(st), ha="center", fontsize=7, color=INK_2)
        fig.tight_layout()
    return fig if fig else ax


# --------------------------------------------------------- circle versus line
def circle_vs_line_figure(st: Stencil, N: int = 120, bc: str = "drop", ax=None):
    """Circle spectrum against line spectrum: what the boundary condition does."""
    import matplotlib.pyplot as plt
    apply_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))

    ec = np.linalg.eigvals(system_matrix_circle(st, N))
    Ml = system_matrix_line_leaderless(st, N, bc=bc)
    el = np.linalg.eigvals(Ml)
    rel = eigen_reliability(Ml)
    ax.plot(ec.real, ec.imag, MARKERS[1], ms=3.4, color=CAT[1], mew=0,
            alpha=0.85, label=f"circle, N={N}", zorder=2)
    ax.plot(el.real, el.imag, MARKERS[2], ms=3.4, mfc="none", mew=0.9,
            color=CAT[2], label=f"line, N={N} (bc={bc})", zorder=3)
    ax.axvline(0, color=INK_2, lw=0.9, ls=":", zorder=1)

    worst_c = np.sort(ec.real)[-3]
    worst_l = el.real.max()
    if worst_l > max(1e-9, rel["tol"]):
        verdict = "line UNSTABLE"
    elif worst_l < -rel["tol"]:
        verdict = "line stable"
    else:
        verdict = "line: undecidable in double precision"
    ax.set_title(f"Circle vs line spectrum  ·  {verdict}")
    ax.set_xlabel(r"Re $\lambda$")
    ax.set_ylabel(r"Im $\lambda$")
    k_rel, t_rel = rel["kappa"], rel["tol"]
    flag = ("" if rel["reliable"] else
            "\n(line eigenvalues unreliable: " + r"$\kappa$" + f"={k_rel:.1e}, "
            + r"$\pm$" + f"{t_rel:.1e})")
    ax.annotate(f"max Re: circle {worst_c:+.2e}\n              line {worst_l:+.2e}" + flag,
                xy=(0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7.5, color=INK_2)
    _legend(ax, loc="center")
    if fig:
        fig.text(0.5, -0.02, _subtitle(st), ha="center", fontsize=7, color=INK_2)
        fig.tight_layout()
    return fig if fig else ax


# ----------------------------------------------------------------- space-time
def spacetime_figure(res, field: str = "spacing", ax=None, rays: bool = True,
                     max_rays: int = 4, robust: bool = True, pct: float = 99.5):
    """Heat map of the disturbance over (agent, time), with predicted arrival rays.

    field: 'spacing'      z_k - z_{k-1}, the deviation in inter-agent spacing (signed)
           'velocity'     zdot_k
           'displacement' z_k - z_0

    The overlaid lines are the leading-order signal path: out at c_+, back at c_-,
    reflecting at the free end and at the leader.
    """
    import matplotlib.pyplot as plt
    apply_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.8))

    if field == "spacing":
        F = np.diff(res.z, axis=1)
        kk = np.arange(1, res.N + 1)
        lab = r"$z_k - z_{k-1}$"
    elif field == "velocity":
        F = res.zdot
        kk = np.arange(res.N + 1)
        lab = r"$\dot z_k$"
    elif field == "displacement":
        F = res.z - res.z[:, [0]]
        kk = np.arange(res.N + 1)
        lab = r"$z_k - z_0$"
    else:
        raise ValueError("field must be 'spacing', 'velocity' or 'displacement'")

    signed = F.min() < -1e-12 and F.max() > 1e-12
    full = float(np.abs(F).max())
    m = float(np.percentile(np.abs(F), pct)) if robust else full
    if m <= 0:
        m = full if full > 0 else 1.0
    clipped = robust and m < 0.98 * full
    if signed:
        im = ax.pcolormesh(kk, res.t, F, cmap=DIVERGING, vmin=-m, vmax=m,
                           shading="auto", rasterized=True)
    else:
        im = ax.pcolormesh(kk, res.t, F, cmap=SEQ, vmin=0.0, vmax=m,
                           shading="auto", rasterized=True)
    ax.grid(False)

    if rays:
        cp, cm = signal_velocities(res.stencil)
        N = res.N
        t, k, direction = 0.0, 0.0, +1
        drawn = 0
        while t < res.t[-1] and drawn < max_rays:
            speed = cp if direction > 0 else cm
            target = N if direction > 0 else 0
            dt = abs((target - k) / speed)
            t2 = min(t + dt, res.t[-1])
            k2 = k + speed * (t2 - t)
            ax.plot([k, k2], [t, t2], color="k", lw=1.1, ls="--", alpha=0.55,
                    zorder=4, label="signal path" if drawn == 0 else None)
            t, k, direction = t2, k2, -direction
            drawn += 1

    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    note = f"\n(colour clipped at {pct:g}th pct;\nmax |·| = {full:.3g})" if clipped else ""
    cb.set_label(lab + note, fontsize=8, color=INK_2)
    cb.outline.set_visible(False)
    ax.set_xlabel("agent $k$")
    ax.set_ylabel("time $t$")
    ax.set_title(f"Disturbance propagation  ·  {lab}")
    if rays:
        ax.legend(loc="upper right")
    if fig:
        fig.text(0.5, -0.02, _subtitle(res.stencil, f"N={res.N}, bc={res.bc}"),
                 ha="center", fontsize=7, color=INK_2)
        fig.tight_layout()
    return fig if fig else ax


# --------------------------------------------------------------- trajectories
def trajectory_figure(res, stride: int = 1, ax=None, highlight_last: bool = True):
    """Each agent's orbit relative to the leader, coloured by position in the flock."""
    import matplotlib.pyplot as plt
    apply_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.6, 4.6))

    n = res.N + 1
    cols = agent_colors(n)
    for k in range(0, n, stride):
        ax.plot(res.t, res.xrel[:, k], lw=0.7, color=cols[k], alpha=0.9, zorder=2)
    if highlight_last:
        ax.plot(res.t, res.xrel[:, -1], lw=1.8, color=CAT[2], zorder=4,
                label=f"last agent (k={res.N})")
        ax.legend(loc="best")

    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, res.N))
    cb = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("agent $k$", fontsize=8.5, color=INK_2)
    cb.outline.set_visible(False)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$x_k(t) - x_0(t)$")
    ax.set_title("Orbits relative to the leader")
    if fig:
        fig.text(0.5, -0.02, _subtitle(res.stencil, f"N={res.N}, bc={res.bc}"),
                 ha="center", fontsize=7, color=INK_2)
        fig.tight_layout()
    return fig if fig else ax


# ------------------------------------------------------------------ transient
def transient_figure(res, nmax: int = 6, ax=None, show_theory: bool = True):
    """The last agent's orbit w = z_N - z_0, with the extrema A_k marked."""
    import matplotlib.pyplot as plt
    apply_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))

    w = res.last_relative
    ax.plot(res.t, w, color=CAT[1], lw=1.5, zorder=3, label="simulation")

    ex = extrema(res.t, w, nmax=nmax)
    if ex:
        ax.plot([e.t for e in ex], [e.value for e in ex], "o", ms=6,
                mfc="none", mew=1.4, color=CAT[2], zorder=5, label=r"extrema $A_k$")
        span = max(abs(w).max(), 1e-30)
        shown, last_t = 0, -np.inf
        for i, e in enumerate(ex, start=1):
            if shown >= 3 or abs(e.value) < 0.05 * span:
                continue
            if e.t - last_t < 0.08 * (res.t[-1] - res.t[0]):
                continue
            ax.annotate(f"$A_{{{i}}}$={e.value:.1f}", xy=(e.t, e.value),
                        xytext=(7, 9 if e.value < 0 else -15),
                        textcoords="offset points", fontsize=7.5, color=INK_2)
            shown, last_t = shown + 1, e.t

    if show_theory:
        th = theory_amplitudes(res.stencil, res.N, v0=res.v0, kmax=nmax)
        tt = [0.0] + list(th["T_k"])
        aa = [0.0] + list(th["A"])
        keep = [i for i, t in enumerate(tt) if t <= res.t[-1]]
        ax.plot([tt[i] for i in keep], [aa[i] for i in keep], color=CAT[3],
                lw=1.3, dashes=(4, 2), zorder=4, label="Theorem 3 (leading order)")

    ax.axhline(0, color=GRID, lw=0.9, zorder=1)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$z_N(t) - z_0(t)$")
    ax.set_title("Transient of the last agent")
    ax.legend(loc="best")
    if fig:
        fig.text(0.5, -0.02, _subtitle(res.stencil, f"N={res.N}, bc={res.bc}"),
                 ha="center", fontsize=7, color=INK_2)
        fig.tight_layout()
    return fig if fig else ax


# --------------------------------------------------------------------- packet
def packet_figure(st: Stencil, N: int = 4096, times=(50, 100, 200, 400),
                  axes=None, gaussian: bool = True, window: float = 5.0):
    """A localised pulse on the circle at several times, against the diffusive prediction.

    Initial data is a unit displacement impulse.  Linearity splits it into two packets
    whose masses are fixed by the signal velocities alone,

        forward (c_+):  |c_-| / (c_+ + |c_-|)        backward (c_-):  c_+ / (c_+ + |c_-|)

    and the theory says each travels at c_sigma while spreading with variance 2 D_sigma t.
    The dashed curves are those Gaussians -- amplitude, centre and width all predicted,
    nothing fitted.
    """
    import matplotlib.pyplot as plt
    from .solve import simulate_circle
    apply_style()

    times = list(times)
    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(times), 1, figsize=(6.8, 1.6 * len(times)),
                                 sharex=False)
        axes = np.atleast_1d(axes)

    z0 = np.zeros(N); z0[0] = 1.0; z0 -= z0.mean()
    zd0 = np.zeros(N)
    res = simulate_circle(st, N, times, z0, zd0)
    k = ((np.arange(N) + N // 2) % N) - N // 2
    order = np.argsort(k)
    ks = k[order]

    cp, cm = signal_velocities(st)
    co = coefficients(st)
    tot = cp + abs(cm)
    weight = {"+": abs(cm) / tot, "-": cp / tot}       # mass carried by each packet
    colour = {"+": CAT[2], "-": CAT[3]}

    for ax, t, row in zip(axes, times, res.z):
        ax.plot(ks, row[order], color=CAT[1], lw=1.3, zorder=3,
                label="simulation" if t == times[0] else None)
        lo, hi = [], []
        for lbl, sl in co.items():
            if sl["D"] <= 0:
                continue
            s = np.sqrt(2 * sl["D"] * t)
            mu = sl["c"] * t
            g = weight[lbl] * np.exp(-0.5 * ((ks - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
            if gaussian:
                ax.plot(ks, g, color=colour[lbl], lw=1.2, dashes=(3, 2), alpha=0.95,
                        zorder=4,
                        label=(rf"$c_{{{lbl}}}$ packet: mass {weight[lbl]:.3f}, "
                               rf"$\sigma=\sqrt{{2D_{{{lbl}}}t}}$") if t == times[0] else None)
            lo.append(mu - window * s); hi.append(mu + window * s)
        if lo:
            ax.set_xlim(min(lo), max(hi))
        sp = np.sqrt(2 * co["+"]["D"] * t); sm_ = np.sqrt(2 * co["-"]["D"] * t)
        ax.set_ylabel(f"$t$={t:g}\n" + rf"$\sigma_+$={sp:.1f}  $\sigma_-$={sm_:.1f}",
                      fontsize=8)
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("agent $k$ (relative to the impulse)")
    axes[0].set_title("Pulse propagation and diffusive spreading")
    axes[0].legend(loc="upper center", ncol=3, fontsize=7.2)
    if fig:
        fig.text(0.5, -0.01, _subtitle(st, f"circle, N={N}"), ha="center",
                 fontsize=7, color=INK_2)
        fig.tight_layout()
    return fig if fig else axes


# ---------------------------------------------------------------- convergence
def convergence_figure(st: Stencil, Ns=(50, 100, 200, 400), bc: str = "regular",
                       v0: float = 1.0, ax=None, nt_per_unit: int = 30):
    """Relative error in |A_1| against N, for the leading-order and corrected formulas.

    Theorem 3 gives |A_1| = N v0/c_+; its relative error decays like C N^(-1/2) with the
    constant C = (1-r) phi(x*) sqrt(2 D_+/c_+) predicted from the theta^2 coefficient
    (solid reference line, nothing fitted).  Adding the diffusive correction leaves a
    residual that decays one order faster.
    """
    import matplotlib.pyplot as plt
    from .solve import simulate_line
    from .analysis import corrected_amplitudes, relative_error_law
    apply_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))

    cp, _ = signal_velocities(st)
    Ns = list(Ns)
    lead, corr = [], []
    for N in Ns:
        tmax = 1.6 * N / cp
        res = simulate_line(st, N, tmax=tmax, nt=int(nt_per_unit * tmax), bc=bc, v0=v0)
        A1 = abs(extrema(res.t, res.last_relative, nmax=1)[0].value)
        lead.append(abs(N * v0 / cp - A1) / A1)
        corr.append(abs(corrected_amplitudes(st, N, v0=v0, kmax=1)["A"][0] - A1) / A1)

    lead = np.array(lead); corr = np.array(corr); Na = np.array(Ns, dtype=float)
    ax.loglog(Na, lead, MARKERS[1], ms=7.5, color=CAT[1], mew=0, zorder=5,
              label=r"leading order  $Nv_0/c_+$")
    ax.loglog(Na, corr, MARKERS[2], ms=6.5, color=CAT[2], mew=0, zorder=5,
              label=r"with diffusive correction")

    xs = np.logspace(np.log10(min(Ns) * 0.75), np.log10(max(Ns) * 1.4), 120)
    C = relative_error_law(st, v0=v0)
    ax.loglog(xs, C / np.sqrt(xs), color=CAT[1], lw=1.3, dashes=(5, 2), alpha=0.8,
              zorder=3, label=rf"predicted  $C N^{{-1/2}}$, $C$={C:.3f}")
    ax.loglog(xs, corr[0] * Na[0] / xs, color=CAT[2], lw=1.2, dashes=(2, 2), alpha=0.8,
              zorder=3, label=r"$\propto N^{-1}$")

    s1 = np.polyfit(np.log(Na), np.log(lead), 1)[0]
    s2 = np.polyfit(np.log(Na), np.log(corr), 1)[0]
    ax.annotate(f"fitted slopes:  leading {s1:+.2f}   corrected {s2:+.2f}",
                xy=(0.02, 0.03), xycoords="axes fraction", fontsize=8, color=INK_2)
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"relative error in $|A_1|$")
    ax.set_title("What the second-order term buys")
    ax.legend(loc="upper right", fontsize=7.6)
    if fig:
        fig.text(0.5, -0.02, _subtitle(st, f"bc={bc}"), ha="center", fontsize=7,
                 color=INK_2)
        fig.tight_layout()
    return fig if fig else ax


# ------------------------------------------------------------------- overview
def overview_figure(st: Stencil, N: int = 150, bc: str = "regular", v0: float = 1.0):
    """Four-panel summary of one stencil: spectrum, circle-vs-line, space-time, transient."""
    import matplotlib.pyplot as plt
    from .solve import simulate_line
    apply_style()

    cp, cm = signal_velocities(st)
    P = N * (1.0 / cp - 1.0 / cm) if cp != 0 and cm != 0 else 4.0 * N
    tmax = N / cp + 2.2 * abs(P)
    res = simulate_line(st, N, tmax=tmax, nt=int(12 * tmax), bc=bc, v0=v0)

    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.4))
    spectrum_figure(st, ax=axs[0, 0])
    circle_vs_line_figure(st, N=min(N, 120), bc=bc, ax=axs[0, 1])
    spacetime_figure(res, ax=axs[1, 0])
    transient_figure(res, ax=axs[1, 1])

    co = coefficients(st)
    head = (f"{st.name or 'stencil'}   ·   $c_+$={cp:.4f}, $c_-$={cm:.4f}   ·   "
            f"$D_+$={co['+']['D']:.4f}, $D_-$={co['-']['D']:.4f}   ·   $a$={st.a:.4f}")
    fig.suptitle(head, fontsize=11)
    fig.text(0.5, 0.005, _subtitle(st, f"N={N}, bc={bc}"), ha="center", fontsize=7.5,
             color=INK_2)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    return fig


# --------------------------------------------------------------------- stability map
def _zero_crossings(x, y):
    """Linearly interpolated abscissae where y changes sign."""
    out = []
    for k in range(len(x) - 1):
        a, b = y[k], y[k + 1]
        if a == 0.0:
            out.append(float(x[k]))
        elif a * b < 0.0:
            out.append(float(x[k] - a * (x[k + 1] - x[k]) / (b - a)))
    if len(y) and y[-1] == 0.0:
        out.append(float(x[-1]))
    return out


def stability_map_figure(smap, figsize=None):
    """Where is the line flock stable?  Numerics over a (parameter, N) grid against theory.

    Left: N * max Re(lambda) of the leaderless line matrix -- blue stable, red unstable --
    scaled by N because the least-stable eigenvalue itself shrinks like 1/N.  The solid line
    is the numerical boundary Re = 0; open circles mark where the theory puts it, one per N.
    Cells whose eigenvalue fails the round-off check are crossed out.

    Right (when the map carries a theory): N * (theory - numerics) in the same units, so the
    size of the error reads directly against the size of the signal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    apply_style()

    has_theory = smap.theory is not None
    ncol = 2 if has_theory else 1
    fig, axs = plt.subplots(1, ncol, figsize=figsize or (6.3 * ncol, 4.9))
    axs = np.atleast_1d(axs)
    X, Y = np.meshgrid(smap.params, smap.Ns)
    Z = smap.max_re * Y
    m = float(np.abs(Z).max()) or 1.0

    ax = axs[0]
    im = ax.pcolormesh(X, Y, Z, cmap=DIVERGING, vmin=-m, vmax=m, shading="nearest",
                       rasterized=True)
    ax.grid(False)
    handles = []
    if Z.min() < 0.0 < Z.max():
        ax.contour(X, Y, Z, levels=[0.0], colors=INK, linewidths=1.6)
        handles.append(Line2D([], [], color=INK, lw=1.6, label="numerical boundary"))
    if has_theory:
        T = smap.theory * Y
        pts = [(qc, N) for N, row in zip(smap.Ns, T) for qc in _zero_crossings(smap.params, row)]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", ms=6.5, mfc=SURFACE,
                    mec=INK, mew=1.2, zorder=5)
            handles.append(Line2D([], [], marker="o", ls="none", ms=6.5, mfc=SURFACE, mec=INK,
                                  mew=1.2, label="predicted boundary"))
    if smap.reliable is not None and (~smap.reliable).any():
        bad = ~smap.reliable
        ax.plot(X[bad], Y[bad], "x", color=INK, ms=5, mew=1.0, zorder=6)
        handles.append(Line2D([], [], color=INK, marker="x", ls="none",
                              label="eigenvalue unreliable"))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$N \cdot \max\,\mathrm{Re}\,\lambda$" + "\n(blue stable, red unstable)",
                 fontsize=8, color=INK_2)
    cb.outline.set_visible(False)
    ax.set_xlabel(f"${smap.param_name}$")
    ax.set_ylabel("$N$")
    ax.set_title("Line stability")
    if handles:
        _legend(ax, loc="upper right", handles=handles)

    if has_theory:
        ax2 = axs[1]
        E = (smap.theory - smap.max_re) * Y
        e = float(np.abs(E).max()) or 1.0
        im2 = ax2.pcolormesh(X, Y, E, cmap=DIVERGING, vmin=-e, vmax=e, shading="nearest",
                             rasterized=True)
        ax2.grid(False)
        cb2 = fig.colorbar(im2, ax=ax2, pad=0.02)
        cb2.set_label(r"$N \cdot$ (theory $-$ numerics)", fontsize=8, color=INK_2)
        cb2.outline.set_visible(False)
        ax2.set_xlabel(f"${smap.param_name}$")
        ax2.set_ylabel("$N$")
        ax2.set_title(f"Theory error  ·  max |error| = {e / m:.2%} of max |signal|")

    head = f"Line stability over (${smap.param_name}$, $N$)"
    if smap.label:
        head += f"  ·  {smap.label}"
    head += f"  ·  bc={smap.bc}"
    fig.suptitle(head, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
