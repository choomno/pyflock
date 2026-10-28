"""Transfer roots and finite spectra for the matched-coupling papers."""

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

from .style import CAT, plot_style


def transfer_roots(omega, rho, f=-1.0, g=-1.0):
    """Eq.(14), using the square root with nonnegative imaginary part.

    For omega>=0 and negative f,g, gamma is in the upper half-plane.
    At zero, mu_plus=max(1,kappa), mu_minus=min(1,kappa).
    """
    omega = np.asarray(omega, dtype=float)
    if (
        not 0 < rho < 1
        or not np.isfinite(f)
        or not np.isfinite(g)
        or f >= 0
        or g >= 0
        or np.any(omega < 0)
        or not np.isfinite(omega).all()
    ):
        raise ValueError("Require 0<rho<1, negative f,g, and finite nonnegative omega")
    nu = 1j * omega
    gamma = (f + g * nu - nu**2) / (f + g * nu)
    discriminant = np.sqrt((gamma**2 - 4 * rho * (1 - rho)).astype(complex))
    discriminant = np.where(discriminant.imag < 0, -discriminant, discriminant)
    plus = (gamma + discriminant) / (2 * rho)
    # Product relation avoids cancellation of the smaller root at high frequency.
    minus = ((1 - rho) / rho) / plus
    return np.stack((plus, minus))


def compute_roots(profile="quick", jobs=1, checkpoint=None):
    omega = np.r_[0.0, np.geomspace(1e-7, 1.0, 1600 if profile == "paper" else 600)]
    rhos = np.array([0.4, 0.5, 0.6])
    roots = np.stack([transfer_roots(omega, rho) for rho in rhos])
    meta = {
        "experiment": "matched_transfer_roots",
        "profile": profile,
        "schema": 1,
        "source": "Tangerman, Veerman & Stosic (2012), Asymmetric decentralized flocks, Eqs.(12)–(14), Fig.3",
        "question": "How do spatial transfer roots change across symmetric coupling?",
        "f": -1.0,
        "g": -1.0,
        "rhos": rhos.tolist(),
        "frequency_interval": [0.0, 1.0],
        "geometry": "Interior recurrence; no finite path or boundary imposed",
        "initial_data": "Not applicable: harmonic spatial recurrence",
        "solver": "Quadratic roots",
        "observable": "Complex transfer roots and their modulus ratio",
        "branch_convention": "Square root has argument in [0,pi); mu_plus has larger modulus for omega>0. At rho=.5, omega0 is a double root.",
        "relation_to_source": "Published gains and rho values. Entire finite-frequency loci and full reference circles replace the source local window; a modulus-ratio panel is added. Source does not state its frequency cutoff.",
        "reference_circles": "kappa=(1-rho)/rho; radii sqrt(kappa) and kappa centered at0.",
        "approximation_status": "Exact interior characteristic roots; the ratio equals1 at symmetric DC, so no uniform strict ratio bound is claimed there.",
    }
    return {"omega": omega, "rhos": rhos, "roots": roots}, meta


def draw_roots(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axes = plt.subplots(
            2, 3, figsize=(11, 6.6), layout="constrained", sharex="row", sharey="row"
        )
        angle = np.linspace(0, 2 * np.pi, 401)
        for i, rho in enumerate(data["rhos"]):
            kappa = (1 - rho) / rho
            for radius, col, ls, label in [
                (np.sqrt(kappa), ".55", ":", r"$\sqrt{\kappa}$ circle"),
                (kappa, ".7", "--", r"$\kappa$ circle"),
            ]:
                curve = radius * np.exp(1j * angle)
                axes[0, i].plot(
                    curve.real, curve.imag, color=col, ls=ls, lw=0.8, label=label
                )
            for j, (col, ls, label) in enumerate(
                [(CAT[1], "-", r"$\mu_+$"), (CAT[2], "--", r"$\mu_-$")]
            ):
                roots = data["roots"][i, j]
                axes[0, i].plot(roots.real, roots.imag, color=col, ls=ls, label=label)
                axes[0, i].plot(roots[0].real, roots[0].imag, "o", color=col, ms=3)
            axes[0, i].axhline(0, color=".85", lw=0.5)
            axes[0, i].axvline(0, color=".85", lw=0.5)
            axes[0, i].set(title=rf"$\rho={rho:g}$", xlabel=r"Re $\mu$", aspect="equal")
            ratio = abs(data["roots"][i, 1] / data["roots"][i, 0])
            axes[1, i].plot(data["omega"], ratio, color=CAT[1])
            axes[1, i].axhline(1, color=".7", lw=0.6, ls=":")
            axes[1, i].set(xlabel=r"Angular frequency $\omega$", ylim=(0, 1.04))
            axes[1, i].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel(r"Im $\mu$")
        axes[1, 0].set_ylabel(r"$|\mu_-/\mu_+|$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
        fig.suptitle(
            r"Spatial transfer roots: $f=g=-1$, $0\leq\omega\leq1$; dots mark zero frequency",
            fontsize=11,
        )
        return fig


def _smallest_high_precision(N, rho):
    """Smallest positive path eigenvalue by an adaptive-precision Sturm count.

    For rho>1/2 the smallest eigenvalue decreases geometrically with N.
    Precision and the bisection resolution account for that scale before
    conversion to the package's binary64 output arrays.
    """
    import mpmath as mp

    with mp.workdps(30):
        r = mp.mpf(str(rho))
        decades = max(0, int(mp.ceil(N * mp.log10(r / (1 - r)))))
    precision = max(70, decades + 40)
    with mp.workdps(precision):
        r = mp.mpf(str(rho))
        off = [r * (1 - r)] * (N - 2) + [r]
        tiny = mp.mpf(10) ** (10 - precision)

        def negative_pivots(value):
            pivot = 1 - value
            count = 0
            for k in range(N):
                if abs(pivot) < tiny:
                    pivot = -tiny
                count += int(pivot < 0)
                if k < N - 1:
                    pivot = 1 - value - off[k] / pivot
            return count

        left = mp.mpf("0")
        right = mp.mpf("1")
        for _ in range(int(mp.ceil((precision - 5) * mp.log(10, 2)))):
            middle = (left + right) / 2
            if negative_pivots(middle):
                right = middle
            else:
                left = middle
        result = float((left + right) / 2)
        if result == 0:
            raise FloatingPointError(
                "Smallest path eigenvalue is below the binary64 range"
            )
        return result


def path_modes(N, rho, f=-1.0, g=-2.0):
    """Path modes after excluding the prescribed leader; exploit exact similarity.

    The follower Laplacian is similar to a symmetric tridiagonal matrix with
    diagonal1, bulk off-diagonal -sqrt(rho(1-rho)), terminal -sqrt(rho).
    """
    if (
        isinstance(N, (bool, np.bool_))
        or not isinstance(N, (int, np.integer))
        or N < 2
        or not 0 < rho < 1
    ):
        raise ValueError("N>=2 and 0<rho<1 required")
    if not np.isfinite(f) or not np.isfinite(g):
        raise ValueError("f and g must be finite")
    off = np.full(N - 1, -np.sqrt(rho * (1 - rho)))
    off[-1] = -np.sqrt(rho)
    laplacian = eigvalsh_tridiagonal(np.ones(N), off)
    if rho > 0.5:
        laplacian[0] = _smallest_high_precision(N, rho)
    if np.any(laplacian <= 0):
        raise RuntimeError("Positive path Laplacian lost numerically")
    root = np.sqrt((g * g * laplacian**2 + 4 * f * laplacian).astype(complex))
    modes = np.stack(((g * laplacian + root) / 2, (g * laplacian - root) / 2))
    return laplacian, modes


def compute_spectra(profile="quick", jobs=1, checkpoint=None):
    N = 100
    rhos = np.array([0.45, 0.5, 0.55])
    data = {"rhos": rhos}
    for i, rho in enumerate(rhos):
        laplacian, modes = path_modes(N, rho)
        data[f"{i}_laplacian"] = laplacian
        data[f"{i}_modes"] = modes
    meta = {
        "experiment": "matched_spectra",
        "schema": 1,
        "profile": profile,
        "source": "Veerman (2010), Symmetry and stability of homogeneous flocks, Fig.4; Tangerman, Veerman & Stosic (2012), Asymmetric decentralized flocks, Theorem2.3",
        "N": N,
        "rhos": rhos.tolist(),
        "f": -1.0,
        "g": -2.0,
        "geometry": "Path, N followers plus prescribed leader",
        "boundary": "Regular: last follower uses the predecessor with unit position and velocity weights",
        "initial_data": "Not applicable: finite-path spectral calculation",
        "solver": "Symmetric tridiagonal similarity; 70-digit Sturm bisection replaces the exponentially small Laplacian eigenvalue for rho>.5",
        "observable": "Nonneutral first-order eigenvalues; eight lowest Laplacian modes; slowest-pair values stated in panel titles",
        "neutral_mode": "Leader double zero omitted by forming the follower subsystem, not by deleting computed rightmost modes.",
        "question": "Why is a path spectral gap a poor guide to transient amplification?",
        "relation_to_source": "Numerical companion to a source figure explicitly described as schematic, not a reconstruction of exact plotted points. N100 and f=-1,g=-2 match its trajectory example. Source prose and caption disagree about left/right order; columns here are labeled.",
        "approximation_status": "Actual finite-N modes via exact algebraic reduction. No inference about transient size from eigenvalues alone. Near-origin panels use shared scales; slowest-pair coordinates are stated explicitly.",
        "verification_dependency": "mpmath1.4.1, work precision70 digits",
    }
    return data, meta


def draw_spectra(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axes = plt.subplots(
            2, 3, figsize=(10.8, 6.5), layout="constrained", sharex="row", sharey="row"
        )
        for i, rho in enumerate(data["rhos"]):
            values = data[f"{i}_modes"]
            axes[0, i].plot(
                values.real.ravel(), values.imag.ravel(), ".", color=CAT[1], ms=3
            )
            low = values[:, :8]
            axes[1, i].plot(
                low.real.ravel(), low.imag.ravel(), "o", mfc="none", mec=CAT[1], ms=3
            )
            pair = values[:, 0]
            axes[1, i].set_title(
                f"Eight lowest Laplacian modes\nSlowest pair: {pair[0].real:.2e} ± {abs(pair[0].imag):.2e} i",
                fontsize=9,
            )
            for row in (0, 1):
                axes[row, i].axvline(0, color=".65", lw=0.6)
                axes[row, i].set_xlabel("Real part")
            axes[0, i].set_title(rf"$\rho={rho:g}$: full spectrum")
        from matplotlib.ticker import FixedLocator, FormatStrFormatter

        for ax in axes[1]:
            ax.xaxis.set_major_locator(FixedLocator([-0.03, -0.02, -0.01, 0.0]))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axes[0, 0].set_ylabel("Imaginary part")
        axes[1, 0].set_ylabel("Imaginary part")
        fig.suptitle(
            "Matched-coupling path spectra: N=100, f=-1, g=-2\nComparison with the paper's spectral schematic",
            fontsize=11,
        )
        return fig


EXPERIMENTS = {
    "matched_transfer_roots": {"compute": compute_roots, "draw": draw_roots},
    "matched_spectra": {"compute": compute_spectra, "draw": draw_spectra},
}
