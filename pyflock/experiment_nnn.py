"""Next nearest neighbour examples and restricted validation inspired by Figure 8.

The paper's ensemble members are unavailable. The ensemble here has an explicit
new selection rule and is distinguished from the sample used in Figure 8.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from math import comb
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse

from .analysis import extrema
from .dispersion import signal_velocities
from .solve import simulate_line
from .stencil import Stencil
from .style import CAT, SEQ, plot_style

SOURCE = "Herbrych et al., Dynamics of locally coupled agents with next nearest neighbor interaction, manuscript Figures 5–8, Section 6, Theorems 5.2–5.3."
EXAMPLES = (
    ("I", 200, (-0.5, 0.25, 1, -0.75, 0), (-1, 0.75, 1, -1, 0.25), 1000),
    ("II", 200, (1, -2, 1, 0, 0), (-0.5, -1, 1, 0.5, 0), 1000),
    ("III", 50, (-2, 15 / 4, 1, -21 / 4, 5 / 2), (-1, 4, 1, -5, 1), 500),
    ("III", 100, (-2, 15 / 4, 1, -21 / 4, 5 / 2), (-1, 4, 1, -5, 1), 2500),
)
BOUNDARIES = ("drop", "fixed_mass")
ALGORITHM_VERSION = "nnn-stable-families-v3"


@cache
def _checkpoint_version():
    """Bind a process's numerical checkpoints to source and arithmetic versions."""
    digest = sha256(
        f"{ALGORITHM_VERSION};numpy={np.__version__};scipy={scipy.__version__}".encode()
    )
    for name in (
        Path(__file__).name,
        "analysis.py",
        "dispersion.py",
        "matrices.py",
        "solve.py",
        "stencil.py",
    ):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _record(st):
    return {"rx": st.rx.tolist(), "rv": st.rv.tolist(), "gx": st.gx, "gv": st.gv}


def _base(name, profile):
    return {
        "experiment": name,
        "schema": 1,
        "profile": profile,
        "source": SOURCE,
        "convention": "N followers and leader 0; z_k=x_k+k; w=z_N-z_0.",
        "initial_data": "Zero displacement; all followers at rest; leader velocity changes from 0 to 1 at t=0.",
        "gains": [-2, -2],
        "approximation_status": "Path-wave predictions depend on the periodic-to-path conjectures; a numerical convergence diagnostic is separate from their asymptotic error.",
    }


def _path_matrix(st, N, bc):
    """Paper Definition 2.2, including its independently documented clamp rule."""
    if bc not in BOUNDARIES or N < 4:
        raise ValueError("requires N>=4 and boundary 'drop' or 'fixed_mass'")
    blocks = []
    for weights in (st.rx, st.rv):
        rows, cols, values = [], [], []
        for k in range(1, N + 1):
            if bc == "fixed_mass":
                for j, value in zip(range(-2, 3), weights):
                    rows.append(k)
                    cols.append(min(max(k + j, 0), N))
                    values.append(value)
            else:
                diagonal = 0.0
                for j, value in zip(range(-2, 3), weights):
                    if j and 0 <= k + j <= N:
                        rows.append(k)
                        cols.append(k + j)
                        values.append(value)
                        diagonal -= value
                rows.append(k)
                cols.append(k)
                values.append(diagonal)
        blocks.append(
            sparse.coo_matrix((values, (rows, cols)), shape=(N + 1, N + 1)).tocsr()
        )
    return sparse.bmat(
        [[None, sparse.eye(N + 1)], [st.gx * blocks[0], st.gv * blocks[1]]],
        format="csr",
    )


def compute_types(profile="quick", jobs=1, checkpoint=None):
    arrays, specs = {}, []
    for index, (kind, N, rx, rv, end) in enumerate(EXAMPLES):
        st = Stencil(rx, rv, -2, -2)
        result = simulate_line(st, N, end, nt=3001, bc="drop")
        arrays[f"{index}_t"] = result.t
        arrays[f"{index}_z"] = result.z
        arrays[f"{index}_velocity"] = result.zdot
        specs.append(
            {
                "type": kind,
                "N": N,
                "boundary": "fixed interaction",
                "stencil": _record(st),
                "speeds": list(signal_velocities(st)),
            }
        )
    meta = _base("nnn-types", profile)
    meta.update(
        question="How do reflected waves, reflectionless waves, and waves directed toward the leader differ?",
        examples=specs,
        solver="Floating-point matrix exponential, dense for N<200 and sparse otherwise.",
        source_corrections=[
            "Figure 7 prints rho_x,-1=-15/4, violating the zero row sum by -15/2. Using +15/4 restores the required zero row sum and zero first position moment."
        ],
        relation_to_source="Figures 5–7, original follower counts, gains, fixed-interaction boundary and displayed time horizons.",
    )
    return arrays, meta


def draw_types(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.3), layout="constrained")
        titles = (
            "Type I: reflected waves",
            "Type II: reflectionless waves",
            "Type III: smaller path",
            "Type III: larger path",
        )
        for index, (ax, spec, title) in enumerate(
            zip(axes.flat, meta["examples"], titles)
        ):
            t, z = data[f"{index}_t"], data[f"{index}_z"]
            N = spec["N"]
            for k in np.unique(np.linspace(0, N, 9, dtype=int))[1:-1]:
                ax.plot(
                    t / N, (z[:, k] - t) / N, color=SEQ(0.25 + 0.65 * k / N), lw=0.8
                )
            ax.plot(
                t / N, (z[:, N] - t) / N, color=CAT[1], lw=1.7, label="Last follower"
            )
            ax.axhline(0, color="#777777", lw=0.65)
            ax.set(
                title=title,
                xlabel=r"Time $t/N$",
                ylabel=r"Leader-relative displacement $(z_k-z_0)/N$",
            )
            cp, cm = spec["speeds"]
            ax.set_title(
                title + "\n" + rf"$N={N},\quad c_+={cp:.2f},\quad c_-={cm:.2f}$",
                fontsize=10,
            )
            if index < 2:
                ax.axvline(
                    1 / cp, color=CAT[2], ls="--", lw=0.9, label=r"Arrival $N/c_+$"
                )
            ax.grid(True, alpha=0.35)
        axes[0, 0].legend(loc="lower right", fontsize=8)
        return fig


def _configuration(kind, index):
    """One member of a finite, analytically stable 20-by-25 parameter grid."""
    i, j = divmod(index, 25)
    if kind == 0:
        u, q = np.linspace(0.05, 0.20, 20)[i], np.linspace(0.10, 0.30, 25)[j]
        return Stencil([0, -0.5, 1, -0.5, 0], [-u, -(1 - u - q), 1, -q, 0], -2, -2)
    a, b = np.linspace(2, 4, 20)[i], np.linspace(0.4, 1, 25)[j]
    return Stencil([1, -2, 1, 0, 0], [0, -1, 1, 0, 0], -a * b, -(a + b))


def select_configurations(count=500):
    """The explicit restricted families; no filtering by measured outcomes."""
    if count < 1 or count > 500:
        raise ValueError("count must be between 1 and 500")
    indices = np.linspace(0, 499, count, dtype=int)
    records, classes, lattice_indices = [], [], []
    for kind in (0, 1):
        for index in indices:
            st = _configuration(kind, int(index))
            records.append(np.r_[st.gx, st.gv, st.rx, st.rv])
            classes.append(kind)
            lattice_indices.append(index)
    return np.asarray(records), np.asarray(classes), np.asarray(lattice_indices)


EXPERIMENTS = {
    "nnn-types": {"compute": compute_types, "draw": draw_types},
}


@dataclass
class TailTrace:
    t: np.ndarray
    relative: np.ndarray
    nfft: int
    damping: float
    period_factor: int
    convergence_error: float | None = None
    convergence_normalization: float | None = None


def _transfer_group(st, N, s, a, lo, hi, bc):
    p = 2 - lo
    q = hi - 2
    degree = p + q
    a = a[:, lo : hi + 1]
    # q(r)=r**p*[gx sum rx_j*r**j+s gv sum rv_j*r**j-s**2].
    # Shift q(1+u), evaluating q(1),q'(1) by structural identities.
    b = np.stack(
        [
            sum(comb(j, k) * a[:, j] for j in range(k, degree + 1))
            for k in range(degree + 1)
        ],
        axis=1,
    )
    b[:, 0] = -s * s
    b[:, 1] = s * st.moment_v(1) - p * s * s
    companion = np.zeros((len(s), degree, degree), complex)
    companion[:, 0, :] = -b[:, degree - 1 :: -1] / b[:, degree, None]
    if degree > 1:
        companion[:, 1:, :-1] = np.eye(degree - 1)
    u = np.linalg.eigvals(companion)
    # log1p retains spatial phase precision for roots 1+O(1/N).
    logroot = np.log1p(u)
    origin = np.where(logroot.real <= 0, 0, N)

    def basis(k):
        return np.exp(logroot * (k - origin))

    equations = [basis(0)]
    rhs = [np.ones(len(s), complex)]
    for k in list(range(1, p)) + list(range(N - q + 1, N + 1)):
        residual = np.zeros_like(u)
        for j in range(-p, q + 1):
            if 0 <= k + j <= N:
                continue
            replacement = k if bc == "drop" else min(max(k + j, 0), N)
            residual += a[:, j + p, None] * (basis(k + j) - basis(replacement))
        equations.append(residual)
        rhs.append(np.zeros(len(s), complex))
    E = np.stack(equations, axis=1)
    scale = np.max(abs(E), axis=2)
    E /= scale[:, :, None]
    rhs = np.stack(rhs, axis=1) / scale
    coeff = np.linalg.solve(E, rhs[..., None])[..., 0]
    result = np.sum(coeff * basis(N), axis=1)
    if not np.isfinite(result).all():
        raise FloatingPointError(
            "Nonfinite spatial transfer; inspect degenerate roots or boundary equations"
        )
    return result


def spatial_transfer(st, N, s, bc="drop"):
    """Return H_N(s)=Z_N(s)/Z_0(s), zero follower initial data.

    Computes a degree<=4 characteristic polynomial and boundary linear system
    at each s. Cost and storage depend on the number of frequencies, not on N.
    """
    if bc not in ("drop", "fixed_mass"):
        raise ValueError("bc must be 'drop' or 'fixed_mass'")
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 4:
        raise ValueError("N must be an integer>=4")
    if not st.is_decentralized(tol=1e-12) or abs(st.moment_x(1)) > 1e-12:
        raise ValueError(
            "Requires decentralized stencils and zero position first moment"
        )
    s = np.atleast_1d(np.asarray(s, dtype=complex))
    if s.ndim != 1 or not np.isfinite(s).all() or np.any(s.real <= 0):
        raise ValueError("s must be finite vector in the open right half-plane")
    a = st.gx * st.rx[None, :] + s[:, None] * st.gv * st.rv[None, :]
    a[:, 2] -= s * s
    # Exact cancellations can reduce spatial order at individual frequencies.
    # Group those frequencies; avoid division by a vanished leading coefficient.
    active = abs(a) > np.max(abs(a), axis=1)[:, None] * 1e-15
    lo = np.argmax(active, axis=1)
    hi = 4 - np.argmax(active[:, ::-1], axis=1)
    result = np.empty(len(s), complex)
    for left, right in sorted(set(zip(lo.tolist(), hi.tolist()))):
        mask = (lo == left) & (hi == right)
        if left >= 2 or right < 2:
            raise ValueError("Requires at least one coupling towards the leader")
        result[mask] = _transfer_group(st, N, s[mask], a[mask], left, right, bc)
    return result


def tail_trace(
    st, N, tmax, *, bc="drop", nfft=8192, damping=32.0, period_factor=4, v0=1.0
):
    """Uniform trace from0 to tmax, using nfft/period_factor+1 points.

    Invert absolute tail displacement H_N(s)*v0/s**2, then subtract v0*t.
    This avoids the artificial high-frequency 1/s**2 tail of relative motion.
    The exponentially damped periodic copies introduce an alias sensitivity, diagnosed
    by changing the contour and frequency resolution; these are not rigorous error bounds.
    """
    if not np.isfinite(tmax) or tmax <= 0 or not np.isfinite(v0):
        raise ValueError("tmax must be positive finite; v0 must be finite")
    if (
        isinstance(period_factor, (bool, np.bool_))
        or not isinstance(period_factor, (int, np.integer))
        or period_factor < 2
    ):
        raise ValueError("period_factor must be an integer>=2")
    if (
        isinstance(nfft, (bool, np.bool_))
        or not isinstance(nfft, (int, np.integer))
        or nfft < 32
        or nfft % period_factor
    ):
        raise ValueError("nfft must be integer>=32 divisible by period_factor")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive finite")
    period = period_factor * tmax
    a = damping / period
    frequencies = a + 2j * np.pi * np.arange(nfft // 2 + 1) / period
    F = v0 * spatial_transfer(st, N, frequencies, bc=bc) / (frequencies * frequencies)
    t = np.arange(nfft // period_factor + 1) * period / nfft
    z = np.fft.irfft(F, n=nfft)[: len(t)] * nfft / period * np.exp(a * t)
    return TailTrace(t, z - v0 * t, nfft, damping, period_factor)


def converged_tail(
    st,
    N,
    tmax,
    *,
    bc="drop",
    start_nfft=2048,
    max_nfft=262144,
    tolerance=1e-8,
    damping=32.0,
    period_factor=4,
    v0=1.0,
):
    """Double frequency/time resolution until max trace change/(N*|v0|)<=tol.

    Raises if convergence is not achieved. Final nfft and measured change are
    returned, so large ensembles can archive per-run numerical diagnostics.
    This comparison controls truncation/discretization, not contour aliasing.
    """
    if (
        isinstance(start_nfft, (bool, np.bool_))
        or not isinstance(start_nfft, (int, np.integer))
        or start_nfft < 32
    ):
        raise ValueError("start_nfft must be an integer>=32")
    if (
        isinstance(max_nfft, (bool, np.bool_))
        or not isinstance(max_nfft, (int, np.integer))
        or max_nfft < 2 * start_nfft
    ):
        raise ValueError("max_nfft must be an integer>=2*start_nfft")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive finite")
    scale = max(N * abs(v0), np.finfo(float).tiny)
    old = tail_trace(
        st,
        N,
        tmax,
        bc=bc,
        nfft=start_nfft,
        damping=damping,
        period_factor=period_factor,
        v0=v0,
    )
    nfft = start_nfft * 2
    while nfft <= max_nfft:
        new = tail_trace(
            st,
            N,
            tmax,
            bc=bc,
            nfft=nfft,
            damping=damping,
            period_factor=period_factor,
            v0=v0,
        )
        error = float(np.max(abs(new.relative[::2] - old.relative)) / scale)
        if error <= tolerance:
            new.convergence_error = error
            new.convergence_normalization = scale
            return new
        old = new
        nfft *= 2
    raise RuntimeError(
        f"No trace convergence at N={N}, nfft<={max_nfft}; last relative change={error:g}"
    )


def _peak(t, w, centre, radius, positive=False):
    indices = np.flatnonzero((t >= max(0, centre - radius)) & (t <= centre + radius))
    if len(indices) < 3:
        return 0.0, 0.0, False
    local = np.argmax(w[indices]) if positive else np.argmin(w[indices])
    i = int(indices[local])
    if i <= 0 or i >= len(t) - 1 or local in (0, len(indices) - 1):
        return 0.0, 0.0, False
    refined = extrema(t[i - 1 : i + 2], w[i - 1 : i + 2], nmax=1)
    if not refined:
        return 0.0, 0.0, False
    return refined[0].t, refined[0].value, True


def _linear_zero(t, w, lo, hi):
    """Zero of the continuous least-squares line on a fixed physical interval."""
    inside = (t > lo) & (t < hi)
    if inside.sum() < 2:
        return 0.0, False
    x = np.r_[lo, t[inside], hi]
    y = np.interp(x, t, w)
    h = np.diff(x)
    integral = np.sum(h * (y[:-1] + y[1:]) / 2)
    first_moment = np.sum(
        h * (x[:-1] * (y[:-1] + y[1:]) / 2 + h * (y[:-1] / 6 + y[1:] / 3))
    )
    width, centre = hi - lo, (hi + lo) / 2
    slope = 12 * (first_moment - centre * integral) / width**3
    intercept = integral / width - slope * centre
    return (float(-intercept / slope), True) if slope > 0 else (0.0, False)


def _measure(kind, st, N, t, w):
    cp, cm = signal_velocities(st)
    first = N / cp
    if kind == 0:
        P = N * (1 / cp - 1 / cm)
        prediction = np.array([first, 2 * P, abs(cm / cp)])
        peaks = [
            _peak(t, w, first + j * P, 0.65 * P, positive=bool(j % 2)) for j in range(3)
        ]
        values = np.array([abs(p[1]) for p in peaks])
        valid = np.array([peaks[0][2], False, all(p[2] for p in peaks)])
        crossing = np.flatnonzero(
            (w[:-1] * w[1:] <= 0) & (t[:-1] > 1.35 * P) & (t[1:] < 2.65 * P)
        )
        period = 0.0
        if len(crossing):
            i = int(crossing[np.argmin(abs(t[crossing] - 2 * P))])
            if w[i + 1] != w[i]:
                period = t[i] - w[i] * (t[i + 1] - t[i]) / (w[i + 1] - w[i])
                valid[1] = True
        attenuation = (
            float(np.mean(values[1:] / values[:-1]))
            if valid[2] and np.all(values[:-1] > 0)
            else 0.0
        )
        measured = np.array([values[0], period, attenuation])
    else:
        second = N / cm
        prediction = np.array([first, first, second])
        _, amplitude, found = _peak(
            t, w, first, min(0.43 * (second - first), 0.8 * first)
        )
        zero, fit_ok = _linear_zero(
            t / N,
            w / N,
            (first + 0.3 * (second - first)) / N,
            (first + 0.7 * (second - first)) / N,
        )
        response = N * zero
        fit_ok = fit_ok and first < response < 1.3 * second
        measured = np.array([abs(amplitude), abs(amplitude), response])
        valid = np.array([found, found, fit_ok])
    return measured, prediction, valid


def _ensemble_case(spec):
    kind, index, record, boundary, N, checkpoint = spec
    st = Stencil(record[2:7], record[7:12], *record[:2])
    fingerprint = sha256(
        (
            _checkpoint_version()
            + json.dumps([kind, index, record.tolist(), boundary, N])
        ).encode()
    ).hexdigest()[:20]
    dest = Path(checkpoint) / f"{fingerprint}.npz" if checkpoint else None
    if dest is not None and dest.exists() and dest.with_suffix(".sha256").exists():
        expected = dest.with_suffix(".sha256").read_text().strip()
        if sha256(dest.read_bytes()).hexdigest() == expected:
            try:
                with np.load(dest, allow_pickle=False) as saved:
                    row = saved["row"]
                if (
                    row.shape == (23,)
                    and np.isfinite(row).all()
                    and np.array_equal(
                        row[:4], [kind, BOUNDARIES.index(boundary), N, index]
                    )
                ):
                    return row
            except (KeyError, ValueError, OSError):
                pass
    cp, cm = signal_velocities(st)
    end = 3.4 * N * (1 / cp - 1 / cm) if kind == 0 else 1.25 * N / cm
    # A row always contains finite values; validity and status are explicit.
    row = np.zeros(23)
    row[:4] = kind, BOUNDARIES.index(boundary), N, index
    trace = converged_tail(st, N, end, bc=boundary, start_nfft=1024, tolerance=2e-8)
    measured, prediction, valid = _measure(kind, st, N, trace.t, trace.relative)
    coarse, _, coarse_valid = _measure(kind, st, N, trace.t[::2], trace.relative[::2])
    change = np.abs(measured - coarse) / prediction
    # Refine the observable interpolation separately from the full-trace test.
    while np.any((valid & coarse_valid) & (change > 2e-6)) and trace.nfft < 262144:
        old = trace
        trace = tail_trace(st, N, end, bc=boundary, nfft=old.nfft * 2)
        trace.convergence_error = float(
            np.max(abs(trace.relative[::2] - old.relative)) / N
        )
        measured, prediction, valid = _measure(kind, st, N, trace.t, trace.relative)
        coarse, _, coarse_valid = _measure(
            kind, st, N, trace.t[::2], trace.relative[::2]
        )
        change = np.abs(measured - coarse) / prediction
    contour = tail_trace(st, N, end, bc=boundary, nfft=trace.nfft, damping=40)
    contour_change = float(np.max(abs(trace.relative - contour.relative)) / N)
    row[4:7], row[7:10] = measured, prediction
    row[10:13] = np.where(valid, abs(measured - prediction) / prediction, 0)
    row[13:16] = change
    row[16:19] = valid.astype(int)
    row[19] = trace.convergence_error
    row[20] = contour_change
    row[21] = trace.nfft
    row[22] = (
        int(not valid.all())
        + 2 * int(contour_change > 1e-7)
        + 4 * int(np.max(change) > 2e-6)
        + 8 * int(trace.convergence_error > 2e-8)
    )
    if not np.isfinite(row).all():
        raise RuntimeError(
            f"Nonfinite NNN measurement: class={kind}, configuration={index}, N={N}, boundary={boundary}"
        )
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Keep a compact audit trail around every detected extremum and a sparse
        # global trace. Full time histories need not accumulate in the parent.
        selected = list(np.linspace(0, len(trace.t) - 1, 81, dtype=int))
        for extremum in extrema(trace.t, trace.relative, nmax=8):
            selected.extend(
                range(max(0, extremum.index - 2), min(len(trace.t), extremum.index + 3))
            )
        selected = np.unique(selected)
        temporary = dest.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            row=row,
            audit_t=trace.t[selected],
            audit_w=trace.relative[selected],
        )
        temporary.replace(dest)
        dest.with_suffix(".sha256").write_text(
            sha256(dest.read_bytes()).hexdigest() + "\n"
        )
    return row


def compute_ensemble(profile="quick", jobs=1, checkpoint=None):
    if profile not in ("quick", "paper") or jobs < 1:
        raise ValueError("profile must be quick/paper and jobs must be positive")
    count = 500 if profile == "paper" else 6
    Ns = [25 * 2**n for n in range(12 if profile == "paper" else 4)]
    records, classes, lattice = select_configurations(count)
    if checkpoint:
        directory = Path(checkpoint)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "configurations.npz",
            configurations=records,
            classes=classes,
            lattice_indices=lattice,
        )
    specs = [
        (int(kind), i, record, bc, N, str(checkpoint) if checkpoint else None)
        for i, (kind, record) in enumerate(zip(classes, records))
        for bc in BOUNDARIES
        for N in Ns
    ]
    rows = []
    if jobs == 1:
        results = map(_ensemble_case, specs)
        for i, row in enumerate(results, 1):
            rows.append(row)
            if profile == "paper" and i % 250 == 0:
                print(f"NNN ensemble: {i}/{len(specs)} cases", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, row in enumerate(pool.map(_ensemble_case, specs), 1):
                rows.append(row)
                if profile == "paper" and i % 250 == 0:
                    print(f"NNN ensemble: {i}/{len(specs)} cases", flush=True)
    rows = np.asarray(rows)
    meta = _base("nnn-ensemble", profile)
    meta.update(
        question="Do the mean and worst errors decrease across a specified ensemble and both physical boundary rules?",
        relation_to_source="New restricted deterministic ensemble inspired by Figure 8. The paper profile matches its class counts, sizes and boundaries. The paper's members and selection grid are unavailable; the new Type II gains vary rather than being fixed at -2.",
        gains={
            "Type I": [-2, -2],
            "Type II": "gx=-a*b, gv=-(a+b), with a in [2,4] and b in [.4,1]",
        },
        source_selection_status="The parameter members and complete selection grid for Figure 8 were unavailable. The deterministic families below specify this experiment completely.",
        deterministic_selection={
            "Type I": "20 equally spaced u in [.05,.20] times 25 equally spaced q in [.10,.30]; p=1-u-q; rx=(0,-.5,1,-.5,0), rv=(-u,-p,1,-q,0), gains(-2,-2)",
            "Type II": "20 equally spaced a in [2,4] times 25 equally spaced b in [.4,1]; rx=(1,-2,1,0,0), rv=(0,-1,1,0,0), gains(-a*b,-a-b)",
            "enumeration": "First parameter outer loop, second inner loop. Quick profile uses six equally spaced flat indices from 0 to 499 in each class; paper uses all 500.",
            "filters": "None. Every specified parameter pair is retained; no selection using measured errors.",
        },
        stability_proof={
            "Type I": "For both boundaries, reduced K=Lx is symmetric positive definite. S=(Lv+Lv.T)/2 has nonpositive off-diagonal entries and weak diagonal dominance, with strictly positive boundary margins and a connected graph, hence is positive definite. With gains -2, E=(||v||^2+2 z^T K z)/2 satisfies E'=-2 v^T S v; the follower system is Hurwitz. The periodic symbols have positive stiffness and strictly positive damping for every nonzero wavenumber.",
            "Type II": "D(theta)=1-exp(-i theta): characteristic polynomial=(lambda+a D)(lambda+b D), so the periodic branches are stable, with speeds a,b. Each physical path is lower triangular in follower 2-by-2 blocks. Their characteristic polynomials are (lambda+a)(lambda+b), except fixed-interaction follower1 has lambda^2+(a+b)lambda+2ab; all are Hurwitz for a,b>0. Every positive Bromwich contour is therefore right of all follower poles.",
        },
        boundaries={
            "drop": "Paper fixed interaction: discard missing interactions and adjust diagonal",
            "fixed_mass": "Paper fixed mass: clamp missing indices to endpoint, without terminal rescaling; use the velocity weights for the velocity endpoint term",
        },
        Ns=Ns,
        configurations_per_class=count,
        completed_cases=len(rows),
        required_cases=len(specs),
        cases_with_all_valid_observables=int(np.all(rows[:, 16:19] == 1, axis=1).sum()),
        cases_passing_numerical_diagnostics=int((rows[:, 22] == 0).sum()),
        solver="Spatial characteristic polynomial (degree at most 4), boundary solve, inverse Laplace FFT. Analytic finite-path stability for the specified restricted families places every follower pole left of the positive Bromwich contour. Resolution and contour variation are numerical sensitivity diagnostics.",
        numerical_checks="Trace doubling until max change/N<=2e-8; observable sampling change/prediction<=2e-6; contour damping 32 versus 40, flagged above 1e-7*N. These are numerical diagnostics, distinct from asymptotic error.",
        estimators={
            "Type I": "Extrema are sought in prediction-guided windows N/c_plus+jP +/- .65P, j=0,1,2. A1 is the first absolute lobe maximum; T is the crossing closest to 2P in [1.35P,2.65P]; |alpha| is the mean of the first two successive absolute peak ratios. Undefined measurements are masked explicitly.",
            "Type II": "A is the first absolute minimum in a window around predicted N/c_plus, T1=A/v0 (so their relative errors are identical by construction); T2 is the zero intercept of a linear fit over the middle 30%–70% between predicted arrival times. This explicitly specified estimator is an adaptation of Figure 8.",
        },
        columns=[
            "class",
            "boundary",
            "N",
            "configuration",
            "measured_A",
            "measured_T_or_T1",
            "measured_alpha_or_T2",
            "predicted_A",
            "predicted_T_or_T1",
            "predicted_alpha_or_T2",
            "relative_error_A",
            "relative_error_T_or_T1",
            "relative_error_alpha_or_T2",
            "sampling_change_A",
            "sampling_change_T_or_T1",
            "sampling_change_alpha_or_T2",
            "valid_A",
            "valid_T_or_T1",
            "valid_alpha_or_T2",
            "trace_refinement_change_over_N",
            "contour_change_over_N",
            "nfft",
            "status",
        ],
        status_codes="0=all checks pass; bit1=undefined observable, bit2=contour variation, bit4=observable sampling variation, bit8=trace refinement variation",
        configuration_columns=[
            "gx",
            "gv",
            "rho_x,-2",
            "rho_x,-1",
            "rho_x,0",
            "rho_x,1",
            "rho_x,2",
            "rho_v,-2",
            "rho_v,-1",
            "rho_v,0",
            "rho_v,1",
            "rho_v,2",
        ],
    )
    return {
        "measurements": rows,
        "configurations": records,
        "classes": classes,
        "lattice_indices": lattice,
        "completion": np.ones(len(rows), dtype=int),
    }, meta


def draw_ensemble(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 6.8), layout="constrained", sharex=True, sharey="row"
        )
        rows = data["measurements"]
        unresolved = []
        meta["rendered_error_sensitivity"] = {
            "rule": "For each case and observable, proxy=max(1e-7,5*saved_observable_sampling_change,10*(saved_trace_refinement_change_over_N+saved_contour_change_over_N)). Compare each arithmetic mean/max error with the corresponding mean/max proxy. Resolved statistics retain their true error value; unresolved statistics use unconnected downward triangles at their proxy level. This is a conservative display convention informed by the recorded diagnostics, not a certified error bound.",
            "raw_data": "All uncensored errors remain in data.npz and case checkpoints.",
        }
        for kind, axisrow in enumerate(axes):
            for bi, ax in enumerate(axisrow):
                subset = rows[(rows[:, 0] == kind) & (rows[:, 1] == bi)]
                series = (
                    ((0, r"$A_1$"), (1, r"$T$"), (2, r"$|\alpha|$"))
                    if kind == 0
                    else ((0, r"$A,T_1$"), (2, r"$T_2$"))
                )
                for j, label in series:
                    Ns, means, worst, mean_sensitivity, max_sensitivity = (
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                    for N in meta["Ns"]:
                        selected = subset[
                            (subset[:, 2] == N)
                            & (subset[:, 16 + j] == 1)
                            & (subset[:, 22] == 0)
                        ]
                        if len(selected):
                            values = selected[:, 10 + j]
                            sensitivity = np.maximum.reduce(
                                (
                                    np.full(len(selected), 1e-7),
                                    5 * selected[:, 13 + j],
                                    10 * (selected[:, 19] + selected[:, 20]),
                                )
                            )
                            Ns.append(N)
                            means.append(float(values.mean()))
                            worst.append(float(values.max()))
                            mean_sensitivity.append(float(sensitivity.mean()))
                            max_sensitivity.append(float(sensitivity.max()))
                    color = CAT[j + 1]
                    for values, sensitivity, statistic, marker, linestyle in (
                        (means, mean_sensitivity, "mean", "o", "-"),
                        (worst, max_sensitivity, "maximum", "s", "--"),
                    ):
                        values, sensitivity = (
                            np.asarray(values),
                            np.asarray(sensitivity),
                        )
                        resolved = values >= sensitivity
                        ax.loglog(
                            Ns,
                            np.where(resolved, values, np.nan),
                            marker=marker,
                            ls=linestyle,
                            color=color,
                            ms=3.5,
                            mfc=color if statistic == "mean" else "white",
                            label=f"{label}: {statistic}",
                        )
                        if np.any(~resolved):
                            x = np.asarray(Ns)[~resolved]
                            ax.loglog(
                                x,
                                sensitivity[~resolved],
                                "v",
                                color=color,
                                ms=5,
                                mfc=color if statistic == "mean" else "white",
                                ls="none",
                            )
                            unresolved.extend(
                                {
                                    "class": kind,
                                    "boundary": bi,
                                    "observable": j,
                                    "statistic": statistic,
                                    "N": int(N),
                                    "raw_error": float(value),
                                    "sensitivity": float(level),
                                }
                                for N, value, level in zip(
                                    x, values[~resolved], sensitivity[~resolved]
                                )
                            )
                boundary = "fixed interaction" if bi == 0 else "fixed mass"
                ax.set_title(f"Type {'I' if kind == 0 else 'II'} · {boundary}")
                ax.grid(True, alpha=0.4)
                ax.legend(fontsize=7.8, ncol=2, loc="best")
        for ax in axes[-1]:
            ax.set_xlabel("Followers $N$")
        for ax in axes[:, 0]:
            ax.set_ylabel("Relative error / prediction")
        fig.suptitle(
            f"Restricted NNN validation: {meta['configurations_per_class']} configurations per class",
            fontsize=11,
        )
        fig.supxlabel(
            "Type II gains vary: $g_x=-ab$, $g_v=-(a+b)$; $A$ and $T_1$ share one estimator.\n"
            + r"$\triangledown$ Unresolved against recorded numerical sensitivity; display floor $10^{-7}$.",
            fontsize=7.8,
        )
        meta["rendered_error_sensitivity"]["unresolved_aggregates"] = unresolved
        return fig


EXPERIMENTS["nnn-ensemble"] = {"compute": compute_ensemble, "draw": draw_ensemble}
