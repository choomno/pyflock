"""Lyons thesis parameter families and tests of the heterogeneous approximation.

The original random samples are unavailable. Fixed seeds below define new,
fully saved samples; no seed is selected for a desired stability outcome.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import find_peaks
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .heterogeneous import (
    HeterogeneousNN,
    c1_slopes,
    gain_moments,
    simulate_circle,
    simulate_path,
    simulate_tail,
    system_matrix,
)
from .style import CAT, plot_style

SOURCE = {
    "title": "Linear Nearest Neighbor Flocks With All Distinct Agents",
    "author": "Robert G. Lyons",
    "year": 2022,
    "doi": "10.15760/etd.7774",
    "url": "https://pdxscholar.library.pdx.edu/open_access_etds/5903/",
}
SEED = 20221016


def _base(name, profile):
    if profile not in ("quick", "paper"):
        raise ValueError("profile must be 'quick' or 'paper'")
    return {
        "experiment": name,
        "schema": 1,
        "profile": profile,
        "source": SOURCE,
        "seed": SEED,
        "random_generator": "numpy PCG64",
        "convention": "N counts all agents 0,...,N-1; positive gain magnitudes with an overall minus sign in the ODE.",
        "position_weights": [-0.5, 1.0, -0.5],
        "velocity_weights": "[-(1+beta_v)/2, 1, (beta_v-1)/2] in offsets [-1,0,+1]",
        "boundary": "regular tail: gamma_x=gamma_v=1; leader 0 autonomous",
        "initial_data": "Path: z=0, follower velocities=0, leader velocity=1 at t=0 (abrupt ramp limit).",
        "solver": "Sparse matrix exponential action; independent DOP853 checks on explicitly recorded shorter intervals.",
        "relation_to_source": "Parameter-family reconstruction with new fixed random samples; the thesis does not supply its random seeds or gain arrays.",
        "approximation_status": "Lyons equation (107) is a repeated-cell low-frequency prediction. Its use for a unique-agent path adds the assumptions in sections 2.5.4, 2.6 and Remark 2.8.1; it is not a finite-system stability test.",
    }


def _theory(model):
    plus, minus = c1_slopes(model)
    cp, cm = -minus, -plus  # velocities along increasing agent index
    length = model.n_agents - 1
    return {
        "c_plus": cp,
        "c_minus": cm,
        "T1": length / cp,
        "P": length * (1 / cp - 1 / cm),
        "ratio": abs(cm / cp),
        **gain_moments(model),
    }


def _refine(t, y, i):
    # Uniform grids only: interpolation is local and never extrapolates.
    a = (y[i - 1] + y[i + 1] - 2 * y[i]) / 2
    b = (y[i + 1] - y[i - 1]) / 2
    shift = -b / (2 * a) if a else 0.0
    if abs(shift) > 1:
        shift = 0.0
    return float(t[i] + shift * (t[i + 1] - t[i])), float(
        y[i] + b * shift + a * shift**2
    )


def measure_response(t, tail):
    """First negative minimum and following positive maximum of a tail trace.

    Undefined quantities use finite zero placeholders and explicit masks.
    The prominence threshold suppresses round-off wiggles; no predicted peak
    time or amplitude enters the estimator.
    """
    t, tail = np.asarray(t, float), np.asarray(tail, float)
    if (
        t.ndim != 1
        or tail.shape != t.shape
        or t.size < 5
        or not np.isfinite(t).all()
        or not np.isfinite(tail).all()
        or np.any(np.diff(t) <= 0)
        or not np.allclose(np.diff(t), t[1] - t[0])
    ):
        raise ValueError(
            "t and tail must be finite equally spaced vectors of length >=5"
        )
    result = {
        "T1": 0.0,
        "A1": 0.0,
        "T2": 0.0,
        "A2": 0.0,
        "ratio": 0.0,
        "arrival_valid": False,
        "ratio_valid": False,
        "status": "no_first_minimum",
    }
    scale = float(np.max(abs(tail)))
    threshold = max(1e-11, scale * 1e-6)
    minima = find_peaks(-tail, prominence=threshold)[0]
    minima = minima[tail[minima] < -threshold]
    if not minima.size:
        return result
    first = int(minima[0])
    result["T1"], result["A1"] = _refine(t, tail, first)
    result.update(arrival_valid=True, status="no_following_positive_maximum")
    maxima = find_peaks(tail, prominence=threshold)[0]
    maxima = maxima[(maxima > first) & (tail[maxima] > threshold)]
    if maxima.size:
        result["T2"], result["A2"] = _refine(t, tail, int(maxima[0]))
        result["ratio"] = abs(result["A2"] / result["A1"])
        result.update(ratio_valid=True, status="ok")
    return result


def _solver_check(model):
    times = np.linspace(0, min(10.0, model.n_agents / 8), 31)
    exponential = simulate_path(model, times)
    rk = simulate_path(model, times, method="rk", rtol=2e-13, atol=2e-15)
    return {
        "interval": [0, float(times[-1])],
        "times": len(times),
        "reference_method": "DOP853",
        "rtol": 2e-13,
        "atol": 2e-15,
        "max_displacement_difference": float(np.max(abs(exponential.r - rk.r))),
        "max_velocity_difference": float(np.max(abs(exponential.rdot - rk.rdot))),
    }


def _spectrum(model, geometry):
    """Sample near zero with residuals; this does not compute spectral abscissa."""
    matrix = system_matrix(model, geometry=geometry, followers_only=geometry == "path")
    count = min(18, matrix.shape[0] - 2)
    shift = (0.1 + 0.2j) / model.n_agents
    status = "converged"
    try:
        values, vectors = eigs(
            matrix.astype(complex),
            k=count,
            sigma=shift,
            which="LM",
            tol=2e-10,
            v0=np.random.default_rng(SEED + 17).standard_normal(matrix.shape[0]),
            maxiter=15000,
        )
    except ArpackNoConvergence as error:
        values, vectors = error.eigenvalues, error.eigenvectors
        status = "partial_ARPACK_convergence"
    if values is None or len(values) == 0:
        return (
            np.empty(0, complex),
            np.empty(0),
            {
                "status": "no_eigenpairs",
                "shift": [shift.real, shift.imag],
                "neutral_modes_removed": 0,
            },
        )
    order = np.argsort(abs(values))
    values, vectors = values[order], vectors[:, order]
    norm = float(np.max(np.asarray(abs(matrix).sum(axis=1))))
    residual = np.linalg.norm(matrix @ vectors - vectors * values, axis=0)
    residual /= (norm + abs(values)) * np.linalg.norm(vectors, axis=0)
    return (
        values,
        residual,
        {
            "status": status,
            "shift": [shift.real, shift.imag],
            "neutral_modes_removed": 0,
            "coherent_modes": "Circle has a known double zero eigenvalue; all returned eigenpairs are retained, including any approximations to those neutral modes."
            if geometry == "circle"
            else "Follower operator excludes the prescribed leader's two coordinates.",
            "meaning": "Nearest eigenpairs to one complex shift; not the complete spectrum. Residuals are backward-error diagnostics, not forward-error bounds for nonnormal eigenvalues. Partial convergence may omit any physical or neutral mode.",
        },
    )


def _map_cases(function, specs, jobs):
    if jobs == 1:
        return list(map(function, specs))
    with ProcessPoolExecutor(max_workers=min(2, jobs)) as executor:
        return list(executor.map(function, specs))


def _boundary_case(spec):
    family, gx, gv, beta, geometry, nt = spec
    started = time.perf_counter()
    model = HeterogeneousNN(gx, gv, beta)
    theory = _theory(model)
    times = np.linspace(0, theory["T1"] + 1.25 * theory["P"], nt)
    chosen = np.unique(np.linspace(0, model.n_agents - 1, 17).astype(int))
    if geometry == "path":
        response = simulate_path(model, times)
        relative = response.r
    else:
        initial_velocity = np.zeros(model.n_agents)
        initial_velocity[0] = 1
        response = simulate_circle(
            model, times, np.zeros(model.n_agents), initial_velocity
        )
        relative = response.z - response.z[:, [0]]
    values, residuals, spectral_meta = _spectrum(model, geometry)
    arrays = {
        "t": times,
        "tail": relative[:, -1],
        "selected_agents": chosen,
        "selected_displacements": relative[:, chosen],
        "max_displacement": np.max(abs(relative), axis=1),
        "eigenvalues": values,
        "eigen_residuals": residuals,
    }
    return arrays, {
        "family": family,
        "N_total": model.n_agents,
        "beta_v": beta,
        "geometry": geometry,
        "theory": theory,
        "spectral_sample": spectral_meta,
        "seconds": time.perf_counter() - started,
        "initial_data": "z=0, velocity[0]=1 and all other velocities=0. Circle: free evolution; path: leader retains velocity 1.",
        "solver_check": _solver_check(model)
        if geometry == "path" and beta > 0
        else None,
    }


def _compute_boundary(profile, jobs):
    meta = _base("lyons_boundary", profile)
    sizes = [400, 1600] if profile == "paper" else [80, 160]
    bounds = [([1, 5], [3, 13]), ([1, 5], [2, 18])]
    rng = np.random.default_rng(SEED)
    arrays, families, specs = {}, [], []
    for family, (n, (xb, vb)) in enumerate(zip(sizes, bounds)):
        gx, gv = rng.uniform(*xb, n), rng.uniform(*vb, n)
        arrays[f"family_{family}_gx"], arrays[f"family_{family}_gv"] = gx, gv
        families.append({"N_total": n, "gx_uniform": xb, "gv_uniform": vb})
        for beta in (0.3, -0.3):
            for geometry in ("circle", "path"):
                specs.append(
                    (
                        family,
                        gx,
                        gv,
                        beta,
                        geometry,
                        1001 if profile == "paper" else 401,
                    )
                )
    cases = []
    for index, (data, record) in enumerate(_map_cases(_boundary_case, specs, jobs)):
        arrays.update({f"case_{index}_{key}": value for key, value in data.items()})
        cases.append(record)
    meta.update(
        question="How do boundaries and velocity-asymmetry direction alter a fixed heterogeneous array?",
        sources="Lyons thesis Figures 8–15, printed pp.85–91; section2.9.",
        families=families,
        cases=cases,
        ensemble_size=1,
        spectral_scope="Near-origin samples only. A sampled positive real part suggests an instability and needs sensitivity checks; absence of one does not establish stability.",
        quick_profile="Uses smaller arrays with the same gain-distribution bounds."
        if profile == "quick"
        else None,
    )
    return arrays, meta


def _response_case(spec):
    family, sample, gx, gv, beta, nt, check = spec
    model = HeterogeneousNN(gx, gv, beta)
    theory = _theory(model)
    initial_nt, refinements = nt, 0
    while True:
        t = np.linspace(0, theory["T1"] + 1.35 * theory["P"], nt)
        _, tail = simulate_tail(model, t)
        fine = measure_response(t, tail)
        coarse = measure_response(t[::2], tail[::2])
        arrival_sampling = fine["arrival_valid"] and coarse["arrival_valid"]
        ratio_sampling = fine["ratio_valid"] and coarse["ratio_valid"]
        arrival_change = abs(fine["T1"] - coarse["T1"]) if arrival_sampling else 0.0
        ratio_change = abs(fine["ratio"] - coarse["ratio"]) if ratio_sampling else 0.0
        arrival_limit = max(
            1e-4 * abs(theory["T1"]), 0.05 * abs(fine["T1"] - theory["T1"])
        )
        ratio_limit = max(
            1e-4 * abs(theory["ratio"]), 0.05 * abs(fine["ratio"] - theory["ratio"])
        )
        arrival_resolved = arrival_sampling and arrival_change <= arrival_limit
        ratio_resolved = ratio_sampling and ratio_change <= ratio_limit
        needs_refinement = (fine["arrival_valid"] and not arrival_resolved) or (
            fine["ratio_valid"] and not ratio_resolved
        )
        if not needs_refinement or refinements == 4:
            break
        nt = 2 * (nt - 1) + 1
        refinements += 1
    row = [
        family,
        sample,
        beta,
        fine["T1"],
        fine["A1"],
        fine["T2"],
        fine["A2"],
        fine["ratio"],
        theory["T1"],
        theory["ratio"],
        int(fine["arrival_valid"] and arrival_resolved),
        int(fine["ratio_valid"] and ratio_resolved),
        arrival_change,
        ratio_change,
        int(fine["arrival_valid"] and coarse["arrival_valid"]),
        int(fine["ratio_valid"] and coarse["ratio_valid"]),
        int(arrival_resolved),
        int(ratio_resolved),
        refinements,
    ]
    return (
        row,
        t,
        tail,
        {
            "family": family,
            "sample": sample,
            "beta_v": beta,
            "status": fine["status"],
            "arrival_found": bool(fine["arrival_valid"]),
            "ratio_found": bool(fine["ratio_valid"]),
            "sampling": {
                "initial_samples": initial_nt,
                "final_samples": nt,
                "refinement_steps": refinements,
                "arrival_resolved": bool(arrival_resolved),
                "ratio_resolved": bool(ratio_resolved),
                "arrival_absolute_limit": arrival_limit,
                "ratio_absolute_limit": ratio_limit,
            },
            "solver_check": _solver_check(model) if check else None,
        },
    )


def _compute_response(profile, jobs):
    meta = _base("lyons_response", profile)
    n, samples = (400, 3) if profile == "paper" else (80, 2)
    betas = (
        np.round(np.linspace(-0.6, 0.6, 13), 10)
        if profile == "paper"
        else np.asarray([-0.3, 0, 0.3, 0.6])
    )
    bounds = [([1, 3], [4, 12]), ([4, 12], [1, 3])]
    rng = np.random.default_rng(SEED + 1)
    gx, gv = np.empty((2, samples, n)), np.empty((2, samples, n))
    specs = []
    for family, (xb, vb) in enumerate(bounds):
        for sample in range(samples):
            gx[family, sample] = rng.uniform(*xb, n)
            gv[family, sample] = rng.uniform(*vb, n)
            for beta in betas:
                specs.append(
                    (
                        family,
                        sample,
                        gx[family, sample],
                        gv[family, sample],
                        float(beta),
                        1201 if profile == "paper" else 601,
                        sample == 0 and beta == 0,
                    )
                )
    rows, ts, tails, cases = [], [], [], []
    for row, t, tail, record in _map_cases(_response_case, specs, jobs):
        rows.append(row)
        ts.append(t)
        tails.append(tail)
        cases.append(record)
    meta.update(
        question="How well do first response time and adjacent peak ratio follow the velocity-asymmetry prediction?",
        sources="Lyons thesis Figures16–17, printed pp.92–93; equations107,137 and Theorem2.6.9.",
        N_total=n,
        ensemble_size=samples,
        beta_values=betas.tolist(),
        families=[{"gx_uniform": x, "gv_uniform": v} for x, v in bounds],
        cases=cases,
        sample_uncertainty="Mean and range of independent saved gain samples, not a confidence interval. Each sample is reused across beta values.",
        peak_estimator="First negative local minimum, followed by a positive local maximum; prominence > max(1e-11, 1e-6*max_abs_trace), local three-point quadratic interpolation. No predicted time enters selection.",
        sampling_check="Compare fine and alternate-point grids from the same trace. Double intervals, up to four times, until sampling_change/abs(prediction) <= max(1e-4, 0.05*abs(measurement-prediction)/abs(prediction)) for each found observable. Validity masks exclude unresolved measurements; found flags, exact grids and refinement counts are recorded. This measures sampling sensitivity separately from independent shorter-interval solver checks and sample spread.",
        undefined_values="Zero placeholders must be excluded using arrival_valid and ratio_valid, and their sampling-valid masks.",
        negative_beta_prediction="Theorem2.6.9 assumes path stability. For beta_v<0 the first-response overlay is a formal extrapolation of the wave formula; these points do not verify that theorem or establish stability. Only extrema actually found within the recorded window enter the measurements.",
        columns=[
            "family",
            "sample",
            "beta_v",
            "T1",
            "A1",
            "T2",
            "A2",
            "ratio",
            "T1_predicted",
            "ratio_predicted",
            "arrival_valid",
            "ratio_valid",
            "T1_sampling_change",
            "ratio_sampling_change",
            "arrival_sampling_valid",
            "ratio_sampling_valid",
            "arrival_sampling_resolved",
            "ratio_sampling_resolved",
            "refinement_steps",
        ],
        grid_note="Beta span and gain distributions follow the thesis; this experiment uses increments0.1 and new saved samples.",
    )
    return {
        "gx": gx,
        "gv": gv,
        "measurements": np.asarray(rows),
        "tail_times": np.concatenate(ts),
        "tail_responses": np.concatenate(tails),
        "offsets": np.r_[0, np.cumsum([len(t) for t in ts])],
    }, meta


def _compute_ordering(profile, jobs):
    meta = _base("lyons_ordering", profile)
    n, p = (240, 12) if profile == "paper" else (80, 8)
    rng = np.random.default_rng(SEED + 2)
    base_cell = rng.uniform(-1, 1, (2, p))
    repeated = np.tile(base_cell, (1, n // p))
    permutation = rng.permutation(n)
    distinct = rng.uniform(-1, 1, (2, n))
    sorted_indices = np.argsort(distinct[0])
    bases = [repeated, repeated[:, permutation], distinct, distinct[:, sorted_indices]]
    kinds = [
        "Repeated cell",
        "Shuffled cell",
        "Distinct agents",
        "Sorted distinct agents",
    ]
    variances = [0, 0.3, 0.75]
    arrays = {
        "cell": base_cell,
        "shuffle_permutation": permutation,
        "sorted_indices": sorted_indices,
        "amplitudes": np.asarray(variances),
    }
    specs, gx_all, gv_all = [], [], []
    for kind, values in enumerate(bases):
        for sample, amplitude in enumerate(variances):
            gx, gv = 2 * (1 + amplitude * values[0]), 1.5 * (1 + amplitude * values[1])
            gx_all.append(gx)
            gv_all.append(gv)
            specs.append(
                (kind, sample, gx, gv, 0.2, 1001 if profile == "paper" else 601, False)
            )
    rows, ts, tails, cases = [], [], [], []
    for row, t, tail, record in _map_cases(_response_case, specs, jobs):
        rows.append(row)
        ts.append(t)
        tails.append(tail)
        cases.append(record)
    # Exact homogeneous reference at the harmonic/effective gains in AppendixE.
    hom_specs = []
    for index, (gx, gv) in enumerate(zip(gx_all, gv_all)):
        m = np.mean(1 / gx)
        v = np.mean(gv / gx)
        hom_specs.append(
            (
                index // 3,
                index % 3,
                np.full(n, 1 / m),
                np.full(n, v / m),
                0.2,
                1001 if profile == "paper" else 601,
                False,
            )
        )
    refs = _map_cases(_response_case, hom_specs, jobs)
    arrays.update(
        gx=np.asarray(gx_all),
        gv=np.asarray(gv_all),
        measurements=np.asarray(rows),
        homogeneous_measurements=np.asarray([r[0] for r in refs]),
        tail_times=np.concatenate(ts),
        tail_responses=np.concatenate(tails),
        homogeneous_tails=np.concatenate([r[2] for r in refs]),
        homogeneous_times=np.concatenate([r[1] for r in refs]),
        homogeneous_offsets=np.r_[0, np.cumsum([len(r[1]) for r in refs])],
        offsets=np.r_[0, np.cumsum([len(t) for t in ts])],
    )
    meta.update(
        question="What changes when a repeated cell is shuffled or replaced by distinct agents?",
        sources="New experiment motivated by Lyons Remark2.8.1 and AppendixE, printed pp.84,119–120.",
        relation_to_source="New controlled investigation, not a reconstruction of a thesis figure.",
        N_total=n,
        cell_size=p,
        cell_repetitions=n // p,
        kinds=kinds,
        cases=cases,
        ensemble_size=1,
        beta_v=0.2,
        amplitudes=variances,
        gain_definition="gx=2*(1+a*ux), gv=1.5*(1+a*uv), with saved ux,uv uniform on[-1,1]. a is heterogeneity amplitude, not variance.",
        pairing="Repeated/shuffled share exactly the same gain pairs; distinct/sorted share exactly the same gain pairs. Every heterogeneous reference uses its own M,V in the homogeneous model gx=1/M,gv=V/M.",
        limitations="One saved realization per construction. Equal empirical gain moments do not imply equal finite-path trajectories, and path endpoints change under ordering.",
        sampling_check="Fine/alternate-point comparison, with up to four doublings until sampling_change/abs(prediction) <= max(1e-4,0.05*abs(measurement-prediction)/abs(prediction)); unresolved measurements are masked. Homogeneous reference grids have their own saved offsets and times.",
        columns=[
            "kind",
            "amplitude_index",
            "beta_v",
            "T1",
            "A1",
            "T2",
            "A2",
            "ratio",
            "T1_predicted",
            "ratio_predicted",
            "arrival_valid",
            "ratio_valid",
            "T1_sampling_change",
            "ratio_sampling_change",
            "arrival_sampling_valid",
            "ratio_sampling_valid",
            "arrival_sampling_resolved",
            "ratio_sampling_resolved",
            "refinement_steps",
        ],
    )
    return arrays, meta


def _cached(name, profile, jobs, checkpoint, function):
    _base(name, profile)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    destination = None
    if checkpoint is not None:
        directory = Path(checkpoint)
        directory.mkdir(parents=True, exist_ok=True)
        # Drawing changes do not invalidate numerical results.
        source = Path(__file__).read_bytes().split(b"\ndef draw_boundary")[0]
        dependencies = b"".join(
            Path(__file__).with_name(name).read_bytes()
            for name in ("heterogeneous.py", "solve.py")
        )
        versions = f"numpy={np.__version__};scipy={scipy.__version__}".encode()
        digest = sha256(source + dependencies + versions).hexdigest()[:20]
        destination = directory / f"{name}-{profile}-{digest}.npz"
        record = destination.with_suffix(".json")
        if destination.exists() and record.exists():
            meta = json.loads(record.read_text())
            if sha256(destination.read_bytes()).hexdigest() == meta.pop(
                "checkpoint_sha256", None
            ):
                with np.load(destination, allow_pickle=False) as saved:
                    return {k: saved[k] for k in saved.files}, meta
    started = time.perf_counter()
    arrays, meta = function(profile, jobs)
    meta["compute_seconds"] = time.perf_counter() - started
    if not all(
        np.issubdtype(a.dtype, np.number) and np.isfinite(a).all()
        for a in arrays.values()
    ):
        raise ValueError(
            "Experiment arrays must be finite and numeric; use explicit validity masks"
        )
    if destination is not None:
        np.savez_compressed(destination, **arrays)
        record.write_text(
            json.dumps(
                {
                    **meta,
                    "checkpoint_sha256": sha256(destination.read_bytes()).hexdigest(),
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    return arrays, meta


def compute_boundary(profile="quick", jobs=1, checkpoint=None):
    return _cached("lyons_boundary", profile, jobs, checkpoint, _compute_boundary)


def compute_response(profile="quick", jobs=1, checkpoint=None):
    return _cached("lyons_response", profile, jobs, checkpoint, _compute_response)


def compute_ordering(profile="quick", jobs=1, checkpoint=None):
    return _cached("lyons_ordering", profile, jobs, checkpoint, _compute_ordering)


def draw_boundary(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axs = plt.subplots(2, 3, figsize=(12, 7.0), layout="constrained")
        for family in (0, 1):
            n = meta["families"][family]["N_total"]
            for bi, beta in enumerate((0.3, -0.3)):
                ax = axs[family, bi]
                for index, case in enumerate(meta["cases"]):
                    if case["family"] != family or case["beta_v"] != beta:
                        continue
                    values = data[f"case_{index}_eigenvalues"]
                    geometry = case["geometry"]
                    if geometry == "circle":
                        ax.scatter(
                            n * values.real,
                            n * values.imag,
                            s=23,
                            marker="o",
                            facecolors="none",
                            edgecolors=CAT[1],
                            label="Circle",
                        )
                    else:
                        ax.scatter(
                            n * values.real,
                            n * values.imag,
                            s=23,
                            marker="x",
                            color=CAT[2],
                            label="Path",
                        )
                ax.axvline(0, color="0.55", lw=0.7)
                ax.set(
                    title=rf"$N={n}$, $\beta_v={beta:+.1f}$",
                    xlabel=r"$N\,\mathrm{Re}\,\lambda$",
                    ylabel=r"$N\,\mathrm{Im}\,\lambda$",
                )
                ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
                if family == bi == 0:
                    ax.legend(loc="best")
            ax = axs[family, 2]
            for index, case in enumerate(meta["cases"]):
                if case["family"] != family or case["geometry"] != "path":
                    continue
                beta = case["beta_v"]
                ax.plot(
                    data[f"case_{index}_t"] / (n - 1),
                    data[f"case_{index}_tail"] / (n - 1),
                    color=CAT[1] if beta > 0 else CAT[2],
                    ls="-" if beta > 0 else "--",
                    label=rf"$\beta_v={beta:+.1f}$",
                )
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set(
                title="Path tail response",
                xlabel=r"Time $t/(N-1)$",
                ylabel=r"$(z_{N-1}-z_0)/(N-1)$",
            )
            ax.legend()
        fig.suptitle(
            "Heterogeneous arrays: eigenvalues sampled near zero and path response",
            fontsize=12,
        )
        return fig


def _sample_summary(rows, beta, value_column, valid_column):
    subset = rows[(rows[:, 2] == beta) & (rows[:, valid_column] > 0)]
    if not len(subset):
        return None
    values = subset[:, value_column]
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def draw_response(data, meta):
    import matplotlib.pyplot as plt

    with plot_style():
        fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
        rows = data["measurements"]
        for family in (0, 1):
            selected = rows[rows[:, 0] == family]
            spec = meta["families"][family]
            for column, (value, prediction, valid, label) in enumerate(
                [
                    (3, 8, 10, r"First response $T_1/(N-1)$"),
                    (7, 9, 11, r"Adjacent peak ratio $|A_2/A_1|$"),
                ]
            ):
                ax = axs[family, column]
                betas = np.asarray(meta["beta_values"])
                if column == 1:
                    # Match thesis Fig17's nonnegative-asymmetry range; negative
                    # beta measurements remain available in the saved arrays.
                    betas = betas[betas >= 0]
                scale = meta["N_total"] - 1 if column == 0 else 1
                summaries = [
                    (b, _sample_summary(selected, b, value, valid)) for b in betas
                ]
                summaries = [(b, s) for b, s in summaries if s is not None]
                if summaries:
                    x = [b for b, _ in summaries]
                    matched = [
                        selected[(selected[:, 2] == b) & (selected[:, valid] > 0)]
                        for b in x
                    ]
                    predicted = [
                        np.mean(subset[:, prediction]) / scale for subset in matched
                    ]
                    ax.plot(x, predicted, "--", color=CAT[1], label="Wave prediction")
                    mean, lo, hi = np.asarray([s for _, s in summaries]).T / scale
                    ax.fill_between(x, lo, hi, color="0.85", label="Sample range")
                    ax.plot(x, mean, "o", color=CAT[2], ms=4, label="Measured mean")
                counts = [
                    np.count_nonzero((selected[:, 2] == b) & (selected[:, valid] > 0))
                    for b in betas
                ]
                if min(counts) < meta["ensemble_size"]:
                    ax.text(
                        0.98,
                        0.97,
                        f"Valid samples per β: {min(counts)}–{max(counts)} / {meta['ensemble_size']}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7.5,
                        color="0.4",
                    )
                ax.set(
                    title=rf"$g_x\sim U{tuple(spec['gx_uniform'])}$, $g_v\sim U{tuple(spec['gv_uniform'])}$",
                    xlabel=r"Velocity asymmetry $\beta_v$",
                    ylabel=label,
                )
                ax.grid(True, alpha=0.35)
                if family == 0 and column == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(
            f"Arrival and attenuation in heterogeneous paths: N={meta['N_total']}, {meta['ensemble_size']} samples per family",
            fontsize=12,
        )
        fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
        fig.text(
            0.5,
            0.012,
            r"Negative $\beta_v$: formal extrapolation beyond the theorem's stability hypothesis.",
            ha="center",
            fontsize=8,
            color="0.35",
        )
        return fig


def draw_ordering(data, meta):
    import matplotlib.pyplot as plt

    colors = [CAT[1], CAT[2], CAT[3], "#805b9c"]
    markers = ["o", "s", "^", "D"]
    dashes = ["-", (0, (5, 2)), (0, (1, 2)), (0, (4, 1, 1, 1))]
    with plot_style():
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained")
        rows, refs = data["measurements"], data["homogeneous_measurements"]
        amplitudes = np.asarray(meta["amplitudes"])
        for kind, (name, col, marker) in enumerate(zip(meta["kinds"], colors, markers)):
            idx = np.flatnonzero(rows[:, 0] == kind)
            valid = (rows[idx, 10] > 0) & (refs[idx, 10] > 0)
            axs[0].plot(
                amplitudes[valid],
                (rows[idx, 3] / rows[idx, 8] - 1)[valid],
                marker=marker,
                color=col,
                ms=4,
                label=name,
            )
            valid_ratio = (rows[idx, 11] > 0) & (refs[idx, 11] > 0)
            axs[1].plot(
                amplitudes[valid_ratio],
                (rows[idx, 7] - refs[idx, 7])[valid_ratio],
                marker=marker,
                color=col,
                ms=4,
            )
            index = idx[-1]
            start, stop = data["offsets"][index : index + 2]
            t = data["tail_times"][start:stop]
            y = data["tail_responses"][start:stop]
            if rows[index, 10] > 0 and rows[index, 4] != 0:
                axs[2].plot(
                    t / rows[index, 8],
                    y / abs(rows[index, 4]),
                    color=col,
                    ls=dashes[kind],
                    lw=1.2,
                )
            if kind == 2 and rows[index, 10] > 0 and rows[index, 4] != 0:
                hs, he = data["homogeneous_offsets"][index : index + 2]
                axs[2].plot(
                    data["homogeneous_times"][hs:he] / rows[index, 8],
                    data["homogeneous_tails"][hs:he] / abs(rows[index, 4]),
                    color="0.25",
                    ls="--",
                    lw=1.2,
                    label="Homogeneous: distinct sample",
                )
        for ax in axs[:2]:
            ax.axhline(0, color="0.5", lw=0.7)
            ax.set_xlabel("Heterogeneity amplitude $a$")
            ax.grid(True, alpha=0.35)
        axs[0].set(
            title="Arrival error against wave prediction",
            ylabel=r"$T_1/T_{1,\mathrm{pred}}-1$",
        )
        axs[1].set(
            title="Attenuation versus homogeneous reference",
            ylabel=r"Measured ratio difference",
        )
        axs[2].set(
            title=r"Tail profiles at $a=0.75$",
            xlabel=r"Time $t/T_{1,\mathrm{pred}}$",
            ylabel=r"$(z_{N-1}-z_0)/|A_1|$",
        )
        axs[0].legend(fontsize=7.7)
        axs[2].legend(fontsize=8)
        fig.suptitle(
            f"New investigation: repeated cells, distinct agents and ordering (N={meta['N_total']}, one realization)",
            fontsize=11,
        )
        return fig


EXPERIMENTS = {
    "lyons_boundary": {"compute": compute_boundary, "draw": draw_boundary},
    "lyons_response": {"compute": compute_response, "draw": draw_response},
    "lyons_ordering": {"compute": compute_ordering, "draw": draw_ordering},
}
