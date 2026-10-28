"""Independent checks of the saved computation and export record."""

import json
import runpy
import shutil
import sys
from hashlib import sha256
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyflock import experiment_nnn as NNN
from pyflock import experiments as E


@pytest.fixture(scope="module")
def reference_export(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reference-export")
    E.run("dispersion", directory)
    return directory


@pytest.fixture
def saved(tmp_path, reference_export):
    shutil.copytree(reference_export / "dispersion", tmp_path / "dispersion")
    return tmp_path


def _record(directory):
    return json.loads((directory / "dispersion/experiment.json").read_text())


def _change_record(directory, function):
    record = _record(directory)
    function(record)
    (directory / "dispersion/experiment.json").write_text(json.dumps(record))


def test_verification_is_read_only_and_does_not_compute_or_draw(saved, monkeypatch):
    before = {p: p.read_bytes() for p in saved.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("verification executed a simulation or renderer")

    monkeypatch.setattr(E, "compute", forbidden)
    monkeypatch.setattr(E, "draw", forbidden)
    report = E.verify("dispersion", saved)
    assert report == {
        "experiment": "dispersion",
        "schema": 2,
        "pass": True,
        "arrays": 5,
        "figures_checked": True,
    }
    assert before == {p: p.read_bytes() for p in saved.rglob("*") if p.is_file()}


@pytest.mark.parametrize("filename", ["data.npz", "source/dispersion.py", "figure.pdf"])
def test_changed_data_source_and_figure_are_rejected(saved, filename):
    path = saved / "dispersion" / filename
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError, match="checksum"):
        E.verify("dispersion", saved)


def test_redraw_repairs_a_figure_without_replacing_numerical_data(saved):
    dest = saved / "dispersion"
    data, source = (
        (dest / "data.npz").read_bytes(),
        (dest / "source/solve.py").read_bytes(),
    )
    (dest / "figure.pdf").unlink()
    assert E.verify("dispersion", saved, artifacts=False)["pass"]
    E.run("dispersion", saved, redraw=True)
    assert E.verify("dispersion", saved)["pass"]
    assert (dest / "data.npz").read_bytes() == data
    assert (dest / "source/solve.py").read_bytes() == source


@pytest.mark.parametrize("schema", [0, 3, True, "2", None])
def test_unknown_record_schema_is_rejected(saved, schema):
    _change_record(saved, lambda r: r.update(schema=schema))
    with pytest.raises(ValueError, match="schema"):
        E.verify("dispersion", saved)


def test_nonobject_record_and_wrong_experiment_identity_are_rejected(saved):
    record = _record(saved)
    path = saved / "dispersion/experiment.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="schema"):
        E.verify("dispersion", saved)
    record["experiment"] = "pulse"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="name"):
        E.verify("dispersion", saved)


def test_legacy_record_is_verified_and_migrated_only_when_redrawn(saved):
    def legacy(record):
        record["schema"] = 1
        record.pop("files")
        record.pop("arrays")

    _change_record(saved, legacy)
    assert E.verify("dispersion", saved)["schema"] == 1
    assert _record(saved)["schema"] == 1
    E.run("dispersion", saved, redraw=True)
    assert E.verify("dispersion", saved)["schema"] == 2


def test_schema_downgrade_cannot_skip_a_present_inventory(saved):
    _change_record(saved, lambda r: r.update(schema=1))
    with pytest.raises(ValueError, match="requires schema 2"):
        E.verify("dispersion", saved)


def test_missing_or_changed_source_is_required_even_for_legacy_records(saved):
    source = saved / "dispersion/source"
    shutil.rmtree(source)
    with pytest.raises(ValueError, match="Missing source snapshot"):
        E.verify("dispersion", saved)


@pytest.mark.parametrize("files", [None, [], {}, [None], [{}]])
def test_malformed_inventory_is_rejected(saved, files):
    _change_record(saved, lambda r: r.update(files=files))
    with pytest.raises(ValueError, match=r"[Ii]nventory"):
        E.verify("dispersion", saved)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../outside",
        "/tmp/outside",
        "source\\outside.py",
        "./data.npz",
        "source//solve.py",
    ],
)
def test_inventory_paths_are_canonical_and_contained(saved, path):
    _change_record(saved, lambda r: r["files"][0].update(path=path))
    with pytest.raises(ValueError, match=r"[Ii]nventory"):
        E.verify("dispersion", saved)


def test_symlink_escape_is_rejected_before_reading_external_file(saved, tmp_path):
    path = saved / "dispersion/figure.pdf"
    outside = tmp_path / "outside.pdf"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="leaves the experiment"):
        E.verify("dispersion", saved)


def test_duplicate_and_incomplete_inventories_are_rejected(saved):
    original = _record(saved)["files"]
    _change_record(saved, lambda r: r.update(files=[*original, original[0]]))
    with pytest.raises(ValueError, match="Repeated"):
        E.verify("dispersion", saved)
    _change_record(saved, lambda r: r.update(files=original[1:]))
    with pytest.raises(ValueError, match="Missing required"):
        E.verify("dispersion", saved)


def test_array_dtype_and_shape_are_checked(saved):
    _change_record(saved, lambda r: r["arrays"]["theta"].update(shape=[1]))
    with pytest.raises(ValueError, match="shapes or dtypes"):
        E.verify("dispersion", saved)


@pytest.mark.parametrize(
    "value", [np.array([np.nan]), np.array(["text"]), np.array([{}])]
)
def test_export_refuses_nonfinite_and_nonnumeric_arrays(saved, monkeypatch, value):
    before = (saved / "dispersion/experiment.json").read_bytes()
    monkeypatch.setattr(E, "compute", lambda *a, **k: ({"values": value}, {}))
    with pytest.raises(ValueError, match="finite and numeric"):
        E.run("dispersion", saved)
    assert (saved / "dispersion/experiment.json").read_bytes() == before


def test_export_refuses_empty_data_mapping(saved, monkeypatch):
    monkeypatch.setattr(E, "compute", lambda *a, **k: ({}, {}))
    with pytest.raises(ValueError, match="at least one array"):
        E.run("dispersion", saved)


def test_failed_render_preserves_the_previous_complete_export(saved, monkeypatch):
    before = {p: p.read_bytes() for p in saved.rglob("*") if p.is_file()}

    def fail(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        E.run("dispersion", saved)
    assert before == {p: p.read_bytes() for p in saved.rglob("*") if p.is_file()}
    assert not list(saved.glob(".dispersion-*"))


def test_successful_recomputation_replaces_stale_source_and_preserves_notes(saved):
    dest = saved / "dispersion"
    (dest / "source/obsolete.py").write_text("pass\n")
    (dest / "notes.txt").write_text("retained notes\n")
    (dest / "cases").mkdir()
    (dest / "cases/saved.npz").write_bytes(b"retained checkpoint")
    E.run("dispersion", saved)
    assert not (dest / "source/obsolete.py").exists()
    assert (dest / "notes.txt").read_text() == "retained notes\n"
    assert (dest / "cases/saved.npz").read_bytes() == b"retained checkpoint"
    assert E.verify("dispersion", saved)["pass"]


def test_measurement_table_is_inventoried_and_agrees_with_arrays(tmp_path, monkeypatch):
    table = np.array([[1.25, -0.5], [2.0, 7.0]])
    monkeypatch.setattr(
        E,
        "compute",
        lambda *a, **k: (
            {"measurements": table},
            {"experiment": "dispersion", "columns": ["x", "y"]},
        ),
    )
    monkeypatch.setattr(E, "draw", lambda *a, **k: plt.figure())
    meta = E.run("dispersion", tmp_path)
    assert "measurements.csv" in {f["path"] for f in meta["files"]}
    assert E.verify("dispersion", tmp_path)["pass"]
    csv = tmp_path / "dispersion/measurements.csv"
    csv.write_text("x,y\n1.25,-0.5\n2.0,8.0\n")
    # A matching file hash cannot conceal disagreement with the primary arrays.
    _change_record(
        tmp_path,
        lambda r: r.update(
            files=[
                E._file_record(csv, f["path"]) if f["path"] == "measurements.csv" else f
                for f in r["files"]
            ]
        ),
    )
    with pytest.raises(ValueError, match="CSV differs"):
        E.verify("dispersion", tmp_path)
    csv.write_text("incorrect,columns\n1.25,-0.5\n2.0,7.0\n")
    _change_record(
        tmp_path,
        lambda r: r.update(
            files=[
                E._file_record(csv, f["path"]) if f["path"] == "measurements.csv" else f
                for f in r["files"]
            ]
        ),
    )
    with pytest.raises(ValueError, match="column names"):
        E.verify("dispersion", tmp_path)


def test_verification_checks_legacy_export_hashes(saved):
    def legacy(record):
        record["schema"] = 1
        record.pop("files")
        record.pop("arrays")

    _change_record(saved, legacy)
    (saved / "dispersion/figure.png").write_bytes(b"not a figure")
    with pytest.raises(ValueError, match="Figure checksum"):
        E.verify("dispersion", saved)
    _change_record(saved, lambda r: r.update(artifacts={}))
    with pytest.raises(ValueError, match="must record"):
        E.verify("dispersion", saved)


@pytest.mark.parametrize(
    "options", [{"name": "../bad"}, {"profile": "unknown"}, {"jobs": 0}, {"jobs": True}]
)
def test_run_rejects_invalid_dispatch_arguments(tmp_path, options):
    args = {"name": "dispersion", "output": tmp_path, **options}
    with pytest.raises(ValueError):
        E.run(**args)


def test_cli_verification_success_and_failure(saved, capsys):
    E.main(["dispersion", "--output", str(saved), "--verify"])
    assert json.loads(capsys.readouterr().out)["pass"]
    (saved / "dispersion/figure.pdf").unlink()
    with pytest.raises(SystemExit) as error:
        E.main(["dispersion", "--output", str(saved), "--verify"])
    assert error.value.code == 1
    assert "verification failed" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        E.main(["dispersion", "--verify", "--redraw"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        E.main(["dispersion", "--jobs", "0"])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="Unknown experiment"):
        E.verify("../bad", saved)


def test_nnn_checkpoint_rejects_hash_mismatch_and_invalid_rows(tmp_path, monkeypatch):
    record = NNN.select_configurations(1)[0][1]
    spec = (1, 1, record, "fixed_mass", 25, str(tmp_path))
    expected = NNN._ensemble_case(spec)
    archive = next(tmp_path.glob("*.npz"))
    original = archive.read_bytes()

    def forbidden(*args, **kwargs):
        raise RuntimeError("recomputed invalid checkpoint")

    monkeypatch.setattr(NNN, "converged_tail", forbidden)
    assert np.array_equal(NNN._ensemble_case(spec), expected)
    archive.write_bytes(original + b"changed")
    with pytest.raises(RuntimeError, match="recomputed"):
        NNN._ensemble_case(spec)

    np.savez_compressed(archive, row=np.zeros(23))
    archive.with_suffix(".sha256").write_text(sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="recomputed"):
        NNN._ensemble_case(spec)
    np.savez_compressed(archive, unrelated=np.ones(3))
    archive.with_suffix(".sha256").write_text(sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="recomputed"):
        NNN._ensemble_case(spec)


@pytest.mark.parametrize("name", ["propagation", "reflection", "boundary", "pulse"])
def test_original_quick_experiments_compute_and_render_from_recorded_arrays(name):
    arrays, meta = E.compute(name)
    assert meta["experiment"] == name
    assert all(np.isfinite(a).all() for a in arrays.values())
    fig = E.draw(name, arrays, meta)
    try:
        fig.canvas.draw()
        assert len(fig.axes) >= 2
        assert any(ax.get_xlabel() for ax in fig.axes)
        assert any(ax.get_ylabel() for ax in fig.axes)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("profile,jobs,count", [("quick", 1, 36), ("paper", 2, 360)])
def test_convergence_profile_enumerates_every_source_parameter_case(
    tmp_path, monkeypatch, profile, jobs, count
):
    seen = []

    def case(spec):
        bi, bc, n, gv, av, checkpoint, source = spec
        seen.append(spec)
        assert bc == ("drop", "regular")[bi]
        assert checkpoint == str(tmp_path)
        assert len(source) == 64
        row = [bi, n, gv, av, 1, 1, 1, 1, 1, 1, 1 / n, 2 / n, 3 / n, 0, 0, 0]
        return row, np.array([0.0, 1.0]), np.array([0.0, -1.0])

    class Pool:
        def __init__(self, max_workers):
            assert max_workers == jobs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, specs):
            return map(function, specs)

    monkeypatch.setattr(E, "_convergence_case", case)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", Pool)
    arrays, meta = E.compute("convergence", profile, jobs=jobs, checkpoint=tmp_path)
    assert len(seen) == count
    assert len({(s[0], s[2], s[3], s[4]) for s in seen}) == count
    assert arrays["measurements"].shape == (count, 16)
    assert np.array_equal(arrays["offsets"], np.arange(count + 1) * 2)
    assert len(arrays["tail_times"]) == len(arrays["tail_responses"]) == 2 * count
    assert meta["profile"] == profile
    if profile == "paper":
        assert meta["Ns"] == [100, 200, 400, 800, 1600, 3200]
        assert len(meta["velocity_weights"]) * len(meta["velocity_gains"]) == 30
    fig = E.draw("convergence", arrays, meta)
    try:
        fig.canvas.draw()
        assert len(fig.axes) == 3
        assert len(fig.legends) == 1
        renderer = fig.canvas.get_renderer()
        legend_top = fig.legends[0].get_window_extent(renderer).y1
        assert legend_top < min(ax.get_window_extent(renderer).y0 for ax in fig.axes)
    finally:
        plt.close(fig)


def test_extension_dispatch_forwards_computational_options_and_drawing(monkeypatch):
    seen = []
    monkeypatch.setattr(
        E,
        "EXTENSIONS",
        {
            "additional": {
                "compute": lambda **kwargs: seen.append(kwargs) or ({}, {}),
                "draw": lambda arrays, meta: (arrays, meta),
            }
        },
    )
    assert E.compute("additional", "paper", jobs=3, checkpoint="saved") == ({}, {})
    assert seen == [{"profile": "paper", "jobs": 3, "checkpoint": "saved"}]
    assert E.draw("additional", {"x": 1}, {"y": 2}) == ({"x": 1}, {"y": 2})
    with pytest.raises(ValueError, match="Unknown experiment"):
        E.compute("unknown")
    with pytest.raises(ValueError, match="Unknown experiment"):
        E.draw("unknown", {}, {})


@pytest.mark.parametrize(
    "registry,error",
    [
        ({"propagation": {"compute": None, "draw": None}}, "Duplicate"),
        ({"additional": {}}, "Incomplete"),
        ({"additional": {"compute": None, "draw": None}}, "Duplicate"),
    ],
)
def test_extension_registry_rejects_ambiguous_or_incomplete_entries(
    monkeypatch, registry, error
):
    monkeypatch.setattr(
        E, "import_module", lambda *args: SimpleNamespace(EXPERIMENTS=registry)
    )
    with pytest.raises(ValueError, match=error):
        E._extensions()


def test_convergence_refuses_unresolved_peak_and_sampling_measurements(monkeypatch):
    spec = (1, "regular", 40, -2.0, 0.0, None, "source")
    monkeypatch.setattr(E, "transient_summary", lambda *a, **k: {"n_extrema": 2})
    with pytest.raises(RuntimeError, match="Insufficient extrema"):
        E._convergence_case(spec)

    def summary(t, *args, **kwargs):
        amplitude = 1.0 if len(t) % 2400 else 100.0
        return {"n_extrema": 3, "A": [amplitude], "T": len(t), "gamma": 1}

    monkeypatch.setattr(E, "transient_summary", summary)
    with pytest.raises(RuntimeError, match="Sampling unresolved"):
        E._convergence_case(spec)


def test_invalid_convergence_checkpoint_is_recomputed(tmp_path, monkeypatch):
    spec = (1, "regular", 40, -2.0, 0.0, str(tmp_path), "source")
    E._convergence_case(spec)
    path = next(tmp_path.glob("*.npz"))

    def fail(*args, **kwargs):
        raise RuntimeError("recomputation")

    monkeypatch.setattr(E, "simulate_tail", fail)
    for values in ({"row": np.zeros(16), "t": [0, 1], "w": [0, 0]}, {"wrong": [1]}):
        np.savez_compressed(path, **values)
        path.with_suffix(".sha256").write_text(sha256(path.read_bytes()).hexdigest())
        with pytest.raises(RuntimeError, match="recomputation"):
            E._convergence_case(spec)


def test_git_metadata_is_optional_outside_a_git_checkout(monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(E.subprocess, "check_output", absent)
    assert E._git_commit() is None
    assert E._git_dirty() is None


def test_cli_dispatches_all_names_and_run_options(monkeypatch):
    calls = []
    monkeypatch.setattr(
        E, "run", lambda name, output, **kwargs: calls.append((name, kwargs))
    )
    E.main(["all", "--profile", "paper", "--jobs", "2"])
    assert [name for name, _ in calls] == list(E.NAMES)
    assert all(
        options == {"profile": "paper", "redraw": False, "jobs": 2}
        for _, options in calls
    )


def test_uncached_convergence_retains_remeasurable_time_trace():
    row, t, w = E._convergence_case((1, "regular", 40, -2.0, 0.0, None, "source"))
    assert row[:4] == [1, 40, -2.0, 0.0]
    assert t.shape == w.shape and len(t) >= 2401
    assert np.all(np.diff(t) > 0)


def test_short_propagation_view_does_not_draw_wave_predictions_beyond_horizon():
    data, meta = E.compute("propagation")
    keep = data["t"] < meta["N"] / meta["speeds"][0] / 2
    data = {key: value[keep] for key, value in data.items()}
    fig = E.draw("propagation", data, meta)
    try:
        predictions = fig.axes[0].lines
        assert len(predictions) == 1
        assert max(predictions[0].get_ydata()) <= data["t"][-1] / meta["N"]
    finally:
        plt.close(fig)


def test_export_records_time_grids_and_largest_sampling_changes(tmp_path, monkeypatch):
    rows = np.ones((2, 16))
    rows[:, 13:16] = [[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]]
    data = {
        "measurements": rows,
        "t": np.array([0.0, 0.5, 1.0]),
        "0_t": np.array([1.0, 2.0]),
        "pred_t": np.array([3.0]),
    }
    monkeypatch.setattr(
        E,
        "compute",
        lambda *a, **k: (
            data,
            {"experiment": "convergence", "columns": [str(i) for i in range(16)]},
        ),
    )
    monkeypatch.setattr(E, "draw", lambda *a, **k: plt.figure())
    meta = E.run("convergence", tmp_path)
    assert meta["maximum_sampling_change"] == {"A1": 0.4, "T": 0.2, "gamma": 0.3}
    assert meta["time_grids"] == {
        "t": {"start": 0.0, "stop": 1.0, "samples": 3},
        "0_t": {"start": 1.0, "stop": 2.0, "samples": 2},
    }
    assert E.verify("convergence", tmp_path)["pass"]


@pytest.mark.parametrize("previous", [False, True])
def test_failed_export_directory_replacement_restores_previous_result(
    saved, monkeypatch, previous
):
    dest = saved / "dispersion"
    if not previous:
        shutil.rmtree(dest)
    before = {p: p.read_bytes() for p in saved.rglob("*") if p.is_file()}
    rename = type(dest).rename

    def failed_replace(path, target):
        if path.name == "dispersion" and path.parent.name.startswith(".dispersion-"):
            raise OSError("replacement failed")
        return rename(path, target)

    monkeypatch.setattr(type(dest), "rename", failed_replace)
    with pytest.raises(OSError, match="replacement failed"):
        E.run("dispersion", saved)
    assert before == {p: p.read_bytes() for p in saved.rglob("*") if p.is_file()}


def test_python_module_entry_point_exposes_the_command_line(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pyflock.experiments", "--help"])
    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(SystemExit) as error,
    ):
        runpy.run_module("pyflock.experiments", run_name="__main__")
    assert error.value.code == 0
    assert "--verify" in capsys.readouterr().out
