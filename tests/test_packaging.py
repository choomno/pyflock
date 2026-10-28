"""Public installation metadata and coverage-accounting regressions."""

import importlib.metadata

import pytest

import pyflock as pf
from ci.check_coverage import check


def test_public_exports_and_distribution_version():
    assert pf.__version__ == importlib.metadata.version("pyflock")
    assert len(pf.__all__) == len(set(pf.__all__))
    assert all(hasattr(pf, name) for name in pf.__all__)
    requirements = importlib.metadata.requires("pyflock") or []
    for dependency in ("numpy", "scipy", "matplotlib", "mpmath"):
        assert any(
            value.startswith(dependency) and ";" not in value for value in requirements
        )


def _report(tmp_path):
    package = tmp_path / "pyflock"
    package.mkdir()
    files = {}
    for name in ("solve.py", "experiments.py"):
        (package / name).write_text("value = 1\n", encoding="utf-8")
        files[f"pyflock/{name}"] = {
            "excluded_lines": [],
            "summary": {
                "num_statements": 100000,
                "covered_lines": 100000,
                "num_branches": 100000,
                "covered_branches": 100000,
            },
        }
    return {"meta": {"branch_coverage": True}, "files": files}


def test_coverage_accepts_complete_separately_measured_groups(tmp_path):
    report = _report(tmp_path)
    result = check(report, root=tmp_path)
    assert len(result) == 2
    assert all(
        value["line_percent"] == value["branch_percent"] == 100
        for value in result.values()
    )


@pytest.mark.parametrize("group", ["solve.py", "experiments.py"])
@pytest.mark.parametrize("metric", ["covered_lines", "covered_branches"])
def test_coverage_rejects_unrounded_threshold_failure(tmp_path, group, metric):
    report = _report(tmp_path)
    # 99.499% displays as 99.50%, but remains below the threshold.
    report["files"][f"pyflock/{group}"]["summary"][metric] = 99499
    with pytest.raises(ValueError, match="before rounding"):
        check(report, root=tmp_path)


def test_coverage_accepts_the_exact_threshold(tmp_path):
    report = _report(tmp_path)
    for data in report["files"].values():
        data["summary"].update(covered_lines=99500, covered_branches=99500)
    result = check(report, root=tmp_path)
    assert all(
        value["line_percent"] == value["branch_percent"] == 99.5
        for value in result.values()
    )


@pytest.mark.parametrize("change", ["missing", "extra", "excluded", "no-branches"])
def test_coverage_rejects_incomplete_or_excluded_measurements(tmp_path, change):
    report = _report(tmp_path)
    if change == "missing":
        del report["files"]["pyflock/solve.py"]
    elif change == "extra":
        report["files"]["unmaintained.py"] = report["files"]["pyflock/solve.py"]
    elif change == "excluded":
        report["files"]["pyflock/solve.py"]["excluded_lines"] = [1]
    else:
        report["meta"]["branch_coverage"] = False
    with pytest.raises(ValueError):
        check(report, root=tmp_path)
