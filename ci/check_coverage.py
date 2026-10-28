"""Require independent line and branch coverage for both maintained package groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINIMUM = 99.5


def check(report, root=ROOT, minimum=MINIMUM):
    """Check complete file accounting and return unrounded percentages by group."""
    root = Path(root).resolve()
    expected = {path.resolve() for path in (root / "pyflock").rglob("*.py")}
    if not expected:
        raise ValueError("no maintained Python source found")
    if not report.get("meta", {}).get("branch_coverage"):
        raise ValueError("branch coverage measurement is required")
    measured = {}
    for name, data in report["files"].items():
        path = Path(name)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path in measured:
            raise ValueError("duplicate source file in coverage report")
        measured[path] = data
    if set(measured) != expected:
        raise ValueError(
            "coverage must report every maintained Python file, and only those files"
        )
    if any(data.get("excluded_lines") for data in measured.values()):
        raise ValueError("maintained source must have no excluded lines")
    result = {}
    for label, experiments in (
        ("Computational and plotting API", False),
        ("Experiments", True),
    ):
        files = [
            data
            for path, data in measured.items()
            if path.stem.startswith("experiment") == experiments
        ]
        totals = {
            key: sum(data["summary"][key] for data in files)
            for key in (
                "num_statements",
                "covered_lines",
                "num_branches",
                "covered_branches",
            )
        }
        if not totals["num_statements"] or not totals["num_branches"]:
            raise ValueError(f"{label}: line and branch measurements are both required")
        lines = 100 * totals["covered_lines"] / totals["num_statements"]
        branches = 100 * totals["covered_branches"] / totals["num_branches"]
        result[label] = {**totals, "line_percent": lines, "branch_percent": branches}
        print(
            f"{label}: {lines:.2f}% lines "
            f"({totals['covered_lines']}/{totals['num_statements']}), "
            f"{branches:.2f}% branches ({totals['covered_branches']}/{totals['num_branches']})"
        )
    if any(
        values[key] < minimum
        for values in result.values()
        for key in ("line_percent", "branch_percent")
    ):
        raise ValueError(
            f"each group requires at least {minimum}% lines AND branches before rounding"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", default="coverage.json")
    args = parser.parse_args()
    try:
        check(json.loads(Path(args.report).read_text(encoding="utf-8")))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Coverage check failed: {error}") from error
