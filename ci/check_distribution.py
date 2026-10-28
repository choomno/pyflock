"""Inspect wheel and source contents before testing an installed distribution."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def single(directory, pattern):
    found = list(Path(directory).glob(pattern))
    if len(found) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(found)}")
    return found[0]


def inspect(directory):
    wheel = single(directory, "*.whl")
    source = single(directory, "*.tar.gz")
    distribution, version, *_ = wheel.name.split("-")
    metadata = f"{distribution}-{version}.dist-info/"
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        if any(not name.startswith(("pyflock/", metadata)) for name in wheel_names):
            raise ValueError(
                "the wheel contains material outside the maintained package"
            )
        expected = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "pyflock").rglob("*.py")
        }
        actual = {name for name in wheel_names if name.startswith("pyflock/")}
        if actual != expected:
            raise ValueError(
                "the wheel must contain exactly the maintained Python source"
            )
        if metadata + "METADATA" not in wheel_names:
            raise ValueError("distribution metadata is missing")
    with tarfile.open(source) as archive:
        source_names = archive.getnames()
        roots = {PurePosixPath(name).parts[0] for name in source_names}
        if len(roots) != 1:
            raise ValueError("source distribution must have one enclosing directory")
        relative = {
            str(PurePosixPath(*PurePosixPath(name).parts[1:])) for name in source_names
        }
        needed = {
            "README.md",
            "pyproject.toml",
            "requirements-tested.txt",
            "requirements-dev.txt",
            "docs/testing.md",
            "docs/coverage.md",
            "docs/mathematical-reference/main.tex",
            "docs/mathematical-reference/mathematical-reference.pdf",
            "tests/test_packaging.py",
            "ci/check_coverage.py",
            "ci/check_installed.py",
            "ci/check_distribution.py",
        }
        if needed - relative:
            raise ValueError(
                f"source distribution is missing {sorted(needed - relative)}"
            )
    forbidden = {
        ".git",
        "__pycache__",
        "private",
        "work",
        "outputs",
        "figures",
        "figures-paper",
        "figures-expanded",
        "figures-resolved",
        "AGENTS.md",
        "CLAUDE.md",
    }
    for name in wheel_names + source_names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"unsafe archive path: {name}")
        if forbidden.intersection(path.parts) or path.suffix in {
            ".pyc",
            ".pyo",
            ".pickle",
            ".pkl",
        }:
            raise ValueError(f"private or generated material in distribution: {name}")
    print(f"Distribution contents passed: {wheel.name}, {source.name}")


if __name__ == "__main__":
    inspect(sys.argv[1] if len(sys.argv) > 1 else "dist")
