"""Build the source archive independently and run its maintained tests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from check_distribution import single


def check(directory, no_isolation=False):
    source = single(directory, "*.tar.gz")
    with tempfile.TemporaryDirectory(prefix="pyflock-sdist-") as directory:
        root = Path(directory).resolve()
        with tarfile.open(source) as archive:
            for entry in archive:
                target = (root / entry.name).resolve()
                if not target.is_relative_to(root) or entry.issym() or entry.islnk():
                    raise ValueError(f"unsafe source archive member: {entry.name}")
            archive.extractall(root)
        trees = [path for path in root.iterdir() if path.is_dir()]
        if len(trees) != 1:
            raise ValueError("source distribution must have one enclosing directory")
        tree = trees[0]
        command = [sys.executable, "-m", "build", "--wheel"]
        if no_isolation:
            command.append("--no-isolation")
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
            MPLBACKEND="Agg",
        )
        env.pop("PYTHONPATH", None)
        subprocess.run(command, cwd=tree, env=env, check=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=tree, env=env, check=True
        )
    print("Source-distribution build and tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="dist")
    parser.add_argument(
        "--no-isolation", action="store_true", help="use already-installed build tools"
    )
    args = parser.parse_args()
    check(args.directory, no_isolation=args.no_isolation)
