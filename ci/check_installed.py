"""Exercise a wheel installation from a directory outside the source checkout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="pyflock-installed-") as directory:
        env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=directory)
        code = """
import importlib.metadata as metadata
import json
from pathlib import Path
import numpy as np
import pyflock as pf
assert pf.__version__ == metadata.version('pyflock')
st = pf.nn_stencil(-1, -1, -.5, -.3)
assert np.allclose(st.rx.sum(), 0)
assert pf.signal_velocities(st)[0] > 0
result = pf.simulate_line(st, N=4, tmax=1, nt=5, bc='regular')
assert result.z.shape == (5, 5)
assert np.allclose(result.z[:, 0], result.t)
print(json.dumps({'package': str(Path(pf.__file__).resolve())}))
"""
        probe = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=directory,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        package = Path(json.loads(probe.stdout)["package"])
        if package.is_relative_to(root):
            raise ValueError(
                "this check requires an installed wheel, not an editable source checkout"
            )
        command = [sys.executable, "-I", "-m", "pyflock.experiments"]
        subprocess.run(
            [*command, "--help"],
            cwd=directory,
            env=env,
            check=True,
            capture_output=True,
        )
        output = Path(directory) / "results"
        subprocess.run(
            [*command, "dispersion", "--output", str(output)],
            cwd=directory,
            env=env,
            check=True,
        )
        record = output / "dispersion" / "experiment.json"
        before = json.loads(record.read_text(encoding="utf-8"))
        subprocess.run(
            [*command, "dispersion", "--output", str(output), "--verify"],
            cwd=directory,
            env=env,
            check=True,
        )
        subprocess.run(
            [*command, "dispersion", "--output", str(output), "--redraw"],
            cwd=directory,
            env=env,
            check=True,
        )
        after = json.loads(record.read_text(encoding="utf-8"))
        if (
            before["data_sha256"] != after["data_sha256"]
            or before["artifacts"] != after["artifacts"]
        ):
            raise ValueError("redraw did not preserve the recorded arrays and figures")
        data = output / "dispersion" / "data.npz"
        data.write_bytes(data.read_bytes() + b"corrupt")
        invalid = subprocess.run(
            [*command, "dispersion", "--output", str(output), "--verify"],
            cwd=directory,
            env=env,
        )
        if invalid.returncode == 0:
            raise ValueError("the installed CLI accepted corrupt experiment data")
    print("Installed-wheel numerical API, regeneration, verification and redraw passed")


if __name__ == "__main__":
    check()
