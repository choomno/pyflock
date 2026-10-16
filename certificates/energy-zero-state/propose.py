"""OPTIONAL, UNTRUSTED proposal conversion; requires NumPy.

This script is not used by verify.py. It converts a numerical candidate NPZ
containing arrays t,x,v into dyadic Hermite data. Its numerical source is used
only to choose coefficients; the exact verifier independently proves the claim.
The distributed trajectory.txt.gz is already complete and needs no rebuilding.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("candidate", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
with np.load(args.candidate, allow_pickle=False) as candidate:
    if not np.array_equal(candidate["t"], np.arange(2049) / 4):
        raise ValueError("Expected 2049 knots t=j/4")
    x = np.rint(candidate["x"] * 2**40).astype(np.int64)
    v = np.rint(candidate["v"] * 2**40).astype(np.int64)
if x.shape != (2049, 128) or v.shape != x.shape:
    raise ValueError("Expected x and v arrays with shape (2049,128)")
header = {
    "format": "dyadic-hermite-v1",
    "N": 128,
    "intervals": 2048,
    "dt_numerator": 1,
    "dt_denominator": 4,
    "bits": 40,
    "columns": "x_1,...,x_N,v_1,...,v_N",
}
raw = (
    json.dumps(header, separators=(",", ":"))
    + "\n"
    + "".join(" ".join(map(str, [*a, *b])) + "\n" for a, b in zip(x, v))
).encode("ascii")
with (
    args.output.open("wb") as stream,
    gzip.GzipFile(
        filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=9
    ) as archive,
):
    archive.write(raw)
