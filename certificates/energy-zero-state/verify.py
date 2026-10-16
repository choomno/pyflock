"""Exact zero-state counterexample certificate; Python standard library only.

The data are exact integers defining a C1 cubic Hermite trajectory, not samples
whose interpolation error must be estimated. Force is DEFINED by the ODE.
All mathematical checks use integers and fractions.Fraction. Decimal strings
in the report are explanatory only. See README.md for the proof and commands.
"""

from __future__ import annotations

import argparse
import gzip
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_GZIP_SHA256 = (
    "b2411735af07bf15f213da435844786c417ae3d180760e8948421b304fe2c97f"
)
EXPECTED_TEXT_SHA256 = (
    "34df4a2bb31dcdddad56e3e9a5bdb1ba6cc3d4b7c627c0d52eba44a73d6cfc7a"
)


def require(condition, message):
    """Checks remain active even when Python runs with -O."""
    if not condition:
        raise ValueError(message)


def exact(value):
    value = Fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def decimal(value):
    value = Fraction(value)
    with localcontext() as context:
        context.prec = 30
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def read_data():
    compressed = (HERE / "trajectory.txt.gz").read_bytes()
    require(
        sha256(compressed).hexdigest() == EXPECTED_GZIP_SHA256,
        "Compressed trajectory checksum differs",
    )
    raw = gzip.decompress(compressed)
    require(
        sha256(raw).hexdigest() == EXPECTED_TEXT_SHA256,
        "Exact integer text checksum differs",
    )
    lines = raw.decode("ascii").splitlines()
    header = json.loads(lines[0])
    require(
        header
        == {
            "format": "dyadic-hermite-v1",
            "N": 128,
            "intervals": 2048,
            "dt_numerator": 1,
            "dt_denominator": 4,
            "bits": 40,
            "columns": "x_1,...,x_N,v_1,...,v_N",
        },
        "Unexpected trajectory definition",
    )
    n = header["N"]
    rows = [[int(token) for token in line.split()] for line in lines[1:]]
    require(
        len(rows) == 2049 and all(len(row) == 2 * n for row in rows),
        "Wrong number of knots or coordinates",
    )
    require(all(value == 0 for value in rows[0]), "Initial state is not zero")
    return [row[:n] for row in rows], [row[n:] for row in rows], header


def apply_k(y):
    """K=B B.T: diagonal 2,...,2,1; adjacent off-diagonals -1."""
    n = len(y)
    return [
        (1 if i == n - 1 else 2) * y[i]
        - (y[i - 1] if i else 0)
        - (y[i + 1] if i + 1 < n else 0)
        for i in range(n)
    ]


def apply_twice_d(y):
    """2D=2(B+abs(B)/2)B.T: diagonal 4,...,4,3; lower -3, upper -1."""
    n = len(y)
    return [
        (3 if i == n - 1 else 4) * y[i]
        - (3 * y[i - 1] if i else 0)
        - (y[i + 1] if i + 1 < n else 0)
        for i in range(n)
    ]


def exact_trajectory_quantities(x, v, bits):
    n, intervals = len(x[0]), len(x) - 1
    force_integral_numerator = 0
    for j in range(intervals):
        # u=4(t-j/4); x(t)=sum C_k*u**k/Q, Q=2**(bits+2).
        c = [
            [4 * z for z in x[j]],
            v[j][:],
            [
                12 * (b - a) - 2 * p - q
                for a, b, p, q in zip(x[j], x[j + 1], v[j], v[j + 1])
            ],
            [8 * (a - b) + p + q for a, b, p, q in zip(x[j], x[j + 1], v[j], v[j + 1])],
        ]
        for i in range(n):
            require(c[0][i] == 4 * x[j][i], "Left position mismatch")
            require(c[1][i] == v[j][i], "Left velocity mismatch")
            require(
                sum(c[k][i] for k in range(4)) == 4 * x[j + 1][i],
                "Right position mismatch",
            )
            require(
                c[1][i] + 2 * c[2][i] + 3 * c[3][i] == v[j + 1][i],
                "Right velocity mismatch",
            )

        # d=x''+D*x'+K*x=sum f_k*u**k/(2Q).
        force = []
        for k in range(4):
            f = [2 * z for z in apply_k(c[k])]
            if k < 3:
                damping = apply_twice_d(c[k + 1])
                f = [a + 4 * (k + 1) * b for a, b in zip(f, damping)]
            if k < 2:
                f = [a + 32 * (k + 1) * (k + 2) * b for a, b in zip(f, c[k + 2])]
            force.append(f)
        # dt=du/4; integral_0^1 u**(k+l)du=1/(k+l+1).
        # 420 is divisible by each integer 1,...,7.
        for k in range(4):
            for l in range(k, 4):
                weight = (1 if k == l else 2) * (420 // (k + l + 1))
                force_integral_numerator += weight * sum(
                    a * b for a, b in zip(force[k], force[l])
                )
    input_energy = Fraction(force_integral_numerator, 420 * 4 * 2 ** (2 * (bits + 3)))
    delta = [-x[-1][0]] + [x[-1][i - 1] - x[-1][i] for i in range(1, n)]
    terminal_energy = Fraction(
        sum(z * z for z in v[-1]) + sum(z * z for z in delta), 2 * 2 ** (2 * bits)
    )
    require(input_energy > 0, "Nonpositive input energy")
    return input_energy, terminal_energy


def singular_value_certificate(n, lower_bound, *, require_positive=True):
    """Exact banded LDL.T for 4*q^2*(D.T D-(m/q)^2 I).

    For s=m/q and D2=2D this integer matrix is q^2 D2.T D2-4m^2 I.
    Positive pivots prove positive definiteness, hence sigma_min(D)>s.
    """
    m, q = lower_bound.numerator, lower_bound.denominator
    d2 = [[0] * n for _ in range(n)]
    for i in range(n):
        d2[i][i] = 3 if i == n - 1 else 4
        if i:
            d2[i][i - 1] = -3
        if i + 1 < n:
            d2[i][i + 1] = -1

    def matrix(i, j):
        return q * q * sum(d2[k][i] * d2[k][j] for k in range(n)) - (
            4 * m * m if i == j else 0
        )

    pivots, lower1, lower2 = [], [], []
    for i in range(n):
        l2 = Fraction(matrix(i, i - 2), pivots[i - 2]) if i >= 2 else Fraction(0)
        l1 = Fraction(0)
        if i:
            off_diagonal = Fraction(matrix(i, i - 1))
            if i >= 2:
                off_diagonal -= l2 * pivots[i - 2] * lower1[i - 1]
            l1 = off_diagonal / pivots[i - 1]
        pivot = Fraction(matrix(i, i))
        if i:
            pivot -= l1 * l1 * pivots[i - 1]
        if i >= 2:
            pivot -= l2 * l2 * pivots[i - 2]
        require(pivot != 0, f"Zero LDL pivot {i} for s={lower_bound}")
        if require_positive:
            require(pivot > 0, f"Nonpositive LDL pivot {i} for s={lower_bound}")
        pivots.append(pivot)
        lower1.append(l1)
        lower2.append(l2)

    # Reconstruct every possibly nonzero entry. Entries outside this band are
    # exactly zero for both G and L diag(pivots) L.T by their bandwidths.
    def entry_l(i, j):
        if i == j:
            return Fraction(1)
        if i - j == 1:
            return lower1[i]
        if i - j == 2:
            return lower2[i]
        return Fraction(0)

    for i in range(n):
        for j in range(max(0, i - 2), i + 1):
            reconstructed = sum(
                entry_l(i, k) * pivots[k] * entry_l(j, k)
                for k in range(max(0, i - 2), j + 1)
            )
            require(reconstructed == matrix(i, j), "LDL reconstruction failed")
    return {
        "s": exact(lower_bound),
        "integer_matrix_scaling": 4 * q * q,
        "integer_matrix_definition": "q^2 (2D)^T(2D)-4m^2 I, where s=m/q",
        "positive_pivot_count": sum(pivot > 0 for pivot in pivots),
        "negative_pivot_count": sum(pivot < 0 for pivot in pivots),
        "pivots": [exact(pivot) for pivot in pivots],
    }


def verify():
    x, v, header = read_data()
    j, h = exact_trajectory_quantities(x, v, header["bits"])
    require(h > 43 and j < Fraction(101, 100), "Simple rational separation failed")
    certificates = []
    for s in (Fraction(3, 250), Fraction(245209, 20000000)):
        certificate = singular_value_certificate(header["N"], s)
        rhs = j / (2 * s)
        gap = h - rhs
        require(gap > 0, "No strict violation")
        certificate.update(
            {
                "conservative_source_rhs": exact(rhs),
                "certified_gap": exact(gap),
                "certified_ratio_lower_bound": exact(h / rhs),
                "decimal_explanations_only": {
                    "s": decimal(s),
                    "rhs": decimal(rhs),
                    "gap": decimal(gap),
                    "ratio_lower_bound": decimal(h / rhs),
                },
            }
        )
        certificates.append(certificate)
    # Optional upper bracket: one negative LDL pivot implies a negative
    # eigenvalue of D.T D-s_upper^2 I by Sylvester's law of inertia.
    s_upper = Fraction(24521, 2000000)
    upper = singular_value_certificate(header["N"], s_upper, require_positive=False)
    require(
        upper["negative_pivot_count"] == 1 and upper["positive_pivot_count"] == 127,
        "Expected inertia (127 positive, 1 negative) for upper bracket",
    )
    upper.update(
        {
            "meaning": "One negative pivot proves sigma_min(D)<s by congruence and Sylvester's law of inertia",
            "source_violation_ratio_strict_upper_bound": exact(2 * s_upper * h / j),
            "decimal_explanations_only": {
                "s": decimal(s_upper),
                "ratio_upper_bound": decimal(2 * s_upper * h / j),
            },
        }
    )
    return {
        "verified": True,
        "proof_arithmetic": "Python integers and fractions.Fraction",
        "compressed_data_sha256": EXPECTED_GZIP_SHA256,
        "rational_text_sha256": EXPECTED_TEXT_SHA256,
        "parameters": {
            "N": 128,
            "T": "512",
            "M": "I",
            "A_stiffness": "I",
            "R": "I",
            "h_delta": "0",
            "h_p": "1/2",
            "leader_velocity": "0",
        },
        "matrices": "B=I-superdiag(1); K=B B^T; D=(B+abs(B)/2)B^T",
        "state_and_force": "x'=v; v'=-Kx-Dv+d; Delta=-B^T x; p=v; d=x''+D x'+Kx",
        "trajectory": header,
        "initial_position_and_velocity_exactly_zero": True,
        "all_endpoint_positions_and_velocities_match_exactly": True,
        "force_extension": "Zero after T; continue trajectory by homogeneous ODE. Values at finitely many knots are irrelevant for L2.",
        "input_energy": exact(j),
        "terminal_energy": exact(h),
        "simple_checks": {"H_greater_than_43": True, "J_less_than_101_over_100": True},
        "decimal_explanations_only": {"J": decimal(j), "H": decimal(h)},
        "singular_value_certificates": certificates,
        "singular_value_upper_certificate": upper,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Optionally save the full exact JSON certificate"
    )
    args = parser.parse_args()
    result = verify()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        key: value
        for key, value in result.items()
        if key
        not in ("singular_value_certificates", "singular_value_upper_certificate")
    }
    summary["singular_value_certificates"] = [
        {key: value for key, value in c.items() if key != "pivots"}
        for c in result["singular_value_certificates"]
    ]
    summary["singular_value_upper_certificate"] = {
        key: value
        for key, value in result["singular_value_upper_certificate"].items()
        if key != "pivots"
    }
    print(json.dumps(summary, indent=2))
