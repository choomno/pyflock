"""Check the exact rational certificate. Python standard library only.

The accompanying README states the analytical norm and Taylor inequalities.
This program verifies their matrix identities and all numerical inequalities
using integers and fractions; it performs no floating-point computation.
"""

import json
from fractions import Fraction
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def multiply(left, right):
    return [
        [sum(a * b for a, b in zip(row, column)) for column in zip(*right)]
        for row in left
    ]


def vector_norm_squared(vector):
    return sum(value * value for value in vector)


def check():
    certificate = json.loads(Path(__file__).with_name("certificate.json").read_text())
    model = certificate["model"]
    energy = certificate["energy_arithmetic"]
    singular = certificate["singular_value_certificate"]
    bounds = certificate["norm_and_taylor_certificate"]
    n = model["N"]
    require(n == 5, "N must equal 5")
    for field in ("h_p", "mass", "stiffness", "damping"):
        require(Fraction(model[field]) == 1, f"Unexpected {field}")
    for field in ("h_delta", "leader_velocity"):
        require(Fraction(model[field]) == 0, f"Unexpected {field}")
    p = list(map(Fraction, model["initial_p"]))
    delta = list(map(Fraction, model["initial_delta"]))
    require(p == [3, 5, 6, 5, 3] and delta == [0] * n, "Unexpected initial state")
    force = [value / 2 for value in p]
    duration = Fraction(model["T"])

    def expected(section, name, actual):
        require(
            Fraction(section[name]) == actual, f"Incorrect certificate field: {name}"
        )

    identity = [[Fraction(i == j) for j in range(n)] for i in range(n)]
    b = [[Fraction(i == j) - Fraction(j == i + 1) for j in range(n)] for i in range(n)]
    c = [[Fraction(j >= i) for j in range(n)] for i in range(n)]
    require(
        multiply(b, c) == identity and multiply(c, b) == identity, "C is not B inverse"
    )
    bt = transpose(b)
    d = [[2 * value for value in row] for row in bt]
    require(
        d == multiply([[value + abs(value) for value in row] for row in b], bt),
        "D does not match h_p=1",
    )
    g = multiply(transpose(c), c)
    require(
        g == [[min(i + 1, j + 1) for j in range(n)] for i in range(n)], "G identity"
    )
    g_squared = multiply(g, g)
    trace = sum(g_squared[i][i] for i in range(n))
    expected(singular, "trace_G_squared", trace)
    inverse_d = [[value / 2 for value in row] for row in transpose(c)]
    require(multiply(d, inverse_d) == identity, "D inverse identity")
    inverse_gram = multiply(inverse_d, transpose(inverse_d))
    require(
        inverse_gram == [[value / 4 for value in row] for row in g],
        "Inverse Gram identity",
    )
    inverse_gram_squared = multiply(inverse_gram, inverse_gram)
    inverse_trace = sum(inverse_gram_squared[i][i] for i in range(n))
    expected(singular, "trace_inverse_gram_squared", inverse_trace)
    sigma_fourth_lower = 1 / inverse_trace
    expected(singular, "sigma_fourth_power_lower_bound", sigma_fourth_lower)
    sigma_comparison = Fraction(singular["comparison_sigma"])
    require(sigma_fourth_lower > sigma_comparison**4, "Strict singular-value bound")
    expected(singular, "comparison_left_integer", 16 * 9**4)
    expected(singular, "comparison_right_integer", 155 * 5**4)
    source_coefficient_upper = 1 / (2 * sigma_comparison)
    require(source_coefficient_upper == Fraction(9, 10), "Source coefficient bound")

    p_norm = vector_norm_squared(p)
    force_norm = vector_norm_squared(force)
    quadratic = sum(p[i] * d[i][j] * p[j] for i in range(n) for j in range(n))
    h_derivative = -quadratic + sum(a * f for a, f in zip(p, force))
    expected(energy, "p_norm_squared", p_norm)
    expected(energy, "p_D_p", quadratic)
    expected(energy, "force_norm_squared", force_norm)
    expected(energy, "initial_energy", p_norm / 2)
    expected(energy, "initial_energy_derivative", h_derivative)
    expected(energy, "total_force_L2_squared", force_norm * duration)

    # ||B||_2^2 <= ||B||_1 ||B||_infinity <= 4, and D=2B.T.
    b_one = max(sum(abs(b[i][j]) for i in range(n)) for j in range(n))
    b_infinity = max(sum(abs(value) for value in row) for row in b)
    b_bound = Fraction(bounds["B_norm_upper_bound"])
    require(b_one * b_infinity <= b_bound**2 and b_bound == 2, "B norm bound")
    d_bound = 2 * b_bound
    expected(bounds, "D_norm_upper_bound", d_bound)
    # A=diag(-D,0)+[[0,B],[-B.T,0]], whose two norms are ||D|| and ||B||.
    a_bound = d_bound + b_bound
    expected(bounds, "state_matrix_norm_upper_bound", a_bound)
    y0_bound = Fraction(bounds["initial_state_norm_strict_upper_bound"])
    f_bound = Fraction(bounds["force_norm_strict_upper_bound"])
    require(
        p_norm < y0_bound**2 and force_norm < f_bound**2, "Strict initial norm bounds"
    )
    endpoint = Fraction(bounds["auxiliary_interval_endpoint"])
    require(0 < duration <= endpoint and a_bound * endpoint < 1, "Time interval")
    # e^u <= 1/(1-u), 0<=u<1, by the nonnegative power series.
    exponential_bound = 1 / (1 - a_bound * endpoint)
    expected(bounds, "exponential_upper_bound", exponential_bound)
    intermediate = exponential_bound * (y0_bound + f_bound * endpoint)
    expected(bounds, "state_norm_intermediate_upper_bound", intermediate)
    y_bound = Fraction(bounds["state_norm_strict_upper_bound"])
    require(intermediate < y_bound, "State norm bound")
    y_prime = a_bound * y_bound + f_bound
    y_second = a_bound * y_prime
    h_second = y_prime**2 + y_bound * y_second
    expected(bounds, "state_derivative_norm_upper_bound", y_prime)
    expected(bounds, "state_second_derivative_norm_upper_bound", y_second)
    expected(bounds, "energy_second_derivative_absolute_upper_bound", h_second)
    comparison_prime = h_derivative - source_coefficient_upper * force_norm
    expected(bounds, "comparison_derivative_at_zero", comparison_prime)
    gap = comparison_prime * duration - h_second * duration**2 / 2
    expected(bounds, "comparison_T_lower_bound", gap)
    require(gap > 0, "The rigorous excess must be strictly positive")
    print("Exact rational certificate verified (no floating-point arithmetic).")
    print("sigma_min(D) > 5/9; source coefficient < 9/10.")
    print(f"T = {duration}; energy excess over the source bound > {gap}.")
    print("Scope: Theorem 4(iii), Eq. (7), with the stated nonzero initial data.")


if __name__ == "__main__":
    check()
