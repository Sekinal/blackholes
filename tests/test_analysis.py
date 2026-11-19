"""Convergence and resummation of the neutral KPW exponent (Fig. 6 and beyond)."""

import itertools
import math

import numpy as np
import pytest
import sympy as sp

from blackholes.analysis.convergence import domb_sykes_fit, endpoint_truncation_errors
from blackholes.analysis.endpoint import build_neutral_endpoint_approximant, two_point_hermite
from blackholes.analysis.kpw_series import horizon_displacement_series, neutral_entropy_series
from blackholes.paper.configurations import FIG2_CHARGE, FIG2_MODEL, QUOTED_VALUES

MODEL, Q = FIG2_MODEL, FIG2_CHARGE
M_I, M_E = MODEL.critical_masses(Q)
E_C = M_I - M_E
R_PLUS = MODEL.horizons(M_I, Q)[-1]
R_MIN = MODEL.stationary_radii(Q)[1]
DS_END = MODEL.entropy(R_MIN) - MODEL.entropy(R_PLUS)


def _exact_entropy_change(E: float) -> float:
    """Neutral ΔS(E) from direct root finding (reference values).

    Parameters
    ----------
    E
        Emitted energy, kept at least ``1e-6 E_c`` away from ``E_c`` so the
        reference root is resolved to ~1e-11.

    Returns
    -------
    float
        ``S(R_+(E)) - S(R_+(0))``.
    """
    return MODEL.entropy(MODEL.horizons(M_I - E, Q)[-1]) - MODEL.entropy(R_PLUS)


def test_first_two_coefficients_are_thermal_eq_2_22() -> None:
    """b_1 = -1/T_H and b_2 = -1/(2 C_Q T_H^2) at the initial state."""
    b = neutral_entropy_series(MODEL, R_PLUS, Q, order=2)
    T = MODEL.temperature(R_PLUS, Q)
    assert b[0] == pytest.approx(-1 / T, rel=1e-12)
    # C_Q = dM/dT along the outer branch, by central differences in M.
    step = 1e-5
    T_up = MODEL.temperature(MODEL.horizons(M_I + step, Q)[-1], Q)
    T_down = MODEL.temperature(MODEL.horizons(M_I - step, Q)[-1], Q)
    heat_capacity = 2 * step / (T_up - T_down)
    assert b[1] == pytest.approx(-1 / (2 * heat_capacity * T**2), rel=1e-6)


def test_appendix_e_closed_form_eq_4_17() -> None:
    """a_n = (κℓ^2/π)^n r_+^(1-2n) A_n(ρ) with A_n from Eqs. (E.1), (E.2), (E.6)."""
    rho = R_PLUS / MODEL.stationary_radii(Q)[0]
    nu = (rho - 1) ** 2 / (rho * (1 - rho + rho * math.log(rho)))
    sigma = (
        (rho - 1)
        * ((rho + 1) * math.log(rho) - 2 * (rho - 1))
        / (rho * (1 - rho + rho * math.log(rho)))
    )
    x = sp.Symbol("x")
    bracket = sigma + x + nu * (1 - (1 + x) * sp.log(1 + x) / x)
    series = sp.series(bracket, x, 0, 6).removeO()
    a = horizon_displacement_series(MODEL, R_PLUS, Q, order=4)
    scale = MODEL.kappa * MODEL.ell**2 / math.pi
    for n in range(1, 5):
        A_n = float((-1) ** n / sp.factorial(n) * sp.diff(series ** (-n), x, n - 1).subs(x, 0))
        assert a[n - 1] == pytest.approx(scale**n * R_PLUS ** (1 - 2 * n) * A_n, rel=1e-9)


def test_fig_6_truncation_errors_are_reproduced() -> None:
    """The quadratic and quartic truncations miss ΔS(E_c) by ~40% and ~29%."""
    b = neutral_entropy_series(MODEL, R_PLUS, Q, order=4)
    errors = endpoint_truncation_errors(b, E_C, DS_END, [2, 4])
    for got, key in zip(
        errors, ["fig6_quadratic_relative_error", "fig6_quartic_relative_error"], strict=True
    ):
        value, tolerance = QUOTED_VALUES[key]
        assert got == pytest.approx(value, abs=tolerance)
    assert pytest.approx(QUOTED_VALUES["fig6_entropy_at_Ec"][0], abs=5e-4) == DS_END


def test_series_converges_up_to_coalescence_with_square_root() -> None:
    """Domb–Sykes: radius = E_c and exponent α = 1/2 (new result)."""
    b = neutral_entropy_series(MODEL, R_PLUS, Q, order=70)
    fit = domb_sykes_fit(b, first_order=20)
    assert fit.radius == pytest.approx(E_C, rel=1e-5)
    assert fit.exponent == pytest.approx(0.5, abs=1e-3)
    assert np.all(b < 0)  # every coefficient has the sign of -1/T_H


def test_endpoint_error_decays_like_inverse_square_root() -> None:
    """With c_n ~ n^(-3/2) E_c^(-n), the endpoint error scales as N^(-1/2)."""
    b = neutral_entropy_series(MODEL, R_PLUS, Q, order=64)
    errors = endpoint_truncation_errors(b, E_C, DS_END, [16, 64])
    assert errors[1] / errors[0] == pytest.approx(0.5, rel=0.05)


def test_domb_sykes_on_a_known_square_root() -> None:
    """Exact coefficients of sqrt(1 - E/R) give back R and α = 1/2."""
    R = 3.0
    c = np.array([float(sp.binomial(sp.Rational(1, 2), n)) * (-1 / R) ** n for n in range(1, 61)])
    fit = domb_sykes_fit(c, first_order=10)
    assert fit.radius == pytest.approx(R, rel=1e-6)
    assert fit.exponent == pytest.approx(0.5, abs=1e-4)
    linear = domb_sykes_fit(c, first_order=10, quadratic=False)
    assert linear.radius == pytest.approx(R, rel=1e-2)


def test_two_point_approximant_converges_geometrically() -> None:
    """The t = sqrt(1 - E/E_c) approximant beats 70 Taylor terms by orders of magnitude."""
    grid = np.linspace(0.0, E_C * (1 - 1e-6), 120)
    reference = np.array([_exact_entropy_change(float(E)) for E in grid])
    max_errors = []
    for K in (4, 6, 8, 10):
        approximant = build_neutral_endpoint_approximant(MODEL, M_I, Q, terms_per_end=K)
        values = np.array([approximant.entropy_change(float(E)) for E in grid])
        max_errors.append(float(np.max(np.abs(values - reference))))
        assert approximant.entropy_change(E_C) == pytest.approx(DS_END, abs=1e-12)
    assert max_errors[-1] < 1e-9
    assert all(later < earlier / 20 for earlier, later in itertools.pairwise(max_errors))


def test_two_point_hermite_reproduces_polynomials() -> None:
    """A cubic is recovered exactly from two coefficients at each end."""
    # P(t) = 1 + 2t + 3t^2 + 4t^3; at t = 1 (s = t - 1): P = 10 + 20 s + ...
    coefficients = two_point_hermite(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    assert [float(c) for c in coefficients] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_approximant_input_validation() -> None:
    """Bad inputs raise informative errors."""
    approximant = build_neutral_endpoint_approximant(MODEL, M_I, Q, terms_per_end=1)
    assert approximant.radius(0.0) == pytest.approx(R_PLUS)
    with pytest.raises(ValueError, match="E must lie"):
        approximant.radius(-0.1)
    with pytest.raises(ValueError, match="terms_per_end"):
        build_neutral_endpoint_approximant(MODEL, M_I, Q, terms_per_end=0)
    with pytest.raises(ValueError, match="outer critical mass"):
        build_neutral_endpoint_approximant(MODEL, M_E - 0.1, Q, terms_per_end=3)
    with pytest.raises(ValueError, match="degenerate"):
        horizon_displacement_series(MODEL, R_MIN, Q, order=2)
    with pytest.raises(ValueError, match="four orders"):
        domb_sykes_fit([1.0, 0.5, 0.25], first_order=1)
    with pytest.raises(ValueError, match="non-zero"):
        domb_sykes_fit([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], first_order=1)


def test_taylor_coefficients_are_stable_at_high_order() -> None:
    """∂^j m / j! matches the derivative formula and stays finite at j = 300."""
    for j in range(8):
        assert MODEL.taylor_coefficient(2.0, Q, j) == pytest.approx(
            MODEL.radial_derivative(2.0, Q, j) / math.factorial(j), rel=1e-12
        )
    assert math.isfinite(MODEL.taylor_coefficient(2.0, Q, 300))
