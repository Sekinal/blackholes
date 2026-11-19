"""Tests for the nonlinear BTZ model against an independent SymPy derivation."""

import math

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from blackholes.models import NonlinearBTZ

FIG2 = NonlinearBTZ(kappa=1.0, ell=1.0, beta=1.2, gamma=1.0)
FIG2_Q = NonlinearBTZ.charge_from_q(1.0)
FIG3 = NonlinearBTZ(kappa=1.0, ell=1.0, beta=27 / 20, gamma=4 / math.pi)
FIG3_Q = 8.0  # q = gamma = 4/pi gives Q = 8 (Fig. 3 caption)

# Independent reference: the blackening function (4.2) written out in SymPy.
_r, _q, _M_par = sp.symbols("r q M_par", positive=True)
_kappa, _ell, _beta, _gamma = sp.symbols("kappa ell beta gamma", positive=True)
_f = (
    _r**2 / _ell**2
    - _M_par
    - 2 * sp.sqrt(2) * _kappa * _q * _r / _gamma * sp.log(_beta * _gamma * _r / (sp.sqrt(2) * _q))
)
# m(r, Q): solve f = 0 for the physical mass M = pi * M_par / kappa.
_m_ref = sp.pi / _kappa * sp.solve(_f, _M_par)[0]
_A_t = sp.sqrt(2) * _r / _gamma * (sp.log(_beta * _gamma * _r / (sp.sqrt(2) * _q)) - 1)

radii = st.floats(min_value=0.05, max_value=20.0)
charges_q = st.floats(min_value=0.1, max_value=5.0)


def _reference(expr: sp.Expr, model: NonlinearBTZ, r: float, q: float) -> float:
    """Evaluate a SymPy reference expression for a model at a point.

    Parameters
    ----------
    expr
        Expression in the module-level symbols.
    model
        Supplies the coupling values.
    r
        Radius.
    q
        Integration constant ``q``.

    Returns
    -------
    float
        The numerical value.
    """
    values = {
        _r: r,
        _q: q,
        _kappa: model.kappa,
        _ell: model.ell,
        _beta: model.beta,
        _gamma: model.gamma,
    }
    return float(expr.subs(values).evalf(30))


@given(r=radii, q=charges_q, n=st.integers(min_value=0, max_value=6))
def test_radial_derivatives_match_sympy(r: float, q: float, n: int) -> None:
    """Eqs. (4.10), (4.18) and (E.3) agree with differentiating Eq. (4.2) directly."""
    expected = _reference(sp.diff(_m_ref, _r, n), FIG2, r, q)
    got = FIG2.radial_derivative(r, FIG2.charge_from_q(q), n)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)


@given(r=radii, q=charges_q)
def test_potential_temperature_and_first_law(r: float, q: float) -> None:
    """Φ = -A_t (4.7) equals ∂m/∂Q, and T dS/dr equals ∂m/∂r (first law (4.8))."""
    Q = FIG2.charge_from_q(q)
    assert FIG2.potential(r, Q) == pytest.approx(-_reference(_A_t, FIG2, r, q), rel=1e-10)
    dm_dQ = _reference(sp.diff(_m_ref, _q) / (2 * sp.pi), FIG2, r, q)
    assert FIG2.potential(r, Q) == pytest.approx(dm_dQ, rel=1e-9, abs=1e-12)
    dS_dr = FIG2.entropy(1.0)  # S is linear in r
    assert FIG2.temperature(r, Q) * dS_dr == pytest.approx(
        FIG2.radial_derivative(r, Q, 1), rel=1e-12, abs=1e-12
    )


@given(r=radii, q=charges_q, scale=st.floats(min_value=0.2, max_value=5.0))
def test_homogeneity_eq_4_29(r: float, q: float, scale: float) -> None:
    """m(λ r, λ Q) = λ^2 m(r, Q)."""
    Q = FIG2.charge_from_q(q)
    assert FIG2.mass_function(scale * r, scale * Q) == pytest.approx(
        scale**2 * FIG2.mass_function(r, Q), rel=1e-10, abs=1e-12
    )


def test_smarr_relation_eq_4_9() -> None:
    """2M = T_H S + Φ_e Q at the event horizon."""
    M = 3.5
    r_plus = FIG2.horizons(M, FIG2_Q)[-1]
    T = FIG2.temperature(r_plus, FIG2_Q)
    smarr = T * FIG2.entropy(r_plus) + FIG2.potential(r_plus, FIG2_Q) * FIG2_Q
    assert smarr == pytest.approx(2 * M, rel=1e-10)


def test_critical_masses_match_closed_forms_and_paper() -> None:
    """Eqs. (4.22)-(4.23)/(4.32)-(4.33) and the Fig. 2 / Fig. 3 energies."""
    w0 = float(sp.LambertW(-1 / (sp.E * sp.Rational(6, 5)), 0))
    wm1 = float(sp.LambertW(-1 / (sp.E * sp.Rational(6, 5)), -1))
    M_i, M_e = FIG2.critical_masses(FIG2_Q)
    prefactor = 2 * math.pi  # 2 pi kappa q^2 ell^2 / gamma^2 with unit parameters
    assert M_i == pytest.approx(-prefactor * w0 * (w0 + 2), rel=1e-12)
    assert M_e == pytest.approx(-prefactor * wm1 * (wm1 + 2), rel=1e-12)
    assert M_i - M_e == pytest.approx(1.856, abs=5e-4)  # Fig. 2 caption
    M_i3, M_e3 = FIG3.critical_masses(FIG3_Q)
    assert M_i3 - M_e3 == pytest.approx(3.9347, abs=5e-5)  # Fig. 4 neutral E_c


def test_positive_mass_window_eq_4_24() -> None:
    """M_e > 0 exactly when 1 < κβℓ^2 < e/2."""
    inside = NonlinearBTZ(1.0, 1.0, 1.3, 1.0)
    outside = NonlinearBTZ(1.0, 1.0, 1.4, 1.0)
    assert inside.critical_masses(FIG2_Q)[1] > 0
    assert outside.critical_masses(FIG2_Q)[1] < 0


def test_horizon_counts_across_the_strip() -> None:
    """One, three, and degenerate configurations around M_e < M < M_i (Fig. 2)."""
    M_i, M_e = FIG2.critical_masses(FIG2_Q)
    r_max, r_min = FIG2.stationary_radii(FIG2_Q)
    middle = 0.5 * (M_i + M_e)
    three = FIG2.horizons(middle, FIG2_Q)
    assert len(three) == 3
    for r in three:
        assert FIG2.mass_function(r, FIG2_Q) == pytest.approx(middle, rel=1e-12)
    assert three[0] < r_max < three[1] < r_min < three[2]
    degenerate_inner = FIG2.horizons(M_i, FIG2_Q)
    assert len(degenerate_inner) == 2
    assert degenerate_inner[0] == pytest.approx(r_max)
    assert len(FIG2.horizons(M_i + 1.0, FIG2_Q)) == 1
    assert len(FIG2.horizons(M_e - 0.5, FIG2_Q)) == 1
    assert FIG2.horizons(M_e, FIG2_Q)[-1] == pytest.approx(r_min)


def test_initial_event_horizon_and_endpoint_entropy_fig_6() -> None:
    """ΔS(E_c) = S(r_min) - S(r_+) ≈ -43.209 (Fig. 6 caption)."""
    M_i, _ = FIG2.critical_masses(FIG2_Q)
    r_plus = FIG2.horizons(M_i, FIG2_Q)[-1]
    r_min = FIG2.stationary_radii(FIG2_Q)[1]
    assert FIG2.entropy(r_min) - FIG2.entropy(r_plus) == pytest.approx(-43.209, abs=5e-4)


def test_one_horizon_sector() -> None:
    """For q/γ < 0 the mass function is convex: one horizon for M > 0 (Sec. 4.2)."""
    model = NonlinearBTZ(kappa=1.0, ell=1.0, beta=-0.3, gamma=1.0)
    Q = model.charge_from_q(-0.5)
    assert len(model.stationary_radii(Q)) == 1
    (r_h,) = model.horizons(2.0, Q)
    assert model.mass_function(r_h, Q) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="critical masses"):
        model.critical_masses(Q)


def test_monotonic_sector_has_no_stationary_points() -> None:
    """For q/γ > 0 and κβℓ^2 < 1 the mass function is monotonic."""
    model = NonlinearBTZ(kappa=1.0, ell=1.0, beta=0.5, gamma=1.0)
    assert model.stationary_radii(FIG2_Q) == ()
    assert len(model.horizons(1.0, FIG2_Q)) == 1


def test_unit_conversions() -> None:
    """Q = 2π q and M = π M_par / κ, Eq. (4.5)."""
    assert FIG3.q_from_charge(FIG3.charge_from_q(0.7)) == pytest.approx(0.7)
    assert FIG3.mass_from_parameter(2.0) == pytest.approx(2 * math.pi)
    assert FIG2.blackening(2.0, 1.0, FIG2_Q) == pytest.approx(
        (FIG2.mass_function(2.0, FIG2_Q) - 1.0) / math.pi
    )


@pytest.mark.parametrize(
    ("couplings", "message"),
    [
        ((0.0, 1.0, 1.0, 1.0), "kappa"),
        ((1.0, -1.0, 1.0, 1.0), "ell"),
        ((1.0, 1.0, 0.0, 1.0), "beta"),
    ],
)
def test_rejects_invalid_couplings(
    couplings: tuple[float, float, float, float], message: str
) -> None:
    """Unphysical couplings are rejected at construction.

    Parameters
    ----------
    couplings
        ``(kappa, ell, beta, gamma)``.
    message
        Expected error fragment.
    """
    with pytest.raises(ValueError, match=message):
        NonlinearBTZ(*couplings)


def test_rejects_invalid_points() -> None:
    """Non-positive radii, the wrong charge sector, and negative orders raise."""
    with pytest.raises(ValueError, match="radius"):
        FIG2.mass_function(0.0, FIG2_Q)
    with pytest.raises(ValueError, match=r"Eq\. \(4\.4\)"):
        FIG2.mass_function(1.0, -FIG2_Q)
    with pytest.raises(ValueError, match=r"Eq\. \(4\.4\)"):
        FIG2.mass_function(1.0, 0.0)
    with pytest.raises(ValueError, match="order"):
        FIG2.radial_derivative(1.0, FIG2_Q, -1)
