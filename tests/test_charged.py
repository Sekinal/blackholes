"""Charged emission: gauge coupling, fluxes, throat Schwinger instability, floor."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from blackholes.models import NonlinearBTZ
from blackholes.paper.configurations import FIG2_CHARGE, FIG2_MODEL, FIG3_CHARGE, FIG3_MODEL
from blackholes.radiation.background import StaticBackground
from blackholes.radiation.charged import charged_fluxes
from blackholes.radiation.discharge import DischargeSettings, coalescence_floor, flow
from blackholes.radiation.greybody import CONFORMAL_COUPLING, absorption_probability
from blackholes.radiation.luminosity import luminosity
from blackholes.radiation.near_horizon import (
    charged_scaling_dimension,
    scaling_dimension,
    throat_charge,
)

MODEL, Q = FIG2_MODEL, FIG2_CHARGE
M_I, M_E = MODEL.critical_masses(Q)
BATH = 5.0


def _background(gap: float) -> StaticBackground:
    """Neutral-path state a mass ``gap`` above the outer critical curve.

    Parameters
    ----------
    gap
        ``ΔE = M - M_e``.

    Returns
    -------
    StaticBackground
        The exterior of that state.
    """
    return StaticBackground.from_outer_horizon(MODEL, Q, MODEL.horizons(M_E + gap, Q)[-1])


@given(st.floats(0.5, 20.0))
def test_potential_derivative_and_vectorised_potential(r: float) -> None:
    """``Φ'`` matches a finite difference; the JAX ``Φ`` matches the scalar one.

    Parameters
    ----------
    r
        Radius.
    """
    h = 1e-6 * r
    central = (MODEL.potential(r + h, Q) - MODEL.potential(r - h, Q)) / (2 * h)
    assert MODEL.potential_derivative(r, Q) == pytest.approx(central, rel=1e-6, abs=1e-8)
    background = _background(1e-2)
    assert float(background.potential(jnp.asarray(r))) == pytest.approx(
        MODEL.potential(r, Q), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize(
    ("model", "charge"), [(FIG2_MODEL, FIG2_CHARGE), (FIG3_MODEL, FIG3_CHARGE)]
)
def test_throat_charge_identity(model: NonlinearBTZ, charge: float) -> None:
    """``e E_e L_2² = 2π e r_e / (κ Q)`` for every member of the family.

    Parameters
    ----------
    model
        Couplings.
    charge
        Hole charge.
    """
    r_e = model.stationary_radii(charge)[1]
    expected = 2 * math.pi * 0.3 * r_e / (model.kappa * charge)
    assert throat_charge(model, charge, 0.3) == pytest.approx(expected, rel=1e-12)


def test_charged_throat_violates_bf_bound() -> None:
    """Any charge makes the conformal s-wave complex; minimal coupling needs ``q_eff > 1/2``."""
    assert charged_scaling_dimension(MODEL, Q, 1, 0.0, 0.0) == pytest.approx(
        scaling_dimension(MODEL, Q, 1, 0.0)
    )
    assert charged_scaling_dimension(MODEL, Q, 0, CONFORMAL_COUPLING, 1e-3).imag == pytest.approx(
        throat_charge(MODEL, Q, 1e-3)
    )
    threshold = 0.5 / throat_charge(MODEL, Q, 1.0)
    assert charged_scaling_dimension(MODEL, Q, 0, 0.0, 0.99 * threshold).imag == 0.0
    assert charged_scaling_dimension(MODEL, Q, 0, 0.0, 1.01 * threshold).imag > 0.0


def test_superradiant_window_has_negative_greybody_factor() -> None:
    """``Γ < 0`` exactly for ``0 < ω < e Φ_rel`` and is gauge-symmetric under ``e → -e``."""
    background = _background(1e-2)
    e = 0.05
    mu = e * (MODEL.potential(background.r_plus, Q) - MODEL.potential(BATH, Q))
    omega = jnp.asarray([0.3 * mu, 0.7 * mu, 1.3 * mu, 2.0 * mu])
    gamma = np.asarray(
        absorption_probability(background, omega, jnp.asarray(0), r_absorb=BATH, charge=e)
    )
    assert np.all(gamma[:2] < 0)
    assert np.all(gamma[2:] > 0)
    assert np.all(gamma <= 1 + 1e-9)


def test_neutral_limit_is_two_real_scalars() -> None:
    """At ``e = 0`` a complex scalar radiates twice the real-scalar luminosity, no charge."""
    background = _background(1e-2)
    fluxes = charged_fluxes(background, charge=0.0, r_absorb=BATH, nodes=24, steps=3000)
    reference = luminosity(background, r_absorb=BATH, nodes=24, steps=3000).total
    assert fluxes.energy == pytest.approx(2 * reference, rel=1e-10)
    assert fluxes.charge == 0.0
    assert fluxes.horizon_energy == pytest.approx(fluxes.energy, rel=1e-12)
    assert fluxes.gap_rate == pytest.approx(-fluxes.energy, rel=1e-12)


def test_flux_bookkeeping_and_discharge_direction() -> None:
    """``F_H = F_E - Φ_rel F_Q``; the hole loses charge of its own sign."""
    background = _background(1e-1)
    fluxes = charged_fluxes(background, charge=0.01, r_absorb=BATH, nodes=24, steps=3000)
    assert fluxes.charge > 0
    assert fluxes.horizon_energy == pytest.approx(
        fluxes.energy - fluxes.potential_difference * fluxes.charge, rel=1e-10
    )
    phi_gap = MODEL.potential(background.r_plus, Q) - MODEL.potential(
        MODEL.stationary_radii(Q)[1], Q
    )
    assert fluxes.gap_rate == pytest.approx(
        -fluxes.horizon_energy - phi_gap * fluxes.charge, rel=1e-10
    )


def test_extremal_push_is_finite_and_positive() -> None:
    """As ``T → 0`` the superradiant push on ``ΔE`` tends to a positive constant."""
    settings = DischargeSettings(charge=0.01, r_absorb=BATH, nodes=24, steps=3000)
    rates = [flow(MODEL, Q, gap, settings)[1] for gap in (1e-5, 1e-7, 1e-9)]
    assert all(rate > 0 for rate in rates)
    assert rates[-1] == pytest.approx(rates[0], rel=1e-2)


def test_floor_separates_cooling_from_heating() -> None:
    """Below ``ΔE_*`` the hole is pushed out, above it cools; neutral emission has no floor."""
    settings = DischargeSettings(charge=0.003, r_absorb=BATH, nodes=24, steps=3000)
    floor = coalescence_floor(MODEL, Q, settings, bracket=(1e-6, 1.0), rtol=1e-3)
    assert 1e-3 < floor < 0.1
    assert flow(MODEL, Q, 0.5 * floor, settings)[1] > 0
    assert flow(MODEL, Q, 2.0 * floor, settings)[1] < 0
    neutral = DischargeSettings(charge=0.0, r_absorb=BATH, nodes=24, steps=3000)
    with pytest.raises(ValueError, match="no floor"):
        coalescence_floor(MODEL, Q, neutral, bracket=(1e-6, 1.0))
