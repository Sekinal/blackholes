"""Energy, charge and entropy fluxes of a complex charged scalar.

Physics
-------
A complex scalar carries quanta of charge ``+e`` and ``-e``. For a quantum of
charge ``e_σ`` (``σ = ±``) the horizon sees the frequency ``ω̃ = ω - μ_σ`` with
chemical potential ``μ_σ = e_σ Φ_rel``, ``Φ_rel = Φ(r_+) - Φ(r_a)``, and the
mean number emitted per mode and unit ``dω/2π`` is ``Γ(ω) / (e^(ω̃/T) - 1)``.
Both factors change sign at ``ω = μ_σ``, so the rate is positive everywhere; for
``0 < ω < μ_σ`` it stays finite as ``T → 0``: that is superradiant (Schwinger-like)
discharge, which never switches off at extremality.

Three fluxes leave the hole (per unit Killing time ``t``):

* ``F_E = Σ ∫ ω Γ n``: energy received by the bath;
* ``F_Q = Σ e_σ ∫ Γ n``: charge, so ``dQ/dt = -F_Q``;
* ``F_H = Σ ∫ ω̃ Γ n = F_E - Φ_rel F_Q``: horizon energy, so ``T dS/dt = -F_H``
  by the first law (4.8). Superradiant quanta have ``ω̃ < 0`` and *raise* ``S``.

The paper's distance from coalescence, ``ΔE = M - M_e(Q) = m(r_+, Q) - m(r_e, Q)``,
then evolves as (using ``∂_r m(r_e) = 0`` and ``∂_Q m = Φ``)

    dΔE/dt = T dS/dt + [Φ(r_+) - Φ(r_e)] dQ/dt = -F_H - [Φ(r_+) - Φ(r_e)] F_Q.

Every quantity here is gauge invariant (only potential differences enter).

Quadrature
----------
For ``μ > 0`` the window ``[0, μ]`` is integrated with Gauss–Legendre, split at
``μ - 40 T`` so the thermal edge of width ``T`` is resolved at any temperature.
The tail ``ω > max(μ, 0)`` uses Gauss–Laguerre in ``x = (ω - max(μ, 0)) / T``.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from blackholes.radiation.background import StaticBackground
from blackholes.radiation.greybody import CONFORMAL_COUPLING, absorption_probability
from blackholes.series.numeric import FloatArray

__all__ = ["ChargedFluxes", "charged_fluxes"]

_EDGE_WIDTH = 40.0
"""Width, in units of ``T``, of the window below ``μ`` given its own quadrature."""


@dataclass(frozen=True, slots=True)
class ChargedFluxes:
    """Fluxes carried away by one complex scalar of charge ``±e``.

    Attributes
    ----------
    energy
        ``F_E``, energy delivered to the bath per unit time.
    charge
        ``F_Q = -dQ/dt``.
    horizon_energy
        ``F_H = -T dS/dt``; negative when superradiance dominates.
    gap_rate
        ``dΔE/dt`` with ``ΔE = M - M_e(Q)``; positive means the hole is pushed
        *away* from coalescence.
    potential_difference
        ``Φ_rel = Φ(r_+) - Φ(r_a)``.
    """

    energy: float
    charge: float
    horizon_energy: float
    gap_rate: float
    potential_difference: float


def _frequency_rule(
    chemical_potential: float, temperature: float, nodes: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Nodes ``ω_i``, weights ``w_i`` and occupations so ``∫ dω Γ n ≈ Σ w_i Γ(ω_i) n_i``.

    Parameters
    ----------
    chemical_potential
        ``μ = e_σ Φ_rel``.
    temperature
        Hawking temperature ``T > 0``.
    nodes
        Points per sub-interval.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]
        ``(ω, w, n)``; for the Laguerre tail ``n`` already includes the ``e^x``
        that undoes the quadrature weight.
    """
    floor = max(chemical_potential, 0.0)
    x, laguerre = np.polynomial.laguerre.laggauss(nodes)
    # y = ω̃/T >= x, and n e^x = e^(x - y) / (1 - e^(-y)) never overflows.
    y = x + (floor - chemical_potential) / temperature
    omegas = [floor + temperature * x]
    weights = [temperature * laguerre]
    occupations = [np.exp(x - y) / -np.expm1(-y)]

    split = max(chemical_potential - _EDGE_WIDTH * temperature, 0.0)
    legendre_x, legendre_w = np.polynomial.legendre.leggauss(nodes)
    for low, high in ((0.0, split), (split, chemical_potential)):
        if high > low:
            half = 0.5 * (high - low)
            omega = low + half * (legendre_x + 1.0)
            omegas.append(omega)
            weights.append(half * legendre_w)
            # ω̃ < 0 here, so expm1 cannot overflow; n < 0 and Γ < 0.
            occupations.append(1.0 / np.expm1((omega - chemical_potential) / temperature))
    return np.concatenate(omegas), np.concatenate(weights), np.concatenate(occupations)


def charged_fluxes(
    background: StaticBackground,
    *,
    charge: float,
    r_absorb: float,
    xi: float = CONFORMAL_COUPLING,
    k_max: int = 6,
    nodes: int = 40,
    steps: int = 6000,
) -> ChargedFluxes:
    """Fluxes of a complex scalar with quanta of charge ``±charge`` into a bath.

    Parameters
    ----------
    background
        The static exterior.
    charge
        ``e`` in the units of ``Q``; ``0`` gives two copies of a neutral scalar.
    r_absorb
        Radius of the absorbing bath.
    xi
        Curvature coupling.
    k_max
        Largest ``|k|`` included.
    nodes
        Quadrature points per frequency sub-interval.
    steps
        Runge–Kutta steps per greybody factor.

    Returns
    -------
    ChargedFluxes
        Energy, charge and horizon-energy fluxes and ``dΔE/dt``.
    """
    temperature = background.temperature
    model, Q = background.model, background.charge
    phi_plus = model.potential(background.r_plus, Q)
    phi_rel = phi_plus - model.potential(r_absorb, Q)
    ks = np.arange(k_max + 1)
    multiplicity = np.where(ks == 0, 1.0, 2.0)

    energy = charge_flux = horizon_energy = 0.0
    for sign in (1.0, -1.0):
        e_sigma = sign * charge
        mu = e_sigma * phi_rel
        omega, weight, occupation = _frequency_rule(mu, temperature, nodes)
        gamma = np.asarray(
            absorption_probability(
                background,
                jnp.asarray(omega[None, :]),
                jnp.asarray(ks[:, None]),
                r_absorb=r_absorb,
                xi=xi,
                charge=e_sigma,
                steps=steps,
            )
        )
        # Number flux per unit dω/2π, summed over modes; Γ n >= 0 term by term.
        density = multiplicity @ (gamma * occupation) * weight / (2.0 * np.pi)
        number = float(density.sum())
        flux_energy = float(density @ omega)
        energy += flux_energy
        charge_flux += e_sigma * number
        horizon_energy += flux_energy - mu * number

    _, r_e = model.stationary_radii(Q)
    gap_rate = -horizon_energy - (phi_plus - model.potential(r_e, Q)) * charge_flux
    return ChargedFluxes(energy, charge_flux, horizon_energy, gap_rate, phi_rel)
