"""Hawking luminosity from greybody factors.

For a scalar in 2+1 dimensions each angular mode ``k`` is a one-dimensional
channel, so the energy flux through the bath is

    L = sum_k ∫_0^∞ (dω / 2π) ω Γ_k(ω) / (e^(ω/T) - 1).

Substituting ``x = ω/T`` gives ``L = (T²/2π) sum_k ∫ dx x Γ_k(xT) / (e^x - 1)``.
The integral is evaluated with Gauss–Laguerre quadrature (weight ``e^(-x)``),
leaving the smooth integrand ``x Γ / (1 - e^(-x))``, which tends to ``Γ(0)`` at
``x = 0``.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from blackholes.radiation.background import StaticBackground
from blackholes.radiation.greybody import CONFORMAL_COUPLING, absorption_probability
from blackholes.series.numeric import FloatArray

__all__ = ["Luminosity", "luminosity"]


@dataclass(frozen=True, slots=True)
class Luminosity:
    """Total luminosity and its decomposition into angular modes.

    Attributes
    ----------
    total
        ``L`` summed over ``k = -k_max ... k_max``.
    per_mode
        Contribution of ``|k| = 0, 1, ..., k_max`` (both signs included for ``k > 0``).
    temperature
        Hawking temperature of the background.
    """

    total: float
    per_mode: FloatArray
    temperature: float


def luminosity(
    background: StaticBackground,
    *,
    r_absorb: float,
    xi: float = CONFORMAL_COUPLING,
    k_max: int = 6,
    nodes: int = 40,
    steps: int = 6000,
) -> Luminosity:
    """Luminosity of a single real scalar into a bath at ``r_absorb``.

    Parameters
    ----------
    background
        The static exterior.
    r_absorb
        Radius of the absorbing bath.
    xi
        Curvature coupling.
    k_max
        Largest ``|k|`` included; check that ``per_mode[-1]`` is negligible.
    nodes
        Gauss–Laguerre nodes in ``x = ω/T``.
    steps
        Runge–Kutta steps per greybody factor.

    Returns
    -------
    Luminosity
        Total and per-mode luminosities.
    """
    x, weights = np.polynomial.laguerre.laggauss(nodes)
    temperature = background.temperature
    ks = np.arange(k_max + 1)
    omega = jnp.asarray(x[None, :] * temperature)
    gamma = np.asarray(
        absorption_probability(
            background, omega, jnp.asarray(ks[:, None]), r_absorb=r_absorb, xi=xi, steps=steps
        )
    )
    integrand = x * gamma / -np.expm1(-x)
    per_k = temperature**2 / (2 * np.pi) * (integrand @ weights)
    multiplicity = np.where(ks == 0, 1.0, 2.0)
    per_mode = (multiplicity * per_k).astype(np.float64)
    return Luminosity(float(per_mode.sum()), per_mode, temperature)
