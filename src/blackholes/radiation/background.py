"""Static background for wave propagation outside the tracked event horizon.

The wave equation needs ``f``, ``f'``, ``f''`` and especially the ratio
``g(r) = f(r) / (r - r_+)`` everywhere between the event horizon and the
absorbing radius. Near extremality ``f`` is the tiny difference of two O(1)
numbers, ``(κ/π)[m(r) - m(r_+)]``, and naive evaluation loses most digits right
where the physics happens (the AdS2 throat). We avoid this by expanding the mass
function about the outer critical radius ``r_min`` (where ``∂_r m = 0``):

    m(r) - m(r_+) = sum_{j>=2} c_j (y^j - y_+^j),   y = r - r_min,  y_+ = r_+ - r_min,

and dividing by ``r - r_+ = y - y_+`` *analytically*:

    g(r) = (κ/π) sum_j c_j S_j,   S_j = (y^j - y_+^j)/(y - y_+) = sum_{i<j} y^i y_+^(j-1-i),

computed by the recurrence ``S_1 = 1``, ``S_j = y S_{j-1} + y_+^(j-1)``. No
subtraction of nearly equal numbers remains. Far from ``r_min`` (where the
Taylor series converges slowly) the direct formula is used instead; there
``r - r_+`` is not small and the direct quotient is accurate.

Everything is written with ``jax.numpy`` so it vectorises over radii and runs on
GPU. Double precision is enabled on import (it is required for the throat).
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from blackholes.models import NonlinearBTZ

# Double precision is required to resolve the throat; JAX ships this setter unannotated.
jax.config.update("jax_enable_x64", val=True)  # type: ignore[no-untyped-call]

__all__ = ["StaticBackground"]

_TAYLOR_TERMS = 90
"""Terms of the expansion about ``r_min``; ``0.6**90 ~ 1e-20`` inside the switch radius."""

_SWITCH_FRACTION = 0.6
"""Use the expansion for ``|r - r_min| < 0.6 r_min`` (its radius of convergence is ``r_min``)."""


@dataclass(frozen=True)
class StaticBackground:
    """Exterior of one stationary member of the nonlinear BTZ family, Eq. (4.2).

    Construct with :meth:`from_outer_horizon`; the state is labelled by its event
    horizon ``r_+`` (rather than the mass) so that ``f(r_+) = 0`` holds exactly.

    Attributes
    ----------
    model
        The fixed-coupling family.
    charge
        Physical charge ``Q``.
    r_plus
        Event-horizon radius (simple root, ``r_+ > r_min``).
    throat_width
        Distance from ``r_+`` to the next root below it (the middle horizon
        ``r_0``, Sec. 4.2), or to ``r_min`` when there is none. It sets the scale
        on which ``g`` varies near the horizon.
    r_min
        Outer critical radius (local minimum of ``m``), expansion point.
    temperature
        Hawking temperature ``T_H = f'(r_+)/4π``.
    taylor
        Coefficients ``c_2 ... c_J`` of ``m(r_min + y) - m(r_min)`` (``c_1 = 0``).
    """

    model: NonlinearBTZ
    charge: float
    r_plus: float
    throat_width: float
    r_min: float
    temperature: float
    taylor: jax.Array

    @classmethod
    def from_outer_horizon(cls, model: NonlinearBTZ, Q: float, r_plus: float) -> "StaticBackground":
        """Build the background whose event horizon sits at ``r_plus``.

        Parameters
        ----------
        model
            The black-hole family (multihorizon sector).
        Q
            Physical charge.
        r_plus
            Event-horizon radius, strictly outside ``r_min``.

        Returns
        -------
        StaticBackground
            The background with mass ``m(r_plus, Q)``.

        Raises
        ------
        ValueError
            If ``r_plus <= r_min`` (no simple outer horizon there).
        """
        _, r_min = model.stationary_radii(Q)
        if not r_plus > r_min:
            msg = f"r_plus = {r_plus} must exceed r_min = {r_min}"
            raise ValueError(msg)
        mass = model.mass_function(r_plus, Q)
        below = [r for r in model.horizons(mass, Q) if r < r_plus]
        throat_width = r_plus - (below[-1] if below else r_min)
        taylor = jnp.array(
            [model.taylor_coefficient(r_min, Q, j) for j in range(2, _TAYLOR_TERMS + 1)]
        )
        return cls(
            model=model,
            charge=Q,
            r_plus=r_plus,
            throat_width=throat_width,
            r_min=r_min,
            temperature=model.temperature(r_plus, Q),
            taylor=taylor,
        )

    @property
    def mass(self) -> float:
        """Physical mass ``m(r_+, Q)`` of this state.

        Returns
        -------
        float
            The mass.
        """
        return self.model.mass_function(self.r_plus, self.charge)

    def _log_constants(self) -> tuple[float, float]:
        """Return ``(C, r_q)`` with ``m ∝ r²/ℓ² - C r ln(r / r_q)``, Eq. (4.10).

        Returns
        -------
        tuple[float, float]
            ``C = 2√2 κ q / γ`` and ``r_q = √2 q / (β γ)``.
        """
        model = self.model
        q = model.q_from_charge(self.charge)
        return (
            2.0 * math.sqrt(2.0) * model.kappa * q / model.gamma,
            math.sqrt(2.0) * q / (model.beta * model.gamma),
        )

    def _direct_mass(self, r: jax.Array) -> jax.Array:
        """Mass function ``m(r, Q)`` by the closed form (4.10), vectorised.

        Parameters
        ----------
        r
            Radii (positive).

        Returns
        -------
        jax.Array
            ``m(r, Q)``.
        """
        model = self.model
        strength, scale = self._log_constants()
        return (math.pi / model.kappa) * (r**2 / model.ell**2 - strength * r * jnp.log(r / scale))

    def g(self, r: jax.Array) -> jax.Array:
        """Cancellation-free ``g(r) = f(r) / (r - r_+)`` (see module docstring).

        Parameters
        ----------
        r
            Radii in ``(r_+, ∞)``.

        Returns
        -------
        jax.Array
            ``g(r)``; ``g(r_+) = 4π T_H``.
        """
        y = r - self.r_min
        y_plus = self.r_plus - self.r_min

        def step(
            carry: tuple[jax.Array, jax.Array, jax.Array], c_j: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
            """Advance ``S_j = y S_{j-1} + y_+^(j-1)`` and accumulate ``c_j S_j``.

            Parameters
            ----------
            carry
                ``(S_{j-1}, y_+^(j-1), partial sum)``.
            c_j
                Next Taylor coefficient.

            Returns
            -------
            tuple
                Updated carry and no per-step output.
            """
            s_prev, y_plus_power, total = carry
            s_j = y * s_prev + y_plus_power
            return (s_j, y_plus_power * y_plus, total + c_j * s_j), None

        ones = jnp.ones_like(y)
        (_, _, series), _ = jax.lax.scan(
            step, (ones, ones * y_plus, jnp.zeros_like(y)), self.taylor
        )
        direct = (self._direct_mass(r) - self.mass) / (r - self.r_plus)
        use_series = jnp.abs(y) < _SWITCH_FRACTION * self.r_min
        return (self.model.kappa / math.pi) * jnp.where(use_series, series, direct)

    def f_derivatives(self, r: jax.Array) -> tuple[jax.Array, jax.Array]:
        """First and second radial derivatives of ``f``, from Eq. (4.18).

        Parameters
        ----------
        r
            Radii (positive).

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(f'(r), f''(r))``.
        """
        model = self.model
        strength, scale = self._log_constants()
        f1 = 2.0 * r / model.ell**2 - strength * (jnp.log(r / scale) + 1.0)
        f2 = 2.0 / model.ell**2 - strength / r
        return f1, f2

    def potential(self, r: jax.Array) -> jax.Array:
        """Electric potential ``Φ(r) = -A_t(r)`` of Eq. (4.7), vectorised.

        Only differences of ``Φ`` are physical; the wave equation uses
        ``Φ(r) - Φ(r_a)`` (gauge in which the bath at ``r_a`` is neutral).

        Parameters
        ----------
        r
            Radii (positive).

        Returns
        -------
        jax.Array
            ``(√2 r / γ) [1 - ln(r / r_q)]``.
        """
        _, scale = self._log_constants()
        return math.sqrt(2.0) * r / self.model.gamma * (1.0 - jnp.log(r / scale))
