"""Greybody factors of a scalar field on the nonlinear BTZ exterior.

Wave equation
-------------
A scalar ``Φ = e^(-iωt + ikφ) ψ(r) / sqrt(r)`` with curvature coupling ``ξ R Φ``
obeys, in the tortoise coordinate ``dr_* = dr / f``,

    d²ψ/dr_*² + [ω² - V(r)] ψ = 0,
    V = f [k²/r² + ξ R + f'/(2r) - f/(4r²)],       R = -f'' - 2 f'/r.

``ξ = 1/8`` is conformal coupling in three dimensions; ``ξ = 0`` minimal coupling.

Boundary conditions
-------------------
* Horizon: purely ingoing, ``ψ = e^(-iω r_*)`` (flux ``ω`` into the hole).
* Absorbing radius ``r_a``: the field is decomposed into local WKB waves
  ``p^(-1/2) e^(±i∫p dr_*)`` with ``p = sqrt(ω² - V(r_a))``; the incoming
  amplitude ``B`` defines the absorption probability ``Γ = ω / |B|²``, which by
  detailed balance is also the emission probability per mode. ``r_a`` models
  where a bath absorbs the radiation; for this family the field strength grows
  logarithmically at infinity, so there is no canonical choice (see README), and
  results should be checked for ``r_a``-independence.

Numerics
--------
In ``s = ln(r - r_+)`` the system ``dψ/ds = φ/g``,
``dφ/ds = [(r - r_+) V/f - ω²/g] ψ`` (with ``φ = dψ/dr_*`` and
``g = f/(r - r_+)``) has bounded coefficients down to the horizon, so a fixed-step
classical Runge–Kutta scheme on a uniform ``s`` grid resolves the throat of
near-extremal states. Background quantities depend only on ``s``; they are
tabulated once and shared by the whole batch of ``(ω, k)``.
"""

import math

import jax
import jax.numpy as jnp

from blackholes.radiation.background import StaticBackground

__all__ = ["CONFORMAL_COUPLING", "absorption_probability", "potential"]

CONFORMAL_COUPLING = 1.0 / 8.0
"""Conformal coupling ``ξ = (n - 2) / (4 (n - 1))`` for ``n = 3`` spacetime dimensions."""

_HORIZON_OFFSET = 1e-7
"""Start the integration at ``r - r_+ = 1e-7 x`` the throat width (ingoing data error O(1e-7))."""


def _coefficients(
    background: StaticBackground, r: jax.Array, xi: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mode-independent pieces of the ODE at radii ``r``.

    Parameters
    ----------
    background
        The static exterior.
    r
        Radii outside the horizon.
    xi
        Curvature coupling.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        ``(1/g, (r - r_+)/r², (r - r_+) * rest)`` where ``V/f = k²/r² + rest``.
    """
    distance = r - background.r_plus
    g = background.g(r)
    f = g * distance
    f1, f2 = background.f_derivatives(r)
    ricci = -f2 - 2.0 * f1 / r
    rest = xi * ricci + f1 / (2.0 * r) - f / (4.0 * r**2)
    return 1.0 / g, distance / r**2, distance * rest


def potential(background: StaticBackground, r: jax.Array, k: jax.Array, xi: float) -> jax.Array:
    """Effective potential ``V(r)`` of the radial equation.

    Parameters
    ----------
    background
        The static exterior.
    r
        Radii outside the horizon.
    k
        Angular quantum numbers (broadcast against ``r``).
    xi
        Curvature coupling.

    Returns
    -------
    jax.Array
        ``f [k²/r² + ξR + f'/(2r) - f/(4r²)]``.
    """
    inverse_g, k_weight, rest = _coefficients(background, r, xi)
    # k_weight and rest already carry the factor (r - r_+); dividing by 1/g = (r - r_+)/f
    # turns (r - r_+) * V/f into V.
    return (k**2 * k_weight + rest) / inverse_g


def absorption_probability(
    background: StaticBackground,
    omega: jax.Array,
    k: jax.Array,
    *,
    r_absorb: float,
    xi: float = CONFORMAL_COUPLING,
    steps: int = 6000,
) -> jax.Array:
    """Greybody factor ``Γ_k(ω)`` for each ``(ω, k)`` pair (broadcast together).

    Parameters
    ----------
    background
        The static exterior.
    omega
        Positive frequencies.
    k
        Angular quantum numbers (only ``k²`` matters).
    r_absorb
        Radius of the absorbing bath, ``r_a > r_+``.
    xi
        Curvature coupling (default: conformal).
    steps
        Number of Runge–Kutta steps in ``s = ln(r - r_+)``.

    Returns
    -------
    jax.Array
        ``Γ`` in ``[0, 1]``; zero where ``V(r_a) >= ω²`` (no propagating wave
        reaches the bath).

    Raises
    ------
    ValueError
        If ``r_absorb <= r_+`` or ``steps < 1``.
    """
    if not r_absorb > background.r_plus:
        msg = f"r_absorb = {r_absorb} must exceed r_+ = {background.r_plus}"
        raise ValueError(msg)
    if steps < 1:
        msg = f"steps must be positive, got {steps}"
        raise ValueError(msg)
    omega_b, k_b = jnp.broadcast_arrays(
        jnp.asarray(omega, dtype=float), jnp.asarray(k, dtype=float)
    )

    s_start = math.log(_HORIZON_OFFSET * min(background.throat_width, background.r_plus))
    s_end = math.log(r_absorb - background.r_plus)
    h = (s_end - s_start) / steps
    # Tabulate on the half-step grid: RK4 needs s_i, s_i + h/2 and s_i + h.
    s_grid = s_start + 0.5 * h * jnp.arange(2 * steps + 1)
    inverse_g, k_weight, rest = _coefficients(background, background.r_plus + jnp.exp(s_grid), xi)

    def derivative(index: jax.Array, psi: jax.Array, phi: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Right-hand side at half-step grid point ``index``.

        Parameters
        ----------
        index
            Position on the half-step grid.
        psi
            Field values.
        phi
            ``dψ/dr_*`` values.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(dψ/ds, dφ/ds)``.
        """
        coupling = k_b**2 * k_weight[index] + rest[index] - omega_b**2 * inverse_g[index]
        return phi * inverse_g[index], coupling * psi

    def rk4_step(
        state: tuple[jax.Array, jax.Array], i: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """One classical Runge–Kutta step from grid point ``2i`` to ``2i + 2``.

        Parameters
        ----------
        state
            ``(ψ, φ)`` at ``s_i``.
        i
            Step index.

        Returns
        -------
        tuple
            ``(ψ, φ)`` at ``s_{i+1}`` and no per-step output.
        """
        psi, phi = state
        k1 = derivative(2 * i, psi, phi)
        k2 = derivative(2 * i + 1, psi + 0.5 * h * k1[0], phi + 0.5 * h * k1[1])
        k3 = derivative(2 * i + 1, psi + 0.5 * h * k2[0], phi + 0.5 * h * k2[1])
        k4 = derivative(2 * i + 2, psi + h * k3[0], phi + h * k3[1])
        psi_next = psi + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        phi_next = phi + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return (psi_next, phi_next), None

    psi0 = jnp.ones_like(omega_b, dtype=complex)
    phi0 = -1j * omega_b * psi0  # ingoing: dψ/dr_* = -iωψ
    (psi, phi), _ = jax.lax.scan(rk4_step, (psi0, phi0), jnp.arange(steps))

    v_absorb = potential(background, jnp.asarray(r_absorb), k_b, xi)
    p_squared = omega_b**2 - v_absorb
    p = jnp.sqrt(jnp.where(p_squared > 0, p_squared, 1.0))
    incoming = (psi - phi / (1j * p)) * jnp.sqrt(p) / 2.0
    gamma = omega_b / jnp.abs(incoming) ** 2
    return jnp.where(p_squared > 0, gamma, 0.0)
