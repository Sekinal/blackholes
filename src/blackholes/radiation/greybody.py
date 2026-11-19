"""Greybody factors of a scalar field on the nonlinear BTZ exterior.

Wave equation
-------------
A scalar ``Φ = e^(-iωt + ikφ) ψ(r) / sqrt(r)`` with curvature coupling ``ξ R Φ``
obeys, in the tortoise coordinate ``dr_* = dr / f``,

    d²ψ/dr_*² + [ω² - V(r)] ψ = 0,
    V = f [k²/r² + ξ R + f'/(2r) - f/(4r²)],       R = -f'' - 2 f'/r.

``ξ = 1/8`` is conformal coupling in three dimensions; ``ξ = 0`` minimal coupling.

A field of charge ``e`` couples to ``A_t = -Φ`` (Eq. (4.7)) through
``∂_t → ∂_t - i e A_t``, which replaces ``ω²`` by ``(ω - e[Φ(r) - Φ(r_a)])²``
in the gauge where the bath at ``r_a`` is neutral (so ``ω`` is the energy the
bath receives). At the horizon the field oscillates with
``ω̃ = ω - e Φ_rel``, ``Φ_rel = Φ(r_+) - Φ(r_a)``, which is gauge invariant.

Boundary conditions
-------------------
* Horizon: purely ingoing, ``ψ = e^(-iω̃ r_*)`` (flux ``ω̃`` into the hole).
* Absorbing radius ``r_a``: the field is decomposed into local WKB waves
  ``p^(-1/2) e^(±i∫p dr_*)`` with ``p = sqrt(ω² - V(r_a))``; the incoming
  amplitude ``B`` defines the absorption probability ``Γ = ω̃ / |B|²``, which by
  detailed balance is also the emission probability per mode. ``Γ < 0`` for
  ``0 < ω < e Φ_rel`` is superradiance (stimulated emission of charge). ``r_a`` models
  where a bath absorbs the radiation; for this family the field strength grows
  logarithmically at infinity, so there is no canonical choice (see README), and
  results should be checked for ``r_a``-independence.

Numerics
--------
In ``s = ln(r - r_+)`` the system ``dψ/ds = φ/g``,
``dφ/ds = [(r - r_+) V/f - (ω - eΦ + eΦ(r_a))²/g] ψ`` (with ``φ = dψ/dr_*`` and
``g = f/(r - r_+)``) has bounded coefficients down to the horizon, so a uniform
``s`` grid resolves the throat of near-extremal states. Near the horizon the
solution oscillates ``|ω̃|/(4π T)`` times per unit ``s``; for superradiant
quanta (``|ω̃| ≫ T``) that is far too stiff for explicit Runge–Kutta. The system
is linear, ``dY/ds = A(s) Y`` with ``A = [[0, 1/g], [c, 0]]``, so we use the
fourth-order Magnus integrator: ``Y_{n+1} = exp(Ω_n) Y_n`` with

    Ω_n = (h/2)(A_1 + A_2) + (√3 h²/12)[A_2, A_1],

``A_{1,2}`` sampled at the two Gauss points of the step. ``Ω`` is a traceless
2×2 matrix, so ``exp(Ω) = cosh(λ) I + (sinh(λ)/λ) Ω`` with ``λ² = -det Ω``
exactly: oscillations inside a step are integrated exactly whatever their
frequency. Background quantities depend only on ``s``; they are tabulated once
and shared by the whole batch of ``(ω, k)``.
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


_SINC_SWITCH = 1e-4
"""Below this ``|λ|`` the series ``1 + λ²/6`` replaces ``sinh(λ)/λ`` (error ~ λ⁴/120)."""


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


def _magnus_propagator(
    a1: jax.Array, a2: jax.Array, c1: jax.Array, c2: jax.Array, h: float
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Entries of ``exp(Ω)`` for ``A_j = [[0, a_j], [c_j, 0]]`` (4th-order Magnus).

    With ``[A_2, A_1] = diag(d, -d)``, ``d = a_2 c_1 - a_1 c_2``, the exponent is
    ``Ω = [[α, β], [γ, -α]]`` with ``α = √3 h² d / 12``, ``β = h (a_1 + a_2)/2``,
    ``γ = h (c_1 + c_2)/2``, and ``exp(Ω) = C I + S Ω`` where ``λ² = α² + βγ``,
    ``C = cosh λ`` and ``S = sinh(λ)/λ`` (``cos`` and ``sin`` when ``λ² < 0``).
    All inputs are real, so this is evaluated in real arithmetic.

    Parameters
    ----------
    a1
        ``1/g`` at the first Gauss point.
    a2
        ``1/g`` at the second Gauss point.
    c1
        Coupling ``c`` at the first Gauss point.
    c2
        Coupling ``c`` at the second Gauss point.
    h
        Step in ``s``.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(m11, m12, m21, m22)`` of the one-step propagator.
    """
    alpha = math.sqrt(3.0) * h**2 / 12.0 * (a2 * c1 - a1 * c2)
    beta = 0.5 * h * (a1 + a2)
    gamma = 0.5 * h * (c1 + c2)
    lam2 = alpha**2 + beta * gamma
    root = jnp.sqrt(jnp.abs(lam2))
    growing = lam2 >= 0
    small = root < _SINC_SWITCH
    cosine = jnp.where(growing, jnp.cosh(root), jnp.cos(root))
    sine = jnp.where(growing, jnp.sinh(root), jnp.sin(root))
    # sinh(λ)/λ -> 1 + λ²/6 as λ -> 0 (exact to rounding below the switch).
    sinc = jnp.where(small, 1.0 + lam2 / 6.0, sine / jnp.where(small, 1.0, root))
    return cosine + sinc * alpha, sinc * beta, sinc * gamma, cosine - sinc * alpha


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
    charge: float = 0.0,
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
    charge
        Charge ``e`` of the field quanta, in the units of ``Q`` (first law (4.8)).
    steps
        Number of Magnus steps in ``s = ln(r - r_+)``.

    Returns
    -------
    jax.Array
        ``Γ <= 1`` with the sign of ``ω̃ = ω - e Φ_rel`` (negative in the
        superradiant window); zero where ``V(r_a) >= ω²`` (no propagating wave
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
    # Two Gauss–Legendre points per step (the 4th-order Magnus scheme samples A there).
    offsets = jnp.array([0.5 - math.sqrt(3.0) / 6.0, 0.5 + math.sqrt(3.0) / 6.0])
    s_nodes = s_start + h * (jnp.arange(steps)[:, None] + offsets[None, :])
    radii = background.r_plus + jnp.exp(s_nodes)
    inverse_g, k_weight, rest = _coefficients(background, radii, xi)
    bath_potential = background.potential(jnp.asarray(r_absorb))
    # e [Φ(r) - Φ(r_a)]: electrostatic energy of a quantum at r relative to the bath.
    energy_shift = charge * (background.potential(radii) - bath_potential)
    horizon_shift = charge * (background.potential(jnp.asarray(background.r_plus)) - bath_potential)
    omega_horizon = omega_b - horizon_shift

    def coupling(i: jax.Array, node: int) -> jax.Array:
        """Lower-left entry ``c = (r - r_+) V/f - (local ω)²/g`` at a Gauss node.

        Parameters
        ----------
        i
            Step index.
        node
            Gauss node within the step (0 or 1).

        Returns
        -------
        jax.Array
            ``c`` for the whole ``(ω, k)`` batch.
        """
        local_omega = omega_b - energy_shift[i, node]
        return k_b**2 * k_weight[i, node] + rest[i, node] - local_omega**2 * inverse_g[i, node]

    def magnus_step(
        state: tuple[jax.Array, jax.Array], i: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Advance ``(ψ, φ)`` across step ``i`` with the 4th-order Magnus propagator.

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
        m11, m12, m21, m22 = _magnus_propagator(
            inverse_g[i, 0], inverse_g[i, 1], coupling(i, 0), coupling(i, 1), h
        )
        return (m11 * psi + m12 * phi, m21 * psi + m22 * phi), None

    psi0 = jnp.ones_like(omega_b, dtype=complex)
    phi0 = -1j * omega_horizon * psi0  # ingoing: dψ/dr_* = -iω̃ψ
    (psi, phi), _ = jax.lax.scan(magnus_step, (psi0, phi0), jnp.arange(steps))

    v_absorb = potential(background, jnp.asarray(r_absorb), k_b, xi)
    p_squared = omega_b**2 - v_absorb
    p = jnp.sqrt(jnp.where(p_squared > 0, p_squared, 1.0))
    incoming = (psi - phi / (1j * p)) * jnp.sqrt(p) / 2.0
    # Flux conservation: |B|² - |A|² = ω̃, so Γ = 1 - |A/B|² = ω̃ / |B|².
    gamma = omega_horizon / jnp.abs(incoming) ** 2
    return jnp.where(p_squared > 0, gamma, 0.0)
