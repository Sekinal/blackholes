"""Analytic near-extremal (AdS2 throat) predictions for the emission rate.

At the outer coalescence the event horizon is a double root of ``f`` at ``r_e``,
and the near-horizon geometry is AdS2 × S¹ with

    L_2² = 2 / f''(r_e),          S¹ radius  r_e.

A mode ``e^(ikφ)`` of a scalar with coupling ``ξ R`` becomes an AdS2 field of
mass ``m² = k²/r_e² + ξ R(r_e)``, where ``R(r_e) = -f''(r_e)`` because
``f'(r_e) = 0``. Its scaling dimension is

    Δ_k = 1/2 + sqrt(1/4 + m² L_2²) = 1/2 + sqrt(1/4 - 2ξ + 2k²/(f''(r_e) r_e²)).

Two consequences, independent of the far region and of where the radiation is
absorbed:

* **Conformal coupling saturates the BF bound** for ``k = 0``: ``ξ = 1/8`` gives
  ``m² L_2² = -1/4`` and ``Δ_0 = 1/2`` for *every* static extremal horizon of the
  form (2.1) in three dimensions. ``ξ > 1/8`` would violate the bound.
* At low temperature the greybody factor of mode ``k`` scales as
  ``Γ_k ∝ T^(2Δ_k - 1)`` (up to logarithms when ``Δ = 1/2``), so the luminosity
  scales as ``L ∝ T^(2Δ_0 + 1)``; with ``T ∝ ΔE^(1/2)`` (Eq. (5.8), ``n = 1``) the
  exponent of Sec. 5.2 is ``ν = Δ_0 + 1/2``. The paper's blackbody estimate
  ``ν = 3/2`` corresponds to ``Δ_0 = 1`` (minimal coupling); conformal coupling
  gives ``ν = 1``, which lies in the non-adiabatic window ``ν < 1 + n/2``.

Charged fields
--------------
The throat also carries the electric field ``E_e = ∂_r A_t(r_e) = -Φ'(r_e)``;
a field of charge ``e`` then has ``m² L_2² → m² L_2² - (e E_e L_2²)²``
(constant-field AdS2). For this family ``m'(r_e) = 0`` implies the identity

    e E_e L_2² = e r_e / (κ q) = 2π e r_e / (κ Q),

so a conformally coupled charged s-wave has complex ``Δ_0`` for *any* ``e ≠ 0``:
the extremal throat pair-produces charge (Schwinger effect) at zero temperature,
which is why charged emission never lets the hole reach coalescence.
"""

import math

from blackholes.models import NonlinearBTZ

__all__ = [
    "charged_scaling_dimension",
    "luminosity_exponent",
    "scaling_dimension",
    "throat_charge",
    "throat_radius_squared",
]


def _f_second_derivative(model: NonlinearBTZ, Q: float) -> tuple[float, float]:
    """Return ``(r_e, f''(r_e))`` at the outer critical radius.

    Parameters
    ----------
    model
        The black-hole family (multihorizon sector).
    Q
        Physical charge.

    Returns
    -------
    tuple[float, float]
        The degenerate horizon radius and the curvature of ``f`` there.
    """
    r_e = model.stationary_radii(Q)[1]
    return r_e, model.kappa / math.pi * model.radial_derivative(r_e, Q, 2)


def throat_radius_squared(model: NonlinearBTZ, Q: float) -> float:
    """AdS2 radius squared ``L_2² = 2 / f''(r_e)`` of the extremal throat.

    Parameters
    ----------
    model
        The black-hole family.
    Q
        Physical charge.

    Returns
    -------
    float
        ``L_2²``.
    """
    _, curvature = _f_second_derivative(model, Q)
    return 2.0 / curvature


def scaling_dimension(model: NonlinearBTZ, Q: float, k: int, xi: float) -> float:
    """Conformal dimension ``Δ_k`` of mode ``k`` in the extremal throat.

    Parameters
    ----------
    model
        The black-hole family.
    Q
        Physical charge.
    k
        Angular quantum number.
    xi
        Curvature coupling.

    Returns
    -------
    float
        ``Δ_k = 1/2 + sqrt(1/4 + m² L_2²)``.

    Raises
    ------
    ValueError
        If ``m² L_2² < -1/4`` (Breitenlohner–Freedman bound violated: the
        extremal throat is unstable for this field).
    """
    r_e, curvature = _f_second_derivative(model, Q)
    mass_squared_l2 = (2.0 / curvature) * (k**2 / r_e**2 - xi * curvature)
    discriminant = 0.25 + mass_squared_l2
    tolerance = 1e-12
    if discriminant < -tolerance:
        msg = f"BF bound violated: m^2 L_2^2 = {mass_squared_l2} < -1/4"
        raise ValueError(msg)
    return 0.5 + math.sqrt(max(discriminant, 0.0))


def luminosity_exponent(model: NonlinearBTZ, Q: float, xi: float) -> float:
    """Predicted ``ν`` in ``L ∝ ΔE^ν`` for neutral emission near coalescence.

    Parameters
    ----------
    model
        The black-hole family.
    Q
        Physical charge.
    xi
        Curvature coupling of the radiated scalar.

    Returns
    -------
    float
        ``ν = Δ_0 + 1/2`` (Sec. 5.2 with ``n = 1``), up to logarithms at ``Δ_0 = 1/2``.
    """
    return scaling_dimension(model, Q, 0, xi) + 0.5


def throat_charge(model: NonlinearBTZ, Q: float, e: float) -> float:
    """Effective AdS2 charge ``e E_e L_2²`` of a quantum of charge ``e``.

    Parameters
    ----------
    model
        The black-hole family.
    Q
        Physical charge of the hole.
    e
        Charge of the field quanta (units of ``Q``).

    Returns
    -------
    float
        ``-e Φ'(r_e) L_2²``, equal to ``2π e r_e / (κ Q)`` for this family.
    """
    r_e, _ = _f_second_derivative(model, Q)
    return -e * model.potential_derivative(r_e, Q) * throat_radius_squared(model, Q)


def charged_scaling_dimension(
    model: NonlinearBTZ, Q: float, k: int, xi: float, e: float
) -> complex:
    """Throat dimension ``Δ = 1/2 + sqrt(1/4 + m² L_2² - (e E_e L_2²)²)`` of a charged mode.

    Parameters
    ----------
    model
        The black-hole family.
    Q
        Physical charge of the hole.
    k
        Angular quantum number.
    xi
        Curvature coupling.
    e
        Charge of the field quanta.

    Returns
    -------
    complex
        ``Δ``; a non-zero imaginary part signals Schwinger pair production in
        the extremal throat (the charged BF bound is violated).
    """
    r_e, curvature = _f_second_derivative(model, Q)
    mass_squared_l2 = (2.0 / curvature) * (k**2 / r_e**2 - xi * curvature)
    discriminant = 0.25 + mass_squared_l2 - throat_charge(model, Q, e) ** 2
    return 0.5 + complex(discriminant) ** 0.5
