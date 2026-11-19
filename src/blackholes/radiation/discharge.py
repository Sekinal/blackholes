"""Where charged Hawking emission actually takes the black hole in ``(Q, ΔE)``.

The paper prescribes the charge profile ``Q(E)`` of each emission channel. With
a charged field in the spectrum the profile is not free: the state ``(Q, ΔE)``,
``ΔE = M - M_e(Q)``, flows with the fluxes of :mod:`~blackholes.radiation.charged`,

    dQ/dt = -F_Q,        dΔE/dt = -F_H - [Φ(r_+) - Φ(r_e)] F_Q.

Thermal emission lowers ``ΔE`` (towards coalescence) at a rate that vanishes
with ``T``; superradiant discharge raises it at a rate that stays finite as
``T → 0`` (see :func:`~blackholes.radiation.near_horizon.charged_scaling_dimension`).
They balance on a *floor* ``ΔE_*(Q)`` that the hole cannot cross: it settles
above the outer critical curve and discharges along it instead of reaching the
coalescence of Sec. 4.
"""

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from blackholes.models import NonlinearBTZ
from blackholes.radiation.background import StaticBackground
from blackholes.radiation.charged import charged_fluxes
from blackholes.radiation.greybody import CONFORMAL_COUPLING

__all__ = ["DischargeSettings", "coalescence_floor", "flow"]


@dataclass(frozen=True, slots=True)
class DischargeSettings:
    """Field content, bath and numerical resolution shared by every evaluation.

    Attributes
    ----------
    charge
        Charge ``e`` of the quanta of one complex scalar.
    r_absorb
        Radius of the absorbing bath.
    xi
        Curvature coupling.
    k_max
        Largest angular mode.
    nodes
        Frequency quadrature points per sub-interval.
    steps
        Magnus steps per greybody factor.
    """

    charge: float
    r_absorb: float
    xi: float = CONFORMAL_COUPLING
    k_max: int = 6
    nodes: int = 32
    steps: int = 4000


def flow(
    model: NonlinearBTZ, Q: float, gap: float, settings: DischargeSettings
) -> tuple[float, float]:
    """Time derivatives ``(dQ/dt, dΔE/dt)`` at the state ``(Q, ΔE)``.

    Parameters
    ----------
    model
        The black-hole family.
    Q
        Physical charge.
    gap
        ``ΔE = M - M_e(Q) > 0``.
    settings
        Field, bath and resolution.

    Returns
    -------
    tuple[float, float]
        ``(-F_Q, dΔE/dt)``.
    """
    _, M_e = model.critical_masses(Q)
    r_plus = model.horizons(M_e + gap, Q)[-1]
    fluxes = charged_fluxes(
        StaticBackground.from_outer_horizon(model, Q, r_plus),
        charge=settings.charge,
        r_absorb=settings.r_absorb,
        xi=settings.xi,
        k_max=settings.k_max,
        nodes=settings.nodes,
        steps=settings.steps,
    )
    return -fluxes.charge, fluxes.gap_rate


def coalescence_floor(
    model: NonlinearBTZ,
    Q: float,
    settings: DischargeSettings,
    *,
    bracket: tuple[float, float],
    rtol: float = 1e-4,
) -> float:
    """Find the gap ``ΔE_*`` where ``dΔE/dt = 0`` (thermal cooling = superradiant heating).

    Parameters
    ----------
    model
        The black-hole family.
    Q
        Physical charge.
    settings
        Field, bath and resolution.
    bracket
        ``(low, high)`` gaps with ``dΔE/dt > 0`` at ``low`` and ``< 0`` at ``high``.
    rtol
        Relative tolerance on ``ΔE_*``.

    Returns
    -------
    float
        The floor ``ΔE_*``, found by Brent's method in ``ln ΔE``.

    Raises
    ------
    ValueError
        If the bracket does not straddle a sign change.
    """
    low, high = (math.log(g) for g in bracket)

    def rate(log_gap: float) -> float:
        """``dΔE/dt`` at ``ΔE = e^log_gap``.

        Parameters
        ----------
        log_gap
            ``ln ΔE``.

        Returns
        -------
        float
            The gap rate.
        """
        return flow(model, Q, math.exp(log_gap), settings)[1]

    rate_low, rate_high = rate(low), rate(high)
    if not (rate_low > 0 > rate_high):
        msg = f"no floor in bracket {bracket}: dΔE/dt = {rate_low:.3e}, {rate_high:.3e}"
        raise ValueError(msg)
    return math.exp(float(brentq(rate, low, high, xtol=rtol)))
