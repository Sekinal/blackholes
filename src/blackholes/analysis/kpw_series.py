"""Power series of the neutral KPW exponent in the emitted energy (paper Sec. 2.2, App. E).

For neutral emission from a simple outer horizon at ``r_+`` the horizon
displacement is ``Δr_+(E) = sum_n a_n E^n`` with the Lagrange–Bürmann coefficients
of ``h(y) = m(r_+) - m(r_+ + y)`` (Eqs. (2.14), (2.16)). In three-dimensional Einstein
gravity ``S = 4π^2 r / κ`` is linear in the radius, so the leading tunneling
exponent is simply ``ΔS(E) = (4π^2/κ) Δr_+(E)`` (Eq. (4.15)).
"""

import numpy as np

from blackholes.models import NonlinearBTZ
from blackholes.series.numeric import FloatArray, lagrange_burmann_inverse_numeric

__all__ = ["horizon_displacement_series", "neutral_entropy_series"]


def horizon_displacement_series(
    model: NonlinearBTZ, r_plus: float, Q: float, order: int
) -> FloatArray:
    """Coefficients ``a_1 ... a_order`` of ``Δr_+(E)`` for neutral emission, Eq. (2.16).

    Parameters
    ----------
    model
        The black-hole family.
    r_plus
        Initial (simple) event-horizon radius.
    Q
        Charge, held fixed during neutral emission.
    order
        Number of coefficients.

    Returns
    -------
    FloatArray
        Shifted coefficients: ``Δr_+ = sum_n a[n-1] E**n``.

    Raises
    ------
    ValueError
        If ``r_plus`` is not a simple horizon (``∂_r m = 0`` there).
    """
    # h(y) = m(r_+) - m(r_+ + y) has shifted coefficients -∂_r^j m(r_+) / j!  (Eq. (A.1)).
    h = np.array([-model.taylor_coefficient(r_plus, Q, j) for j in range(1, order + 1)])
    if h[0] == 0:
        msg = "r_plus is a degenerate horizon; use the Puiseux expansion"
        raise ValueError(msg)
    return lagrange_burmann_inverse_numeric(h, order)


def neutral_entropy_series(model: NonlinearBTZ, r_plus: float, Q: float, order: int) -> FloatArray:
    """Coefficients ``b_n`` of the neutral exponent ``ΔS(E) = sum_n b_n E^n``, Eq. (4.16).

    Parameters
    ----------
    model
        The black-hole family.
    r_plus
        Initial (simple) event-horizon radius.
    Q
        Charge, held fixed.
    order
        Number of coefficients.

    Returns
    -------
    FloatArray
        Shifted coefficients of ``ΔS``; ``b_1 = -1/T_H`` (the Boltzmann factor).
    """
    return model.entropy(1.0) * horizon_displacement_series(model, r_plus, Q, order)
