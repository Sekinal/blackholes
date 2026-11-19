"""Radius of convergence and singularity type from series coefficients (Domb–Sykes).

The paper notes that the Maclaurin series of the KPW exponent "need not converge
throughout the emission interval" (Sec. 2.2) but does not determine where it
converges. If the nearest singularity of ``f(E) = sum_n c_n E^n`` is an algebraic
branch point on the positive real axis,

    f(E) ~ A (1 - E/R)^α,

the coefficients behave as ``c_n ~ n^(-α-1) R^(-n)`` and their ratios satisfy

    c_n / c_{n-1} = (1/R) [1 - (α + 1)/n + O(1/n^2)].

A straight-line fit of the ratio against ``1/n`` (the Domb–Sykes plot) therefore
gives ``1/R`` as the intercept and ``α`` from the slope. For the nonlinear BTZ
exponent this yields ``R = E_c`` and ``α = 1/2``: the series converges on the whole
emission interval, but the ``n^(-3/2)`` decay makes the truncation error at the
endpoint fall only like ``N^(-1/2)`` (see :func:`endpoint_truncation_errors`).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from blackholes.series.numeric import FloatArray, evaluate_series

__all__ = ["DombSykesFit", "domb_sykes_fit", "endpoint_truncation_errors"]


@dataclass(frozen=True, slots=True)
class DombSykesFit:
    """Result of a Domb–Sykes ratio fit.

    Attributes
    ----------
    radius
        Estimated radius of convergence ``R``.
    exponent
        Estimated singularity exponent ``α`` in ``(1 - E/R)^α``.
    orders
        Orders ``n`` used in the fit.
    ratios
        Coefficient ratios ``c_n / c_{n-1}`` at those orders.
    """

    radius: float
    exponent: float
    orders: FloatArray
    ratios: FloatArray


def domb_sykes_fit(
    coefficients: npt.ArrayLike, first_order: int, *, quadratic: bool = True
) -> DombSykesFit:
    """Fit ``c_n/c_{n-1}`` against ``1/n`` for ``n >= first_order``.

    Parameters
    ----------
    coefficients
        Shifted coefficients ``[c_1, c_2, ...]`` with a single dominant real
        singularity (ratios must not oscillate in sign).
    first_order
        Smallest ``n`` included in the fit (drop the pre-asymptotic orders).
    quadratic
        Include an ``O(1/n^2)`` term in the fit, which removes the leading
        correction and sharpens both estimates considerably.

    Returns
    -------
    DombSykesFit
        Radius, exponent and the data used.

    Raises
    ------
    ValueError
        If fewer than four orders are available or a coefficient vanishes.
    """
    c = np.asarray(coefficients, dtype=np.float64)
    n = np.arange(1, c.size + 1, dtype=np.float64)
    mask = n >= max(first_order, 2)
    min_points = 4
    if int(mask.sum()) < min_points:
        msg = "need at least four orders beyond first_order for a stable fit"
        raise ValueError(msg)
    if np.any(c[:-1] == 0):
        msg = "coefficients must be non-zero for ratio analysis"
        raise ValueError(msg)
    ratios = c[1:] / c[:-1]  # ratio at order n = 2, 3, ...
    orders = n[1:]
    keep = orders >= max(first_order, 2)
    x = 1.0 / orders[keep]
    degree = 2 if quadratic else 1
    poly = np.polynomial.Polynomial.fit(x, ratios[keep], degree).convert()
    intercept, slope = float(poly.coef[0]), float(poly.coef[1])
    radius = 1.0 / intercept
    exponent = -slope * radius - 1.0
    return DombSykesFit(radius, exponent, orders[keep], ratios[keep])


def endpoint_truncation_errors(
    coefficients: npt.ArrayLike, endpoint: float, exact: float, orders: npt.ArrayLike
) -> FloatArray:
    """Relative error of ``N``-term truncations evaluated at ``E = endpoint``.

    Parameters
    ----------
    coefficients
        Shifted series coefficients (at least ``max(orders)`` of them).
    endpoint
        Evaluation point (``E_c`` for the KPW exponent).
    exact
        The exact function value at ``endpoint``.
    orders
        Truncation orders ``N``.

    Returns
    -------
    FloatArray
        ``|S_N(endpoint) - exact| / |exact|`` for each ``N``.
    """
    c = np.asarray(coefficients, dtype=np.float64)
    return np.array(
        [
            abs(evaluate_series(c[: int(N)], endpoint) - exact) / abs(exact)
            for N in np.asarray(orders)
        ],
        dtype=np.float64,
    )
