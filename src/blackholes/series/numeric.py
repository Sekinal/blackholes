"""Floating-point (float64) counterparts of the exact series inversions.

The exact SymPy routines are ideal for checking closed forms but too slow for the
70-term expansions used in the convergence analysis. The functions here implement
the same Lagrange–Bürmann / Puiseux formulas (Eqs. (2.16), (C.5)) on NumPy arrays.

Accuracy
--------
Powers of series are computed with J. C. P. Miller's recurrence: for
``b = a**p`` with ``a[0] != 0``,

    b_0 = a_0^p,    b_k = (1 / (k a_0)) sum_{j=1}^{k} ((p + 1) j - k) a_j b_{k-j}.

It needs no subtraction of large intermediate terms, and for the nonlinear BTZ
horizon series it agrees with 60-digit arithmetic to ~1e-14 relative error up to
order 70 (checked in the test suite against the exact implementation).

Layout
------
All public functions use the *shifted* layout: index ``k`` holds the coefficient of
``y**(k + 1)`` (inputs) or of ``E**(k + 1)`` / ``u**(k + 1)`` (outputs).
"""

import math

import numpy as np
import numpy.typing as npt

__all__ = [
    "FloatArray",
    "compose_numeric",
    "evaluate_series",
    "lagrange_burmann_inverse_numeric",
    "puiseux_inverse_numeric",
]

type FloatArray = npt.NDArray[np.float64]
"""One-dimensional float64 coefficient array."""


def _as_array(coefficients: npt.ArrayLike, length: int) -> FloatArray:
    """Convert coefficients to a float64 array truncated or zero-padded to ``length``.

    Parameters
    ----------
    coefficients
        Any one-dimensional array-like of real numbers.
    length
        Required length.

    Returns
    -------
    FloatArray
        A new contiguous float64 array of exactly ``length`` entries.

    Raises
    ------
    ValueError
        If the input is not one-dimensional or contains non-finite values.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 1:
        msg = f"coefficients must be one-dimensional, got shape {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "coefficients must be finite"
        raise ValueError(msg)
    out = np.zeros(length, dtype=np.float64)
    count = min(length, values.size)
    out[:count] = values[:count]
    return out


def _series_power(a: FloatArray, exponent: float) -> FloatArray:
    """Raise a dense series with ``a[0] > 0`` to a real power (Miller recurrence).

    Parameters
    ----------
    a
        Dense coefficients (index ``k`` multiplies ``y**k``); ``a[0]`` must be
        positive so that ``a[0]**exponent`` is real and unambiguous.
    exponent
        Real exponent ``p``.

    Returns
    -------
    FloatArray
        Dense coefficients of ``a(y)**p`` to the same length as ``a``.

    Raises
    ------
    ValueError
        If ``a[0]`` is not positive.
    """
    if not a[0] > 0:
        msg = f"series power needs a positive constant term, got {a[0]}"
        raise ValueError(msg)
    length = a.size
    b = np.zeros(length, dtype=np.float64)
    b[0] = a[0] ** exponent
    for k in range(1, length):
        j = np.arange(1, k + 1)
        weights = (exponent + 1.0) * j - k
        b[k] = np.dot(weights * a[1 : k + 1], b[k - j]) / (k * a[0])
    return b


def _real_leading_root(h_m: float, multiplicity: int) -> float:
    """Return the real ``m``-th root of ``1 / h_m`` used as the default ``b_1``.

    Parameters
    ----------
    h_m
        Non-zero leading coefficient.
    multiplicity
        Root order ``m``.

    Returns
    -------
    float
        The positive root if ``1/h_m > 0``; the negative real root if
        ``1/h_m < 0`` and ``m`` is odd.

    Raises
    ------
    ValueError
        If ``1/h_m < 0`` and ``m`` is even (no real branch exists).
    """
    target = 1.0 / h_m
    if target > 0:
        return float(target ** (1.0 / multiplicity))
    if multiplicity % 2 == 1:
        return -float((-target) ** (1.0 / multiplicity))
    msg = f"no real branch: 1/h_{multiplicity} = {target} < 0 with even multiplicity"
    raise ValueError(msg)


def puiseux_inverse_numeric(
    h: npt.ArrayLike,
    multiplicity: int,
    order: int,
    *,
    leading: float | None = None,
) -> FloatArray:
    """Float64 Puiseux inverse ``y = sum_n b_n u^n`` with ``u^m = E``, Eq. (C.5).

    Parameters
    ----------
    h
        Shifted coefficients of ``h`` (``h[0]`` multiplies ``y``); the first
        ``multiplicity - 1`` must be exactly zero.
    multiplicity
        Multiplicity ``m >= 1`` of the zero.
    order
        Number of coefficients to return (``>= 1``).
    leading
        Real branch choice for ``b_1`` (must satisfy ``b_1**m h_m = 1`` to
        relative accuracy ``1e-12``). Defaults to the real root, see
        :func:`_real_leading_root`.

    Returns
    -------
    FloatArray
        ``[b_1, ..., b_order]``.

    Raises
    ------
    ValueError
        If ``order`` or ``multiplicity`` is below one, the lower coefficients do
        not vanish, ``h_m`` is zero, or ``leading`` is not a valid root.
    """
    if order < 1 or multiplicity < 1:
        msg = f"order and multiplicity must be >= 1, got {order} and {multiplicity}"
        raise ValueError(msg)
    coefficients = _as_array(h, multiplicity - 1 + order)
    if np.any(coefficients[: multiplicity - 1] != 0.0):
        msg = f"h_1 ... h_{multiplicity - 1} must vanish exactly"
        raise ValueError(msg)
    h_m = float(coefficients[multiplicity - 1])
    if h_m == 0.0:
        msg = f"leading coefficient h_{multiplicity} vanishes"
        raise ValueError(msg)
    b_1 = _real_leading_root(h_m, multiplicity) if leading is None else leading
    if not math.isclose(b_1**multiplicity * h_m, 1.0, rel_tol=1e-12):
        msg = f"leading = {b_1} is not a real {multiplicity}-th root of 1/h_{multiplicity}"
        raise ValueError(msg)

    normalized = coefficients[multiplicity - 1 :] / h_m  # g(y)/h_m, constant term 1
    out = np.empty(order, dtype=np.float64)
    for n in range(1, order + 1):
        # Coefficient n is b_1**n / n times the y**(n-1) term of (g/h_m) to the power -n/m.
        expansion = _series_power(normalized[:n], -n / multiplicity)
        out[n - 1] = b_1**n * expansion[n - 1] / n
    return out


def lagrange_burmann_inverse_numeric(h: npt.ArrayLike, order: int) -> FloatArray:
    """Float64 Lagrange–Bürmann inverse of a simple zero, Eq. (2.16).

    Parameters
    ----------
    h
        Shifted coefficients of ``h`` with ``h[0] != 0``.
    order
        Number of inverse coefficients ``a_1 ... a_order`` to return.

    Returns
    -------
    FloatArray
        ``[a_1, ..., a_order]``.
    """
    return puiseux_inverse_numeric(h, multiplicity=1, order=order)


def compose_numeric(outer: npt.ArrayLike, inner: npt.ArrayLike, order: int) -> FloatArray:
    """Float64 composition ``outer(inner(E))`` of series without constant term.

    Evaluated by Horner's scheme in series arithmetic:
    ``outer(x) = x (o_1 + x (o_2 + x (o_3 + ...)))``.

    Parameters
    ----------
    outer
        Shifted coefficients of the outer series.
    inner
        Shifted coefficients of the inner series.
    order
        Number of output coefficients (``>= 1``).

    Returns
    -------
    FloatArray
        Shifted coefficients ``[c_1, ..., c_order]``.

    Raises
    ------
    ValueError
        If ``order < 1``.
    """
    if order < 1:
        msg = f"order must be >= 1, got {order}"
        raise ValueError(msg)
    o = _as_array(outer, order)
    x = np.concatenate(([0.0], _as_array(inner, order)))  # dense, length order + 1
    acc = np.zeros(order + 1, dtype=np.float64)
    for coefficient in o[::-1]:
        acc = np.convolve(acc, x)[: order + 1]
        acc[0] += coefficient
    result = np.convolve(acc, x)[: order + 1]
    return result[1:]


def evaluate_series(coefficients: npt.ArrayLike, x: float) -> float:
    """Evaluate a shifted series ``sum_k c_k x**(k + 1)`` by Horner's rule.

    Parameters
    ----------
    coefficients
        Shifted coefficients ``[c_1, c_2, ...]``.
    x
        Evaluation point.

    Returns
    -------
    float
        The truncated series value.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    total = 0.0
    for c in values[::-1]:
        total = (total + float(c)) * x
    return total
