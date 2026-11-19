"""Exact truncated power-series arithmetic on *dense* coefficient lists.

A dense list ``a`` represents ``a[0] + a[1] y + a[2] y**2 + ...`` known only modulo
``y**length``. Every function takes the truncation ``length`` explicitly and returns
a new list of exactly that many SymPy expressions, so truncation can never be
forgotten or applied inconsistently.

This module is private to :mod:`blackholes.series`; the public routines
(:mod:`~blackholes.series.lagrange_burmann`, :mod:`~blackholes.series.puiseux`)
translate between this layout and their user-facing "shifted" layout.
"""

from collections.abc import Sequence
from fractions import Fraction

import sympy as sp

__all__ = [
    "Scalar",
    "as_exprs",
    "binomial_power",
    "multiply",
    "pad",
    "power",
    "reciprocal",
]

type Scalar = sp.Expr | int | Fraction
"""Anything SymPy can turn into an exact expression (floats are excluded on purpose)."""


def as_exprs(coefficients: Sequence[Scalar]) -> list[sp.Expr]:
    """Convert user-supplied coefficients into exact SymPy expressions.

    Parameters
    ----------
    coefficients
        Integers, fractions or SymPy expressions.

    Returns
    -------
    list[sp.Expr]
        The same values as SymPy objects (``Fraction`` becomes ``Rational``).
    """
    return [sp.sympify(c) for c in coefficients]


def pad(coefficients: Sequence[sp.Expr], length: int) -> list[sp.Expr]:
    """Truncate or zero-pad a dense coefficient list to exactly ``length`` entries.

    Parameters
    ----------
    coefficients
        Dense coefficients (index ``k`` holds the ``y**k`` coefficient).
    length
        Number of coefficients to keep (the series is known modulo ``y**length``).

    Returns
    -------
    list[sp.Expr]
        A new list of exactly ``length`` entries.
    """
    padded = list(coefficients[:length])
    padded.extend([sp.Integer(0)] * (length - len(padded)))
    return padded


def multiply(a: Sequence[sp.Expr], b: Sequence[sp.Expr], length: int) -> list[sp.Expr]:
    """Multiply two dense series modulo ``y**length`` (Cauchy product).

    Parameters
    ----------
    a
        Dense coefficients of the first factor; missing entries count as zero.
    b
        Dense coefficients of the second factor; missing entries count as zero.
    length
        Truncation length of the result.

    Returns
    -------
    list[sp.Expr]
        Dense coefficients ``c_k = sum_{i=0}^{k} a_i b_{k-i}`` for ``k < length``.
    """
    a_full = pad(a, length)
    b_full = pad(b, length)
    return [
        sp.expand(sum((a_full[i] * b_full[k - i] for i in range(k + 1)), sp.Integer(0)))
        for k in range(length)
    ]


def reciprocal(a: Sequence[sp.Expr], length: int) -> list[sp.Expr]:
    """Compute ``1 / a(y)`` modulo ``y**length``.

    Uses the recursion obtained from ``a(y) b(y) = 1`` order by order:
    ``b_0 = 1 / a_0`` and ``b_k = -(1 / a_0) sum_{i=1}^{k} a_i b_{k-i}``.

    Parameters
    ----------
    a
        Dense coefficients with an invertible constant term ``a[0]``.
    length
        Truncation length of the result.

    Returns
    -------
    list[sp.Expr]
        Dense coefficients of the reciprocal series.

    Raises
    ------
    ValueError
        If the constant term is (provably) zero, so no reciprocal exists.
    """
    a_full = pad(a, length)
    if a_full[0].is_zero is True:
        msg = "series reciprocal requires a non-zero constant term"
        raise ValueError(msg)
    inverse_a0 = 1 / a_full[0]
    b: list[sp.Expr] = [inverse_a0]
    for k in range(1, length):
        tail = sum((a_full[i] * b[k - i] for i in range(1, k + 1)), sp.Integer(0))
        b.append(sp.expand(-inverse_a0 * tail))
    return b


def power(a: Sequence[sp.Expr], exponent: int, length: int) -> list[sp.Expr]:
    """Raise a dense series to a non-negative integer power modulo ``y**length``.

    Binary exponentiation keeps the number of truncated products at
    ``O(log exponent)``.

    Parameters
    ----------
    a
        Dense coefficients of the base series.
    exponent
        Non-negative integer power.
    length
        Truncation length of the result.

    Returns
    -------
    list[sp.Expr]
        Dense coefficients of ``a(y) ** exponent``.

    Raises
    ------
    ValueError
        If ``exponent`` is negative (use :func:`reciprocal` first).
    """
    if exponent < 0:
        msg = f"exponent must be non-negative, got {exponent}"
        raise ValueError(msg)
    result = pad([sp.Integer(1)], length)
    base = pad(a, length)
    remaining = exponent
    while remaining > 0:
        if remaining & 1:
            result = multiply(result, base, length)
        base = multiply(base, base, length)
        remaining >>= 1
    return result


def binomial_power(t: Sequence[sp.Expr], exponent: sp.Expr, length: int) -> list[sp.Expr]:
    """Compute ``(1 + t(y)) ** exponent`` for a series ``t`` with ``t(0) = 0``.

    The binomial series ``sum_k binom(exponent, k) t^k`` terminates modulo
    ``y**length`` because ``t^k`` starts at order ``y**k``. This is how fractional
    powers such as ``(h / y^m)^(-n/m)`` in the Puiseux construction (App. C.1) are
    expanded without choosing a branch for anything but the leading constant.

    Parameters
    ----------
    t
        Dense coefficients of ``t``; ``t[0]`` must vanish.
    exponent
        Any exact exponent (typically a rational such as ``-n/m``).
    length
        Truncation length of the result.

    Returns
    -------
    list[sp.Expr]
        Dense coefficients of ``(1 + t) ** exponent``.

    Raises
    ------
    ValueError
        If ``t`` has a non-zero constant term (the expansion would not terminate).
    """
    t_full = pad(t, length)
    if t_full[0].is_zero is not True:
        msg = "binomial_power requires t(0) = 0"
        raise ValueError(msg)
    total = pad([sp.Integer(1)], length)
    t_to_k = pad([sp.Integer(1)], length)
    for k in range(1, length):
        t_to_k = multiply(t_to_k, t_full, length)
        weight = sp.binomial(exponent, k)
        total = [s + weight * term for s, term in zip(total, t_to_k, strict=True)]
    return [sp.expand(c) for c in total]
