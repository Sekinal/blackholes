"""Lagrange–Bürmann inversion of formal power series (paper Sec. 2.2, Appendix A).

Problem
-------
Given a series without constant term and with a non-vanishing linear coefficient,

    E = h(y) = h_1 y + h_2 y^2 + h_3 y^3 + ...,        h_1 != 0,

find the unique inverse series y(E) = a_1 E + a_2 E^2 + ... with h(y(E)) = E.
In the paper, ``y`` is the horizon displacement Δr_+ and ``E`` the emitted energy
(Eq. (2.14)); the coefficients a_n are those of Eq. (2.16).

Method
------
The Lagrange–Bürmann formula, Eq. (2.16),

    a_n = (1/n!) d^{n-1}/dy^{n-1} [ (y / h(y))^n ]  at y = 0
        = (1/n) [y^{n-1}] (y / h(y))^n,

is evaluated by coefficient extraction on truncated power series. We never
differentiate symbolically: ``y / h(y)`` is the reciprocal of
``g(y) = h(y) / y = h_1 + h_2 y + ...``, which is computed by the standard
recursion for series reciprocals, then raised to the n-th power by truncated
multiplication. All arithmetic is exact (SymPy), so the output can be compared
symbol-for-symbol with closed forms such as Eq. (A.5).

Indexing conventions
--------------------
Two coefficient layouts appear, and mixing them up is the classic bug in this kind
of code, so they are kept strictly apart:

* **Dense** lists (private module :mod:`blackholes.series._dense`) store the
  coefficient of ``y**k`` at index ``k``, starting from the constant term.
* **Shifted** sequences (public API) describe series *without* constant term and
  store the coefficient of ``y**(k + 1)`` at index ``k``. So ``h[0]`` is h_1.
"""

from collections.abc import Sequence

import sympy as sp

from blackholes.series._dense import Scalar, as_exprs, multiply, pad, power, reciprocal

__all__ = ["compose", "lagrange_burmann_inverse"]


def lagrange_burmann_inverse(
    h: Sequence[Scalar], order: int, *, simplify: bool = True
) -> list[sp.Expr]:
    """Invert ``E = h(y)`` to ``y = a_1 E + ... + a_order E**order`` (Eq. (2.16)).

    Parameters
    ----------
    h
        Shifted coefficients of ``h``: ``h[0]`` multiplies ``y``, ``h[1]``
        multiplies ``y**2``, and so on. Entries beyond ``order`` are ignored
        because they cannot affect ``a_1 ... a_order``; missing entries are zero.
    order
        Number of inverse coefficients to return (``>= 1``).
    simplify
        If true, pass each coefficient through ``sympy.factor`` so symbolic
        results take the compact rational form used in the paper. Disable for
        speed when only numerical values are needed.

    Returns
    -------
    list[sp.Expr]
        ``[a_1, ..., a_order]``, the shifted coefficients of the inverse series.

    Raises
    ------
    ValueError
        If ``order < 1`` or the linear coefficient ``h[0]`` is zero (the inverse
        is then not a power series; see the Puiseux construction of Appendix C).

    Examples
    --------
    With ``h(y) = -m_1 y - m_2 y^2/2`` this reproduces Eq. (A.5):

    >>> import sympy as sp
    >>> m1, m2 = sp.symbols("m1 m2")
    >>> lagrange_burmann_inverse([-m1, -m2 / 2], order=2)
    [-1/m1, -m2/(2*m1**3)]
    """
    if order < 1:
        msg = f"order must be >= 1, got {order}"
        raise ValueError(msg)
    h_exprs = as_exprs(h)
    # g(y) = h(y) / y in dense layout: dense index k <-> shifted index k.
    g_dense = pad(h_exprs, order)
    if g_dense[0].is_zero is True:
        msg = "the linear coefficient h[0] vanishes: use a Puiseux expansion instead"
        raise ValueError(msg)
    phi = reciprocal(g_dense, order)  # phi(y) = y / h(y)
    coefficients: list[sp.Expr] = []
    for n in range(1, order + 1):
        # a_n = (1/n) [y^(n-1)] phi(y)^n; only n terms of phi^n are needed.
        phi_to_n = power(phi, n, n)
        a_n = phi_to_n[n - 1] / n
        coefficients.append(sp.factor(a_n) if simplify else sp.expand(a_n))
    return coefficients


def compose(outer: Sequence[Scalar], inner: Sequence[Scalar], order: int) -> list[sp.Expr]:
    """Compose two series without constant term: ``outer(inner(E))``.

    Used to verify inversions (``h(y(E)) = E``) and, later, to build entropy
    series ``ΔS(E) = S(r_+ + Δr_+(E)) - S(r_+)`` as in Eq. (2.17).

    Parameters
    ----------
    outer
        Shifted coefficients of the outer series (``outer[0]`` multiplies ``y``).
    inner
        Shifted coefficients of the inner series (``inner[0]`` multiplies ``E``).
    order
        Number of coefficients of the composition to return (``>= 1``).

    Returns
    -------
    list[sp.Expr]
        Shifted coefficients ``[c_1, ..., c_order]`` of ``outer(inner(E))``.

    Raises
    ------
    ValueError
        If ``order < 1``.
    """
    if order < 1:
        msg = f"order must be >= 1, got {order}"
        raise ValueError(msg)
    length = order + 1  # dense layout includes the zero constant term
    outer_exprs = pad(as_exprs(outer), order)
    inner_dense = [sp.Integer(0), *as_exprs(inner)]
    total = pad([], length)
    inner_power = pad([sp.Integer(1)], length)
    for j in range(1, order + 1):
        inner_power = multiply(inner_power, inner_dense, length)
        total = [t + outer_exprs[j - 1] * p for t, p in zip(total, inner_power, strict=True)]
    return [sp.expand(c) for c in total[1:]]
