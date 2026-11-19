"""Puiseux inversion about a horizon of multiplicity ``m`` (paper Sec. 3, App. C.1).

Problem
-------
At a degenerate horizon the linear coefficient of the mass–radius relation vanishes,

    E = h(y) = h_m y^m + h_{m+1} y^{m+1} + ...,        h_m != 0,

so ``y(E)`` is no longer a power series in ``E``. Its ``m`` local solutions are
power series in ``E**(1/m)`` (Puiseux series), Eq. (C.5).

Method
------
Following App. C.1, write ``h(y) = y^m g(y)`` with ``g(0) = h_m`` and introduce

    u = y g(y)^(1/m),        u^m = E.                                     (C.4)

Since ``du/dy = h_m^(1/m) != 0``, the ordinary Lagrange–Bürmann formula inverts
``u(y)``:

    y = sum_n b_n u^n,     b_n = (1/n) [y^{n-1}] g(y)^(-n/m).

Only the *constant* ``g(0)^(-1/m) = b_1`` requires a choice of branch; the rest of
``g^(-n/m) = b_1^n (1 + t)^(-n/m)`` with ``t = g/h_m - 1`` is a terminating binomial
series. The ``m`` physical branches then follow from ``u = ζ_j E^(1/m)`` with
``ζ_j = exp(2πij/m)``, i.e. the branch-``j`` coefficients are ``ζ_j^n b_n``.

Layout
------
Inputs and outputs use the *shifted* layout of
:mod:`blackholes.series.lagrange_burmann`: ``h[0]`` multiplies ``y``, and the
returned ``b[0]`` multiplies ``u = E**(1/m)``.
"""

from collections.abc import Sequence

import sympy as sp

from blackholes.series._dense import Scalar, as_exprs, binomial_power, pad

__all__ = ["branch_coefficients", "puiseux_inverse"]


def _validate_degenerate_series(h: Sequence[sp.Expr], multiplicity: int) -> sp.Expr:
    """Check that ``h`` has an exact zero of the stated multiplicity at ``y = 0``.

    Parameters
    ----------
    h
        Shifted coefficients of ``h`` (``h[0]`` multiplies ``y``).
    multiplicity
        The claimed multiplicity ``m >= 1``.

    Returns
    -------
    sp.Expr
        The leading coefficient ``h_m``.

    Raises
    ------
    ValueError
        If ``multiplicity < 1``, if a lower coefficient ``h_1 ... h_{m-1}`` is
        provably non-zero, or if ``h_m`` is missing or provably zero.
    """
    if multiplicity < 1:
        msg = f"multiplicity must be >= 1, got {multiplicity}"
        raise ValueError(msg)
    padded = pad(h, multiplicity)
    for index, coefficient in enumerate(padded[: multiplicity - 1], start=1):
        if coefficient.is_zero is not True:
            msg = f"h_{index} = {coefficient} must vanish for a zero of multiplicity {multiplicity}"
            raise ValueError(msg)
    leading = padded[multiplicity - 1]
    if leading.is_zero is True:
        msg = f"leading coefficient h_{multiplicity} vanishes: the multiplicity is higher"
        raise ValueError(msg)
    return leading


def puiseux_inverse(
    h: Sequence[Scalar],
    multiplicity: int,
    order: int,
    *,
    leading: Scalar | None = None,
    simplify: bool = True,
) -> list[sp.Expr]:
    """Invert ``E = h(y)`` near an ``m``-fold zero as ``y = sum_n b_n u^n``, ``u^m = E``.

    Implements Eq. (C.5) of the paper for the branch selected by ``leading``.

    Parameters
    ----------
    h
        Shifted coefficients of ``h``: ``h[0]`` multiplies ``y``. The first
        ``multiplicity - 1`` entries must be exactly zero.
    multiplicity
        Multiplicity ``m >= 1`` of the zero (``m = 1`` reduces to Lagrange–Bürmann).
    order
        Number of coefficients ``b_1 ... b_order`` to return (``>= 1``).
    leading
        The chosen value of ``b_1``, which must satisfy ``b_1**m * h_m = 1``.
        Defaults to SymPy's principal root ``(1/h_m)**(1/m)``. Pass a real root
        explicitly when a real branch is wanted for negative ``1/h_m`` and odd ``m``
        (as for the triple zero of App. C.2, Eq. (C.9)).
    simplify
        If true, factor each coefficient into compact rational form.

    Returns
    -------
    list[sp.Expr]
        ``[b_1, ..., b_order]``, coefficients of ``u, u**2, ...``.

    Raises
    ------
    ValueError
        If ``order < 1``, if ``h`` does not have a zero of exactly this
        multiplicity, or if ``leading`` is not an ``m``-th root of ``1/h_m``.

    Examples
    --------
    A double zero ``E = y**2 - y**3`` (positive branch):

    >>> puiseux_inverse([0, 1, -1], multiplicity=2, order=3)
    [1, 1/2, 5/8]
    """
    if order < 1:
        msg = f"order must be >= 1, got {order}"
        raise ValueError(msg)
    h_exprs = as_exprs(h)
    h_m = _validate_degenerate_series(h_exprs, multiplicity)
    b_1 = sp.root(1 / h_m, multiplicity) if leading is None else sp.sympify(leading)
    if sp.simplify(b_1**multiplicity * h_m - 1) != 0:
        msg = f"leading = {b_1} is not a {multiplicity}-th root of 1/h_{multiplicity}"
        raise ValueError(msg)

    # g(y) = h(y) / y^m in dense layout, and t = g / h_m - 1 (so t(0) = 0).
    g_dense = pad(h_exprs[multiplicity - 1 :], order)
    t_dense = [sp.Integer(0), *(c / h_m for c in g_dense[1:])]

    coefficients: list[sp.Expr] = []
    for n in range(1, order + 1):
        # b_n = (1/n) [y^(n-1)] g^(-n/m) = (b_1^n / n) [y^(n-1)] (1 + t)^(-n/m).
        expansion = binomial_power(t_dense, sp.Rational(-n, multiplicity), n)
        b_n = b_1**n * expansion[n - 1] / n
        coefficients.append(sp.factor(b_n) if simplify else sp.expand(b_n))
    return coefficients


def branch_coefficients(
    coefficients: Sequence[Scalar], multiplicity: int, branch: int
) -> list[sp.Expr]:
    """Coefficients of branch ``j``: substitute ``u = ζ_j E^(1/m)``, Eq. (C.5).

    Parameters
    ----------
    coefficients
        Output ``[b_1, ...]`` of :func:`puiseux_inverse`.
    multiplicity
        Multiplicity ``m`` used to compute them.
    branch
        Branch index ``j`` (taken modulo ``m``); ``j = 0`` returns the input.

    Returns
    -------
    list[sp.Expr]
        ``[ζ_j b_1, ζ_j**2 b_2, ...]`` so that ``y = sum_n c_n E**(n/m)``.

    Raises
    ------
    ValueError
        If ``multiplicity < 1``.

    Examples
    --------
    For a double zero, branch 1 flips the sign of odd orders (``a^(-)_n = (-1)^n a^(+)_n``):

    >>> branch_coefficients([1, 2, 3], multiplicity=2, branch=1)
    [-1, 2, -3]
    """
    if multiplicity < 1:
        msg = f"multiplicity must be >= 1, got {multiplicity}"
        raise ValueError(msg)
    zeta = sp.exp(2 * sp.pi * sp.I * sp.Rational(branch % multiplicity, multiplicity))
    return [sp.expand_complex(zeta**n) * c for n, c in enumerate(as_exprs(coefficients), start=1)]
