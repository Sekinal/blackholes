"""Tests for the Puiseux inversion against App. C of the paper."""

from fractions import Fraction

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from blackholes.series import (
    branch_coefficients,
    compose,
    lagrange_burmann_inverse,
    puiseux_inverse,
)

c, f2, f3, f4, f5 = sp.symbols("c f2 f3 f4 f5", nonzero=True)
rationals = st.fractions(min_value=-4, max_value=4, max_denominator=5)
nonzero_rationals = rationals.filter(lambda value: value != 0)


def _affine_h(*derivatives: sp.Expr) -> list[sp.Expr]:
    """Shifted coefficients of ``h(y) = -f_0(r_d + y) / c`` for the affine family.

    Parameters
    ----------
    *derivatives
        ``∂_r^j f_0(r_d)`` for ``j = 1, 2, ...`` (the first ones vanish at a
        degenerate zero and are passed as ``0``).

    Returns
    -------
    list[sp.Expr]
        ``[-f_1/(1! c), -f_2/(2! c), ...]``.
    """
    return [-d / (sp.factorial(j) * c) for j, d in enumerate(derivatives, start=1)]


def test_double_zero_matches_eqs_3_5_and_c_8() -> None:
    """Double zero: b_1 = sqrt(-2c/f2), Eq. (3.5) for b_2 and Eq. (C.8) for b_3."""
    b1 = sp.sqrt(-2 * c / f2)
    b = puiseux_inverse(_affine_h(0, f2, f3, f4), multiplicity=2, order=3, leading=b1)
    assert sp.simplify(b[0] - b1) == 0
    assert sp.simplify(b[1] - c * f3 / (3 * f2**2)) == 0
    expected_b3 = c * b1 / (36 * f2**3) * (3 * f2 * f4 - 5 * f3**2)
    assert sp.simplify(b[2] - expected_b3) == 0


def test_triple_zero_matches_eqs_c_9_and_c_10() -> None:
    """Triple zero with the real cube root of Eq. (C.9) reproduces Eq. (C.10)."""
    a0 = sp.cbrt(-6 * c / f3)
    b = puiseux_inverse(_affine_h(0, 0, f3, f4, f5), multiplicity=3, order=3, leading=a0)
    assert sp.simplify(b[1] - (-(a0**2) * f4 / (12 * f3))) == 0
    expected_b3 = a0**3 * (f4**2 / (48 * f3**2) - f5 / (60 * f3))
    assert sp.simplify(b[2] - expected_b3) == 0


def test_leading_coefficient_matches_eq_c_6() -> None:
    """(b_1)^m = -c m! / ∂^m f_0 for every multiplicity, Eq. (C.6)."""
    fm = sp.Symbol("fm", nonzero=True)
    for m in range(1, 5):
        h = _affine_h(*([0] * (m - 1)), fm)
        (b1,) = puiseux_inverse(h, multiplicity=m, order=1)
        assert sp.simplify(b1**m - (-c * sp.factorial(m) / fm)) == 0


@given(
    leading=nonzero_rationals,
    multiplicity=st.integers(min_value=1, max_value=4),
    higher=st.lists(rationals, max_size=4),
    order=st.integers(min_value=1, max_value=6),
)
def test_round_trip_gives_u_to_the_m(
    leading: Fraction, multiplicity: int, higher: list[Fraction], order: int
) -> None:
    """``h(y(u)) = u^m`` exactly through the requested order."""
    h_m = 1 / leading**multiplicity
    h = [Fraction(0)] * (multiplicity - 1) + [h_m, *higher]
    b = puiseux_inverse(h, multiplicity=multiplicity, order=order, leading=leading)
    expected = [sp.Integer(1 if k == multiplicity else 0) for k in range(1, order + 1)]
    assert compose(h, b, order=order) == expected


@given(linear=nonzero_rationals, higher=st.lists(rationals, max_size=4))
def test_multiplicity_one_reduces_to_lagrange_burmann(
    linear: Fraction, higher: list[Fraction]
) -> None:
    """A simple zero is the ordinary inverse of Sec. 2.2."""
    h = [linear, *higher]
    assert puiseux_inverse(h, multiplicity=1, order=5) == lagrange_burmann_inverse(h, order=5)


def test_double_zero_branches_alternate_sign() -> None:
    """a^(-)_n = (-1)^n a^(+)_n, stated below Eq. (C.8)."""
    b = puiseux_inverse([0, 1, -1, 2], multiplicity=2, order=4)
    minus = branch_coefficients(b, multiplicity=2, branch=1)
    assert minus == [(-1) ** n * coefficient for n, coefficient in enumerate(b, start=1)]


def test_triple_zero_branches_are_complex_conjugates() -> None:
    """Branches j = 1, 2 of a real triple zero are conjugate (App. C.2)."""
    b = puiseux_inverse([0, 0, -1, 1], multiplicity=3, order=3, leading=-1)
    first = branch_coefficients(b, multiplicity=3, branch=1)
    second = branch_coefficients(b, multiplicity=3, branch=2)
    for x, y in zip(first, second, strict=True):
        assert sp.simplify(sp.conjugate(x) - y) == 0
    assert branch_coefficients(b, multiplicity=3, branch=3) == branch_coefficients(
        b, multiplicity=3, branch=0
    )


@pytest.mark.parametrize(
    ("h", "multiplicity", "message"),
    [
        ([1, 1], 2, "must vanish"),
        ([0, 0, 1], 2, "multiplicity is higher"),
        ([0, 1], 0, "multiplicity must be"),
    ],
)
def test_rejects_wrong_multiplicity(h: list[int], multiplicity: int, message: str) -> None:
    """Inputs that do not have a zero of the stated multiplicity are refused.

    Parameters
    ----------
    h
        Shifted coefficients of the series.
    multiplicity
        The (wrong) claimed multiplicity.
    message
        Expected fragment of the error message.
    """
    with pytest.raises(ValueError, match=message):
        puiseux_inverse(h, multiplicity=multiplicity, order=2)


def test_rejects_wrong_leading_root_and_order() -> None:
    """``leading`` must be an m-th root of 1/h_m, and ``order`` must be positive."""
    with pytest.raises(ValueError, match="root"):
        puiseux_inverse([0, 4], multiplicity=2, order=1, leading=1)
    with pytest.raises(ValueError, match="order"):
        puiseux_inverse([0, 4], multiplicity=2, order=0)
    with pytest.raises(ValueError, match="multiplicity"):
        branch_coefficients([1], multiplicity=0, branch=0)
