"""Tests for the Lagrange–Bürmann inversion against the paper's closed forms."""

from fractions import Fraction

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from blackholes.series import compose, lagrange_burmann_inverse

m1, m2, m3, m4 = sp.symbols("m1:5", nonzero=True)
c, f1, f2 = sp.symbols("c f1 f2", nonzero=True)

# Small rationals keep exact arithmetic fast while still exercising signs and
# non-trivial denominators.
rationals = st.fractions(min_value=-5, max_value=5, max_denominator=7)
nonzero_rationals = rationals.filter(lambda value: value != 0)


def _mass_function_h(*derivatives: sp.Expr) -> list[sp.Expr]:
    """Shifted coefficients of h(y) = m(r_+) - m(r_+ + y), Eq. (2.14).

    Parameters
    ----------
    *derivatives
        Radial derivatives ``∂_r^j m(r_+)`` for ``j = 1, 2, ...``.

    Returns
    -------
    list[sp.Expr]
        ``[-m_1/1!, -m_2/2!, ...]`` as in Eq. (A.1).
    """
    return [-d / sp.factorial(j) for j, d in enumerate(derivatives, start=1)]


def test_reproduces_appendix_a_closed_forms() -> None:
    """The first four coefficients equal Eq. (A.5) of the paper."""
    a1, a2, a3, a4 = lagrange_burmann_inverse(_mass_function_h(m1, m2, m3, m4), order=4)
    expected = [
        -1 / m1,
        -m2 / (2 * m1**3),
        (m1 * m3 - 3 * m2**2) / (6 * m1**5),
        (10 * m1 * m2 * m3 - m1**2 * m4 - 15 * m2**3) / (24 * m1**7),
    ]
    for got, want in zip([a1, a2, a3, a4], expected, strict=True):
        assert sp.simplify(got - want) == 0


def test_affine_specialization_matches_eq_2_21() -> None:
    """For f = c (m - M), h(y) = -f_0(r_+ + y)/c gives Eq. (2.21)."""
    a1, a2 = lagrange_burmann_inverse([-f1 / c, -f2 / (2 * c)], order=2)
    assert sp.simplify(a1 - (-c / f1)) == 0
    assert sp.simplify(a2 - (-(c**2) * f2 / (2 * f1**3))) == 0


@given(
    linear=nonzero_rationals,
    higher=st.lists(rationals, max_size=5),
    order=st.integers(min_value=1, max_value=6),
)
def test_inverse_round_trip(linear: Fraction, higher: list[Fraction], order: int) -> None:
    """Composing h with its computed inverse gives the identity series E."""
    h = [linear, *higher]
    inverse = lagrange_burmann_inverse(h, order=order, simplify=False)
    identity = [sp.Integer(1)] + [sp.Integer(0)] * (order - 1)
    assert compose(h, inverse, order=order) == identity


@given(
    linear=nonzero_rationals,
    higher=st.lists(rationals, max_size=4),
)
def test_inverse_is_two_sided(linear: Fraction, higher: list[Fraction]) -> None:
    """The inverse also undoes h from the other side: y(h(y)) = y."""
    order = 5
    h = [linear, *higher]
    inverse = lagrange_burmann_inverse(h, order=order, simplify=False)
    identity = [sp.Integer(1)] + [sp.Integer(0)] * (order - 1)
    assert compose(inverse, h, order=order) == identity


def test_rejects_vanishing_linear_coefficient() -> None:
    """A degenerate root (h_1 = 0) has no power-series inverse."""
    with pytest.raises(ValueError, match="Puiseux"):
        lagrange_burmann_inverse([0, 1], order=2)


@pytest.mark.parametrize("order", [0, -3])
def test_rejects_nonpositive_order(order: int) -> None:
    """Truncation orders below one are meaningless and must be rejected.

    Parameters
    ----------
    order
        An invalid truncation order.
    """
    with pytest.raises(ValueError, match="order"):
        lagrange_burmann_inverse([1], order=order)
    with pytest.raises(ValueError, match="order"):
        compose([1], [1], order=order)


def test_missing_and_extra_coefficients_are_handled() -> None:
    """Short inputs are zero-padded; extra inputs beyond ``order`` are ignored."""
    assert lagrange_burmann_inverse([2], order=3) == [Fraction(1, 2), 0, 0]
    assert lagrange_burmann_inverse([2, 0, 0, 99], order=2) == [Fraction(1, 2), 0]
