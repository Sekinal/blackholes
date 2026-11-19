"""Tests for the private dense truncated-series arithmetic."""

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from blackholes.series._dense import binomial_power, multiply, pad, power, reciprocal

rationals = st.fractions(min_value=-4, max_value=4, max_denominator=5).map(sp.Rational)
series = st.lists(rationals, min_size=1, max_size=6)


@given(a=series.filter(lambda s: s[0] != 0))
def test_reciprocal_times_series_is_one(a: list[sp.Rational]) -> None:
    """``a * (1/a) = 1`` modulo ``y**length``."""
    length = 6
    product = multiply(a, reciprocal(a, length), length)
    assert product == pad([sp.Integer(1)], length)


@given(a=series, exponent=st.integers(min_value=0, max_value=5))
def test_power_matches_repeated_multiplication(a: list[sp.Rational], exponent: int) -> None:
    """Binary exponentiation agrees with naive repeated products."""
    length = 6
    naive = pad([sp.Integer(1)], length)
    for _ in range(exponent):
        naive = multiply(naive, a, length)
    assert power(a, exponent, length) == naive


@given(tail=st.lists(rationals, max_size=5), k=st.integers(min_value=1, max_value=3))
def test_binomial_power_inverts_integer_power(tail: list[sp.Rational], k: int) -> None:
    """``((1 + t)^(1/k))^k = 1 + t``: fractional powers are consistent."""
    length = 6
    t = [sp.Integer(0), *tail]
    root = binomial_power(t, sp.Rational(1, k), length)
    assert power(root, k, length) == pad([sp.Integer(1), *tail], length)


def test_guards() -> None:
    """Invalid inputs fail loudly instead of producing ``zoo`` or nonsense."""
    with pytest.raises(ValueError, match="non-zero constant term"):
        reciprocal([sp.Integer(0), sp.Integer(1)], 2)
    with pytest.raises(ValueError, match="non-negative"):
        power([sp.Integer(1)], -1, 2)
    with pytest.raises(ValueError, match=r"t\(0\) = 0"):
        binomial_power([sp.Integer(1)], sp.Rational(1, 2), 2)
