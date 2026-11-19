"""Float64 series routines must agree with the exact SymPy implementations."""

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from blackholes.series import compose, lagrange_burmann_inverse, puiseux_inverse
from blackholes.series.numeric import (
    compose_numeric,
    evaluate_series,
    lagrange_burmann_inverse_numeric,
    puiseux_inverse_numeric,
)

rationals = st.fractions(min_value=-3, max_value=3, max_denominator=4)
positive = st.fractions(min_value=Fraction(1, 4), max_value=3, max_denominator=4)


def _floats(values: list[object]) -> list[float]:
    """Convert exact SymPy/Fraction values to floats for comparison.

    Parameters
    ----------
    values
        Exact numbers.

    Returns
    -------
    list[float]
        Their float64 values.
    """
    return [float(v) for v in values]  # type: ignore[arg-type]


@given(linear=positive, sign=st.sampled_from([-1, 1]), higher=st.lists(rationals, max_size=5))
def test_lagrange_burmann_matches_exact(
    linear: Fraction, sign: int, higher: list[Fraction]
) -> None:
    """Numeric and exact Lagrange–Bürmann coefficients agree to round-off."""
    h = [sign * linear, *higher]
    exact = _floats(lagrange_burmann_inverse(h, order=6, simplify=False))
    numeric = lagrange_burmann_inverse_numeric(_floats(list(h)), order=6)
    np.testing.assert_allclose(numeric, exact, rtol=1e-11, atol=1e-11)


@given(
    leading=positive,
    sign=st.sampled_from([-1, 1]),
    multiplicity=st.integers(min_value=1, max_value=3),
    higher=st.lists(rationals, max_size=4),
)
def test_puiseux_matches_exact(
    leading: Fraction, sign: int, multiplicity: int, higher: list[Fraction]
) -> None:
    """Numeric and exact Puiseux coefficients agree for real branches."""
    b1 = sign * leading
    h = [Fraction(0)] * (multiplicity - 1) + [1 / b1**multiplicity, *higher]
    exact = _floats(puiseux_inverse(h, multiplicity, order=5, leading=b1, simplify=False))
    numeric = puiseux_inverse_numeric(_floats(list(h)), multiplicity, order=5, leading=float(b1))
    np.testing.assert_allclose(numeric, exact, rtol=1e-11, atol=1e-11)


@given(outer=st.lists(rationals, min_size=1, max_size=5), inner=st.lists(rationals, max_size=5))
def test_compose_matches_exact(outer: list[Fraction], inner: list[Fraction]) -> None:
    """Horner composition agrees with the exact composition."""
    exact = _floats(compose(outer, inner, order=5))
    numeric = compose_numeric(_floats(list(outer)), _floats(list(inner)), order=5)
    np.testing.assert_allclose(numeric, exact, rtol=1e-12, atol=1e-12)


def test_default_real_branches() -> None:
    """Default ``b_1`` is the positive root, or the negative real root for odd m."""
    assert puiseux_inverse_numeric([0.0, 4.0], 2, order=1)[0] == pytest.approx(0.5)
    assert puiseux_inverse_numeric([0.0, 0.0, -8.0], 3, order=1)[0] == pytest.approx(-0.5)


def test_evaluate_series() -> None:
    """Horner evaluation of ``x + 2x^2 + 3x^3`` at ``x = 2`` gives 34."""
    assert evaluate_series([1.0, 2.0, 3.0], 2.0) == pytest.approx(34.0)


@pytest.mark.parametrize(
    ("h", "multiplicity", "kwargs", "message"),
    [
        ([0.0, -4.0], 2, {}, "no real branch"),
        ([1.0, 1.0], 2, {}, "must vanish"),
        ([0.0, 0.0], 2, {}, "vanishes"),
        ([0.0, 4.0], 2, {"leading": 0.3}, "not a real"),
        ([1.0], 0, {}, ">= 1"),
        ([[1.0]], 1, {}, "one-dimensional"),
        ([float("nan")], 1, {}, "finite"),
    ],
)
def test_input_validation(
    h: list[float], multiplicity: int, kwargs: dict[str, float], message: str
) -> None:
    """Invalid inputs raise informative errors.

    Parameters
    ----------
    h
        Shifted coefficients.
    multiplicity
        Claimed multiplicity.
    kwargs
        Extra keyword arguments (e.g. a bad ``leading``).
    message
        Expected error fragment.
    """
    with pytest.raises(ValueError, match=message):
        puiseux_inverse_numeric(h, multiplicity, order=2, **kwargs)


def test_compose_and_power_guards() -> None:
    """Order and positivity guards fire."""
    from blackholes.series.numeric import _series_power  # noqa: PLC0415

    with pytest.raises(ValueError, match="order"):
        compose_numeric([1.0], [1.0], order=0)
    with pytest.raises(ValueError, match="positive constant"):
        _series_power(np.array([-1.0, 1.0]), 0.5)
