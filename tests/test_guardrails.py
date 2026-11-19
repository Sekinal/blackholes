"""Meta-tests that confirm the project's safety nets are actually switched on."""

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from blackholes.series import lagrange_burmann_inverse


def test_runtime_type_checking_is_active() -> None:
    """Annotation violations raise at call time (beartype claw from conftest)."""
    with pytest.raises(BeartypeCallHintParamViolation):
        lagrange_burmann_inverse([1], order="3")  # type: ignore[arg-type]


def test_floats_are_rejected() -> None:
    """Floats would silently break exact arithmetic, so the API refuses them."""
    with pytest.raises(BeartypeCallHintParamViolation):
        lagrange_burmann_inverse([0.5], order=1)
