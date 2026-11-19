"""Emission channels and horizon histories against Secs. 4.2-4.3 of the paper."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from blackholes.emission import (
    ChargeProfile,
    EmissionPath,
    ExponentialDepletion,
    FractionalLoss,
    LinearProfile,
    NeutralProfile,
    ReturningProfile,
)
from blackholes.paper.configurations import (
    FIG2_CHARGE,
    FIG2_MODEL,
    FIG3_CHARGE,
    FIG3_MODEL,
    QUOTED_VALUES,
    fig3_paths,
    fig5_paths,
)

PATHS = fig3_paths()
M_I3, M_E3 = FIG3_MODEL.critical_masses(FIG3_CHARGE)


def _quoted(name: str) -> object:
    """Return a ``pytest.approx`` matcher for a number printed in the paper.

    Parameters
    ----------
    name
        Key of :data:`QUOTED_VALUES`.

    Returns
    -------
    object
        Matcher accepting values within half a unit of the last printed digit.
    """
    value, tolerance = QUOTED_VALUES[name]
    return pytest.approx(value, abs=tolerance)


@pytest.mark.parametrize(
    ("key", "quoted"),
    [
        ("neutral", "fig4_Ec_neutral"),
        ("linear_half", "fig4_Ec_linear_half"),
        ("linear_three_quarters", "fig4_Ec_linear_three_quarters"),
        ("exponential", "fig4_Ec_exponential"),
        ("fractional", "fig4_Ec_fractional"),
    ],
)
def test_coalescence_energies_fig_4(key: str, quoted: str) -> None:
    """All five E_c values quoted in the Fig. 4 caption are reproduced.

    Parameters
    ----------
    key
        Path name in :func:`fig3_paths`.
    quoted
        Key of the quoted value.
    """
    assert PATHS[key].coalescence_energy() == _quoted(quoted)


def test_delayed_recovery_energy_eq_4_39() -> None:
    """The 3/4 linear path re-meets M_i(Q) at E_s ≈ 3.6477, matching Eq. (4.39)."""
    path = PATHS["linear_three_quarters"]
    E_s = path.inner_recovery_energy(path.coalescence_energy())
    assert E_s == _quoted("fig4_Es_linear_three_quarters")
    eta = 0.75 * FIG3_CHARGE / M_I3
    closed_form = FIG3_CHARGE**2 / (eta**2 * M_I3) * (2 * eta * M_I3 / FIG3_CHARGE - 1)
    assert E_s == pytest.approx(closed_form, rel=1e-10)


def test_fractional_exponent_eq_4_43() -> None:
    """P ≈ 0.1063 for Q_f/Q = 0.65, and 0 < p < 1/2 automatically."""
    profile = PATHS["fractional"].profile
    assert isinstance(profile, FractionalLoss)
    assert profile.p == _quoted("fig3_fractional_p")


@given(E=st.floats(min_value=0.0, max_value=3.0))
def test_linear_inner_gap_is_exact_quadratic_eq_4_38(E: float) -> None:
    """ΔM_i(E) = (1 - 2ηM_i/Q) E + η^2 M_i E^2 / Q^2 exactly."""
    path = PATHS["linear_three_quarters"]
    profile = path.profile
    assert isinstance(profile, LinearProfile)
    eta = profile.eta
    expected = (1 - 2 * eta * M_I3 / FIG3_CHARGE) * E + eta**2 * M_I3 * E**2 / FIG3_CHARGE**2
    assert path.mass_gaps(E)[0] == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("key", ["neutral", "linear_half", "exponential", "fractional"])
def test_paths_stay_in_strip_with_positive_horizon_energy(key: str) -> None:
    """These paths stay in the three-horizon strip with 1 - Φ dQ/dE > 0 (Sec. 4.2).

    Parameters
    ----------
    key
        Path name.
    """
    path = PATHS[key]
    E_c = path.coalescence_energy()
    for E in np.linspace(0.0, E_c, 60)[1:-1]:
        assert path.in_three_horizon_strip(float(E))
        assert path.horizon_energy_factor(float(E)) > 0


def test_returning_and_preserving_channels_fig_5() -> None:
    """Eq. (4.46) gap, common endpoint, and common exponent Eq. (4.47)."""
    paths = fig5_paths()
    returning, preserving = paths["returning"], paths["preserving"]
    profile = returning.profile
    assert isinstance(profile, ReturningProfile)
    E_r = profile.return_energy
    for E in np.linspace(0.0, E_r, 25)[1:-1]:
        E_f = float(E)
        assert returning.mass_gaps(E_f)[0] == pytest.approx(E_f * (1 - E_f / E_r), rel=1e-9)
        assert returning.in_three_horizon_strip(E_f)
        assert returning.horizon_energy_factor(E_f) > 0
        assert preserving.mass_gaps(E_f)[0] == pytest.approx(0.0, abs=1e-9)
    assert returning.state(E_r) == pytest.approx(preserving.state(E_r), rel=1e-12)
    M_i = FIG2_MODEL.critical_masses(FIG2_CHARGE)[0]
    S0 = FIG2_MODEL.entropy(returning.outer_horizon(0.0))
    expected = S0 * (math.sqrt(1 - E_r / M_i) - 1)
    assert returning.entropy_change(E_r) == pytest.approx(expected, rel=1e-9)
    assert preserving.entropy_change(E_r) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("key", ["neutral", "linear_half", "exponential"])
def test_outer_horizon_velocity_eq_4_37(key: str) -> None:
    """dR_+/dE from Eq. (4.37) matches a central finite difference.

    Parameters
    ----------
    key
        Path name.
    """
    path = PATHS[key]
    E, step = 1.0, 1e-5
    numeric = (path.outer_horizon(E + step) - path.outer_horizon(E - step)) / (2 * step)
    assert path.outer_horizon_velocity(E) == pytest.approx(numeric, rel=1e-6)


def test_entropy_endpoint_eq_5_5() -> None:
    """Lim ln Γ = (4π^2/κ)(r_min(Q_f) - r_+) for the neutral path."""
    path = PATHS["neutral"]
    E_c = path.coalescence_energy()
    r_min = FIG3_MODEL.stationary_radii(FIG3_CHARGE)[1]
    expected = FIG3_MODEL.entropy(r_min) - FIG3_MODEL.entropy(path.outer_horizon(0.0))
    assert path.entropy_change(E_c * (1 - 1e-12)) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "profile",
    [
        NeutralProfile(),
        LinearProfile(0.3),
        ExponentialDepletion(8.0, 2.0, 3.0, 4.0),
        FractionalLoss(8.0, 4.0, 0.3),
        ReturningProfile(8.0, 4.0, 1.5),
    ],
)
def test_profiles_start_at_zero_and_rate_is_derivative(profile: ChargeProfile) -> None:
    """Q_em(0) = 0 and rate = dQ_em/dE for every profile.

    Parameters
    ----------
    profile
        The charge profile under test.
    """
    assert isinstance(profile, ChargeProfile)
    assert profile.charge(0.0) == 0.0
    E, step = 1.0, 1e-6
    numeric = (profile.charge(E + step) - profile.charge(E - step)) / (2 * step)
    assert profile.rate(E) == pytest.approx(numeric, rel=1e-6, abs=1e-9)


def test_profile_domains() -> None:
    """Evaluations outside a profile's domain raise instead of returning NaN."""
    with pytest.raises(ValueError, match="non-negative"):
        LinearProfile(0.1).charge(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        NeutralProfile().rate(-1.0)
    with pytest.raises(ValueError, match="E < M_0"):
        FractionalLoss(1.0, 2.0, 0.5).charge(2.0)
    with pytest.raises(ValueError, match="E <= E_r"):
        ReturningProfile(1.0, 2.0, 0.5).rate(0.6)


def test_path_without_outer_coalescence() -> None:
    """With ηM_i/Q >= 1 the charge is exhausted before M_e(Q') is reached (Sec. 4.2)."""
    path = EmissionPath(FIG3_MODEL, M_I3, FIG3_CHARGE, LinearProfile(1.2 * FIG3_CHARGE / M_I3))
    limit = FIG3_CHARGE / path.profile.rate(0.0) * (1 - 1e-6)
    with pytest.raises(ValueError, match="outer critical curve"):
        path.coalescence_energy(E_max=limit)
    assert PATHS["neutral"].inner_recovery_energy(1.0) is None
