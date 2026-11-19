"""Greybody factors, luminosity and near-extremal predictions."""

import math

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import solve_ivp

from blackholes.paper.configurations import FIG2_CHARGE, FIG2_MODEL
from blackholes.radiation.background import StaticBackground
from blackholes.radiation.greybody import CONFORMAL_COUPLING, absorption_probability, potential
from blackholes.radiation.luminosity import luminosity
from blackholes.radiation.near_horizon import (
    luminosity_exponent,
    scaling_dimension,
    throat_radius_squared,
)

MODEL, Q = FIG2_MODEL, FIG2_CHARGE
M_I, M_E = MODEL.critical_masses(Q)
KAPPA_OVER_PI = MODEL.kappa / math.pi


def _background(gap: float) -> StaticBackground:
    """Neutral-path state a mass ``gap`` above the outer critical curve.

    Parameters
    ----------
    gap
        ``ΔE = M - M_e``.

    Returns
    -------
    StaticBackground
        The exterior of that state.
    """
    return StaticBackground.from_outer_horizon(MODEL, Q, MODEL.horizons(M_E + gap, Q)[-1])


def _reference_gamma(gap: float, omega: float, k: int, r_absorb: float) -> float:
    """Independent greybody factor: adaptive SciPy integration with scalar model methods.

    Uses a Taylor expansion about ``r_+`` (not about ``r_min``) near the horizon and
    the direct mass function elsewhere, so it shares no background code with the
    JAX solver.

    Parameters
    ----------
    gap
        ``ΔE = M - M_e``.
    omega
        Frequency.
    k
        Angular quantum number.
    r_absorb
        Absorbing radius.

    Returns
    -------
    float
        ``Γ_k(ω)``.
    """
    M = M_E + gap
    roots = MODEL.horizons(M, Q)
    r_plus, width = roots[-1], roots[-1] - roots[-2]
    taylor = [KAPPA_OVER_PI * MODEL.taylor_coefficient(r_plus, Q, j) for j in range(1, 12)]

    def pieces(r: float) -> tuple[float, float, float, float]:
        """Return ``(f, f/(r - r_+), f', f'')`` at ``r``.

        Parameters
        ----------
        r
            Radius outside the horizon.

        Returns
        -------
        tuple[float, float, float, float]
            Background values.
        """
        d = r - r_plus
        if d < 1e-3 * width:
            g = sum(c * d**j for j, c in enumerate(taylor))
        else:
            g = KAPPA_OVER_PI * (MODEL.mass_function(r, Q) - M) / d
        f1 = KAPPA_OVER_PI * MODEL.radial_derivative(r, Q, 1)
        f2 = KAPPA_OVER_PI * MODEL.radial_derivative(r, Q, 2)
        return g * d, g, f1, f2

    def bracket(r: float) -> float:
        """``V/f`` at ``r``.

        Parameters
        ----------
        r
            Radius.

        Returns
        -------
        float
            ``k²/r² + ξR + f'/(2r) - f/(4r²)``.
        """
        f, _, f1, f2 = pieces(r)
        return (
            k * k / r**2 + CONFORMAL_COUPLING * (-f2 - 2 * f1 / r) + f1 / (2 * r) - f / (4 * r * r)
        )

    def rhs(s: float, y: npt.NDArray[np.float64]) -> list[float]:
        """Real form of the first-order system in ``s = ln(r - r_+)``.

        Parameters
        ----------
        s
            Log-distance from the horizon.
        y
            ``(Re ψ, Im ψ, Re φ, Im φ)``.

        Returns
        -------
        list[float]
            Derivatives.
        """
        d = math.exp(s)
        _, g, _, _ = pieces(r_plus + d)
        psi, phi = complex(y[0], y[1]), complex(y[2], y[3])
        dpsi = phi / g
        dphi = (d * bracket(r_plus + d) - omega**2 / g) * psi
        return [dpsi.real, dpsi.imag, dphi.real, dphi.imag]

    span = (math.log(1e-7 * min(width, r_plus)), math.log(r_absorb - r_plus))
    sol = solve_ivp(rhs, span, [1.0, 0.0, 0.0, -omega], method="DOP853", rtol=1e-10, atol=1e-12)
    psi = complex(sol.y[0, -1], sol.y[1, -1])
    phi = complex(sol.y[2, -1], sol.y[3, -1])
    f = pieces(r_absorb)[0]
    p = math.sqrt(omega**2 - f * bracket(r_absorb))
    incoming = (psi - phi / (1j * p)) * math.sqrt(p) / 2
    return omega / abs(incoming) ** 2


@pytest.mark.parametrize(
    ("gap", "x", "k"), [(1e-1, 1.0, 0), (1e-1, 5.0, 1), (1e-4, 0.1, 0), (1e-4, 1.0, 1)]
)
def test_greybody_matches_independent_solver(gap: float, x: float, k: int) -> None:
    """Batched JAX RK4 agrees with adaptive SciPy (different background code).

    Parameters
    ----------
    gap
        Mass above coalescence.
    x
        Frequency in units of ``T_H``.
    k
        Angular quantum number.
    """
    background = _background(gap)
    omega = x * background.temperature
    r_absorb = 2 * background.r_plus
    got = float(
        absorption_probability(background, jnp.asarray(omega), jnp.asarray(k), r_absorb=r_absorb)[
            ()
        ]
    )
    assert got == pytest.approx(_reference_gamma(gap, omega, k, r_absorb), rel=1e-6)


def test_background_is_cancellation_free() -> None:
    """``g(r_+) = 4π T_H`` and the series and direct branches agree where both apply."""
    for gap in (1e-1, 1e-6):
        background = _background(gap)
        assert float(background.g(jnp.asarray(background.r_plus + 1e-13))) == pytest.approx(
            4 * math.pi * background.temperature, rel=1e-6
        )
    background = _background(1e-2)
    r = jnp.asarray(background.r_min * 1.5)
    direct = (
        KAPPA_OVER_PI
        * (MODEL.mass_function(float(r), Q) - background.mass)
        / (float(r) - background.r_plus)
    )
    assert float(background.g(r)) == pytest.approx(direct, rel=1e-10)
    with pytest.raises(ValueError, match="r_plus"):
        StaticBackground.from_outer_horizon(MODEL, Q, background.r_min)


def test_greybody_bounds_and_input_validation() -> None:
    """0 < Γ <= 1, Γ grows with frequency, and bad inputs raise."""
    background = _background(1e-2)
    omega = jnp.linspace(0.05, 8.0, 12) * background.temperature
    gamma = np.asarray(absorption_probability(background, omega, jnp.asarray(0), r_absorb=8.0))
    assert np.all(gamma > 0)
    assert np.all(gamma <= 1 + 1e-9)
    assert np.all(np.diff(gamma) > 0)
    with pytest.raises(ValueError, match="r_absorb"):
        absorption_probability(background, omega, jnp.asarray(0), r_absorb=background.r_plus)
    with pytest.raises(ValueError, match="steps"):
        absorption_probability(background, omega, jnp.asarray(0), r_absorb=8.0, steps=0)


def test_evanescent_bath_gives_zero() -> None:
    """A minimally coupled field cannot propagate where V(r_a) > ω²."""
    background = _background(1e-2)
    r_absorb = 30.0
    omega = 0.5 * background.temperature
    assert float(potential(background, jnp.asarray(r_absorb), jnp.asarray(0.0), 0.0)) > omega**2
    gamma = absorption_probability(
        background, jnp.asarray(omega), jnp.asarray(0), r_absorb=r_absorb, xi=0.0
    )
    assert float(gamma[()]) == 0.0


def test_near_horizon_dimensions() -> None:
    """Conformal s-wave saturates the BF bound; minimal s-wave has Δ = 1."""
    assert scaling_dimension(MODEL, Q, 0, CONFORMAL_COUPLING) == pytest.approx(0.5)
    assert scaling_dimension(MODEL, Q, 0, 0.0) == pytest.approx(1.0)
    assert luminosity_exponent(MODEL, Q, CONFORMAL_COUPLING) == pytest.approx(1.0)
    assert luminosity_exponent(MODEL, Q, 0.0) == pytest.approx(1.5)
    r_e = MODEL.stationary_radii(Q)[1]
    expected = 0.5 + math.sqrt(throat_radius_squared(MODEL, Q) / r_e**2)
    assert scaling_dimension(MODEL, Q, 1, CONFORMAL_COUPLING) == pytest.approx(expected)
    with pytest.raises(ValueError, match="BF bound"):
        scaling_dimension(MODEL, Q, 0, 0.2)


@pytest.mark.slow
def test_k1_greybody_follows_throat_scaling() -> None:
    """Γ_1(ω = T) ∝ T^(2Δ_1 - 1) deep in the throat (bath-independent exponent)."""
    expected = 2 * scaling_dimension(MODEL, Q, 1, CONFORMAL_COUPLING) - 1
    for r_absorb in (6.0, 10.0):
        values = []
        temperatures = []
        for gap in (1e-6, 1e-8):
            background = _background(gap)
            T = background.temperature
            values.append(
                float(
                    absorption_probability(
                        background, jnp.asarray(T), jnp.asarray(1), r_absorb=r_absorb
                    )[()]
                )
            )
            temperatures.append(T)
        slope = math.log(values[0] / values[1]) / math.log(temperatures[0] / temperatures[1])
        assert slope == pytest.approx(expected, abs=0.03)


def test_luminosity_is_dominated_by_low_modes() -> None:
    """The mode sum converges quickly and the total is positive."""
    result = luminosity(_background(1e-2), r_absorb=8.0, k_max=4, nodes=24, steps=3000)
    assert result.total > 0
    assert result.per_mode[-1] < 1e-6 * result.total
    assert result.per_mode[0] == max(result.per_mode)
