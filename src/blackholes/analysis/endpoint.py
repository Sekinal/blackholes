"""Uniform approximation of the neutral event horizon up to outer coalescence.

Why the paper's truncations fail
--------------------------------
``R_+(E)`` has a square-root branch point at the coalescence energy ``E_c``
(Eq. (3.17)), so no polynomial in ``E`` can reproduce it near the endpoint
(Fig. 6 of the paper). Its Maclaurin series does converge at ``E_c``, but only like
``N^(-1/2)`` (:mod:`blackholes.analysis.convergence`).

The fix
-------
In the variable ``t = sqrt(1 - E/E_c)`` (so ``E = E_c (1 - t^2)``), the branch point
is unfolded: the Puiseux series of App. C.4 becomes an ordinary Taylor series in
``t`` about ``t = 0``, and the Lagrange–Bürmann series about ``E = 0`` becomes a
Taylor series about ``t = 1``. ``R_+(t)`` is analytic on a neighbourhood of
``[0, 1]``; its nearest singularity is at ``t = -1`` (the *other* member of the
outer pair meeting the inner degenerate root at ``E = 0``). A polynomial matching
``K`` Taylor coefficients at each end (two-point Hermite interpolation) therefore
converges geometrically on the whole emission interval.

Numerics: the confluent Vandermonde system is solved and the polynomial evaluated
in 50-digit arithmetic (mpmath), so the monomial basis causes no loss of accuracy.
"""

from dataclasses import dataclass

import mpmath
import numpy as np

from blackholes.models import NonlinearBTZ
from blackholes.series.numeric import (
    FloatArray,
    compose_numeric,
    lagrange_burmann_inverse_numeric,
    puiseux_inverse_numeric,
)

__all__ = ["EndpointApproximant", "build_neutral_endpoint_approximant", "two_point_hermite"]

_WORKING_DIGITS = 50


def two_point_hermite(left: FloatArray, right: FloatArray) -> list[mpmath.mpf]:
    """Monomial coefficients of the polynomial matching Taylor data at ``t = 0`` and ``t = 1``.

    Parameters
    ----------
    left
        Taylor coefficients at ``t = 0``: ``P(t) = left[0] + left[1] t + ... + O(t^K)``.
    right
        Taylor coefficients at ``t = 1`` in ``s = t - 1``:
        ``P = right[0] + right[1] s + ... + O(s^L)``.

    Returns
    -------
    list[mpmath.mpf]
        Coefficients ``p_0 ... p_{K+L-1}`` (in increasing powers of ``t``) of the
        unique polynomial of degree ``< K + L`` with that data.
    """
    K, L = len(left), len(right)
    degree = K + L
    with mpmath.workdps(_WORKING_DIGITS):
        matrix = mpmath.zeros(degree, degree)
        rhs = mpmath.zeros(degree, 1)
        for i in range(K):  # the t**i coefficient of P equals left[i]
            matrix[i, i] = 1
            rhs[i] = mpmath.mpf(float(left[i]))
        for j in range(L):  # the s**j coefficient at t = 1 is sum_k binom(k, j) p_k
            for k in range(j, degree):
                matrix[K + j, k] = mpmath.binomial(k, j)
            rhs[K + j] = mpmath.mpf(float(right[j]))
        solution = mpmath.lu_solve(matrix, rhs)
        return [solution[k] for k in range(degree)]


@dataclass(frozen=True, slots=True)
class EndpointApproximant:
    """Uniform polynomial approximation of ``R_+(E)`` on ``[0, E_c]`` in ``t = sqrt(1 - E/E_c)``.

    Attributes
    ----------
    coalescence_energy
        ``E_c``.
    initial_radius
        ``R_+(0)``.
    coefficients
        Polynomial coefficients in ``t`` (increasing powers), 50-digit.
    entropy_per_radius
        ``dS/dr = 4π^2/κ``, converting radii into the KPW exponent.
    """

    coalescence_energy: float
    initial_radius: float
    coefficients: tuple[mpmath.mpf, ...]
    entropy_per_radius: float

    def radius(self, E: float) -> float:
        """Approximate event-horizon radius ``R_+(E)``.

        Parameters
        ----------
        E
            Emitted energy in ``[0, E_c]``.

        Returns
        -------
        float
            The polynomial evaluated at ``t = sqrt(1 - E/E_c)``.

        Raises
        ------
        ValueError
            If ``E`` lies outside ``[0, E_c]``.
        """
        if not 0 <= E <= self.coalescence_energy:
            msg = f"E must lie in [0, E_c = {self.coalescence_energy}], got {E}"
            raise ValueError(msg)
        with mpmath.workdps(_WORKING_DIGITS):
            t = mpmath.sqrt(1 - mpmath.mpf(E) / mpmath.mpf(self.coalescence_energy))
            return float(mpmath.polyval(list(reversed(self.coefficients)), t))

    def entropy_change(self, E: float) -> float:
        """Approximate leading KPW exponent ``ΔS(E) = (4π^2/κ)(R_+(E) - R_+(0))``.

        Parameters
        ----------
        E
            Emitted energy in ``[0, E_c]``.

        Returns
        -------
        float
            The exponent ``ln Γ``.
        """
        return self.entropy_per_radius * (self.radius(E) - self.initial_radius)


def build_neutral_endpoint_approximant(
    model: NonlinearBTZ, M: float, Q: float, terms_per_end: int
) -> EndpointApproximant:
    """Build the two-point approximant for neutral emission from ``(M, Q)``.

    The initial state must lie in (or on the upper edge of) the three-horizon strip
    so that neutral emission ends at the outer coalescence ``M - E_c = M_e(Q)``.

    Parameters
    ----------
    model
        The black-hole family.
    M
        Initial mass, ``M_e(Q) < M <= M_i(Q)``.
    Q
        Charge (fixed for neutral emission).
    terms_per_end
        Number ``K`` of Taylor coefficients matched at each end; the error
        decreases geometrically in ``K``.

    Returns
    -------
    EndpointApproximant
        The approximant of degree ``2K - 1`` in ``t``.

    Raises
    ------
    ValueError
        If ``M`` is not above the outer critical mass or ``terms_per_end < 1``.
    """
    if terms_per_end < 1:
        msg = f"terms_per_end must be >= 1, got {terms_per_end}"
        raise ValueError(msg)
    _, M_e = model.critical_masses(Q)
    if not M_e < M:
        msg = f"initial mass {M} must exceed the outer critical mass {M_e}"
        raise ValueError(msg)
    E_c = M - M_e
    K = terms_per_end
    r_min = model.stationary_radii(Q)[1]
    r_plus = model.horizons(M, Q)[-1]

    # Left end (t = 0): m(r_min + y) - M_e = E_c - E = E_c t^2 is a double zero in y.
    # The positive Puiseux branch in u = sqrt(E_c - E) = sqrt(E_c) t is the outer horizon.
    h_left = [model.taylor_coefficient(r_min, Q, j) for j in range(1, K + 1)]
    h_left[0] = 0.0  # ∂_r m vanishes at the minimum; remove round-off
    b = puiseux_inverse_numeric(h_left, multiplicity=2, order=K - 1) if K > 1 else np.array([])
    left = np.concatenate(([r_min], b * np.sqrt(E_c) ** np.arange(1, K)))

    # Right end (t = 1, s = t - 1): E = -E_c (2 s + s^2); compose with the LB series.
    h_right = [-model.taylor_coefficient(r_plus, Q, j) for j in range(1, K)]
    if K > 1:
        a = lagrange_burmann_inverse_numeric(h_right, order=K - 1)
        tail = compose_numeric(a, [-2.0 * E_c, -E_c], order=K - 1)
        right = np.concatenate(([r_plus], tail))
    else:
        right = np.array([r_plus])

    coefficients = two_point_hermite(left, right)
    return EndpointApproximant(E_c, r_plus, tuple(coefficients), model.entropy(1.0))
