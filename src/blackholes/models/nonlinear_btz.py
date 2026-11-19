"""The logarithmically (nonlinearly) charged BTZ family of paper Sec. 4 and App. D.

Geometry
--------
Static metric ``ds^2 = -f dt^2 + dr^2/f + r^2 dφ^2`` with, Eq. (4.2),

    f(r) = r^2/ℓ^2 - M_par - (2√2 κ q r / γ) ln(β γ r / (√2 q)),

where ``M_par`` and ``q`` are integration constants and ``(κ, ℓ, β, γ)`` are fixed
couplings (κ = 8πG_3, Λ = -1/ℓ^2). Physical charges, Eq. (4.5):

    M = π M_par / κ,     Q = 2π q,     S = 4π^2 r_+ / κ.

Mass function
-------------
Every horizon solves ``m(r, Q) = M`` with, Eq. (4.10),

    m(r, Q) = (π/κ) [ r^2/ℓ^2 - (2√2 κ q r / γ) ln(β γ r / (√2 q)) ],   q = Q / 2π,

so that ``f = (κ/π) [m(r, Q) - M]`` (Eq. (4.11)) is affine in the mass with
``c = κ/π``. All public methods take the *physical* mass ``M`` and charge ``Q``.

Validity
--------
The logarithm requires ``β γ / q > 0`` (Eq. (4.4)); methods raise ``ValueError``
outside that sector rather than returning NaN.
"""

import math
from dataclasses import dataclass
from itertools import pairwise

from scipy.optimize import brentq
from scipy.special import lambertw

__all__ = ["NonlinearBTZ"]

_SQRT2 = math.sqrt(2.0)
_LAST_POLYNOMIAL_ORDER = 2  # r**2/ell**2 stops contributing beyond the 2nd derivative
_EXTREMA_PAIR = 2  # a local maximum and a local minimum of m(r)


@dataclass(frozen=True, slots=True)
class NonlinearBTZ:
    """Fixed couplings of the Einstein–NLED theory supporting Eq. (4.2).

    Attributes
    ----------
    kappa
        Gravitational coupling ``κ = 8πG_3 > 0``.
    ell
        AdS radius ``ℓ > 0`` (Λ = -1/ℓ^2).
    beta
        Nonlinear-electrodynamics coupling ``β`` (non-zero).
    gamma
        Nonlinear-electrodynamics coupling ``γ`` (non-zero).

    Examples
    --------
    The configuration of the paper's Fig. 2:

    >>> model = NonlinearBTZ(kappa=1.0, ell=1.0, beta=1.2, gamma=1.0)
    >>> Q = model.charge_from_q(1.0)
    >>> M_i, M_e = model.critical_masses(Q)
    >>> round(M_i - M_e, 4)
    1.8559
    """

    kappa: float
    ell: float
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        """Validate the couplings once, so every method can rely on them.

        Raises
        ------
        ValueError
            If ``κ`` or ``ℓ`` is not positive and finite, or ``β γ = 0``.
        """
        for name in ("kappa", "ell"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                msg = f"{name} must be positive and finite, got {value}"
                raise ValueError(msg)
        if not (math.isfinite(self.beta * self.gamma) and self.beta * self.gamma != 0):
            msg = f"beta * gamma must be finite and non-zero, got {self.beta * self.gamma}"
            raise ValueError(msg)

    # ------------------------------------------------------------------ units --
    @staticmethod
    def charge_from_q(q: float) -> float:
        """Physical charge ``Q = 2π q`` from the integration constant ``q``, Eq. (4.5).

        Parameters
        ----------
        q
            Integration constant appearing in Eq. (4.2).

        Returns
        -------
        float
            The physical charge ``Q``.
        """
        return 2.0 * math.pi * q

    @staticmethod
    def q_from_charge(Q: float) -> float:
        """Integration constant ``q = Q / 2π`` from the physical charge, Eq. (4.5).

        Parameters
        ----------
        Q
            Physical charge.

        Returns
        -------
        float
            The constant ``q`` of Eq. (4.2).
        """
        return Q / (2.0 * math.pi)

    def mass_from_parameter(self, M_par: float) -> float:
        """Physical mass ``M = π M_par / κ`` from the metric parameter, Eq. (4.5).

        Parameters
        ----------
        M_par
            The constant ``M`` appearing in the blackening function (4.2).

        Returns
        -------
        float
            The physical (Hamiltonian) mass.
        """
        return math.pi * M_par / self.kappa

    @property
    def coupling_ratio(self) -> float:
        """Dimensionless combination ``κ β ℓ^2`` controlling the horizon structure.

        Returns
        -------
        float
            ``κ β ℓ^2``; two stationary points exist for ``> 1`` (Sec. 4.2) and the
            outer coalescence has positive mass for ``1 < κβℓ^2 < e/2`` (Eq. (4.24)).
        """
        return self.kappa * self.beta * self.ell**2

    # --------------------------------------------------------- local functions --
    def _log_term(self, r: float, Q: float) -> float:
        """Return ``ln(β γ r / (√2 q))`` after checking it is real.

        Parameters
        ----------
        r
            Areal radius, must be positive.
        Q
            Physical charge, must satisfy ``β γ / q > 0`` (Eq. (4.4)).

        Returns
        -------
        float
            The logarithm entering Eqs. (4.2)-(4.3).

        Raises
        ------
        ValueError
            If ``r <= 0`` or ``β γ / q <= 0``.
        """
        if not r > 0:
            msg = f"radius must be positive, got {r}"
            raise ValueError(msg)
        q = self.q_from_charge(Q)
        argument = self.beta * self.gamma * r / (_SQRT2 * q) if q != 0 else -1.0
        if not argument > 0:
            msg = f"need beta*gamma/q > 0 (Eq. (4.4)); got q = {q}"
            raise ValueError(msg)
        return math.log(argument)

    def _log_strength(self, Q: float) -> float:
        """Return ``C = 2√2 κ q / γ``, the coefficient of ``r ln r`` in ``f``.

        Parameters
        ----------
        Q
            Physical charge.

        Returns
        -------
        float
            The coefficient ``C`` (sign follows ``q / γ``).
        """
        return 2.0 * _SQRT2 * self.kappa * self.q_from_charge(Q) / self.gamma

    def mass_function(self, r: float, Q: float) -> float:
        """Mass ``m(r, Q)`` of the solution with a horizon at ``r``, Eq. (4.10).

        Parameters
        ----------
        r
            Horizon radius candidate (positive).
        Q
            Physical charge.

        Returns
        -------
        float
            The physical mass for which ``f(r) = 0``.
        """
        log_term = self._log_term(r, Q)
        return (math.pi / self.kappa) * (r**2 / self.ell**2 - self._log_strength(Q) * r * log_term)

    def radial_derivative(self, r: float, Q: float, n: int) -> float:
        """``n``-th radial derivative ``∂_r^n m(r, Q)``, Eqs. (4.18) and (E.3).

        For ``n >= 3`` only the ``r ln r`` term contributes:
        ``∂_r^n m = (π/κ) (-1)^(n+1) (n-2)! C / r^(n-1)`` with ``C = 2√2κq/γ``.

        Parameters
        ----------
        r
            Radius (positive).
        Q
            Physical charge.
        n
            Derivative order, ``n >= 0`` (``n = 0`` returns ``m`` itself).

        Returns
        -------
        float
            The derivative value.

        Raises
        ------
        ValueError
            If ``n < 0``.
        """
        if n < 0:
            msg = f"derivative order must be >= 0, got {n}"
            raise ValueError(msg)
        if n == 0:
            return self.mass_function(r, Q)
        log_term = self._log_term(r, Q)  # also validates (r, Q)
        strength = self._log_strength(Q)
        prefactor = math.pi / self.kappa
        if n == 1:
            return prefactor * (2.0 * r / self.ell**2 - strength * (log_term + 1.0))
        if n == _LAST_POLYNOMIAL_ORDER:
            return prefactor * (2.0 / self.ell**2 - strength / r)
        return prefactor * (-1.0) ** (n + 1) * math.factorial(n - 2) * strength / r ** (n - 1)

    def taylor_coefficient(self, r: float, Q: float, j: int) -> float:
        """Taylor coefficient ``∂_r^j m(r, Q) / j!`` of the mass function about ``r``.

        Computed in closed form so that it stays finite at any order: for
        ``j >= 3`` the factorials of Eq. (E.3) cancel to
        ``(π/κ) (-1)^(j+1) C / (j (j-1) r^(j-1))`` with ``C = 2√2κq/γ``.

        Parameters
        ----------
        r
            Expansion point (positive).
        Q
            Physical charge.
        j
            Order, ``j >= 0``.

        Returns
        -------
        float
            The coefficient of ``y**j`` in ``m(r + y, Q)``.
        """
        if j <= _LAST_POLYNOMIAL_ORDER:
            return self.radial_derivative(r, Q, j) / math.factorial(max(j, 0))
        self._log_term(r, Q)  # validates (r, Q)
        return (
            (math.pi / self.kappa)
            * (-1.0) ** (j + 1)
            * self._log_strength(Q)
            / (j * (j - 1) * r ** (j - 1))
        )

    def blackening(self, r: float, M: float, Q: float) -> float:
        """Blackening function ``f(r; M, Q) = (κ/π) [m(r, Q) - M]``, Eq. (4.11).

        Parameters
        ----------
        r
            Radius (positive).
        M
            Physical mass.
        Q
            Physical charge.

        Returns
        -------
        float
            ``f(r)``; positive outside the event horizon.
        """
        return (self.kappa / math.pi) * (self.mass_function(r, Q) - M)

    def temperature(self, r_h: float, Q: float) -> float:
        """Hawking temperature ``T_H = f'(r_h)/4π = κ ∂_r m / 4π^2`` at a horizon, Eq. (4.6).

        Parameters
        ----------
        r_h
            A horizon radius (the caller guarantees ``f(r_h) = 0``).
        Q
            Physical charge.

        Returns
        -------
        float
            The surface-gravity temperature (negative on inner horizons where
            ``f' < 0``; physically its magnitude is the inner surface gravity / 2π).
        """
        return self.kappa * self.radial_derivative(r_h, Q, 1) / (4.0 * math.pi**2)

    def potential(self, r: float, Q: float) -> float:
        """Electric potential ``Φ(r) = -A_t(r)`` in the gauge of Eq. (4.3), Eq. (4.7).

        This equals ``∂m/∂Q`` at fixed ``r``, the conjugate potential in the first
        law (4.8).

        Parameters
        ----------
        r
            Radius (positive).
        Q
            Physical charge.

        Returns
        -------
        float
            ``(√2 r / γ) [1 - ln(β γ r / (√2 q))]``.
        """
        return _SQRT2 * r / self.gamma * (1.0 - self._log_term(r, Q))

    def potential_derivative(self, r: float, Q: float) -> float:
        """Radial derivative ``∂_r Φ = -(√2/γ) ln(β γ r / (√2 q))`` of :meth:`potential`.

        ``-∂_r Φ = ∂_r A_t`` is the radial electric field ``F_rt``; it grows
        logarithmically, which is why this family has no canonical zero of the
        potential at infinity.

        Parameters
        ----------
        r
            Radius (positive).
        Q
            Physical charge.

        Returns
        -------
        float
            ``∂_r Φ(r)``.
        """
        return -_SQRT2 / self.gamma * self._log_term(r, Q)

    def entropy(self, r_h: float) -> float:
        """Bekenstein–Hawking entropy ``S = 4π^2 r_h / κ``, Eq. (4.5).

        Parameters
        ----------
        r_h
            Horizon radius.

        Returns
        -------
        float
            The entropy (``ħ = 1``).
        """
        return 4.0 * math.pi**2 * r_h / self.kappa

    # ------------------------------------------------------ critical structure --
    def stationary_radii(self, Q: float) -> tuple[float, ...]:
        """Positive radii where ``∂_r m = 0``, via Lambert ``W``, Eqs. (4.19)-(4.21).

        Writing ``r = -r_t w`` with ``r_t = √2 κ q ℓ^2 / γ`` turns ``∂_r m = 0`` into
        ``w e^w = -1/(e κ β ℓ^2)``. Each real branch giving ``r > 0`` is a
        stationary point.

        Parameters
        ----------
        Q
            Physical charge (``β γ / q > 0``).

        Returns
        -------
        tuple[float, ...]
            Sorted stationary radii: ``(r_max, r_min)`` in the multihorizon sector
            (``q/γ > 0``, ``κβℓ^2 > 1``), one minimum when ``q/γ < 0``, and none
            when ``m`` is monotonic.
        """
        self._log_term(1.0, Q)  # validates the charge sector
        r_t = _SQRT2 * self.kappa * self.q_from_charge(Q) * self.ell**2 / self.gamma
        x = -1.0 / (math.e * self.coupling_ratio)
        branches = [0] if x >= 0 else ([0, -1] if x > -1.0 / math.e else [])
        radii = {-r_t * float(lambertw(x, k).real) for k in branches}
        return tuple(sorted(r for r in radii if r > 0))

    def critical_masses(self, Q: float) -> tuple[float, float]:
        """Inner and outer critical masses ``(M_i(Q), M_e(Q))``, Eqs. (4.32)-(4.33).

        ``M_i`` (local maximum of ``m``) carries a degenerate inner horizon; ``M_e``
        (local minimum) a degenerate, zero-temperature event horizon.

        Parameters
        ----------
        Q
            Physical charge in the multihorizon sector (``q/γ > 0``, ``κβℓ^2 > 1``).

        Returns
        -------
        tuple[float, float]
            ``(M_i, M_e)`` with ``M_i > M_e``.

        Raises
        ------
        ValueError
            If the charge/coupling sector has no maximum–minimum pair.
        """
        radii = self.stationary_radii(Q)
        if len(radii) != _EXTREMA_PAIR:
            msg = "critical masses need q/gamma > 0 and kappa*beta*ell^2 > 1"
            raise ValueError(msg)
        r_max, r_min = radii
        return self.mass_function(r_max, Q), self.mass_function(r_min, Q)

    def horizons(self, M: float, Q: float, *, degenerate_tol: float = 1e-12) -> tuple[float, ...]:
        """All positive horizon radii for mass ``M`` and charge ``Q``, sorted.

        Roots of ``m(r, Q) = M`` are bracketed between consecutive stationary
        points (where ``m`` is monotonic) and polished with Brent's method. A
        stationary point whose mass equals ``M`` to within ``degenerate_tol``
        (relative) is returned once, as a degenerate horizon.

        Parameters
        ----------
        M
            Physical mass.
        Q
            Physical charge.
        degenerate_tol
            Relative tolerance for recognising a degenerate (double) horizon.

        Returns
        -------
        tuple[float, ...]
            Horizon radii in increasing order (possibly empty).
        """
        breakpoints = list(self.stationary_radii(Q))
        scale = max(1.0, abs(M))
        lower = 1e-9 * (breakpoints[0] if breakpoints else 1.0)
        upper = max([2.0, *breakpoints])
        while self.mass_function(upper, Q) <= M:
            upper *= 2.0
        nodes = [lower, *breakpoints, upper]
        values = [self.mass_function(r, Q) - M for r in nodes]

        roots: list[float] = [
            r
            for r, v in zip(breakpoints, values[1:-1], strict=True)
            if abs(v) <= degenerate_tol * scale
        ]
        for (a, fa), (b, fb) in pairwise(zip(nodes, values, strict=True)):
            if fa * fb < 0 and min(abs(fa), abs(fb)) > degenerate_tol * scale:
                roots.append(
                    float(brentq(lambda r: self.mass_function(r, Q) - M, a, b, xtol=1e-14))
                )
        return tuple(sorted(roots))
