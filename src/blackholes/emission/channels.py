"""Charge-emission profiles ``Q_em(E)`` defining emission channels (paper Sec. 2.4, 4.2-4.3).

An emission channel connects the initial state ``(M, Q)`` to ``(M - E, Q - Q_em(E))``
(Eq. (2.24)). The paper studies the profiles implemented here; each is a frozen
dataclass exposing

* ``charge(E)`` — cumulative emitted charge ``Q_em(E)``, and
* ``rate(E)`` — its derivative ``dQ_em/dE``, which enters the horizon-energy
  factor ``1 - Φ_+ dQ_em/dE`` of Eq. (2.28) and the horizon velocities (4.37).

Every profile satisfies ``Q_em(0) = 0`` and is defined for ``E >= 0`` within its
stated domain; evaluation outside the domain raises ``ValueError``.
"""

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "ChargeProfile",
    "ExponentialDepletion",
    "FractionalLoss",
    "LinearProfile",
    "NeutralProfile",
    "ReturningProfile",
]


def _require_nonnegative(E: float) -> None:
    """Reject negative emitted energies.

    Parameters
    ----------
    E
        Emitted energy.

    Raises
    ------
    ValueError
        If ``E < 0``.
    """
    if E < 0:
        msg = f"emitted energy must be non-negative, got {E}"
        raise ValueError(msg)


@runtime_checkable
class ChargeProfile(Protocol):
    """Interface shared by all emission channels."""

    def charge(self, E: float) -> float:
        """Cumulative emitted charge ``Q_em(E)``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            Charge carried away after emitting ``E``.
        """
        ...

    def rate(self, E: float) -> float:
        """Charge emitted per unit energy, ``dQ_em/dE``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            The derivative of :meth:`charge` at ``E``.
        """
        ...


@dataclass(frozen=True, slots=True)
class NeutralProfile:
    """Neutral emission: ``Q_em = 0`` (Secs. 2.1, 3.1)."""

    def charge(self, E: float) -> float:
        """Return zero emitted charge.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            ``0``.
        """
        _require_nonnegative(E)
        return 0.0

    def rate(self, E: float) -> float:
        """Return zero charge rate.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            ``0``.
        """
        _require_nonnegative(E)
        return 0.0


@dataclass(frozen=True, slots=True)
class LinearProfile:
    """Linear extraction ``Q_em = η E`` (Eq. (4.38)).

    Attributes
    ----------
    eta
        Charge emitted per unit energy. For an initial state on the inner critical
        curve, ``η M_i / Q = 1/2`` is the tangent case (ii) of Sec. 3.2.
    """

    eta: float

    def charge(self, E: float) -> float:
        """Return ``η E``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            Cumulative emitted charge.
        """
        _require_nonnegative(E)
        return self.eta * E

    def rate(self, E: float) -> float:
        """Return the constant rate ``η``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            ``η``.
        """
        _require_nonnegative(E)
        return self.eta


@dataclass(frozen=True, slots=True)
class ExponentialDepletion:
    """Exponential relaxation towards a residual charge, Eq. (4.40).

    ``Q_em(E) = (Q - Q_res) (1 - exp(-a E / E_scale))``, the solution of
    ``dQ'/dE = -(a / E_scale)(Q' - Q_res)`` with ``Q'(0) = Q``.

    Attributes
    ----------
    initial_charge
        Initial black-hole charge ``Q``.
    residual_charge
        Asymptotic charge ``Q_res`` (``0 <= Q_res < Q`` in the paper).
    a
        Dimensionless relaxation rate (``a > 0``).
    energy_scale
        Energy scale in the exponent (the paper uses the path's coalescence
        energy ``E_c``).
    """

    initial_charge: float
    residual_charge: float
    a: float
    energy_scale: float

    @classmethod
    def from_endpoint(
        cls, initial_charge: float, a: float, final_fraction: float, energy_scale: float
    ) -> "ExponentialDepletion":
        """Choose ``Q_res`` so that ``Q'(E_scale) / Q = s_f``, Eq. (4.43).

        Parameters
        ----------
        initial_charge
            Initial charge ``Q``.
        a
            Relaxation rate.
        final_fraction
            Remaining charge fraction ``s_f`` at ``E = E_scale``.
        energy_scale
            The energy at which ``s_f`` is reached (``E_c`` from Eq. (4.42)).

        Returns
        -------
        ExponentialDepletion
            The profile with ``Q_res / Q = (s_f - e^-a) / (1 - e^-a)``.
        """
        decay = math.exp(-a)
        residual = initial_charge * (final_fraction - decay) / (1.0 - decay)
        return cls(initial_charge, residual, a, energy_scale)

    def charge(self, E: float) -> float:
        """Return ``(Q - Q_res)(1 - exp(-a E / E_scale))``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            Cumulative emitted charge.
        """
        _require_nonnegative(E)
        amplitude = self.initial_charge - self.residual_charge
        return amplitude * -math.expm1(-self.a * E / self.energy_scale)

    def rate(self, E: float) -> float:
        """Return ``(a / E_scale)(Q - Q_res) exp(-a E / E_scale)``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            ``dQ_em/dE``; equals ``η_0 = a (Q - Q_res)/E_scale`` at ``E = 0``.
        """
        _require_nonnegative(E)
        amplitude = self.initial_charge - self.residual_charge
        return self.a * amplitude / self.energy_scale * math.exp(-self.a * E / self.energy_scale)


@dataclass(frozen=True, slots=True)
class FractionalLoss:
    """Fractional charge loss proportional to fractional mass loss, Eq. (4.41).

    ``dQ'/Q' = p dM'/M'`` with ``M' = M_0 - E`` gives
    ``Q_em(E) = Q [1 - (1 - E/M_0)^p]`` for ``0 <= E < M_0``. The choice ``p = 1/2``
    with ``M_0 = M_i`` keeps the state on the inner critical curve (Eq. (4.44)).

    Attributes
    ----------
    initial_charge
        Initial charge ``Q``.
    initial_mass
        Initial mass ``M_0`` (``M_i`` in the paper).
    p
        Exponent ``p > 0``.
    """

    initial_charge: float
    initial_mass: float
    p: float

    @classmethod
    def from_endpoint(
        cls, initial_charge: float, M_i: float, M_e: float, final_fraction: float
    ) -> "FractionalLoss":
        """Choose ``p`` so the path meets ``M_e(Q')`` with ``Q'/Q = s_f``, Eq. (4.43).

        Parameters
        ----------
        initial_charge
            Initial charge ``Q``.
        M_i
            Initial (inner-critical) mass.
        M_e
            Outer critical mass at the initial charge.
        final_fraction
            Charge fraction ``s_f`` remaining at coalescence.

        Returns
        -------
        FractionalLoss
            The profile with ``p = ln s_f / ln[(M_e/M_i) s_f^2]``.
        """
        p = math.log(final_fraction) / math.log(M_e / M_i * final_fraction**2)
        return cls(initial_charge, M_i, p)

    @classmethod
    def degeneracy_preserving(cls, initial_charge: float, M_i: float) -> "FractionalLoss":
        """Profile (4.44) that stays on the inner critical curve (``p = 1/2``).

        Parameters
        ----------
        initial_charge
            Initial charge ``Q``.
        M_i
            Initial mass on the inner critical curve.

        Returns
        -------
        FractionalLoss
            The ``p = 1/2`` profile.
        """
        return cls(initial_charge, M_i, 0.5)

    def _remaining_fraction(self, E: float) -> float:
        """Return ``1 - E / M_0`` after domain checks.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            The remaining mass fraction.

        Raises
        ------
        ValueError
            If ``E`` lies outside ``[0, M_0)``.
        """
        _require_nonnegative(E)
        if self.initial_mass <= E:
            msg = f"FractionalLoss is defined for E < M_0 = {self.initial_mass}, got {E}"
            raise ValueError(msg)
        return 1.0 - E / self.initial_mass

    def charge(self, E: float) -> float:
        """Return ``Q [1 - (1 - E/M_0)^p]``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            Cumulative emitted charge.
        """
        return self.initial_charge * (1.0 - math.pow(self._remaining_fraction(E), self.p))

    def rate(self, E: float) -> float:
        """Return ``(p Q / M_0)(1 - E/M_0)^(p-1)``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            ``dQ_em/dE``.
        """
        fraction = self._remaining_fraction(E)
        return self.p * self.initial_charge / self.initial_mass * math.pow(fraction, self.p - 1.0)


@dataclass(frozen=True, slots=True)
class ReturningProfile:
    """Channel that splits the inner horizon and re-merges it at ``E_r``, Eq. (4.45).

    ``Q_em(E) = Q [1 - sqrt(1 - E^2 / (M_i E_r))]`` for ``0 <= E <= E_r``, which gives
    the exact inner mass gap ``ΔM_i = E (1 - E/E_r)`` (Eq. (4.46)).

    Attributes
    ----------
    initial_charge
        Initial charge ``Q``.
    M_i
        Initial mass on the inner critical curve.
    return_energy
        ``E_r``; the paper requires ``0 < E_r < M_i - M_e``.
    """

    initial_charge: float
    M_i: float
    return_energy: float

    def _radicand(self, E: float) -> float:
        """Return ``1 - E^2 / (M_i E_r)`` after domain checks.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            The quantity under the square root.

        Raises
        ------
        ValueError
            If ``E`` lies outside ``[0, E_r]``.
        """
        _require_nonnegative(E)
        if self.return_energy < E:
            msg = f"ReturningProfile is defined for E <= E_r = {self.return_energy}, got {E}"
            raise ValueError(msg)
        return 1.0 - E**2 / (self.M_i * self.return_energy)

    def charge(self, E: float) -> float:
        """Return ``Q [1 - sqrt(1 - E^2/(M_i E_r))]``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            Cumulative emitted charge (starts at order ``E^2``).
        """
        return self.initial_charge * (1.0 - math.sqrt(self._radicand(E)))

    def rate(self, E: float) -> float:
        """Return ``Q E / (M_i E_r sqrt(1 - E^2/(M_i E_r)))``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            ``dQ_em/dE`` (zero at ``E = 0``).
        """
        radicand = self._radicand(E)
        return self.initial_charge * E / (self.M_i * self.return_energy * math.sqrt(radicand))
