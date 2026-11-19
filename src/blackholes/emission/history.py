"""Horizon histories along an emission channel (paper Secs. 3.2, 4.2-4.3, 5.1).

Given a model, an initial state ``(M, Q)`` and a charge profile, an
:class:`EmissionPath` answers the questions the paper asks of each channel:

* which stationary state is reached after emitting ``E`` (Eq. (2.24));
* where it sits relative to the two critical curves (mass gaps, Eq. (4.36));
* where the horizons are, and how fast the event horizon moves (Eq. (4.37));
* the leading KPW exponent ``ΔS(E)`` (Eqs. (2.27), (4.15));
* the horizon-energy factor ``1 - Φ_+ dQ_em/dE`` of Eq. (2.28);
* the energies of outer coalescence ``E_c`` (Eq. (5.1)) and of any return to the
  inner critical curve (``E_s``, Eq. (4.39)).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from blackholes.emission.channels import ChargeProfile
from blackholes.models import NonlinearBTZ

__all__ = ["EmissionPath"]


@dataclass(frozen=True, slots=True)
class EmissionPath:
    """A quasistationary emission channel through the nonlinear BTZ family.

    Attributes
    ----------
    model
        The fixed-coupling black-hole family.
    initial_mass
        Initial physical mass ``M``.
    initial_charge
        Initial physical charge ``Q`` (sign sector ``Q / γ > 0``).
    profile
        The charge profile ``Q_em(E)``.
    """

    model: NonlinearBTZ
    initial_mass: float
    initial_charge: float
    profile: ChargeProfile

    # ------------------------------------------------------------ state space --
    def state(self, E: float) -> tuple[float, float]:
        """Return the black-hole state ``(M', Q')`` after emitting ``E``, Eq. (2.24).

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        tuple[float, float]
            ``(M - E, Q - Q_em(E))``.
        """
        return self.initial_mass - E, self.initial_charge - self.profile.charge(E)

    def mass_gaps(self, E: float) -> tuple[float, float]:
        """Signed gaps to the critical curves, ``(ΔM_i, ΔM_e)``, Eq. (4.36).

        ``ΔM_l = M_l(Q') - M'``. The three-horizon strip is ``ΔM_i > 0 > ΔM_e``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        tuple[float, float]
            ``(M_i(Q') - M', M_e(Q') - M')``.
        """
        M_prime, Q_prime = self.state(E)
        M_i, M_e = self.model.critical_masses(Q_prime)
        return M_i - M_prime, M_e - M_prime

    def horizons(self, E: float) -> tuple[float, ...]:
        """All positive horizons of the state reached after emitting ``E``.

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        tuple[float, ...]
            Sorted radii (``r_-, r_0, R_+`` in the three-horizon strip).
        """
        return self.model.horizons(*self.state(E))

    def outer_horizon(self, E: float) -> float:
        """Tracked event horizon ``R_+(E)`` (largest root), valid for ``E < E_c``.

        Parameters
        ----------
        E
            Emitted energy, below the outer coalescence energy.

        Returns
        -------
        float
            The event-horizon radius.
        """
        return self.horizons(E)[-1]

    # ---------------------------------------------------------- KPW quantities --
    def entropy_change(self, E: float) -> float:
        """Leading KPW exponent ``ln Γ ≃ ΔS = S(R_+(E)) - S(R_+(0))``, Eqs. (2.27), (4.15).

        Parameters
        ----------
        E
            Emitted energy (``E <= E_c``; at ``E_c`` the limit through simple
            horizons equals the value at the degenerate horizon).

        Returns
        -------
        float
            The entropy difference (negative for ordinary emission).
        """
        return self.model.entropy(self.outer_horizon(E)) - self.model.entropy(
            self.outer_horizon(0.0)
        )

    def horizon_energy_factor(self, E: float) -> float:
        """Return ``1 - Φ_+ dQ_em/dE``; ordinary channels require it positive, Eq. (2.28).

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        float
            The factor multiplying ``dE / T_H`` in the KPW integrand.
        """
        _, Q_prime = self.state(E)
        return 1.0 - self.model.potential(self.outer_horizon(E), Q_prime) * self.profile.rate(E)

    def outer_horizon_velocity(self, E: float) -> float:
        """Velocity ``dR_+/dE = -(1 - Φ dQ_em/dE) / ∂_r m`` of the event horizon, Eq. (4.37).

        Parameters
        ----------
        E
            Emitted energy (the outer horizon must be simple).

        Returns
        -------
        float
            ``dR_+/dE``.
        """
        _, Q_prime = self.state(E)
        r = self.outer_horizon(E)
        return -self.horizon_energy_factor(E) / self.model.radial_derivative(r, Q_prime, 1)

    # ------------------------------------------------------- critical crossings --
    def _first_crossing(self, gap_index: int, E_max: float, samples: int) -> float | None:
        """Smallest ``E`` in ``(0, E_max]`` where a mass gap changes sign.

        Parameters
        ----------
        gap_index
            ``0`` for ``ΔM_i``, ``1`` for ``ΔM_e``.
        E_max
            Upper end of the search interval.
        samples
            Number of grid intervals used to bracket the first sign change.

        Returns
        -------
        float | None
            The crossing energy (polished with Brent's method), or ``None``.
        """

        def gap(E: float) -> float:
            """Evaluate the selected mass gap.

            Parameters
            ----------
            E
                Emitted energy.

            Returns
            -------
            float
                ``ΔM_i(E)`` or ``ΔM_e(E)``.
            """
            return self.mass_gaps(E)[gap_index]

        grid = np.linspace(0.0, E_max, samples + 1)[1:]
        previous_E, previous_gap = float(grid[0]), gap(float(grid[0]))
        for E in grid[1:]:
            current = gap(float(E))
            if previous_gap * current <= 0:
                return float(brentq(gap, previous_E, float(E), xtol=1e-14))
            previous_E, previous_gap = float(E), current
        return None

    def coalescence_energy(self, E_max: float | None = None, samples: int = 400) -> float:
        """Energy ``E_c`` at which the path meets the outer critical curve, Eq. (5.1).

        Parameters
        ----------
        E_max
            Search limit; defaults to just below the initial mass.
        samples
            Bracketing resolution.

        Returns
        -------
        float
            ``E_c`` solving ``M - E_c = M_e(Q - Q_em(E_c))``.

        Raises
        ------
        ValueError
            If the path never reaches the outer critical curve before ``E_max``.
        """
        limit = (1.0 - 1e-9) * self.initial_mass if E_max is None else E_max
        crossing = self._first_crossing(1, limit, samples)
        if crossing is None:
            msg = "the path does not reach the outer critical curve"
            raise ValueError(msg)
        return crossing

    def inner_recovery_energy(self, E_max: float, samples: int = 400) -> float | None:
        """First ``E > 0`` where the path returns to the inner critical curve, e.g. Eq. (4.39).

        Parameters
        ----------
        E_max
            Search limit (typically the coalescence energy).
        samples
            Bracketing resolution.

        Returns
        -------
        float | None
            The energy ``E_s`` at which ``ΔM_i`` changes sign, or ``None``.
        """
        return self._first_crossing(0, E_max, samples)

    def in_three_horizon_strip(self, E: float) -> bool:
        """Return whether ``M_e(Q') < M' < M_i(Q')`` (Eq. (4.34)).

        Parameters
        ----------
        E
            Emitted energy.

        Returns
        -------
        bool
            ``True`` inside the open strip.
        """
        gap_i, gap_e = self.mass_gaps(E)
        return gap_i > 0 > gap_e
