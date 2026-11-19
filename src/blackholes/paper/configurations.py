"""Exact parameter sets of the paper's figures and the numbers quoted in them.

Keeping every configuration in one module means tests, scripts and figures can
never drift apart: they all import the same objects.

* Figs. 2, 5, 6: ``κ = ℓ = γ = q = 1``, ``β = 6/5``.
* Figs. 3, 4: ``κ = ℓ = 1``, ``β = 27/20``, ``q = γ = 4/π`` (so ``Q = 8``).
"""

import math
from types import MappingProxyType

from blackholes.emission import (
    ChargeProfile,
    EmissionPath,
    ExponentialDepletion,
    FractionalLoss,
    LinearProfile,
    NeutralProfile,
    ReturningProfile,
)
from blackholes.models import NonlinearBTZ

__all__ = [
    "FIG2_CHARGE",
    "FIG2_MODEL",
    "FIG3_CHARGE",
    "FIG3_MODEL",
    "QUOTED_VALUES",
    "fig3_paths",
    "fig5_paths",
]

FIG2_MODEL = NonlinearBTZ(kappa=1.0, ell=1.0, beta=6 / 5, gamma=1.0)
"""Couplings of Figs. 2, 5 and 6."""

FIG2_CHARGE = NonlinearBTZ.charge_from_q(1.0)
"""Initial charge ``Q = 2π`` (``q = 1``) of Figs. 2, 5 and 6."""

FIG3_MODEL = NonlinearBTZ(kappa=1.0, ell=1.0, beta=27 / 20, gamma=4 / math.pi)
"""Couplings of Figs. 3 and 4."""

FIG3_CHARGE = NonlinearBTZ.charge_from_q(4 / math.pi)
"""Initial charge ``Q = 8`` (``q = 4/π``) of Figs. 3 and 4."""

QUOTED_VALUES = MappingProxyType(
    {
        "fig2_neutral_Ec": (1.856, 5e-4),
        "fig6_entropy_at_Ec": (-43.209, 5e-4),
        "fig6_quadratic_relative_error": (0.40, 5e-3),
        "fig6_quartic_relative_error": (0.29, 5e-3),
        "fig4_Ec_neutral": (3.9347, 5e-5),
        "fig4_Ec_linear_half": (4.0606, 5e-5),
        "fig4_Ec_linear_three_quarters": (4.0930, 5e-5),
        "fig4_Ec_exponential": (3.9787, 5e-5),
        "fig4_Ec_fractional": (4.0323, 5e-5),
        "fig4_Es_linear_three_quarters": (3.6477, 5e-5),
        "fig3_fractional_p": (0.1063, 5e-5),
    }
)
"""Numbers printed in the paper, as ``name -> (value, half a unit in the last digit)``."""


def fig3_paths() -> dict[str, EmissionPath]:
    """Build the five emission paths of Figs. 3 and 4.

    All start on the inner critical curve, ``(Q, M_i(Q))`` with ``Q = 8``.

    Returns
    -------
    dict[str, EmissionPath]
        Keys ``neutral``, ``linear_half`` (``η M_i/Q = 1/2``),
        ``linear_three_quarters`` (``3/4``), ``exponential`` (``a = 3``,
        ``Q_f/Q = 0.86``) and ``fractional`` (``Q_f/Q = 0.65``).
    """
    model, Q = FIG3_MODEL, FIG3_CHARGE
    M_i, M_e = model.critical_masses(Q)

    def path(profile: ChargeProfile) -> EmissionPath:
        """Wrap a profile into a path from the common initial state.

        Parameters
        ----------
        profile
            A charge profile.

        Returns
        -------
        EmissionPath
            The path starting at ``(M_i, Q)``.
        """
        return EmissionPath(model, M_i, Q, profile)

    exponential_fraction = 0.86
    fractional_fraction = 0.65
    return {
        "neutral": path(NeutralProfile()),
        "linear_half": path(LinearProfile(0.5 * Q / M_i)),
        "linear_three_quarters": path(LinearProfile(0.75 * Q / M_i)),
        "exponential": path(
            ExponentialDepletion.from_endpoint(
                Q,
                a=3.0,
                final_fraction=exponential_fraction,
                energy_scale=M_i - M_e * exponential_fraction**2,
            )
        ),
        "fractional": path(FractionalLoss.from_endpoint(Q, M_i, M_e, fractional_fraction)),
    }


def fig5_paths() -> dict[str, EmissionPath]:
    """Build the two channels of Fig. 5 sharing initial and final states.

    Returns
    -------
    dict[str, EmissionPath]
        ``preserving`` (Eq. (4.44), stays on the inner critical curve) and
        ``returning`` (Eq. (4.45) with ``E_r = (M_i - M_e)/2``).
    """
    model, Q = FIG2_MODEL, FIG2_CHARGE
    M_i, M_e = model.critical_masses(Q)
    return {
        "preserving": EmissionPath(model, M_i, Q, FractionalLoss.degeneracy_preserving(Q, M_i)),
        "returning": EmissionPath(model, M_i, Q, ReturningProfile(Q, M_i, 0.5 * (M_i - M_e))),
    }
