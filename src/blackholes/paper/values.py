"""Recompute every number quoted in the paper and compare with the printed value."""

from dataclasses import dataclass

from blackholes.analysis.convergence import endpoint_truncation_errors
from blackholes.analysis.kpw_series import neutral_entropy_series
from blackholes.emission import FractionalLoss
from blackholes.paper.configurations import (
    FIG2_CHARGE,
    FIG2_MODEL,
    QUOTED_VALUES,
    fig3_paths,
)

__all__ = ["Comparison", "compare_with_paper", "reproduced_values"]


def reproduced_values() -> dict[str, float]:
    """Recompute each entry of :data:`~blackholes.paper.configurations.QUOTED_VALUES`.

    Returns
    -------
    dict[str, float]
        Same keys as ``QUOTED_VALUES``, with independently computed values.

    Raises
    ------
    RuntimeError
        If the Fig. 3 configuration was edited so that the delayed path no longer
        returns to the inner critical curve or the fractional profile changed type.
    """
    model, Q = FIG2_MODEL, FIG2_CHARGE
    M_i, M_e = model.critical_masses(Q)
    r_plus = model.horizons(M_i, Q)[-1]
    r_min = model.stationary_radii(Q)[1]
    entropy_at_Ec = model.entropy(r_min) - model.entropy(r_plus)
    series = neutral_entropy_series(model, r_plus, Q, order=4)
    quadratic, quartic = endpoint_truncation_errors(series, M_i - M_e, entropy_at_Ec, [2, 4])

    paths = fig3_paths()
    delayed = paths["linear_three_quarters"]
    E_s = delayed.inner_recovery_energy(delayed.coalescence_energy())
    fractional = paths["fractional"].profile
    if E_s is None or not isinstance(fractional, FractionalLoss):  # pragma: no cover
        msg = "Fig. 3 configuration no longer matches the paper"
        raise RuntimeError(msg)

    return {
        "fig2_neutral_Ec": M_i - M_e,
        "fig6_entropy_at_Ec": entropy_at_Ec,
        "fig6_quadratic_relative_error": float(quadratic),
        "fig6_quartic_relative_error": float(quartic),
        "fig4_Ec_neutral": paths["neutral"].coalescence_energy(),
        "fig4_Ec_linear_half": paths["linear_half"].coalescence_energy(),
        "fig4_Ec_linear_three_quarters": delayed.coalescence_energy(),
        "fig4_Ec_exponential": paths["exponential"].coalescence_energy(),
        "fig4_Ec_fractional": paths["fractional"].coalescence_energy(),
        "fig4_Es_linear_three_quarters": E_s,
        "fig3_fractional_p": fractional.p,
    }


@dataclass(frozen=True, slots=True)
class Comparison:
    """One printed number next to its reproduction.

    Attributes
    ----------
    name
        Key in ``QUOTED_VALUES``.
    quoted
        Value printed in the paper.
    tolerance
        Half a unit in the last printed digit.
    reproduced
        Value computed by this package.
    """

    name: str
    quoted: float
    tolerance: float
    reproduced: float

    @property
    def agrees(self) -> bool:
        """Whether the reproduction rounds to the printed value.

        Returns
        -------
        bool
            ``|reproduced - quoted| <= tolerance``.
        """
        return abs(self.reproduced - self.quoted) <= self.tolerance


def compare_with_paper() -> list[Comparison]:
    """Pair every quoted number with its reproduction.

    Returns
    -------
    list[Comparison]
        One row per entry of ``QUOTED_VALUES``, in its order.
    """
    values = reproduced_values()
    return [
        Comparison(name, quoted, tolerance, values[name])
        for name, (quoted, tolerance) in QUOTED_VALUES.items()
    ]
