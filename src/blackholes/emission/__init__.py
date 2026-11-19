"""Emission channels and the horizon histories they select (paper Secs. 2.4, 4.2-4.3).

* :mod:`~blackholes.emission.channels` — charge profiles ``Q_em(E)``.
* :mod:`~blackholes.emission.history` — :class:`EmissionPath`: states, mass gaps,
  horizons, KPW exponent and critical crossings along a channel.
"""

from blackholes.emission.channels import (
    ChargeProfile,
    ExponentialDepletion,
    FractionalLoss,
    LinearProfile,
    NeutralProfile,
    ReturningProfile,
)
from blackholes.emission.history import EmissionPath

__all__ = [
    "ChargeProfile",
    "EmissionPath",
    "ExponentialDepletion",
    "FractionalLoss",
    "LinearProfile",
    "NeutralProfile",
    "ReturningProfile",
]
