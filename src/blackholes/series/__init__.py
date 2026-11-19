"""Formal power-series tools used to invert horizon conditions.

The horizon displacement caused by emitting an energy ``E`` is the local inverse of
the mass–radius relation (paper Sec. 2.2). The routines here perform that inversion
order by order with exact (SymPy) arithmetic, so the results can be compared
symbol-for-symbol with the closed forms printed in the paper:

* :func:`lagrange_burmann_inverse` — simple horizons (Eq. (2.16), App. A).
* :func:`puiseux_inverse`, :func:`branch_coefficients` — degenerate horizons of
  any multiplicity (Eq. (C.5), App. C).
* :func:`compose` — series composition, e.g. entropy along the inverse (Eq. (2.17)).
"""

from blackholes.series.lagrange_burmann import compose, lagrange_burmann_inverse
from blackholes.series.puiseux import branch_coefficients, puiseux_inverse

__all__ = ["branch_coefficients", "compose", "lagrange_burmann_inverse", "puiseux_inverse"]
