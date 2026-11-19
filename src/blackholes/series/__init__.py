"""Formal power-series tools used to invert horizon conditions.

The horizon displacement caused by emitting an energy ``E`` is the local inverse of
the mass–radius relation (paper Sec. 2.2). The routines here perform that inversion
order by order with exact (SymPy) arithmetic, so the results can be compared
symbol-for-symbol with the closed forms printed in the paper.
"""

from blackholes.series.lagrange_burmann import compose, lagrange_burmann_inverse

__all__ = ["compose", "lagrange_burmann_inverse"]
