"""Concrete black-hole families on which the general series machinery is applied.

* :class:`NonlinearBTZ` — the logarithmically charged BTZ family of paper Sec. 4
  (Ref. [19] of the paper), with its mass function, thermodynamics and critical
  (degenerate-horizon) curves.
"""

from blackholes.models.nonlinear_btz import NonlinearBTZ

__all__ = ["NonlinearBTZ"]
