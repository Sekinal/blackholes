"""Companion numerics for finite-energy Hawking emission and horizon kinematics.

This package reproduces and extends the calculations of

    M. A. Aguirre Cortés, M. Bravo-Gaete, J. A. Méndez-Zavaleta and E. Rojas Marcial,
    "Horizon kinematics and the fine structure induced by finite-energy Hawking
    radiation", arXiv:2611.23376.

Subpackages
-----------
series
    Formal power-series tools: Lagrange–Bürmann inversion (paper Sec. 2.2 and
    Appendix A) and, later, Puiseux expansions about degenerate horizons
    (Sec. 3 and Appendix C).

Conventions
-----------
Units are ħ = 1 throughout, as in the paper. Symbols in code follow the paper's
notation (``M``, ``Q``, ``E``, ``T_H`` ...), and every function that implements an
equation cites it as "Eq. (x.y)" in its docstring.
"""

__all__: list[str] = []
