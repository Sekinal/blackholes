# blackholes

Reproducible, independently verified numerics for

> M. A. Aguirre Cortés, M. Bravo-Gaete, J. A. Méndez-Zavaleta, E. Rojas Marcial,
> *Horizon kinematics and the fine structure induced by finite-energy Hawking radiation*,
> [arXiv:2611.23376](https://arxiv.org/abs/2611.23376) (2026).

The goal is a small, well-tested library that regenerates every figure and quoted
number of the paper, exposes the series machinery (Lagrange–Bürmann and Puiseux
inversion of the horizon condition) to arbitrary order, and extends the analysis
where the paper stops (convergence radii, resummed endpoint behaviour, and later
greybody factors and evaporation trajectories).

This is an independent companion project; it is not affiliated with the authors.

## Status

| Component | Paper reference | State |
|---|---|---|
| Lagrange–Bürmann inversion, series composition | Sec. 2.2, App. A | done, tested against Eq. (A.5) |
| Puiseux expansions about degenerate horizons | Sec. 3, App. C | planned |
| Nonlinearly charged BTZ model (mass function, critical curves) | Sec. 4, App. D | planned |
| Figure and number reproduction (Figs. 2–6) | Secs. 4–5 | planned |
| Convergence radius and endpoint resummation of the KPW series | Fig. 6 | planned |

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create the environment from uv.lock
uv run pre-commit install
uv run poe check        # every quality gate: format, lint, types, docstrings, spelling, tests
```

Other tasks: `uv run poe --help` (e.g. `poe fmt`, `poe test-fast`).

## Layout

```
src/blackholes/
    series/              formal power-series tools
        lagrange_burmann.py
tests/                   pytest + Hypothesis property tests (beartype-instrumented)
```

## Quality gates

Every commit must pass, locally (pre-commit) and in CI:

- **ruff**: formatting and nearly every lint rule (`select = ["ALL"]`); each opt-out is justified in `pyproject.toml`.
- **mypy --strict**: plus extra error codes; `# type: ignore` requires an error code.
- **interrogate**: 100% docstring coverage, including private helpers.
- **pydoclint**: NumPy-style docstrings must match signatures, returns and raises.
- **codespell**: spelling in code and prose.
- **pytest**: warnings are errors, test order randomized, doctests run, 95% branch coverage minimum.
- **beartype**: every function in the package is type-checked at runtime during tests.
- **Hypothesis**: property-based tests for mathematical invariants (e.g. inversion round trips).

Physics conventions: ħ = 1; code symbols follow the paper (`M`, `Q`, `E`, `T_H`);
functions implementing an equation cite it as "Eq. (x.y)" in the docstring.
