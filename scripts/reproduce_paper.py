"""Regenerate every figure and quoted number of arXiv:2611.23376.

Usage::

    uv run python scripts/reproduce_paper.py [--output-dir outputs]

Prints a table comparing each number printed in the paper with its reproduction
(exit status 1 if any disagrees) and writes the figures as PDF and PNG.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from blackholes.paper import figures
from blackholes.paper.values import compare_with_paper

FIGURES = {
    "fig2_mass_function": figures.figure_2_mass_function,
    "fig3_state_space": figures.figure_3_state_space,
    "fig4_horizon_histories": figures.figure_4_horizon_histories,
    "fig5_returning_channel": figures.figure_5_returning_channel,
    "fig6_convergence": figures.figure_6_convergence,
}


def main() -> int:
    """Run the reproduction and report.

    Returns
    -------
    int
        Process exit status: 0 if every quoted number is reproduced.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    rows = compare_with_paper()
    width = max(len(row.name) for row in rows)
    print(f"{'quantity':<{width}}  {'paper':>10}  {'reproduced':>14}  ok")
    for row in rows:
        mark = "yes" if row.agrees else "NO"
        print(f"{row.name:<{width}}  {row.quoted:>10.4f}  {row.reproduced:>14.8f}  {mark}")

    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, build in FIGURES.items():
        fig = build()
        for suffix in ("pdf", "png"):
            fig.savefig(figure_dir / f"{name}.{suffix}")
        plt.close(fig)
        print(f"wrote {figure_dir / name}.{{pdf,png}}")

    return 0 if all(row.agrees for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
