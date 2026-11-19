"""Shared plotting style: a validated categorical palette and recessive axes.

The five categorical colours (in fixed order, never cycled) passed the
colour-vision-deficiency and lightness checks of the reference data-viz palette
for adjacent line series. Three of them sit below 3:1 contrast on white, so every
series also carries a distinct line style and a legend or direct label: identity
never depends on colour alone, and figures survive greyscale printing.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from types import MappingProxyType

import matplotlib as mpl
from matplotlib.rcsetup import cycler

__all__ = ["LINESTYLES", "SERIES", "TEXT", "figure_style"]

SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
"""Categorical slots 1-5 (blue, orange, aqua, yellow, magenta), assigned in order."""

LINESTYLES: tuple[str | tuple[int, tuple[int, ...]], ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1, 1, 1, 1, 1)),
)
"""Secondary encoding paired one-to-one with :data:`SERIES`."""

TEXT = MappingProxyType({"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e4e3df"})
"""Ink colours for text, axes and the recessive grid."""


@contextmanager
def figure_style() -> Iterator[None]:
    """Temporarily apply the project's matplotlib style.

    Yields
    ------
    None
        Control returns to the caller with the style active.
    """
    # Passed inline so mypy checks every key against matplotlib's typed rcParams.
    with mpl.rc_context(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.edgecolor": TEXT["secondary"],
            "axes.labelcolor": TEXT["primary"],
            "axes.titlecolor": TEXT["primary"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": TEXT["grid"],
            "grid.linewidth": 0.6,
            "xtick.color": TEXT["secondary"],
            "ytick.color": TEXT["secondary"],
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "axes.prop_cycle": cycler(color=list(SERIES)),
        }
    ):
        yield
