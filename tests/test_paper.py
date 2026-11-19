"""Reproduction of the paper's quoted numbers and figures."""

import matplotlib as mpl
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from blackholes.paper import figures
from blackholes.paper.values import compare_with_paper


def test_every_quoted_number_is_reproduced() -> None:
    """Each number printed in the paper rounds to its reproduction."""
    rows = compare_with_paper()
    assert rows, "no quoted values configured"
    failures = [row for row in rows if not row.agrees]
    assert not failures, failures


@pytest.mark.slow
@pytest.mark.parametrize(
    ("build", "panels"),
    [
        (figures.figure_2_mass_function, 1),
        (figures.figure_3_state_space, 1),
        (figures.figure_4_horizon_histories, 5),
        (figures.figure_5_returning_channel, 2),
        (figures.figure_6_convergence, 3),
    ],
)
def test_figures_build(build: object, panels: int) -> None:
    """Every figure builds with the expected number of panels.

    Parameters
    ----------
    build
        Figure factory.
    panels
        Expected number of axes.
    """
    assert callable(build)
    fig = build()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == panels
    plt.close(fig)
