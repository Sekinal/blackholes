"""Regenerate the paper's figures (2-6) plus the new convergence figure.

Every function returns a :class:`matplotlib.figure.Figure` and has no side
effects; saving is left to ``scripts/reproduce_paper.py``. Data come from the same
configuration objects the test suite checks, so a figure cannot silently drift
from the verified numbers.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from blackholes.analysis.convergence import domb_sykes_fit
from blackholes.analysis.endpoint import build_neutral_endpoint_approximant
from blackholes.analysis.kpw_series import neutral_entropy_series
from blackholes.emission import EmissionPath
from blackholes.paper.configurations import (
    FIG2_CHARGE,
    FIG2_MODEL,
    FIG3_CHARGE,
    FIG3_MODEL,
    fig3_paths,
    fig5_paths,
)
from blackholes.paper.style import LINESTYLES, SERIES, TEXT, figure_style
from blackholes.series.numeric import FloatArray, evaluate_series

__all__ = [
    "figure_2_mass_function",
    "figure_3_state_space",
    "figure_4_horizon_histories",
    "figure_5_returning_channel",
    "figure_6_convergence",
]

_PATH_LABELS = {
    "neutral": "neutral",
    "linear_half": r"linear, $\eta M_i/Q=1/2$",
    "linear_three_quarters": r"linear, $\eta M_i/Q=3/4$",
    "exponential": "exponential depletion",
    "fractional": "fractional loss",
}
_HORIZON_LABELS = (r"$r_-$ (innermost)", r"$r_0$ (middle)", r"$R_+$ (event)")


def _horizon_tracks(path: EmissionPath, energies: FloatArray) -> FloatArray:
    """Sort the horizons along a path into (innermost, middle, event) tracks.

    Parameters
    ----------
    path
        The emission path.
    energies
        Emitted energies, all below the coalescence energy.

    Returns
    -------
    FloatArray
        Shape ``(len(energies), 3)``; missing horizons are NaN. A degenerate
        inner root fills both inner tracks; a lone root is the event horizon.
    """
    tracks = np.full((energies.size, 3), np.nan)
    for i, E in enumerate(energies):
        match path.horizons(float(E)):
            case (inner, middle, event):  # the three-horizon strip
                tracks[i] = (inner, middle, event)
            case (degenerate, event):  # degenerate inner pair + event horizon
                tracks[i] = (degenerate, degenerate, event)
            case roots:  # a single (event) horizon
                tracks[i, 2] = roots[-1]
    return tracks


def _critical_curves(ax: Axes, M_i: float, M_e: float, Q0: float, Q_max: float) -> None:
    """Draw ``M_i(Q)``, ``M_e(Q)`` and shade the three-horizon strip (Eq. (4.34)).

    Parameters
    ----------
    ax
        Target axes.
    M_i
        Inner critical mass at ``Q0``.
    M_e
        Outer critical mass at ``Q0``.
    Q0
        Reference charge.
    Q_max
        Right end of the charge axis.
    """
    charges = np.linspace(0.0, Q_max, 300)
    upper, lower = M_i * (charges / Q0) ** 2, M_e * (charges / Q0) ** 2  # homogeneity (4.29)
    ax.fill_between(charges, lower, upper, color=TEXT["grid"], alpha=0.7, linewidth=0)
    ax.plot(charges, upper, color=TEXT["primary"], linewidth=1.0)
    ax.plot(charges, lower, color=TEXT["secondary"], linewidth=1.0)
    ax.annotate(
        r"$M_i(Q)$",
        (charges[-1], upper[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=TEXT["primary"],
    )
    ax.annotate(
        r"$M_e(Q)$",
        (charges[-1], lower[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=TEXT["secondary"],
    )


def figure_2_mass_function() -> Figure:
    """Fig. 2: the mass function ``m(r, Q)`` and its two critical levels.

    Returns
    -------
    Figure
        One panel: ``m(r)``, the levels ``M_i`` and ``M_e``, and the horizons on
        each level.
    """
    model, Q = FIG2_MODEL, FIG2_CHARGE
    M_i, M_e = model.critical_masses(Q)
    radii = np.linspace(1e-3, 4.2, 500)
    masses = [model.mass_function(float(r), Q) for r in radii]
    with figure_style():
        fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
        ax.plot(radii, masses, color=SERIES[0])
        for level, name in ((M_i, "M_i"), (M_e, "M_e")):
            ax.axhline(level, color=TEXT["secondary"], linewidth=0.8, linestyle="--")
            ax.annotate(
                rf"${name}$",
                (radii[-1], level),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=TEXT["secondary"],
            )
            roots = model.horizons(level, Q)
            ax.plot(roots, [level] * len(roots), "o", color=SERIES[1], markersize=5, zorder=3)
        ax.set_xlim(0.0, 4.75)
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$m(r, Q)$")
        ax.set_title(rf"Mass function, $\mathcal{{E}}_c = M_i - M_e = {M_i - M_e:.3f}$", loc="left")
    return fig


def figure_3_state_space() -> Figure:
    """Fig. 3: critical curves and the five emission paths in the (Q, M) plane.

    Returns
    -------
    Figure
        One panel with the shaded three-horizon strip and five paths.
    """
    M_i, M_e = FIG3_MODEL.critical_masses(FIG3_CHARGE)
    with figure_style():
        fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
        _critical_curves(ax, M_i, M_e, FIG3_CHARGE, 1.15 * FIG3_CHARGE)
        for slot, (key, path) in enumerate(fig3_paths().items()):
            energies = np.linspace(0.0, path.coalescence_energy(), 200)
            states = np.array([path.state(float(E)) for E in energies])
            ax.plot(
                states[:, 1],
                states[:, 0],
                color=SERIES[slot],
                linestyle=LINESTYLES[slot],
                label=_PATH_LABELS[key],
            )
            ax.plot(states[-1, 1], states[-1, 0], "o", color=SERIES[slot], markersize=5)
        ax.set_xlabel(r"charge $Q$")
        ax.set_ylabel(r"mass $M$")
        ax.set_xlim(left=1.5)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left")
        ax.set_title(
            "Emission paths from the inner critical curve (shaded: three horizons)", loc="left"
        )
    return fig


def figure_4_horizon_histories() -> Figure:
    """Fig. 4: horizon radii along each of the five paths (small multiples).

    Returns
    -------
    Figure
        Five panels sharing axes; radii normalised by ``R_+(0)``, energy by ``E_c``.
    """
    paths = fig3_paths()
    with figure_style():
        fig, axes = plt.subplots(
            1, 5, figsize=(11.0, 3.2), sharex=True, sharey=True, layout="constrained"
        )
        for ax, (key, path) in zip(axes, paths.items(), strict=True):
            E_c = path.coalescence_energy()
            energies = np.linspace(0.0, E_c * (1 - 1e-9), 240)
            tracks = _horizon_tracks(path, energies) / path.outer_horizon(0.0)
            for slot in range(3):
                ax.plot(
                    energies / E_c,
                    tracks[:, slot],
                    color=SERIES[slot],
                    linestyle=LINESTYLES[slot],
                    label=_HORIZON_LABELS[slot],
                )
            ax.set_title(f"{_PATH_LABELS[key]}\n$\\mathcal{{E}}_c = {E_c:.4f}$", loc="left")
            r_degenerate = path.horizons(0.0)[0] / path.outer_horizon(0.0)
            ax.plot(0.0, r_degenerate, "o", color=TEXT["primary"], markersize=4, zorder=3)
            ax.set_xlabel(r"$\mathcal{E}/\mathcal{E}_c$")
        axes[0].set_ylabel(r"$r / R_+(0)$")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
    return fig


def figure_5_returning_channel() -> Figure:
    """Fig. 5: degeneracy-preserving vs. returning channels with a common endpoint.

    Returns
    -------
    Figure
        (a) both paths in the (Q, M) plane; (b) horizons along the returning one.
    """
    model, Q = FIG2_MODEL, FIG2_CHARGE
    M_i, M_e = model.critical_masses(Q)
    paths = fig5_paths()
    E_r = 0.5 * (M_i - M_e)
    energies = np.linspace(0.0, E_r, 200)
    with figure_style():
        fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.4), layout="constrained")
        _critical_curves(left, M_i, M_e, Q, 1.1 * Q)
        for slot, key in enumerate(("preserving", "returning")):
            states = np.array([paths[key].state(float(E)) for E in energies])
            left.plot(
                states[:, 1],
                states[:, 0],
                color=SERIES[slot],
                linestyle=LINESTYLES[slot + 1],
                label=f"{key} channel",
            )
        left.set_xlim(0.75 * Q, 1.1 * Q)
        Q_label = 0.8 * Q
        left.annotate(
            r"$M_i(Q)$",
            (Q_label, M_i * (Q_label / Q) ** 2),
            xytext=(-6, 8),
            textcoords="offset points",
            ha="right",
            color=TEXT["primary"],
        )
        left.set_ylim(0.5 * M_e, 1.1 * M_i)
        left.set_xlabel(r"charge $Q$")
        left.set_ylabel(r"mass $M$")
        left.legend(loc="upper left")
        left.set_title("(a) same endpoints, different interiors", loc="left")
        tracks = _horizon_tracks(paths["returning"], energies)
        for slot in range(3):
            right.plot(
                energies / E_r,
                tracks[:, slot],
                color=SERIES[slot],
                linestyle=LINESTYLES[slot],
                label=_HORIZON_LABELS[slot],
            )
        right.set_xlabel(r"$\mathcal{E}/\mathcal{E}_r$")
        right.set_ylabel(r"$r$")
        right.legend(loc="center right")
        right.set_title("(b) horizons along the returning channel", loc="left")
    return fig


def figure_6_convergence() -> Figure:
    """Fig. 6 extended: the KPW exponent, its truncations, and the uniform approximant.

    Returns
    -------
    Figure
        (a) ``ΔS(E)`` with the thermal tangent, the paper's quadratic/quartic
        truncations and the two-point approximant; (b) absolute errors on a log
        scale; (c) the Domb–Sykes plot locating the singularity at ``E_c``.
    """
    model, Q = FIG2_MODEL, FIG2_CHARGE
    M_i, M_e = model.critical_masses(Q)
    E_c = M_i - M_e
    r_plus = model.horizons(M_i, Q)[-1]
    series = neutral_entropy_series(model, r_plus, Q, order=70)
    energies = np.linspace(0.0, E_c * (1 - 1e-6), 240)
    exact = np.array(
        [
            model.entropy(model.horizons(M_i - float(E), Q)[-1]) - model.entropy(r_plus)
            for E in energies
        ]
    )
    approximants = {K: build_neutral_endpoint_approximant(model, M_i, Q, K) for K in (4, 8)}

    def truncated(order: int) -> FloatArray:
        """Evaluate the ``order``-term Maclaurin truncation on the grid.

        Parameters
        ----------
        order
            Number of terms.

        Returns
        -------
        FloatArray
            Truncated ``ΔS`` values.
        """
        return np.array([evaluate_series(series[:order], float(E)) for E in energies])

    x = energies / E_c
    with figure_style():
        fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(12.0, 3.6), layout="constrained")
        ax_a.plot(x, exact, color=SERIES[0], label=r"exact $\Delta S$")
        ax_a.plot(
            x,
            series[0] * energies,
            color=SERIES[1],
            linestyle=LINESTYLES[1],
            label=r"thermal $-\mathcal{E}/T_H$",
        )
        ax_a.plot(
            x, truncated(2), color=SERIES[2], linestyle=LINESTYLES[2], label="paper: 2nd order"
        )
        ax_a.plot(
            x, truncated(4), color=SERIES[3], linestyle=LINESTYLES[3], label="paper: 4th order"
        )
        ax_a.plot(
            x[::16],
            [approximants[8].entropy_change(float(E)) for E in energies[::16]],
            "o",
            color=SERIES[4],
            markersize=4,
            label=r"two-point in $t$, $K=8$",
        )
        ax_a.set_xlabel(r"$\mathcal{E}/\mathcal{E}_c$")
        ax_a.set_ylabel(r"$\ln\Gamma \simeq \Delta S$")
        ax_a.legend(loc="lower left")
        ax_a.set_title("(a) exponent and approximations", loc="left")

        floor = 1e-14
        for slot, (label, values) in enumerate(
            (
                ("Maclaurin, 4 terms", truncated(4)),
                ("Maclaurin, 70 terms", truncated(70)),
                (
                    r"two-point, $K=4$",
                    np.array([approximants[4].entropy_change(float(E)) for E in energies]),
                ),
                (
                    r"two-point, $K=8$",
                    np.array([approximants[8].entropy_change(float(E)) for E in energies]),
                ),
            )
        ):
            ax_b.semilogy(
                x,
                np.maximum(np.abs(values - exact), floor),
                color=SERIES[slot],
                linestyle=LINESTYLES[slot],
                label=label,
            )
        ax_b.set_xlabel(r"$\mathcal{E}/\mathcal{E}_c$")
        ax_b.set_ylabel(r"$|\,\mathrm{error\ in}\ \Delta S\,|$")
        ax_b.legend(loc="upper left")
        ax_b.set_title("(b) errors (reference resolved to ~1e-11)", loc="left")

        fit = domb_sykes_fit(series, first_order=20)
        ratio_all = series[1:] / series[:-1]
        n_all = np.arange(2, series.size + 1)
        ax_c.plot(
            1 / n_all, ratio_all, "o", color=SERIES[0], markersize=3, label=r"$b_n / b_{n-1}$"
        )
        grid = np.linspace(0.0, 1 / fit.orders[0], 50)
        model_ratio = (1 / fit.radius) * (1 - (fit.exponent + 1) * grid)
        ax_c.plot(
            grid,
            model_ratio,
            color=SERIES[1],
            linestyle=LINESTYLES[1],
            label=rf"fit: $R={fit.radius:.5f}$, $\alpha={fit.exponent:.4f}$",
        )
        ax_c.plot(
            0.0,
            1 / E_c,
            "s",
            color=TEXT["primary"],
            markersize=5,
            label=rf"$1/\mathcal{{E}}_c$, $\mathcal{{E}}_c={E_c:.5f}$",
        )
        ax_c.set_xlim(-0.005, 0.2)
        ax_c.set_xlabel(r"$1/n$")
        ax_c.set_ylabel("coefficient ratio")
        ax_c.legend(loc="lower left")
        ax_c.set_title("(c) Domb–Sykes: radius $=\\mathcal{E}_c$, square root", loc="left")
    return fig
