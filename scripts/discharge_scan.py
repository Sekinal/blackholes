"""Charged Hawking emission from the initial state of Fig. 2: floors and trajectories.

One massless, conformally coupled complex scalar with quanta of charge ``±e`` is
radiated into a bath at ``r_a``. Starting from the Fig. 2 state on the inner
critical curve (``Q = 2π``, ``ΔE = M_i - M_e``), this computes

* the floor ``ΔE_*(e)`` at ``Q = 2π`` where thermal cooling and superradiant
  heating balance, for several baths (the hole cannot get closer to coalescence);
* full trajectories ``(Q(t), ΔE(t))`` for a few charges, integrated in Killing
  time with an implicit Radau scheme (the approach to the floor is fast, the
  discharge along it slow).

Usage::

    uv run python scripts/discharge_scan.py --output-dir outputs/discharge

Writes ``discharge.npz``, ``summary.json`` and ``discharge_scan.{pdf,png}``.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

mpl.use("Agg")

import matplotlib.pyplot as plt

from blackholes.paper.configurations import FIG2_CHARGE, FIG2_MODEL
from blackholes.paper.style import LINESTYLES, SERIES, TEXT, figure_style
from blackholes.radiation.discharge import DischargeSettings, coalescence_floor, flow

MODEL, Q0 = FIG2_MODEL, FIG2_CHARGE
M_I, M_E = MODEL.critical_masses(Q0)
GAP0 = M_I - M_E

BATHS = (5.0, 7.0, 10.0)
"""Absorbing radii for the floor scan."""

TRAJECTORY_BATH = 5.0
"""Bath used for the time-evolved trajectories."""

TRAJECTORY_CHARGES = (1e-3, 3e-3, 1e-2)
"""Charges ``e`` of the time-evolved trajectories."""

FLOOR_BRACKET = (1e-9, 0.97 * GAP0)
"""Search interval for ``ΔE_*``: from deep in the throat to just below the initial state."""

type Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Trajectory:
    """One time-evolved discharge history.

    Attributes
    ----------
    charge
        ``e`` of the radiated field.
    times
        Killing times.
    charges
        ``Q(t)``.
    gaps
        ``ΔE(t) = M - M_e(Q)``.
    final_floor
        ``ΔE_*`` at the final charge (the trajectory should sit on it).
    """

    charge: float
    times: Array
    charges: Array
    gaps: Array
    final_floor: float


def floor_scan(charges: Array) -> Array:
    """Floors ``ΔE_*`` at ``Q = 2π`` for every bath and charge (NaN: no floor).

    Parameters
    ----------
    charges
        Values of ``e``.

    Returns
    -------
    Array
        Shape ``(len(BATHS), len(charges))``.
    """
    floors = np.full((len(BATHS), charges.size), np.nan)
    for b, r_absorb in enumerate(BATHS):
        for j, e in enumerate(charges):
            settings = DischargeSettings(charge=float(e), r_absorb=r_absorb)
            try:
                floors[b, j] = coalescence_floor(MODEL, Q0, settings, bracket=FLOOR_BRACKET)
            except ValueError as error:
                # Superradiance wins already at the initial state: no approach at all.
                print(f"r_a={r_absorb:g} e={e:.2e}: {error}")
            print(f"r_a={r_absorb:g} e={e:.2e} floor={floors[b, j]:.4e}")
    return floors


def evolve(e: float, t_max: float, q_stop: float) -> Trajectory:
    """Integrate ``(Q, ln ΔE)`` from the Fig. 2 initial state.

    Parameters
    ----------
    e
        Charge of the radiated quanta.
    t_max
        Final Killing time.
    q_stop
        Stop once ``Q`` has fallen to this value.

    Returns
    -------
    Trajectory
        The sampled history and the floor at its final charge.
    """
    settings = DischargeSettings(charge=e, r_absorb=TRAJECTORY_BATH)

    def rhs(_t: float, y: Array) -> list[float]:
        """``d(Q, ln ΔE)/dt``.

        Parameters
        ----------
        _t
            Killing time (the flow is autonomous).
        y
            ``(Q, ln ΔE)``.

        Returns
        -------
        list[float]
            Time derivatives.
        """
        gap = math.exp(y[1])
        dq, dgap = flow(MODEL, float(y[0]), gap, settings)
        return [dq, dgap / gap]

    def discharged(_t: float, y: Array) -> float:
        """Event: ``Q`` reaches ``q_stop``.

        Parameters
        ----------
        _t
            Killing time.
        y
            ``(Q, ln ΔE)``.

        Returns
        -------
        float
            ``Q - q_stop``.
        """
        return float(y[0]) - q_stop

    discharged.terminal = True  # type: ignore[attr-defined]  # solve_ivp event protocol
    solution = solve_ivp(
        rhs,
        (0.0, t_max),
        [Q0, math.log(GAP0)],
        method="Radau",
        rtol=1e-5,
        atol=[1e-9, 1e-6],
        events=discharged,
    )
    charges, gaps = solution.y[0], np.exp(solution.y[1])
    upper = max(1.5 * float(gaps[-1]), 1e-6)
    final_floor = coalescence_floor(
        MODEL, float(charges[-1]), settings, bracket=(1e-3 * float(gaps[-1]), upper)
    )
    print(f"e={e:.1e}: {solution.t.size} steps, Q_end={charges[-1]:.4f}, gap={gaps[-1]:.4e}")
    return Trajectory(e, solution.t, charges, gaps, final_floor)


def plot(charges: Array, floors: Array, trajectories: list[Trajectory], output_dir: Path) -> None:
    """Two panels: floors versus ``e``; trajectories in the ``(Q, ΔE)`` plane.

    Parameters
    ----------
    charges
        ``e`` grid of the floor scan.
    floors
        Output of :func:`floor_scan`.
    trajectories
        Time-evolved histories.
    output_dir
        Directory receiving ``discharge_scan.{pdf,png}``.
    """
    with figure_style():
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 3.8), layout="constrained")
        for slot, r_absorb in enumerate(BATHS):
            ax_a.loglog(
                charges,
                floors[slot],
                color=SERIES[slot],
                linestyle=LINESTYLES[slot],
                marker="o",
                markersize=4,
                label=rf"bath $r_a = {r_absorb:g}$",
            )
        ax_a.axhline(GAP0, color=TEXT["secondary"], linewidth=0.8, linestyle="--")
        ax_a.annotate(
            r"initial state ($M = M_i$)",
            (charges[0], GAP0),
            xytext=(4, -12),
            textcoords="offset points",
            color=TEXT["secondary"],
        )
        ax_a.set_xlabel(r"charge of the radiated quanta $e$")
        ax_a.set_ylabel(r"floor $\Delta\mathcal{E}_* = M - M_e$ at $Q = 2\pi$")
        ax_a.legend(loc="lower right")
        ax_a.set_title("(a) closest approach to coalescence", loc="left")

        ax_b.axvline(Q0, color=TEXT["secondary"], linewidth=0.8, linestyle="--")
        ax_b.annotate(
            "neutral channel\n(paper: reaches $\\Delta\\mathcal{E} = 0$)",
            (Q0, 5e-3),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            color=TEXT["secondary"],
        )
        for slot, trajectory in enumerate(trajectories):
            ax_b.semilogy(
                trajectory.charges,
                trajectory.gaps,
                color=SERIES[slot],
                linestyle=LINESTYLES[slot],
                label=rf"$e = {trajectory.charge:g}$",
            )
        ax_b.set_xlabel(r"charge $Q$")
        ax_b.set_ylabel(r"$\Delta\mathcal{E} = M - M_e(Q)$")
        ax_b.legend(loc="upper center")
        ax_b.set_title(rf"(b) charged evaporation, bath $r_a = {TRAJECTORY_BATH:g}$", loc="left")
        for suffix in ("pdf", "png"):
            fig.savefig(output_dir / f"discharge_scan.{suffix}")


def main() -> None:
    """Parse arguments, run both scans, and write data, summary and figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/discharge"))
    parser.add_argument("--floor-points", type=int, default=7)
    parser.add_argument("--t-max", type=float, default=1e10)
    parser.add_argument("--q-stop-fraction", type=float, default=0.7)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    charges = np.logspace(-3.5, -2.0, args.floor_points)
    floors = floor_scan(charges)
    trajectories = [evolve(e, args.t_max, args.q_stop_fraction * Q0) for e in TRAJECTORY_CHARGES]
    valid = np.isfinite(floors[0])
    slope = np.polyfit(np.log(charges[valid][:3]), np.log(floors[0][valid][:3]), 1)[0]
    summary = {
        "seconds": time.perf_counter() - started,
        "initial_gap": GAP0,
        "charges": charges.tolist(),
        "baths": list(BATHS),
        "floors": floors.tolist(),
        "floor_exponent_smallest_charges_bath5": float(slope),
        "trajectories": [
            {
                "charge": t.charge,
                "steps": int(t.times.size),
                "final_time": float(t.times[-1]),
                "final_charge": float(t.charges[-1]),
                "minimum_gap": float(t.gaps.min()),
                "final_gap": float(t.gaps[-1]),
                "floor_at_final_charge": t.final_floor,
            }
            for t in trajectories
        ],
    }
    arrays: dict[str, Array] = {"charges": charges, "floors": floors}
    for t in trajectories:
        tag = f"{t.charge:g}"
        arrays[f"t_{tag}"] = t.times
        arrays[f"Q_{tag}"] = t.charges
        arrays[f"gap_{tag}"] = t.gaps
    np.savez(args.output_dir / "discharge.npz", **arrays)  # type: ignore[arg-type]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    plot(charges, floors, trajectories, args.output_dir)


if __name__ == "__main__":
    main()
