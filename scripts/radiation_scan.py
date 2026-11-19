"""Near-extremal Hawking emission along the neutral path of Fig. 2 (GPU-friendly scan).

For a conformally coupled scalar radiated into a bath at radius ``r_a``, this
computes, on a logarithmic grid of the remaining energy ``ΔE = M - M_e``:

* greybody factors ``Γ_k(ω = T_H)`` for ``k = 0, 1, 2`` (throat scaling test);
* the luminosity ``L`` and its local exponent ``ν_eff = d ln L / d ln ΔE``;
* the adiabaticity parameter ``ε_ad = |dT_H/dt| / (2π T_H²)`` of Eq. (5.11),
  using the exact ``dT_H/dΔE`` along the neutral branch, for one scalar field.

Usage::

    uv run python scripts/radiation_scan.py --output-dir outputs/radiation

Writes ``scan.npz``, ``summary.json`` and ``radiation_scan.{pdf,png}``.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import numpy.typing as npt

mpl.use("Agg")

import matplotlib.pyplot as plt

from blackholes.paper.configurations import FIG2_CHARGE, FIG2_MODEL
from blackholes.paper.style import LINESTYLES, SERIES, TEXT, figure_style
from blackholes.radiation.background import StaticBackground
from blackholes.radiation.greybody import absorption_probability
from blackholes.radiation.luminosity import luminosity
from blackholes.radiation.near_horizon import scaling_dimension

MODEL, Q = FIG2_MODEL, FIG2_CHARGE
XI = 1.0 / 8.0


def temperature_slope(r_plus: float) -> float:
    """Exact ``dT_H/dΔE`` along the neutral branch at event horizon ``r_plus``.

    Parameters
    ----------
    r_plus
        Event-horizon radius.

    Returns
    -------
    float
        ``κ m''(r_+) / (4π² m'(r_+))`` from ``T = κ m'/4π²`` and ``dr_+/dM = 1/m'``.
    """
    first = MODEL.radial_derivative(r_plus, Q, 1)
    second = MODEL.radial_derivative(r_plus, Q, 2)
    return MODEL.kappa * second / (4 * math.pi**2 * first)


BATHS = (5.0, 7.0, 10.0)
"""Absorbing radii ``r_a`` used to test bath independence."""

KS = (0, 1)
"""Angular modes tracked (k >= 2 cannot propagate to baths at r_a <= 10 at low frequency)."""

type Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Scan:
    """Raw scan output on the grid of remaining energies.

    Attributes
    ----------
    gaps
        ``ΔE = M - M_e``, decreasing.
    temperatures
        ``T_H`` at each gap.
    slopes
        Exact ``dT_H/dΔE`` at each gap.
    gamma_at_t
        ``Γ_k(ω = T_H)``, shape ``(len(BATHS), len(KS), len(gaps))``.
    total
        Luminosity, shape ``(len(BATHS), len(gaps))``.
    seconds
        Wall-clock time of the scan.
    """

    gaps: Array
    temperatures: Array
    slopes: Array
    gamma_at_t: Array
    total: Array
    seconds: float


def run_scan(gaps: Array, steps: int) -> Scan:
    """Compute greybody factors and luminosities on the gap grid.

    Parameters
    ----------
    gaps
        Remaining energies ``ΔE``.
    steps
        Runge–Kutta steps per solve.

    Returns
    -------
    Scan
        The raw results.
    """
    _, M_e = MODEL.critical_masses(Q)
    temperatures = np.empty(gaps.size)
    slopes = np.empty(gaps.size)
    gamma_at_t = np.empty((len(BATHS), len(KS), gaps.size))
    total = np.empty((len(BATHS), gaps.size))
    started = time.perf_counter()
    for j, gap in enumerate(gaps):
        r_plus = MODEL.horizons(M_e + float(gap), Q)[-1]
        background = StaticBackground.from_outer_horizon(MODEL, Q, r_plus)
        temperatures[j] = background.temperature
        slopes[j] = temperature_slope(r_plus)
        for b, r_absorb in enumerate(BATHS):
            gamma_at_t[b, :, j] = np.asarray(
                absorption_probability(
                    background,
                    jnp.asarray(background.temperature),
                    jnp.asarray(KS),
                    r_absorb=r_absorb,
                    steps=steps,
                )
            )
            total[b, j] = luminosity(background, r_absorb=r_absorb, steps=steps).total
        print(f"[{j + 1}/{gaps.size}] gap={gap:.3e} T={temperatures[j]:.3e} L={total[:, j]}")
    return Scan(gaps, temperatures, slopes, gamma_at_t, total, time.perf_counter() - started)


def summarize(scan: Scan, steps: int) -> dict[str, object]:
    """Compute derived quantities, predictions and a step-convergence check.

    Parameters
    ----------
    scan
        Raw results.
    steps
        Runge–Kutta steps used in the scan.

    Returns
    -------
    dict[str, object]
        JSON-serialisable summary.
    """
    _, M_e = MODEL.critical_masses(Q)
    worst = StaticBackground.from_outer_horizon(
        MODEL, Q, MODEL.horizons(M_e + float(scan.gaps[-1]), Q)[-1]
    )
    coarse = luminosity(worst, r_absorb=BATHS[0], steps=steps).total
    fine = luminosity(worst, r_absorb=BATHS[0], steps=2 * steps).total
    nu_eff = np.gradient(np.log(scan.total), np.log(scan.gaps), axis=1)
    adiabaticity = scan.total * scan.slopes / (2 * math.pi * scan.temperatures**2)
    deltas = [scaling_dimension(MODEL, Q, k, XI) for k in KS]
    greybody_slopes = np.gradient(np.log(scan.gamma_at_t), np.log(scan.temperatures), axis=2)
    return {
        "device": str(jax.devices()[0]),
        "seconds": scan.seconds,
        "step_convergence_relative": abs(fine - coarse) / abs(fine),
        "predicted_delta": deltas,
        "predicted_greybody_exponent": [2 * d - 1 for d in deltas],
        "measured_greybody_exponent_smallest_gap": greybody_slopes[:, :, -1].tolist(),
        "nu_eff_smallest_gap": nu_eff[:, -1].tolist(),
        "nu_eff": nu_eff.tolist(),
        "adiabaticity": adiabaticity.tolist(),
        "L_over_T2_first_last": [
            (scan.total[:, 0] / scan.temperatures[0] ** 2).tolist(),
            (scan.total[:, -1] / scan.temperatures[-1] ** 2).tolist(),
        ],
    }


def plot(scan: Scan, summary: dict[str, object], output_dir: Path) -> None:
    """Draw the three-panel figure: greybody scaling, ν_eff, adiabaticity.

    Parameters
    ----------
    scan
        Raw results.
    summary
        Output of :func:`summarize`.
    output_dir
        Directory receiving ``radiation_scan.{pdf,png}``.
    """
    nu_eff = np.asarray(summary["nu_eff"])
    adiabaticity = np.asarray(summary["adiabaticity"])
    deltas = [scaling_dimension(MODEL, Q, k, XI) for k in KS]
    with figure_style():
        fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(12.0, 3.6), layout="constrained")
        for slot, (k, delta) in enumerate(zip(KS, deltas, strict=True)):
            ax_a.loglog(
                scan.temperatures,
                scan.gamma_at_t[0, slot],
                color=SERIES[slot],
                linestyle=LINESTYLES[slot],
                label=rf"$k={k}$, throat $\Delta_k={delta:.3f}$",
            )
        ax_a.set_xlabel(r"$T_H$")
        ax_a.set_ylabel(r"$\Gamma_k(\omega = T_H)$")
        ax_a.legend(loc="lower right")
        ax_a.set_title(r"(a) greybody factors, bath $r_a=5$", loc="left")
        for slot, r_absorb in enumerate(BATHS):
            ax_b.semilogx(
                scan.gaps,
                nu_eff[slot],
                color=SERIES[slot],
                linestyle=LINESTYLES[slot],
                label=rf"bath $r_a = {r_absorb:g}$",
            )
        for value, text in (
            (1.0, r"$\nu=1$ (conformal throat)"),
            (1.5, r"$\nu=3/2$ (blackbody, paper)"),
        ):
            ax_b.axhline(value, color=TEXT["secondary"], linewidth=0.8, linestyle="--")
            ax_b.annotate(
                text,
                (scan.gaps[0], value),
                xytext=(4, 4),
                textcoords="offset points",
                color=TEXT["secondary"],
            )
        ax_b.invert_xaxis()
        ax_b.set_xlabel(r"$\Delta\mathcal{E} = M - M_e$")
        ax_b.set_ylabel(r"$\nu_{\rm eff} = d\ln L / d\ln\Delta\mathcal{E}$")
        ax_b.legend(loc="center right")
        ax_b.set_title("(b) local luminosity exponent", loc="left")
        for slot, r_absorb in enumerate(BATHS):
            ax_c.loglog(
                scan.gaps,
                adiabaticity[slot],
                color=SERIES[slot],
                linestyle=LINESTYLES[slot],
                label=rf"bath $r_a = {r_absorb:g}$",
            )
        ax_c.invert_xaxis()
        ax_c.set_xlabel(r"$\Delta\mathcal{E} = M - M_e$")
        ax_c.set_ylabel(r"$\epsilon_{\rm ad}$ (one scalar, $\kappa = 1$)")
        ax_c.legend(loc="upper left")
        ax_c.set_title("(c) adiabaticity parameter, Eq. (5.11)", loc="left")
        for suffix in ("pdf", "png"):
            fig.savefig(output_dir / f"radiation_scan.{suffix}")


def main() -> None:
    """Parse arguments, run the scan, and write data, summary and figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radiation"))
    parser.add_argument("--points", type=int, default=33)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--min-log-gap", type=float, default=-9.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"JAX devices: {jax.devices()}")

    M_i, M_e = MODEL.critical_masses(Q)
    gaps = np.logspace(math.log10(0.5 * (M_i - M_e)), args.min_log_gap, args.points)
    scan = run_scan(gaps, args.steps)
    summary = summarize(scan, args.steps)
    np.savez(
        args.output_dir / "scan.npz",
        gaps=scan.gaps,
        temperatures=scan.temperatures,
        baths=np.array(BATHS),
        ks=np.array(KS),
        gamma_at_t=scan.gamma_at_t,
        luminosity=scan.total,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"nu_eff", "adiabaticity"}}, indent=2
        )
    )
    plot(scan, summary, args.output_dir)


if __name__ == "__main__":
    main()
