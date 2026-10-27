from pathlib import Path

import numpy as np

from src.config import MV_PER_V, PlotConfig, Technique
from src.plots import figure_grid
from src.soc_error import BiasStudy
from src.temperature import ArrheniusFit

FIGSIZE = (11.0, 8.0)
SOC_FIGSIZE = (12.0, 4.4)
PASS_COLOR = "#229954"
FAIL_COLOR = "#c0392b"


def plot_params_vs_temperature(block: dict[str, object], stem: Path, config: PlotConfig) -> None:
    temperatures = np.array(block["temperatures_c"], dtype=float)
    per = [block["per_temperature"][str(int(t))] for t in temperatures]
    fit = block["arrhenius"]
    arrhenius = ArrheniusFit(fit["activation_ev"], float(np.log(fit["r0_at_reference_ohm"])), 25.0)
    r0_mohm = np.array(fit["r0_measured_ohm"]) * MV_PER_V
    dense = np.linspace(temperatures.min() - 3.0, temperatures.max() + 3.0, 100)
    thresholds = block["thresholds"]
    with figure_grid(stem, config, 2, 2, FIGSIZE) as axes:
        ax = axes[0, 0]
        ax.plot(
            dense,
            arrhenius.r0_ohm(dense) * MV_PER_V,
            color=config.reference_color,
            label="Arrhenius fit",
        )
        ax.scatter(
            temperatures,
            r0_mohm,
            color=config.technique_colors[Technique.BUCK_BOOST],
            zorder=3,
            label="Measured R0",
        )
        ax.set(
            xlabel="Temperature (°C)",
            ylabel="R0 (mΩ)",
            yscale="log",
            title=f"R0 vs temperature, Ea = {fit['activation_ev']:.3f} eV",
        )
        ax.legend(loc="upper right")
        ax = axes[0, 1]
        ax.plot(
            temperatures, [p["capacity_ah"] for p in per], marker="o", color=config.neutral_color
        )
        ax.set(xlabel="Temperature (°C)", ylabel="C/20 capacity (Ah)", title="Capacity")
        ax2 = ax.twinx()
        ax2.plot(
            temperatures,
            [p["ocv_at_50_percent_v"] * MV_PER_V for p in per],
            marker="s",
            color=config.cell_colors[1],
        )
        ax2.set_ylabel("OCV at 50% SOC (mV)")
        ax2.grid(False)
        ax = axes[1, 0]
        for cycle, marker in (("UDDS", "o"), ("LA92", "s")):
            points = [
                (
                    t,
                    p["validation"][cycle]["rmse_whole_v"] * MV_PER_V,
                    p["validation"][cycle]["passes"],
                )
                for t, p in zip(temperatures, per, strict=True)
                if cycle in p["validation"]
            ]
            ax.scatter(
                [x for x, _, _ in points],
                [y for _, y, _ in points],
                marker=marker,
                color=[PASS_COLOR if ok else FAIL_COLOR for _, _, ok in points],
                label=cycle,
                zorder=3,
            )
        ax.axhline(
            thresholds["rmse_whole_cycle_v"] * MV_PER_V,
            color=config.reference_color,
            linestyle="--",
            linewidth=config.reference_line_width,
            label="30 mV limit",
        )
        ax.set(
            xlabel="Temperature (°C)",
            ylabel="Whole-cycle RMSE (mV)",
            title="Validation (green pass, red fail)",
        )
        ax.legend(loc="upper right")
        ax = axes[1, 1]
        r1 = [p["simulator_cell"]["r1_ohm"] * MV_PER_V for p in per]
        tau = [p["simulator_cell"]["r1_ohm"] * p["simulator_cell"]["c1_farad"] for p in per]
        ax.plot(temperatures, r1, marker="o", color=config.cell_colors[2])
        ax.set(xlabel="Temperature (°C)", ylabel="Single-branch R1 (mΩ)", title="Polarisation")
        ax3 = ax.twinx()
        ax3.plot(temperatures, tau, marker="s", color=config.neutral_color)
        ax3.set_ylabel("Time constant (s)")
        ax3.grid(False)


def plot_soc_error(
    study: BiasStudy,
    sigma_percent: tuple[float, ...],
    tolerance: float,
    stem: Path,
    config: PlotConfig,
) -> None:
    percent = 100.0
    with figure_grid(stem, config, 1, 3, SOC_FIGSIZE) as axes:
        for ax, technique in zip(axes[0], Technique, strict=True):
            color = config.technique_colors[technique]
            for position, label in enumerate(study.labels):
                spread = study.final_spread[technique][label] * percent
                low, mid, high = np.percentile(spread, [5.0, 50.0, 95.0])
                ax.errorbar(
                    position,
                    mid,
                    yerr=[[mid - low], [high - mid]],
                    fmt="o",
                    color=color,
                    capsize=5,
                )
            ax.axhline(
                tolerance * percent,
                color=config.reference_color,
                linestyle="--",
                linewidth=config.reference_line_width,
            )
            ax.set_xticks(range(len(sigma_percent)), [f"{s:g}" for s in sigma_percent])
            ax.set(
                xlabel="SOC estimation error sigma (% SOC)",
                ylabel="True final SOC spread (%)",
                title=technique.label,
            )
