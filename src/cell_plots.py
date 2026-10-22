from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from src.config import (
    MOHM_PER_OHM,
    MV_PER_V,
    PERCENT,
    FloatArray,
    PlotConfig,
    Technique,
)
from src.hg2 import DRIVE_CYCLES, Recording, TestKind
from src.identify import RcFit, TwoRcFit
from src.params import SECONDS_PER_HOUR
from src.plots import figure_grid
from src.validate import Replay, ReplayMetrics

OVERVIEW_ORDER = (
    TestKind.HPPC,
    TestKind.C20,
    TestKind.CAPACITY_1C,
    TestKind.DISCHARGE_HALF_C,
    TestKind.DISCHARGE_2C,
    TestKind.UDDS,
    TestKind.HWFET,
    TestKind.LA92,
    TestKind.US06,
)
OVERVIEW_TITLES = {
    TestKind.HPPC: "HPPC",
    TestKind.C20: "C/20",
    TestKind.CAPACITY_1C: "1C",
    TestKind.DISCHARGE_HALF_C: "0.5C",
    TestKind.DISCHARGE_2C: "2C",
    TestKind.UDDS: "UDDS",
    TestKind.HWFET: "HWFET",
    TestKind.LA92: "LA92",
    TestKind.US06: "US06",
}
OVERVIEW_FIGSIZE = (24.0, 7.5)
PARAMS_FIGSIZE = (11.0, 8.0)
VALIDATION_FIGSIZE = (11.0, 7.0)
SUMMARY_FIGSIZE = (11.0, 5.0)
COMPARISON_FIGSIZE = (11.0, 5.0)
MODEL_LABELS = {
    "assumed": "Assumed parameters",
    "1rc": "Fitted 1RC",
    "2rc": "Fitted 2RC",
    "lumped": "Fitted 2RC, single-branch equivalent",
}


def model_colors(config: PlotConfig) -> dict[str, str]:
    return {
        "assumed": config.cell_colors[3],
        "1rc": config.neutral_color,
        "2rc": config.cell_colors[1],
        "lumped": config.cell_colors[0],
    }


def plot_hg2_overview(
    recordings: Mapping[TestKind, Recording], stem: Path, config: PlotConfig
) -> None:
    rows = (
        ("voltage_v", "Voltage (V)"),
        ("current_a", "Current (A)"),
        ("temperature_c", "Battery temperature (°C)"),
    )
    with figure_grid(stem, config, len(rows), len(OVERVIEW_ORDER), OVERVIEW_FIGSIZE) as axes:
        for column, kind in enumerate(OVERVIEW_ORDER):
            recording = recordings[kind]
            hours = recording.time_s / SECONDS_PER_HOUR
            for row, (field, label) in enumerate(rows):
                ax = axes[row, column]
                ax.plot(
                    hours,
                    getattr(recording, field),
                    color=config.neutral_color,
                    linewidth=config.reference_line_width * 0.6,
                )
                if row == 0:
                    ax.set_title(OVERVIEW_TITLES[kind])
                if column == 0:
                    ax.set_ylabel(label)
                if row == len(rows) - 1:
                    ax.set_xlabel("Time (h)")


def plot_hppc_params(
    one_rc: list[RcFit],
    two_rc: list[TwoRcFit],
    window: tuple[float, float],
    stem: Path,
    config: PlotConfig,
) -> None:
    soc = np.array([fit.soc for fit in one_rc]) * PERCENT
    colors = model_colors(config)
    bounded_1 = np.array([f.at_bound for f in one_rc])
    bounded_2 = np.array([f.at_bound for f in two_rc])
    none = np.zeros(soc.size, dtype=bool)
    panels = (
        (
            (0, 0),
            "Resistance (mΩ)",
            "Ohmic resistance R0",
            False,
            (("R0 from current step", [f.r0_ohm * MOHM_PER_OHM for f in one_rc], none, "text"),),
        ),
        (
            (0, 1),
            "Resistance (mΩ)",
            "Polarisation resistance",
            False,
            (
                ("1RC branch", [f.r1_ohm * MOHM_PER_OHM for f in one_rc], bounded_1, "1rc"),
                ("2RC fast", [f.r1_ohm * MOHM_PER_OHM for f in two_rc], bounded_2, "lumped"),
                ("2RC slow", [f.r2_ohm * MOHM_PER_OHM for f in two_rc], bounded_2, "2rc"),
            ),
        ),
        (
            (1, 0),
            "Time constant (s)",
            "Time constants",
            True,
            (
                ("1RC", [f.tau_s for f in one_rc], bounded_1, "1rc"),
                ("2RC fast", [f.tau1_s for f in two_rc], bounded_2, "lumped"),
                ("2RC slow", [f.tau2_s for f in two_rc], bounded_2, "2rc"),
            ),
        ),
        (
            (1, 1),
            "Fit RMSE (mV)",
            "Fit residual per SOC point",
            False,
            (
                ("1RC", [f.rmse_v * MV_PER_V for f in one_rc], bounded_1, "1rc"),
                ("2RC", [f.rmse_v * MV_PER_V for f in two_rc], bounded_2, "2rc"),
            ),
        ),
    )
    colors["text"] = config.text_color
    with figure_grid(stem, config, 2, 2, PARAMS_FIGSIZE) as axes:
        for (row, column), ylabel, title, log, series in panels:
            ax = axes[row, column]
            ax.axvspan(
                window[0] * PERCENT,
                window[1] * PERCENT,
                color=config.band_color,
                alpha=config.band_alpha,
                label="Window used for constants",
            )
            for label, values, bound, key in series:
                _series(ax, soc, np.array(values), bound, colors[key], label)
            ax.set(
                xlabel="State of charge (%)",
                ylabel=ylabel,
                title=title,
                yscale="log" if log else "linear",
            )
            ax.legend(loc="best")


def _series(
    ax: Axes, soc: FloatArray, values: FloatArray, at_bound: np.ndarray, color: str, label: str
) -> None:
    ax.plot(soc, values, color=color, marker="o", label=label)
    if at_bound.any():
        ax.plot(
            soc[at_bound],
            values[at_bound],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markersize=9,
        )


def plot_cell_validation(
    runs: Mapping[str, Replay],
    kept: str,
    cycle: str,
    stem: Path,
    config: PlotConfig,
) -> None:
    colors = model_colors(config)
    first = runs[kept]
    hours = first.time_s / SECONDS_PER_HOUR
    with figure_grid(stem, config, 2, 1, VALIDATION_FIGSIZE, sharex=True) as axes:
        axes[0, 0].plot(
            hours, first.measured_v, color=config.text_color, linewidth=1.0, label="Measured"
        )
        for name in ("assumed", kept):
            axes[0, 0].plot(
                hours,
                runs[name].model_v,
                color=colors[name],
                linewidth=1.0,
                label=MODEL_LABELS[name],
            )
            axes[1, 0].plot(
                hours,
                (runs[name].model_v - runs[name].measured_v) * MV_PER_V,
                color=colors[name],
                linewidth=1.0,
                label=MODEL_LABELS[name],
            )
        axes[0, 0].set(ylabel="Terminal voltage (V)", title=f"{cycle} at 25 °C: measured vs model")
        axes[0, 0].legend(loc="lower left")
        axes[1, 0].set(xlabel="Time (h)", ylabel="Model minus measured (mV)")


def plot_validation_summary(
    metrics: Mapping[str, Mapping[str, ReplayMetrics]],
    unusable: tuple[str, ...],
    kept: str,
    limits_v: tuple[float, float],
    stem: Path,
    config: PlotConfig,
) -> None:
    colors = model_colors(config)
    cycles = [kind.value for kind in DRIVE_CYCLES]
    models = ("assumed", kept)
    width = 0.8 / len(models)
    panels = (
        ("rmse_whole_v", "Whole cycle RMSE (mV)", limits_v[0]),
        ("rmse_window_v", "20 to 80 % SOC window RMSE (mV)", limits_v[1]),
    )
    with figure_grid(stem, config, 1, 2, SUMMARY_FIGSIZE) as axes:
        for ax, (field, label, limit) in zip(axes[0], panels, strict=True):
            for k, name in enumerate(models):
                heights = [
                    getattr(metrics[cycle][name], field) * MV_PER_V if cycle in metrics else 0.0
                    for cycle in cycles
                ]
                ax.bar(
                    np.arange(len(cycles)) + (k - 0.5) * width,
                    heights,
                    width,
                    color=colors[name],
                    label=MODEL_LABELS[name],
                )
            ax.axhline(
                limit * MV_PER_V,
                color=config.reference_color,
                linestyle="--",
                linewidth=config.reference_line_width,
                label="Pass limit (UDDS, LA92)",
            )
            for index, cycle in enumerate(cycles):
                if cycle in unusable:
                    ax.text(index, 0.0, "no current\nin file", ha="center", va="bottom")
            ax.set_xticks(np.arange(len(cycles)), cycles)
            ax.set(ylabel=label)
        axes[0, 0].legend(loc="upper left")


def plot_baseline_comparison(
    rows: list[dict[str, float | str]], stem: Path, config: PlotConfig
) -> None:
    panels = (
        ("time_h", "Balancing time (h)", "Balancing time"),
        ("energy_j", "Energy dissipated (J)", "Energy dissipated"),
    )
    with figure_grid(stem, config, 1, 2, COMPARISON_FIGSIZE) as axes:
        for ax, (field, label, title) in zip(axes[0], panels, strict=True):
            for k, version in enumerate(("assumed", "fitted")):
                for index, technique in enumerate(Technique):
                    row = next(r for r in rows if r["technique"] == technique.key)
                    ax.bar(
                        index + (k - 0.5) * 0.4,
                        float(row[f"{field}_{version}"]),
                        0.4,
                        color=config.technique_colors[technique],
                        alpha=0.45 if version == "assumed" else 1.0,
                        hatch="//" if version == "assumed" else None,
                        edgecolor=config.text_color,
                        label=f"{version.capitalize()} parameters" if index == 0 else None,
                    )
            ax.set_xticks(range(len(Technique)), [t.label for t in Technique])
            ax.set(yscale="log", ylabel=label, title=title)
        axes[0, 0].legend(loc="best")
