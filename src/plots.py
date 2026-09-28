from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from src.balancing import BalanceResult
from src.config import (
    PERCENT,
    SECONDS_PER_HOUR,
    FloatArray,
    IntArray,
    PlotConfig,
    Technique,
)

PNG_METADATA = {"Software": None}
SVG_METADATA = {"Creator": None, "Date": None, "Format": None, "Type": None}


def _style(config: PlotConfig) -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(config.font_family),
        "font.size": config.font_size,
        "axes.titlesize": config.title_size,
        "axes.titleweight": "bold",
        "axes.labelsize": config.font_size,
        "axes.labelcolor": config.text_color,
        "axes.edgecolor": config.text_color,
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": config.grid_color,
        "grid.linewidth": config.grid_line_width,
        "xtick.labelsize": config.tick_size,
        "ytick.labelsize": config.tick_size,
        "xtick.color": config.text_color,
        "ytick.color": config.text_color,
        "legend.fontsize": config.legend_size,
        "legend.frameon": False,
        "lines.linewidth": config.line_width,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": config.svg_hash_salt,
    }


@contextmanager
def _figure(
    stem: Path,
    config: PlotConfig,
    vector: bool = False,
    figsize: tuple[float, float] | None = None,
) -> Iterator[Axes]:
    with mpl.rc_context(_style(config)):
        figure = Figure(figsize=figsize or config.figsize, layout="constrained")
        yield figure.subplots()
        figure.savefig(
            stem.with_suffix(".png"),
            dpi=config.dpi,
            bbox_inches="tight",
            metadata=PNG_METADATA,
        )
        if vector:
            figure.savefig(
                stem.with_suffix(".svg"), bbox_inches="tight", metadata=SVG_METADATA
            )


def _outside_legend(ax: Axes, config: PlotConfig) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=config.legend_anchor, borderaxespad=0.0)


def plot_soc_trace(
    result: BalanceResult,
    technique: Technique,
    tolerance: float,
    stem: Path,
    config: PlotConfig,
) -> None:
    hours = result.trace_time_s / SECONDS_PER_HOUR
    soc = result.trace_soc[:, 0, :] * PERCENT
    band_low = soc[-1].min()
    initial_spread = np.ptp(soc[0])
    with _figure(stem, config) as ax:
        ax.axhspan(
            band_low,
            band_low + tolerance * PERCENT,
            color=config.band_color,
            alpha=config.band_alpha,
            label=f"{tolerance * PERCENT:.0f}% balancing band",
        )
        for cell, color in enumerate(config.cell_colors):
            ax.plot(hours, soc[:, cell], color=color, label=f"Cell {cell + 1}")
        ax.axvline(
            result.balancing_time_s[0] / SECONDS_PER_HOUR,
            color=config.reference_color,
            linestyle="--",
            linewidth=config.reference_line_width,
            label="Balanced",
        )
        ax.set(
            xlabel="Time (h)",
            ylabel="Cell state of charge (%)",
            title=f"{technique.label}: cell SOC during balancing\n"
            f"({initial_spread:.0f}% initial spread, 4 cells in series, at rest)",
        )
        _outside_legend(ax, config)


def plot_spread_comparison(
    results: Mapping[Technique, BalanceResult],
    tolerance: float,
    stem: Path,
    config: PlotConfig,
) -> None:
    threshold = tolerance * PERCENT
    floor = threshold * config.spread_axis_floor
    with _figure(stem, config, vector=True) as ax:
        ax.axhspan(
            floor,
            threshold,
            color=config.band_color,
            alpha=config.band_alpha,
            label=f"Balanced (spread < {threshold:.0f}%)",
        )
        for technique, result in results.items():
            ax.plot(
                result.trace_time_s / SECONDS_PER_HOUR,
                np.ptp(result.trace_soc[:, 0, :], axis=1) * PERCENT,
                color=config.technique_colors[technique],
                label=technique.label,
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=config.log_tick_subs))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set(
            ylim=(floor, None),
            xlabel="Time (h)",
            ylabel="SOC spread, max minus min (%)",
            title="SOC spread vs time for three balancing techniques\n"
            "(same 4-cell pack, mid-range component values)",
        )
        _outside_legend(ax, config)


def plot_sweep(
    values: FloatArray,
    balancing_time_s: FloatArray,
    xlabel: str,
    title: str,
    color: str,
    stem: Path,
    config: PlotConfig,
) -> None:
    with _figure(stem, config, vector=True) as ax:
        ax.plot(
            values,
            balancing_time_s / SECONDS_PER_HOUR,
            marker="o",
            color=color,
        )
        ax.set(xlabel=xlabel, ylabel="Balancing time (h)", title=title)


def plot_parity(
    simulated: FloatArray,
    predicted: FloatArray,
    technique: IntArray,
    quantity: str,
    unit: str,
    scale: float,
    r2: float,
    stem: Path,
    config: PlotConfig,
) -> None:
    bounds = np.array([simulated.min(), simulated.max()]) * scale
    with _figure(stem, config, vector=True) as ax:
        for t in Technique:
            mask = technique == t
            ax.scatter(
                simulated[mask] * scale,
                predicted[mask] * scale,
                s=config.marker_size,
                alpha=config.scatter_alpha,
                color=config.technique_colors[t],
                edgecolors="none",
                label=t.label,
            )
        ax.plot(
            bounds,
            bounds,
            color=config.reference_color,
            linestyle="--",
            linewidth=config.reference_line_width,
            label="Perfect prediction",
        )
        ax.text(
            *config.annotation_position,
            f"Test R² = {r2:.3f}",
            transform=ax.transAxes,
            va="top",
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=f"Simulated {quantity.lower()} ({unit})",
            ylabel=f"Predicted {quantity.lower()} ({unit})",
            title=f"Surrogate model: predicted vs simulated {quantity.lower()}\n"
            "(held-out test set, log scale)",
        )
        _outside_legend(ax, config)


def plot_importance(
    labels: tuple[str, ...],
    mean: FloatArray,
    std: FloatArray,
    quantity: str,
    stem: Path,
    config: PlotConfig,
) -> None:
    order = np.argsort(mean)
    with _figure(stem, config, figsize=config.importance_figsize) as ax:
        ax.barh(
            np.asarray(labels)[order],
            mean[order],
            xerr=std[order],
            color=config.neutral_color,
            error_kw={
                "ecolor": config.reference_color,
                "linewidth": config.reference_line_width,
            },
        )
        ax.grid(axis="y", visible=False)
        ax.set(
            xlabel="Drop in test R² when the input is shuffled (unitless)",
            ylabel="Model input",
            title=f"Permutation feature importance for {quantity.lower()}",
        )
