from pathlib import Path

import numpy as np

from src.config import PERCENT, PlotConfig, Technique
from src.plots import figure_grid
from src.voltage_balancing import setting_key

FIGSIZE = (12.0, 4.6)
LINESTYLES = ("-", "--", ":")


def plot_voltage_balancing(block: dict[str, object], stem: Path, config: PlotConfig) -> None:
    sigmas = np.array(block["sigma_mv"])
    thresholds = block["stop_thresholds"]
    with figure_grid(stem, config, 1, 3, FIGSIZE) as axes:
        for ax, technique in zip(axes[0], Technique, strict=True):
            rows = block["results"][technique.key]
            for threshold, style in zip(thresholds, LINESTYLES, strict=False):
                fractions = [
                    rows[setting_key(threshold, s)]["fraction_true_spread_below_target"] * PERCENT
                    for s in sigmas
                ]
                ax.plot(
                    sigmas,
                    fractions,
                    linestyle=style,
                    marker="o",
                    color=config.technique_colors[technique],
                    label=f"Stop at {threshold * PERCENT:g}% estimated spread",
                )
            ax.axhline(
                block["required_fraction"] * PERCENT,
                color=config.reference_color,
                linewidth=config.reference_line_width,
                linestyle="-.",
            )
            ax.set(
                xscale="log",
                xlabel="Cell voltage sensor error sigma (mV)",
                ylabel="Packs with true spread below 1% (%)",
                title=technique.label,
                ylim=(-3.0, 103.0),
            )
            ax.set_xticks(sigmas, [f"{s:g}" for s in sigmas])
            ax.legend(loc="lower left", fontsize=config.legend_size - 2)
