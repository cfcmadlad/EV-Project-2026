from pathlib import Path

import numpy as np

from src.config import FloatArray, PlotConfig, Technique
from src.plots import figure_grid
from src.thermal import TRACE_INTERVAL_S

FIGSIZE = (12.0, 4.6)
SECONDS_PER_HOUR = 3600.0


def plot_thermal(
    block: dict[str, object], traces: dict[str, FloatArray], stem: Path, config: PlotConfig
) -> None:
    ceiling = block["passive_limited"]["ceiling_rise_k"]
    red = config.technique_colors[Technique.PASSIVE]
    with figure_grid(stem, config, 1, 2, FIGSIZE) as axes:
        trace_ax, bar_ax = axes[0]
        for key, label, style in (
            ("unconstrained", "Continuous bleed", "-"),
            ("limited", "Duty-cycled at the ceiling", "--"),
        ):
            rise = traces[key]
            time_h = np.arange(rise.size) * TRACE_INTERVAL_S / SECONDS_PER_HOUR
            trace_ax.plot(time_h, rise, color=red, linestyle=style, label=label)
        trace_ax.axhline(
            ceiling,
            color=config.reference_color,
            linewidth=config.reference_line_width,
            linestyle=":",
        )
        trace_ax.set(
            xlabel="Time (h)",
            ylabel="Hottest resistor rise (K)",
            title="Passive bleed resistor (assumed parameters)",
        )
        trace_ax.legend(loc="lower right", fontsize=config.legend_size - 2)
        rises = [block["baseline"][t.key]["peak_cell_rise_k"] for t in Technique]
        bar_ax.bar(
            [t.label.lower() for t in Technique],
            rises,
            color=[config.technique_colors[t] for t in Technique],
        )
        bar_ax.set(
            yscale="log",
            ylabel="Peak cell rise (K)",
            title="Cell R0 heating (assumed parameters)",
        )
