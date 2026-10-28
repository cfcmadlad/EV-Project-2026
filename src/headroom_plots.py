from pathlib import Path

from src.config import PlotConfig, Technique
from src.headroom import CELLS, HEADROOM_P90, NO_HEADROOM_MEDIAN
from src.plots import figure_grid

FIGSIZE = (8.0, 5.0)
PERCENT = 100.0


def plot_headroom(block: dict, stem: Path, config: PlotConfig) -> None:
    with figure_grid(stem, config, 1, 1, FIGSIZE) as axes:
        ax = axes[0, 0]
        for technique in Technique:
            stats = [block["results"][technique.key][str(n)]["pooled"]["rule"] for n in CELLS]
            color = config.technique_colors[technique]
            ax.plot(
                CELLS,
                [s["median"] * PERCENT for s in stats],
                color=color,
                marker="o",
                label=technique.label,
            )
            ax.fill_between(
                CELLS,
                [s["p10"] * PERCENT for s in stats],
                [s["p90"] * PERCENT for s in stats],
                color=color,
                alpha=config.band_alpha,
            )
        ax.axhline(NO_HEADROOM_MEDIAN * PERCENT, color=config.reference_color, linestyle=":")
        ax.axhline(HEADROOM_P90 * PERCENT, color=config.reference_color, linestyle="--")
        ax.set(
            xscale="log",
            xlabel="Cells in the string",
            ylabel="Rule gap to ideal transfer (% of delivered Ah)",
            title="Controller headroom",
        )
        ax.set_xticks(CELLS, [str(n) for n in CELLS])
        ax.legend(loc="upper left")
