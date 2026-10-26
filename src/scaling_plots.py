from pathlib import Path

import numpy as np

from src.config import PlotConfig, Technique
from src.params import SECONDS_PER_HOUR
from src.plots import figure_grid

FIGSIZE = (12.0, 4.8)
TITLES = {
    "random": "Random initial SOCs (project statistics)",
    "step": "Step profile: top half high, bottom half low",
}


def plot_scaling(block: dict[str, object], stem: Path, config: PlotConfig) -> None:
    cells = np.array(block["cells"])
    anchor = block["large_n_min"]
    with figure_grid(stem, config, 1, 2, FIGSIZE) as axes:
        for ax, (profile, techniques) in zip(axes[0], block["profiles"].items(), strict=True):
            for technique in Technique:
                row = techniques[technique.key]
                hours = np.array([row["median_time_s"][str(n)] for n in cells]) / SECONDS_PER_HOUR
                color = config.technique_colors[technique]
                ax.plot(
                    cells,
                    hours,
                    marker="o",
                    color=color,
                    label=f"{technique.label}, slope {row['fitted_slope_large_n']:.2f}",
                )
                expected = row["theory_slope_large_n"]
                if expected is not None:
                    base = hours[list(cells).index(anchor)]
                    ax.plot(
                        cells,
                        base * (cells / anchor) ** expected,
                        linestyle="--",
                        color=color,
                        alpha=0.7,
                    )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Cells in series, N",
                ylabel="Median balancing time (h)",
                title=TITLES[profile],
            )
            ax.set_xticks(cells, [str(n) for n in cells])
            ax.legend(loc="upper left", fontsize=config.legend_size - 1)
