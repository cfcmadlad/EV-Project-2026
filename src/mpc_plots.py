from pathlib import Path

from src.config import PERCENT, PlotConfig
from src.mpc import TECHNIQUES
from src.plots import figure_grid

FIGSIZE = (12.0, 4.6)


def plot_mpc(
    block: dict[str, object], cells: tuple[int, ...], stem: Path, config: PlotConfig
) -> None:
    with figure_grid(stem, config, 1, 2, FIGSIZE) as axes:
        gain_ax, win_ax = axes[0]
        for technique in TECHNIQUES:
            rows = [block["results"][technique.key][str(n)] for n in cells]
            color = config.technique_colors[technique]
            label = technique.label.lower()
            gain_ax.plot(
                cells,
                [row["mean_gain_ah"] * 1000.0 for row in rows],
                color=color,
                marker="o",
                label=f"MPC gain, {label}",
            )
            gain_ax.plot(
                cells,
                [row["gap_to_reference_ah"] * 1000.0 for row in rows],
                color=color,
                linestyle="--",
                marker="s",
                label=f"Rule to reference gap, {label}",
            )
            win_ax.plot(
                cells,
                [row["win_fraction"] * PERCENT for row in rows],
                color=color,
                marker="o",
                label=label,
            )
        gain_ax.axhline(0.0, color=config.reference_color, linewidth=config.reference_line_width)
        gain_ax.set(
            xscale="log",
            xlabel="Cells in the string",
            ylabel="Delivered Ah relative to the rule (mAh)",
            title="MPC gain against the room left by the rule",
        )
        gain_ax.set_xticks(cells, [str(n) for n in cells])
        gain_ax.minorticks_off()
        gain_ax.legend(
            loc="lower left", bbox_to_anchor=(0.28, 0.16), fontsize=config.legend_size - 3
        )
        win_ax.axhline(
            block["min_win_fraction"] * PERCENT,
            color=config.reference_color,
            linewidth=config.reference_line_width,
            linestyle=":",
        )
        win_ax.set(
            xscale="log",
            xlabel="Cells in the string",
            ylabel="Packs where MPC matches the rule (%)",
            title="Win fraction (dotted: 90 % requirement)",
        )
        win_ax.set_xticks(cells, [str(n) for n in cells])
        win_ax.minorticks_off()
        win_ax.legend(loc="lower left", fontsize=config.legend_size - 2)
