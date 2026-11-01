from pathlib import Path

from src.config import PERCENT, PlotConfig
from src.plots import figure_grid
from src.selector import budget_key
from src.shift import factor_key

FIGSIZE = (12.0, 4.6)


def plot_shift(block: dict[str, object], stem: Path, config: PlotConfig) -> None:
    factors = block["factors"]
    rows = [block["by_factor"][factor_key(f)] for f in factors]
    budget = budget_key(block["budget_h"])
    with figure_grid(stem, config, 1, 2, FIGSIZE) as axes:
        error_ax, violation_ax = axes[0]
        for key, label, color in (
            ("boosting_time_mdape", "Boosting", config.cell_colors[2]),
            ("hybrid_time_mdape", "Hybrid", config.cell_colors[1]),
        ):
            error_ax.plot(
                factors,
                [row[key] * PERCENT for row in rows],
                color=color,
                marker="o",
                label=f"{label}, balancing time",
            )
        error_ax.plot(
            factors,
            [row["hybrid_time_interval_coverage"] * PERCENT for row in rows],
            color=config.neutral_color,
            linestyle="--",
            marker="s",
            label="Hybrid interval coverage",
        )
        error_ax.set(
            xlabel="Parameter range relative to training",
            ylabel="Median error or coverage (%)",
            title="Error and coverage under range shift",
            yscale="log",
        )
        error_ax.legend(loc="center right", fontsize=config.legend_size - 2)
        for key, label, color in (
            ("selector", "Boosting selector", config.cell_colors[2]),
            ("conformal_abstain", "Hybrid abstaining selector", config.cell_colors[3]),
        ):
            violation_ax.plot(
                factors,
                [row["violation_by_budget"][key][budget] * PERCENT for row in rows],
                color=color,
                marker="o",
                label=label,
            )
        violation_ax.axhline(
            block["max_violation"] * PERCENT,
            color=config.reference_color,
            linewidth=config.reference_line_width,
            linestyle=":",
        )
        violation_ax.set(
            xlabel="Parameter range relative to training",
            ylabel="Packs missing the 1 h budget (%)",
            title="Violations among feasible packs (dotted: 5 %)",
        )
        violation_ax.legend(loc="upper left", fontsize=config.legend_size - 2)
