from pathlib import Path

import numpy as np

from src.config import PERCENT, PlotConfig, Technique
from src.dataset import TARGET_NAMES
from src.plots import figure_grid
from src.selector import budget_key

FIGSIZE = (13.0, 4.8)
TARGET_LABELS = ("time", "energy")
SHORT = {"passive": "passive", "buck_boost": "buck-boost", "flying_capacitor": "fly cap"}
BAR_WIDTH = 0.38
FIXED_ALPHA = 0.45


def plot_abstaining(
    block: dict[str, object],
    selector: dict[str, object],
    stem: Path,
    config: PlotConfig,
    low: float,
    high: float,
) -> None:
    budgets = selector["budgets_h"]
    rows = selector["by_budget"]
    abstain_rows = block["by_budget"]
    series = (
        ("selector", rows, "Boosting selector", config.cell_colors[2], "-"),
        ("conformal_point", abstain_rows, "Hybrid, point forecast", config.neutral_color, "-"),
        ("conformal_upper", abstain_rows, "Hybrid, upper bound", config.cell_colors[1], "-"),
        (
            "conformal_abstain",
            abstain_rows,
            "Hybrid, upper bound and abstain",
            config.cell_colors[3],
            "--",
        ),
    )
    with figure_grid(stem, config, 1, 2, FIGSIZE) as axes:
        left = axes[0, 0]
        groups = [(name, t) for name in TARGET_NAMES for t in Technique]
        x = np.arange(len(groups))
        for offset, key, alpha, label in (
            (-BAR_WIDTH / 2, "fixed_coverage", FIXED_ALPHA, "Fixed width"),
            (BAR_WIDTH / 2, "coverage", 1.0, "Normalised"),
        ):
            values = [
                block["calibration"][n]["techniques"][t.key][key] * PERCENT for n, t in groups
            ]
            left.bar(
                x + offset,
                values,
                BAR_WIDTH,
                color=[config.technique_colors[t] for _, t in groups],
                alpha=alpha,
                label=label,
            )
        left.axhspan(
            low * PERCENT, high * PERCENT, color=config.band_color, alpha=config.band_alpha
        )
        left.set_xticks(
            x,
            [f"{SHORT[t.key]}\n{TARGET_LABELS[TARGET_NAMES.index(n)]}" for n, t in groups],
            fontsize=config.tick_size,
        )
        left.set(
            ylim=(80.0, 96.0),
            ylabel="Test coverage of the 90 % interval (%)",
            title="Interval coverage (shaded: 87 to 93 %)",
        )
        left.legend(loc="upper right", fontsize=config.legend_size - 2)
        right = axes[0, 1]
        for key, source, label, color, style in series:
            violation = [
                source[budget_key(h)]["policies"][key]["budget_violation_fraction_among_feasible"]
                * PERCENT
                for h in budgets
            ]
            right.plot(budgets, violation, color=color, linestyle=style, marker="o", label=label)
        right.axhline(
            selector["max_violation"] * PERCENT,
            color=config.reference_color,
            linewidth=config.reference_line_width,
            linestyle=":",
        )
        right.set(
            xscale="log",
            xlabel="Parking time budget (h)",
            ylabel="Packs missing the budget (%)",
            title="Budget violations among feasible packs (dotted: 5 % limit)",
        )
        right.set_xticks(budgets, [f"{h:g}" for h in budgets])
        right.minorticks_off()
        right.legend(loc="upper right", fontsize=config.legend_size - 2)
