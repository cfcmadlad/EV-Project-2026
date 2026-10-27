from pathlib import Path

from src.config import PERCENT, PlotConfig, Technique
from src.plots import figure_grid
from src.selector import budget_key

FIGSIZE = (12.0, 4.6)


def plot_selector(block: dict[str, object], stem: Path, config: PlotConfig) -> None:
    budgets = block["budgets_h"]
    rows = block["by_budget"]
    series = (
        ("selector", "Surrogate selector", config.cell_colors[2], "-"),
        ("oracle", "Oracle (simulation)", config.reference_color, "--"),
        *(
            (f"always_{t.key}", f"Always {t.label.lower()}", config.technique_colors[t], ":")
            for t in Technique
        ),
    )
    with figure_grid(stem, config, 1, 2, FIGSIZE) as axes:
        for key, label, color, style in series:
            violation = [
                rows[budget_key(h)]["policies"][key]["budget_violation_fraction_among_feasible"]
                * PERCENT
                for h in budgets
            ]
            energy = [rows[budget_key(h)]["policies"][key]["mean_energy_j"] for h in budgets]
            axes[0, 0].plot(
                budgets, violation, color=color, linestyle=style, marker="o", label=label
            )
            axes[0, 1].plot(budgets, energy, color=color, linestyle=style, marker="o", label=label)
        axes[0, 0].set(
            xscale="log",
            xlabel="Parking time budget (h)",
            ylabel="Packs missing the budget (%)",
            title="Budget violations (packs where a technique is fast enough)",
        )
        axes[0, 1].set(
            xscale="log",
            yscale="log",
            xlabel="Parking time budget (h)",
            ylabel="Mean energy dissipated (J)",
            title="Energy of the chosen technique",
        )
        for ax in axes[0]:
            ax.set_xticks(budgets, [f"{h:g}" for h in budgets])
            ax.minorticks_off()
        axes[0, 1].legend(loc="upper right", fontsize=config.legend_size - 2)
