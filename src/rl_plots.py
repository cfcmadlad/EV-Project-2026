from pathlib import Path

import numpy as np

from src.config import PlotConfig
from src.plots import figure_grid

CURVE_FIGSIZE = (8.0, 4.8)
BAR_FIGSIZE = (12.0, 4.8)
CONTROLLERS = (
    ("none", "No balancing"),
    ("rule_based", "Rule-based (midsem)"),
    ("greedy", "Greedy max transfer"),
    ("ppo_mean_of_seeds", "PPO (3 seeds)"),
    ("ideal_reference", "Ideal-transfer reference"),
)


def plot_training_curves(curves: dict[str, list], stem: Path, config: PlotConfig) -> None:
    with figure_grid(stem, config, 1, 1, CURVE_FIGSIZE) as axes:
        ax = axes[0, 0]
        for index, (seed, points) in enumerate(curves.items()):
            steps, values = np.array(points).T
            ax.plot(steps, values, color=config.cell_colors[index], label=f"Seed {seed}")
        ax.set(
            xlabel="Training steps (10 s control steps)",
            ylabel="Mean episode return (Ah-equivalent)",
            title="PPO training on Mixed 1 to 6",
        )
        ax.legend(loc="lower right")


def plot_rl_vs_baselines(test: dict[str, dict], stem: Path, config: PlotConfig) -> None:
    cycles = list(test)
    colors = (
        config.neutral_color,
        config.technique_colors[1],
        config.cell_colors[3],
        config.cell_colors[2],
        config.reference_color,
    )
    panels = (("pack_ah", "Delivered pack Ah"), ("reward", "Reward (Ah minus loss term)"))
    with figure_grid(stem, config, 1, 2, BAR_FIGSIZE) as axes:
        for ax, (field, label) in zip(axes[0], panels, strict=True):
            width = 0.16
            for index, ((key, name), color) in enumerate(zip(CONTROLLERS, colors, strict=True)):
                means = [test[c][key][field]["mean"] for c in cycles]
                errors = [test[c][key][field].get("std_over_seeds", 0.0) for c in cycles]
                ax.bar(
                    np.arange(len(cycles)) + (index - 2) * width,
                    means,
                    width,
                    yerr=errors,
                    color=color,
                    label=name,
                    capsize=2,
                )
            ax.set_xticks(range(len(cycles)), cycles)
            lows = [test[c]["none"][field]["mean"] for c in cycles]
            ax.set(ylabel=label, ylim=(0.9 * min(lows), None))
        axes[0, 0].legend(loc="upper left", fontsize=config.legend_size - 2)
