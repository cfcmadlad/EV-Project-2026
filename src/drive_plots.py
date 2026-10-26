from pathlib import Path

from src.config import PERCENT, FloatArray, PlotConfig, Technique
from src.drive_cycle import STRATEGY_KEYS
from src.plots import figure_grid

SPREAD_FIGSIZE = (11.0, 4.6)
CAPACITY_FIGSIZE = (12.0, 4.6)
SECONDS_PER_MINUTE = 60.0
LABELS = ("No balancing", *(t.label for t in Technique))
SCENARIO_TITLES = {
    "rest": "Rest balancing, then drive",
    "driving": "Driving (Mixed 1, repeated)",
    "charging": "Charging at 0.5C",
}


def strategy_colors(config: PlotConfig) -> tuple[str, ...]:
    return (config.neutral_color, *config.technique_colors)


def plot_drive_spread(
    traces: dict[str, dict[str, tuple[FloatArray, FloatArray]]],
    tolerance: float,
    stem: Path,
    config: PlotConfig,
) -> None:
    colors = strategy_colors(config)
    with figure_grid(stem, config, 1, 2, SPREAD_FIGSIZE) as axes:
        for ax, scenario in zip(axes[0], ("driving", "charging"), strict=True):
            trace = traces[scenario]
            for key, label, color in zip(STRATEGY_KEYS, LABELS, colors, strict=True):
                time_s, spread = trace[key]
                ax.plot(time_s / SECONDS_PER_MINUTE, spread * PERCENT, color=color, label=label)
            ax.axhline(
                tolerance * PERCENT,
                color=config.reference_color,
                linestyle="--",
                linewidth=config.reference_line_width,
                label="1% criterion",
            )
            ax.set(
                xlabel="Time (min)",
                ylabel="SOC spread, max minus min (%)",
                title=SCENARIO_TITLES[scenario],
            )
        axes[0, 0].legend(loc="upper right")


def plot_drive_capacity(block: dict[str, object], stem: Path, config: PlotConfig) -> None:
    scenarios = block["scenarios"]
    reference = block["reference_packs"]["charge_equalised"]
    colors = strategy_colors(config)
    keys = {
        "rest": ("passive", "buck_boost", "flying_capacitor"),
        "driving": STRATEGY_KEYS,
        "charging": STRATEGY_KEYS,
    }
    with figure_grid(stem, config, 1, 3, CAPACITY_FIGSIZE) as axes:
        for ax, scenario in zip(axes[0], ("rest", "driving", "charging"), strict=True):
            for position, key in enumerate(keys[scenario]):
                row = scenarios[scenario][key]
                value = row["followup_usable_ah"] if scenario == "rest" else row["usable_ah"]
                color = colors[STRATEGY_KEYS.index(key)]
                ax.bar(position, value, color=color, label=LABELS[STRATEGY_KEYS.index(key)])
                ax.text(position, value, f"{value:.3f}", ha="center", va="bottom")
            if scenario == "rest":
                ax.axhline(
                    scenarios["driving"]["none"]["usable_ah"],
                    color=colors[0],
                    linestyle=":",
                    linewidth=config.reference_line_width,
                    label="No balancing, no rest",
                )
            else:
                ax.axhline(
                    reference[f"{scenario}_usable_ah"],
                    color=config.reference_color,
                    linestyle="--",
                    linewidth=config.reference_line_width,
                    label="Ideal-transfer reference",
                )
            ax.set_xticks([])
            ax.set(
                ylabel="Pack Ah until first cell hits a limit",
                title=SCENARIO_TITLES[scenario],
                ylim=(1.0, 1.5),
            )
        axes[0, 1].legend(loc="lower right")
        axes[0, 0].legend(loc="lower right")
