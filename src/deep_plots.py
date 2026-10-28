from pathlib import Path

import numpy as np

from src.config import PlotConfig, Technique
from src.plots import figure_grid

SURROGATES = (
    ("hgb", "Boosting"),
    ("mlp", "MLP"),
    ("cell_gru", "Cell GRU"),
    ("cell_transformer", "Cell Transformer"),
    ("hgb_physics", "Boosting, physics"),
    ("mlp_physics", "MLP, physics"),
    ("cell_gru_physics", "Cell GRU, physics"),
)
CONTROLLERS = (
    ("none", "None"),
    ("rule_based", "Rule"),
    ("greedy", "Greedy"),
    ("residual_plain", "Residual PPO"),
    ("imitation_plain", "Imitation PPO"),
    ("ideal_reference", "Ideal"),
)
SOC_MODELS = (
    ("ekf2", "2RC EKF"),
    ("gru_augmented", "GRU, augmented"),
    ("ekf_gru_hybrid", "EKF + GRU"),
)
R2_FLOOR = 0.7
CAPSIZE = 3


def _r2(block: dict, model: str, target: str, group: str) -> tuple[float, float]:
    leaf = block["models"][model]["metrics"][target][group]["r2"]
    return leaf["mean"], leaf["std"]


def plot_surrogates(block: dict, stem: Path, config: PlotConfig) -> None:
    green = config.technique_colors[Technique.FLYING_CAPACITOR]
    with figure_grid(stem, config, 1, 2, (11.0, 4.8)) as axes:
        for ax, target, title in (
            (axes[0, 0], "balancing_time_s", "Balancing time, test R²"),
            (axes[0, 1], "energy_dissipated_j", "Energy dissipated, test R²"),
        ):
            x = np.arange(len(SURROGATES))
            for offset, group, color, label in (
                (-0.2, "overall", config.neutral_color, "All techniques"),
                (0.2, "flying_capacitor", green, "Flying capacitor"),
            ):
                values = [_r2(block, m, target, group) for m, _ in SURROGATES]
                ax.bar(
                    x + offset,
                    [v[0] for v in values],
                    0.38,
                    yerr=[v[1] for v in values],
                    capsize=CAPSIZE,
                    color=color,
                    label=label,
                )
            ax.set_xticks(x, [name for _, name in SURROGATES], rotation=30, ha="right")
            ax.set(ylim=(R2_FLOOR, 1.0), title=title)
        axes[0, 0].legend(loc="lower right")


def plot_trajectories(block: dict, stem: Path, config: PlotConfig) -> None:
    grid = np.linspace(0.0, 1.0, block["grid_points"])
    colors = config.technique_colors
    with figure_grid(stem, config, 2, 3, (12.0, 6.5), sharex=True) as axes:
        for row, model in enumerate(("gru_decoder", "transformer_decoder")):
            for column, tech in enumerate(Technique):
                ax = axes[row, column]
                for k, example in enumerate(block["examples"][model][tech.key]):
                    ax.plot(
                        grid,
                        np.array(example["true_spread"]) * 100.0,
                        color=colors[tech],
                        alpha=1.0 - 0.25 * k,
                    )
                    ax.plot(
                        grid,
                        np.array(example["pred_spread"]) * 100.0,
                        color=config.reference_color,
                        linestyle="--",
                        alpha=0.8 - 0.2 * k,
                    )
                if row == 0:
                    ax.set_title(tech.label)
                if column == 0:
                    ax.set_ylabel(f"{model.replace('_', ' ').capitalize()}\nSOC spread (%)")
                if row == 1:
                    ax.set_xlabel("Normalised time")
        axes[0, 0].plot([], [], color=colors[0], label="Simulated")
        axes[0, 0].plot([], [], color=config.reference_color, linestyle="--", label="Predicted")
        axes[0, 0].legend(loc="upper right")


def plot_soc_deep(block: dict, stem: Path, config: PlotConfig) -> None:
    with figure_grid(stem, config, 1, 2, (11.0, 4.8)) as axes:
        ax = axes[0, 0]
        x = np.arange(len(SOC_MODELS))
        for offset, split, shade in ((-0.2, "test_a", 0.9), (0.2, "test_b", 0.5)):
            values = [block["results"][m]["nominal"][split]["rmse_percent"] for m, _ in SOC_MODELS]
            ax.bar(
                x + offset,
                values,
                0.38,
                color=config.neutral_color,
                alpha=shade,
                label=split.replace("_", " ").capitalize(),
            )
        ax.set_xticks(x, [name for _, name in SOC_MODELS], rotation=15, ha="right")
        ax.set(ylabel="Nominal RMSE (%)", title="SOC estimation error")
        ax.legend(loc="upper right")
        ax = axes[0, 1]
        for model, name in SOC_MODELS[1:]:
            levels = block["quantisation"][model]
            bits = [32, 8, 4, 1]
            ax.plot(
                range(len(bits)),
                [levels[str(b)]["test_rmse_percent"] for b in bits],
                marker="o",
                label=name,
            )
        ax.set_xticks(range(4), ["float32", "int8", "int4", "1 bit"])
        ax.set(ylabel="Test RMSE (%)", title="Weight quantisation", yscale="log")
        ax.legend(loc="upper left")


def plot_rl_second(block: dict, stem: Path, config: PlotConfig) -> None:
    blue = config.technique_colors[Technique.BUCK_BOOST]
    with figure_grid(stem, config, 1, 2, (11.0, 4.8)) as axes:
        for ax, size, title in ((axes[0, 0], "n4", "4 cells"), (axes[0, 1], "n8", "8 cells")):
            cycles = list(block[size]["pack_ah"])
            x = np.arange(len(cycles))
            width = 0.8 / len(CONTROLLERS)
            for k, (key, name) in enumerate(CONTROLLERS):
                values = [block[size]["pack_ah"][c][key] for c in cycles]
                color = blue if key.endswith("ppo") or "plain" in key else config.neutral_color
                ax.bar(
                    x + (k - 2.5) * width,
                    values,
                    width,
                    color=color,
                    alpha=0.35 + 0.1 * k,
                    label=name,
                )
            ax.set_xticks(x, cycles)
            ax.set(ylim=(1.2, 1.4), ylabel="Usable pack capacity (Ah)", title=title)
        axes[0, 0].legend(loc="upper left", ncols=2, fontsize=8)
