from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

from src.config import PlotConfig, Technique
from src.dataset import TARGET_NAMES
from src.ml_eval import GROUPS, MODELS, SCENARIOS, EvalData
from src.plots import figure_grid

TARGET_LABELS = {"balancing_time_s": "Balancing time", "energy_dissipated_j": "Energy dissipated"}
TARGET_UNITS = {"balancing_time_s": "s", "energy_dissipated_j": "J"}
MODEL_LABELS = {
    "physics": "Physics formula",
    "linear": "Log-linear regression",
    "hgb": "HGB (midsem)",
    "hybrid": "Hybrid (HGB on physics residual)",
}
MODEL_HATCH = {"physics": "", "linear": "//", "hgb": "..", "hybrid": "xx"}
CV_FIGSIZE = (13.0, 5.2)
RANGE_FIGSIZE = (15.0, 13.0)
COVERAGE_FIGSIZE = (11.0, 4.8)
BAR_WIDTH = 0.2
MARKER_AREA = 5.0
SCATTER_ALPHA = 0.4
PERCENT = 100.0


def plot_cv_comparison(cv: Mapping[str, object], config: PlotConfig, stem: Path) -> None:
    groups = GROUPS[1:]
    with figure_grid(stem, config, 1, len(TARGET_NAMES), CV_FIGSIZE) as axes:
        for column, name in enumerate(TARGET_NAMES):
            ax = axes[0, column]
            for g, group in enumerate(groups):
                color = config.technique_colors[Technique[group.upper()]]
                for m, model in enumerate(MODELS):
                    entry = cv["scores"][name][model][group]["mae"]
                    ax.bar(
                        g + (m - 1.5) * BAR_WIDTH,
                        entry["mean"],
                        BAR_WIDTH,
                        yerr=entry["std"],
                        color=color,
                        edgecolor=config.reference_color,
                        hatch=MODEL_HATCH[model],
                        error_kw={"elinewidth": config.reference_line_width, "capsize": 3},
                    )
            ax.set(
                yscale="log",
                xticks=range(len(groups)),
                xticklabels=[t.label for t in Technique],
                ylabel=f"Mean absolute error ({TARGET_UNITS[name]})",
                title=f"{TARGET_LABELS[name]}: 5-fold cross-validation",
            )
            ax.grid(True, axis="y", color=config.grid_color, linewidth=config.grid_line_width)
        handles = [
            Patch(
                facecolor="white",
                edgecolor=config.reference_color,
                hatch=MODEL_HATCH[m],
                label=MODEL_LABELS[m],
            )
            for m in MODELS
        ]
        axes[0, -1].legend(handles=handles, loc="upper left", bbox_to_anchor=config.legend_anchor)


def plot_out_of_range(
    arrays: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    data: EvalData,
    config: PlotConfig,
    stem: Path,
) -> None:
    rows = [(scenario, name) for scenario in SCENARIOS for name in TARGET_NAMES]
    titles = {"component": "train: component <= p80", "spread": "train: spread <= 8 %"}
    with figure_grid(stem, config, len(rows), len(MODELS), RANGE_FIGSIZE) as axes:
        for r, (scenario, name) in enumerate(rows):
            block = arrays[scenario][name]
            technique = data.technique[block["test_index"]]
            true = block["true"]
            bounds = np.array([true.min(), true.max()]) * np.array([0.8, 1.25])
            for c, model in enumerate(MODELS):
                ax = axes[r, c]
                for t in Technique:
                    mask = technique == t
                    ax.scatter(
                        true[mask],
                        block[model][mask],
                        s=MARKER_AREA,
                        alpha=SCATTER_ALPHA,
                        color=config.technique_colors[t],
                        edgecolors="none",
                        label=t.label if (r, c) == (0, 0) else None,
                    )
                ax.plot(
                    bounds,
                    bounds,
                    color=config.reference_color,
                    linestyle="--",
                    linewidth=config.reference_line_width,
                )
                ax.set(xscale="log", yscale="log", xlim=bounds, ylim=bounds)
                if r == 0:
                    ax.set_title(MODEL_LABELS[model])
                if c == 0:
                    ax.set_ylabel(
                        f"{TARGET_LABELS[name]} ({TARGET_UNITS[name]})\n{titles[scenario]}"
                    )
                if r == len(rows) - 1:
                    ax.set_xlabel("Simulated")
        axes[0, 0].legend(loc="upper left")


def plot_conformal_coverage(
    conformal: Mapping[str, object], config: PlotConfig, low: float, high: float, stem: Path
) -> None:
    with figure_grid(stem, config, 1, len(TARGET_NAMES), COVERAGE_FIGSIZE) as axes:
        for column, name in enumerate(TARGET_NAMES):
            ax = axes[0, column]
            block = conformal[name]
            for i, technique in enumerate(Technique):
                entry = block["techniques"][technique.key]
                ax.bar(i, entry["coverage"] * PERCENT, color=config.technique_colors[technique])
                ax.text(
                    i,
                    entry["coverage"] * PERCENT + 0.4,
                    f"x{entry['half_width_factor']:.2f}",
                    ha="center",
                    va="bottom",
                )
            repeated = conformal["repeated_splits"][name]
            ax.errorbar(
                range(len(Technique)),
                [repeated[t.key]["mean_coverage"] * PERCENT for t in Technique],
                yerr=[repeated[t.key]["std_coverage"] * PERCENT for t in Technique],
                fmt="D",
                color=config.reference_color,
                capsize=4,
                label=f"Mean ± std, {conformal['repeated_splits']['n_repeats']} random splits",
            )
            ax.axhspan(
                low * PERCENT,
                high * PERCENT,
                color=config.band_color,
                alpha=config.band_alpha,
                label="87 to 93 % pass band",
            )
            ax.axhline(
                conformal["nominal"] * PERCENT,
                color=config.reference_color,
                linestyle="--",
                linewidth=config.reference_line_width,
                label="Nominal 90 %",
            )
            ax.set(
                ylim=(80.0, 100.0),
                xticks=range(len(Technique)),
                xticklabels=[t.label for t in Technique],
                ylabel="Test coverage (%)",
                title=f"{TARGET_LABELS[name]}: {MODEL_LABELS[block['model']]}",
            )
            ax.legend(loc="upper left", framealpha=1.0)
            ax.grid(True, axis="y", color=config.grid_color, linewidth=config.grid_line_width)
