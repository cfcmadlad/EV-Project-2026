from pathlib import Path

import numpy as np

from src.config import PlotConfig
from src.plots import figure_grid

TRACE_FIGSIZE = (12.0, 4.6)
SUMMARY_FIGSIZE = (12.0, 4.6)
CURVE_FIGSIZE = (12.0, 4.6)
SECONDS_PER_MINUTE = 60.0
PERCENT = 100.0
ESTIMATORS = (
    ("cc", "offset_injected", "Coulomb counting, 0.5% FS offset"),
    ("ekf", "nominal", "EKF"),
    ("gru", "nominal", "GRU"),
)
TRACE_KEYS = (
    ("coulomb_offset_injected", "Coulomb counting, 0.5% FS offset", 0),
    ("ekf", "EKF", 1),
    ("gru", "GRU", 2),
)
PANEL_TITLES = {"test_a": "Test A (0, 25, 40 °C)", "test_b": "Test B (10 °C, unseen)"}


def estimator_colors(config: PlotConfig) -> tuple[str, str, str]:
    return config.neutral_color, config.cell_colors[1], config.cell_colors[2]


def plot_traces(traces: dict[str, dict[str, object]], stem: Path, config: PlotConfig) -> None:
    colors = estimator_colors(config)
    with figure_grid(stem, config, 1, len(traces), TRACE_FIGSIZE) as axes:
        for ax, (name, trace) in zip(axes[0], traces.items(), strict=False):
            step = trace["step_s"]
            true = np.array(trace["true"], dtype=float)
            minutes = (np.arange(true.size) + 1) * step / SECONDS_PER_MINUTE
            ax.plot(minutes, true * PERCENT, color=config.reference_color, label="Reference")
            for key, label, index in TRACE_KEYS:
                values = np.array([np.nan if v is None else v for v in trace[key]])
                ax.plot(minutes, values * PERCENT, color=colors[index], label=label, alpha=0.85)
            temperature, cycle = name.split("_")[-2:]
            test = "Test A" if name.startswith("test_a") else "Test B"
            ax.set(
                xlabel="Time (min)",
                ylabel="State of charge (%)",
                title=f"{cycle} at {temperature} °C ({test})",
            )
        axes[0, 0].legend(loc="upper right")


def plot_error_summary(results: dict[str, object], stem: Path, config: PlotConfig) -> None:
    colors = estimator_colors(config)
    groups = (("test_a", "0"), ("test_a", "25"), ("test_a", "40"), ("test_b", "10"))
    panels = (("rmse_percent", "RMSE (% SOC)"), ("max_abs_percent", "Max absolute error (% SOC)"))
    with figure_grid(stem, config, 1, 2, SUMMARY_FIGSIZE) as axes:
        for ax, (field, label) in zip(axes[0], panels, strict=True):
            for e, (name, scenario, text) in enumerate(ESTIMATORS):
                values, errors = [], []
                for test_set, temperature in groups:
                    leaf = results[name][scenario][test_set][temperature]
                    values.append(leaf[field])
                    errors.append(leaf.get(f"{field}_std", 0.0))
                ax.bar(
                    np.arange(len(groups)) + (e - 1) * 0.27,
                    values,
                    0.27,
                    yerr=errors,
                    color=colors[e],
                    label=text,
                    capsize=3,
                )
            ax.set_xticks(
                range(len(groups)),
                [f"{t} °C\n{'A' if s == 'test_a' else 'B'}" for s, t in groups],
            )
            ax.set(ylabel=label)
        axes[0, 0].legend(loc="upper left")


def plot_wrong_start(curves: dict[str, dict[str, object]], stem: Path, config: PlotConfig) -> None:
    colors = estimator_colors(config)
    with figure_grid(stem, config, 1, 2, CURVE_FIGSIZE) as axes:
        for ax, test_set in zip(axes[0], ("test_a", "test_b"), strict=True):
            for name, label, index in (
                ("cc", "Coulomb counting (+0.5% FS offset)", 0),
                ("ekf", "EKF", 1),
            ):
                curve = curves[f"{name}_{test_set}"]
                values = np.array(curve["mean_abs_error_percent"])
                time_s = np.arange(values.size) * curve["step_s"]
                ax.plot(time_s / SECONDS_PER_MINUTE, values, color=colors[index], label=label)
            ax.axvline(
                5.0,
                color=config.reference_color,
                linestyle="--",
                linewidth=config.reference_line_width,
                label="300 s",
            )
            ax.set(
                xlabel="Time since start (min)",
                ylabel="Mean absolute error (% SOC)",
                title=f"{PANEL_TITLES[test_set]}: start 20% too low",
            )
        axes[0, 0].legend(loc="upper right")
