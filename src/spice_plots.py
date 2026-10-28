from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from src.config import FloatArray, ProjectConfig, Technique
from src.plots import figure_grid
from src.spice import RawTrace, SpiceCase, measure_start_s

WINDOW_PERIODS = 3
WAVEFORM_WIDTH_IN = 11.0
WAVEFORM_ROW_HEIGHT_IN = 2.6
PARITY_FIGSIZE = (7.5, 6.5)
SINK_COLOR_ALPHA = 0.55
HOLLOW_ALPHA = 0.9


def _window(
    trace: RawTrace, case: SpiceCase, config: ProjectConfig
) -> tuple[FloatArray, FloatArray]:
    spec = config.converter_check
    stop = measure_start_s(case, spec) + spec.measured_periods * case.period_s
    start = stop - WINDOW_PERIODS * case.period_s
    inside = (trace.time_s >= start) & (trace.time_s <= stop)
    return (trace.time_s[inside] - start) / case.period_s, inside


def _traces_for(
    traces: Mapping[str, RawTrace], cases: Sequence[SpiceCase], converter: str
) -> list[tuple[SpiceCase, RawTrace]]:
    return [
        (case, traces[case.name])
        for case in cases
        if case.converter == converter and case.name in traces
    ]


def _label(case: SpiceCase) -> str:
    if case.converter == "buck_boost":
        return f"L = {case.component * 1e6:g} µH, f = {case.frequency_hz / 1e3:g} kHz"
    return f"C = {case.component * 1e3:g} mF, f = {case.frequency_hz / 1e3:g} kHz"


def _cell_currents(ax: Axes, case: SpiceCase, trace: RawTrace, x: FloatArray, inside, color: str):
    ax.plot(x, -trace.signals["I(VA)"][inside], color=color, label="Source cell")
    ax.plot(
        x,
        trace.signals["I(VB)"][inside],
        color=color,
        alpha=SINK_COLOR_ALPHA,
        linestyle="--",
        label="Sink cell",
    )
    ax.set(ylabel="Cell current (A)")


def plot_buck_boost_waveform(
    traces: Mapping[str, RawTrace],
    cases: Sequence[SpiceCase],
    config: ProjectConfig,
    stem: Path,
) -> None:
    color = config.plot.technique_colors[Technique.BUCK_BOOST]
    selected = _traces_for(traces, cases, "buck_boost")
    with figure_grid(
        stem,
        config.plot,
        len(selected),
        2,
        (WAVEFORM_WIDTH_IN, WAVEFORM_ROW_HEIGHT_IN * len(selected)),
    ) as axes:
        for row, (case, trace) in enumerate(selected):
            x, inside = _window(trace, case, config)
            axes[row, 0].plot(x, trace.signals["I(L1)"][inside], color=color)
            axes[row, 0].set(ylabel="Inductor current (A)", title=_label(case))
            _cell_currents(axes[row, 1], case, trace, x, inside, color)
            axes[row, 1].set(title="Cell currents")
            axes[row, 1].legend(loc="upper right")
        for ax in axes[-1]:
            ax.set_xlabel("Time (switching periods)")
        for ax in axes.ravel():
            ax.grid(True, color=config.plot.grid_color, linewidth=config.plot.grid_line_width)


def plot_flying_capacitor_waveform(
    traces: Mapping[str, RawTrace],
    cases: Sequence[SpiceCase],
    config: ProjectConfig,
    stem: Path,
) -> None:
    color = config.plot.technique_colors[Technique.FLYING_CAPACITOR]
    selected = _traces_for(traces, cases, "flying_capacitor")
    with figure_grid(
        stem,
        config.plot,
        len(selected),
        2,
        (WAVEFORM_WIDTH_IN, WAVEFORM_ROW_HEIGHT_IN * len(selected)),
    ) as axes:
        for row, (case, trace) in enumerate(selected):
            x, inside = _window(trace, case, config)
            axes[row, 0].plot(x, trace.signals["I(C1)"][inside], color=color)
            axes[row, 0].set(ylabel="Capacitor current (A)", title=_label(case))
            _cell_currents(axes[row, 1], case, trace, x, inside, color)
            axes[row, 1].set(title="Cell currents")
            axes[row, 1].legend(loc="upper right")
        for ax in axes[-1]:
            ax.set_xlabel("Time (switching periods)")
        for ax in axes.ravel():
            ax.grid(True, color=config.plot.grid_color, linewidth=config.plot.grid_line_width)


def plot_current_parity(
    rows: Sequence[Mapping[str, object]], config: ProjectConfig, stem: Path
) -> None:
    colors = config.plot.technique_colors
    groups = (
        ("buck_boost", "averaged", "Buck-boost", colors[Technique.BUCK_BOOST], True),
        (
            "flying_capacitor",
            "series",
            "Flying capacitor, series formula",
            colors[Technique.FLYING_CAPACITOR],
            False,
        ),
        (
            "flying_capacitor",
            "two_phase",
            "Flying capacitor, two-phase formula",
            colors[Technique.FLYING_CAPACITOR],
            True,
        ),
    )
    spice = np.array([row["spice_sink_a"] for row in rows], dtype=float)
    bound = np.array([spice.min(), spice.max()]) * 1.3
    with figure_grid(stem, config.plot, 1, 1, PARITY_FIGSIZE) as axes:
        ax = axes[0, 0]
        for converter, model, label, color, filled in groups:
            chosen = [r for r in rows if r["converter"] == converter and r["model"] == model]
            if not chosen:
                continue
            ax.scatter(
                [r["spice_sink_a"] for r in chosen],
                [r["averaged_sink_a"] for r in chosen],
                s=config.plot.marker_size * 4,
                facecolors=color if filled else "none",
                edgecolors=color,
                marker="o" if filled else "s",
                alpha=HOLLOW_ALPHA,
                label=label,
            )
        ax.plot(
            bound,
            bound,
            color=config.plot.reference_color,
            linestyle="--",
            label="Perfect agreement",
        )
        ax.plot(
            bound, bound * 1.05, color=config.plot.neutral_color, linestyle=":", label="± 5 % limit"
        )
        ax.plot(bound, bound * 0.95, color=config.plot.neutral_color, linestyle=":")
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="SPICE mean current into the sink cell (A)",
            ylabel="Averaged-model mean current (A)",
            title="Averaged converter models vs LTspice\n(55 % / 50 % and 51 % / 50 % SOC links)",
        )
        ax.legend(loc="upper left")
        ax.grid(True, color=config.plot.grid_color, linewidth=config.plot.grid_line_width)
