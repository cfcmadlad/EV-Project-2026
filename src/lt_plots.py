from pathlib import Path

import numpy as np

from src.calibration import LTSPICE_DIR
from src.config import PlotConfig, Technique
from src.lt_report import calibrated_prediction
from src.plots import figure_grid
from src.table_io import read_rows

WAVEFORM_SIZE = (11.0, 4.6)
PARITY_SIZE = (11.0, 4.8)
BAR_SIZE = (8.0, 5.0)
MARKER_ALPHA = 0.55
BREAKDOWN_LABELS = (
    ("p_switch_cond_w", "Switch conduction"),
    ("p_switch_trans_w", "Switch transitions"),
    ("p_gate_w", "Gate drive"),
    ("p_diode_w", "Diode"),
    ("p_ind_w", "Inductor"),
    ("p_esr_w", "Capacitor ESR"),
    ("p_r0_src_w", "Cell R0, source"),
    ("p_r0_snk_w", "Cell R0, sink"),
)
LABEL_FRACTION = 0.06
SHADES = (0.25, 0.4, 0.55, 0.7, 0.85, 0.35, 0.5, 0.65)


def _colors(config: PlotConfig) -> dict[str, str]:
    return {
        "buck_boost": config.technique_colors[Technique.BUCK_BOOST],
        "flying_capacitor": config.technique_colors[Technique.FLYING_CAPACITOR],
    }


def plot_waveforms(
    kind: str, stem: Path, config: PlotConfig, directory: Path = LTSPICE_DIR
) -> None:
    rows = read_rows(directory / f"waveform_{kind}.csv")
    color = _colors(config)[kind]
    t = np.array([r["t_us"] for r in rows])
    main = "inductor_a" if kind == "buck_boost" else "capacitor_a"
    top = "switch_node_v" if kind == "buck_boost" else "capacitor_top_v"
    titles = {
        "inductor_a": "Inductor current (A)",
        "capacitor_a": "Capacitor current (A)",
        "switch_node_v": "Switch node voltage (V)",
        "capacitor_top_v": "Capacitor top plate voltage (V)",
    }
    with figure_grid(stem, config, 1, 3, WAVEFORM_SIZE) as axes:
        axes[0, 0].plot(t, [r[main] for r in rows], color=color)
        axes[0, 0].set(title=titles[main], xlabel="Time (µs)")
        axes[0, 1].plot(t, [r[top] for r in rows], color=color)
        axes[0, 1].set(title=titles[top], xlabel="Time (µs)")
        axes[0, 2].plot(t, [r["source_a"] for r in rows], color=color, label="Source cell")
        axes[0, 2].plot(
            t,
            [r["sink_a"] for r in rows],
            color=color,
            alpha=0.55,
            linestyle="--",
            label="Sink cell",
        )
        axes[0, 2].set(title="Cell currents (A)", xlabel="Time (µs)")
        axes[0, 2].legend(loc="upper right")


def plot_averaged_vs_ltspice(stem: Path, config: PlotConfig, directory: Path = LTSPICE_DIR) -> None:
    colors = _colors(config)
    flying = [r for r in read_rows(directory / "flying_cap.csv") if r["switch"] == "real"]
    buck = [
        r
        for r in read_rows(directory / "buck_boost.csv")
        if r["switch"] == "real" and r["rectifier"] == "sync" and r["saturates"] == 0.0
    ]
    with figure_grid(stem, config, 1, 2, PARITY_SIZE) as axes:
        ax = axes[0, 0]
        ax.scatter(
            [r["i_snk_a"] for r in flying],
            [r["py_i_coth_a"] for r in flying],
            s=config.marker_size,
            color=colors["flying_capacitor"],
            alpha=MARKER_ALPHA,
            label="Flying capacitor, averaged",
        )
        ax.scatter(
            [r["i_snk_a"] for r in buck],
            [r["py_i_snk_mean_a"] for r in buck],
            s=config.marker_size,
            color=colors["buck_boost"],
            alpha=MARKER_ALPHA,
            label="Buck-boost, averaged",
        )
        ax.scatter(
            [r["i_snk_a"] for r in buck],
            [calibrated_prediction(r, True)["i_snk"] for r in buck],
            s=config.marker_size,
            facecolors="none",
            edgecolors=colors["buck_boost"],
            label="Buck-boost, calibrated",
        )
        low = min(r["i_snk_a"] for r in flying + buck)
        high = max(r["i_snk_a"] for r in flying + buck)
        ax.plot([low, high], [low, high], color=config.reference_color, linewidth=1.0)
        ax.set(xscale="log", yscale="log", xlabel="LTspice (A)", ylabel="Model (A)")
        ax.set_title("Mean sink current")
        ax.legend(loc="upper left")
        ax = axes[0, 1]
        ax.scatter(
            [r["efficiency_with_gate"] * 100 for r in flying],
            [r["py_efficiency"] * 100 for r in flying],
            s=config.marker_size,
            color=colors["flying_capacitor"],
            alpha=MARKER_ALPHA,
        )
        ax.scatter(
            [r["efficiency"] * 100 for r in buck],
            [r["py_efficiency_mean"] * 100 for r in buck],
            s=config.marker_size,
            color=colors["buck_boost"],
            alpha=MARKER_ALPHA,
        )
        ax.scatter(
            [r["efficiency_with_gate"] * 100 for r in buck],
            [calibrated_prediction(r, True)["efficiency"] * 100 for r in buck],
            s=config.marker_size,
            facecolors="none",
            edgecolors=colors["buck_boost"],
        )
        ax.plot([80, 100], [80, 100], color=config.reference_color, linewidth=1.0)
        ax.set(xlabel="LTspice (%)", ylabel="Model (%)", title="Link efficiency")


def plot_loss_breakdown(
    block: dict[str, object], stem: Path, config: PlotConfig, directory: Path = LTSPICE_DIR
) -> None:
    flying = next(
        r
        for r in read_rows(directory / "flying_cap.csv")
        if r["switch"] == "real"
        and r["c_f"] == 5.5e-3
        and r["f_hz"] == 5.5e3
        and abs(r["soc_spread"] - 0.05) < 1e-9
    )
    entries = {
        "Buck-boost, sync": (block["sync"], "buck_boost", 1.0),
        "Buck-boost, Schottky": (block["diode"], "buck_boost", 0.6),
        "Flying capacitor": (flying, "flying_capacitor", 1.0),
    }
    colors = _colors(config)
    with figure_grid(stem, config, 1, 1, BAR_SIZE) as axes:
        ax = axes[0, 0]
        for position, (data, kind, _) in enumerate(entries.values()):
            bottom = 0.0
            total = sum(float(data.get(key, 0.0)) for key, _ in BREAKDOWN_LABELS) * 1e3
            for (key, name), shade in zip(BREAKDOWN_LABELS, SHADES, strict=True):
                value = float(data.get(key, 0.0)) * 1e3
                if value <= 0.0:
                    continue
                ax.bar(
                    position,
                    value,
                    bottom=bottom,
                    color=colors[kind],
                    alpha=shade,
                    edgecolor="white",
                )
                if value > LABEL_FRACTION * total:
                    ax.text(
                        position, bottom + value / 2, name, ha="center", va="center", fontsize=8
                    )
                bottom += value
        ax.set_xticks(range(len(entries)), list(entries))
        ax.set(ylabel="Loss at the baseline link (mW)", title="Loss breakdown, LTspice real parts")
