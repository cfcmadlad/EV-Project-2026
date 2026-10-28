from dataclasses import dataclass

import numpy as np

from src.config import CONFIG
from src.lt_circuits import Built, Cells, Solver
from src.lt_measure import (
    Trace,
    Window,
    cell_terms,
    conduction_fractions,
    device_terms,
    mean_of,
    mean_product,
    waveform_limits,
)

WINDOW_S = 5e-3
STARTUP_WINDOWS = 2
MIN_PERIODS = 20
STEP_DIVISOR = 200
RELTOL = 3e-4


@dataclass(frozen=True, slots=True)
class Timing:
    window_s: float
    stop_s: float
    solver: Solver

    @property
    def last(self) -> Window:
        return Window(self.stop_s - self.window_s, self.stop_s)

    @property
    def previous(self) -> Window:
        return Window(self.stop_s - 2.0 * self.window_s, self.stop_s - self.window_s)


def timing(frequency_hz: float, step_divisor: int = STEP_DIVISOR, reltol: float = RELTOL) -> Timing:
    periods = max(round(WINDOW_S * frequency_hz), MIN_PERIODS)
    window = periods / frequency_hz
    stop = (STARTUP_WINDOWS + 2) * window
    return Timing(window, stop, Solver(stop, 1.0 / frequency_hz / step_divisor, reltol))


def _group(device_rows: list[dict[str, float]], kinds: list[str]) -> dict[str, float]:
    out = {"p_switch_cond_w": 0.0, "p_switch_trans_w": 0.0, "p_gate_w": 0.0, "p_diode_w": 0.0}
    for kind, terms in zip(kinds, device_rows, strict=True):
        if kind == "diode":
            out["p_diode_w"] += terms["conduction"]
            continue
        out["p_switch_cond_w"] += terms["conduction"]
        out["p_switch_trans_w"] += terms["transition"]
        out["p_gate_w"] += terms["gate"]
    return out


def common_row(
    trace: Trace, built: Built, cells: Cells, clock: Timing, passive: dict[str, tuple[float, str]]
) -> dict[str, float]:
    window = clock.last
    row = dict(cell_terms(trace, window, cells.v_source, cells.v_sink, cells.r0_ohm))
    terms = [device_terms(trace, window, device) for device in built.devices]
    row.update(_group(terms, [device.kind for device in built.devices]))
    for key, (resistance, current) in passive.items():
        row[key] = resistance * mean_product(trace, window, {current: 1.0}, {current: 1.0})
    losses = (
        row["p_r0_src_w"]
        + row["p_r0_snk_w"]
        + row["p_switch_cond_w"]
        + row["p_switch_trans_w"]
        + row["p_diode_w"]
        + sum(row[key] for key in passive)
    )
    removed = row["p_src_w"] - row["p_snk_w"]
    row["p_loss_total_w"] = losses + row["p_gate_w"]
    row["efficiency"] = row["p_snk_w"] / row["p_src_w"]
    row["efficiency_with_gate"] = row["p_snk_w"] / (row["p_src_w"] + row["p_gate_w"])
    row["balance_residual_rel"] = abs(removed - losses) / removed
    earlier = mean_of(trace, clock.previous, {"I(VB)": 1.0})
    row["pss_change_snk"] = abs(row["i_snk_a"] / earlier - 1.0)
    earlier_src = mean_of(trace, clock.previous, {"I(VA)": -1.0})
    row["pss_change_src"] = abs(row["i_src_a"] / earlier_src - 1.0)
    return row


def flying_row(
    trace: Trace, built: Built, cells: Cells, clock: Timing, esr_ohm: float
) -> dict[str, float]:
    row = common_row(trace, built, cells, clock, {"p_esr_w": (esr_ohm, "I(RE)")})
    row["i_cap_rms_a"] = float(
        np.sqrt(mean_product(trace, clock.last, {"I(RE)": 1.0}, {"I(RE)": 1.0}))
    )
    row["i_cap_pk_a"] = max(map(abs, waveform_limits(trace, clock.last, "I(RE)")))
    row["r_eff_ohm"] = (cells.v_source - cells.v_sink) / row["i_snk_a"]
    return row


def buck_boost_row(
    trace: Trace,
    built: Built,
    cells: Cells,
    clock: Timing,
    frequency_hz: float,
    duty: float,
    r_inductor_ohm: float,
) -> dict[str, float]:
    row = common_row(trace, built, cells, clock, {"p_ind_w": (r_inductor_ohm, "I(L1)")})
    peak, low = waveform_limits(trace, clock.last, "I(L1)")
    row["i_pk_a"] = peak
    row["i_min_a"] = low
    row["i_l_rms_a"] = float(
        np.sqrt(mean_product(trace, clock.last, {"I(L1)": 1.0}, {"I(L1)": 1.0}))
    )
    d2, idle = conduction_fractions(trace, clock.last, 1.0 / frequency_hz, duty)
    row["d2_measured"] = d2
    row["idle_fraction"] = idle
    row["d2_formula"] = duty * cells.v_source / cells.v_sink
    return row


def ideal_ron(converter: str) -> float:
    if converter == "flying_capacitor":
        return CONFIG.flying_capacitor.r_ds_on_ohm
    return CONFIG.buck_boost.r_ds_on_ohm
