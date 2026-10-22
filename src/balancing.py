from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from src.cell import (
    build_ocv_curve,
    open_circuit_voltage,
    rc_step_gain,
    rc_step_gain_per_pack,
    soc_per_amp_step,
)
from src.config import (
    BoolArray,
    FloatArray,
    ProjectConfig,
    Technique,
)
from src.flow import StepFlow
from src.ledger import (
    ElementLossModel,
    RCSquareWeights,
    buck_boost_element_loss_w,
    flying_capacitor_element_loss_w,
    passive_element_loss_w,
    per_link_r0_loss_w,
    rc_branch_square_integral,
    rc_square_weights,
)
from src.pack import PackBatch
from src.params import BuckBoostSpec
from src.switched_cap import link_resistances_ohm

CurrentModel = Callable[[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray], StepFlow]

LEDGER_KEYS = ("element_loss", "r0_loss", "r1_loss", "r0_per_link_loss")
TILT_PIVOT_SOC = 0.5


@dataclass(frozen=True, slots=True)
class BalanceResult:
    balancing_time_s: FloatArray
    energy_removed_j: FloatArray
    energy_transferred_j: FloatArray
    balanced: BoolArray
    trace_time_s: FloatArray
    trace_soc: FloatArray
    final_soc: FloatArray
    element_loss_j: FloatArray
    r0_loss_j: FloatArray
    r1_loss_j: FloatArray
    c1_stored_j: FloatArray
    r0_per_link_loss_j: FloatArray

    @property
    def energy_dissipated_j(self) -> FloatArray:
        return self.energy_removed_j - self.energy_transferred_j

    @property
    def ledger_j(self) -> FloatArray:
        return self.element_loss_j + self.r0_loss_j + self.r1_loss_j + self.c1_stored_j

    @property
    def per_link_ledger_j(self) -> FloatArray:
        return self.element_loss_j + self.r0_per_link_loss_j + self.r1_loss_j + self.c1_stored_j

    @property
    def efficiency(self) -> FloatArray:
        return np.divide(
            self.energy_transferred_j,
            self.energy_removed_j,
            out=np.zeros_like(self.energy_removed_j),
            where=self.energy_removed_j > 0.0,
        )


def passive_currents(
    soc: FloatArray,
    ocv: FloatArray,
    emf: FloatArray,
    r0: FloatArray,
    r_bleed: FloatArray,
) -> StepFlow:
    bleeding = soc > soc.mean(axis=1, keepdims=True)
    current = np.where(bleeding, emf / (r_bleed[:, None] + r0), 0.0)
    removed = np.einsum("ij,ij->i", ocv, current)
    no_links = np.zeros((soc.shape[0], 0))
    return StepFlow(
        cell_current_a=current,
        removed_w=removed,
        delivered_w=np.zeros_like(removed),
        link_source_a=no_links,
        link_sink_a=no_links,
        link_forward=no_links.astype(bool),
    )


def buck_boost_currents(
    soc: FloatArray,
    ocv: FloatArray,
    emf: FloatArray,
    r0: FloatArray,
    inverse_lf: FloatArray,
    hardware: BuckBoostSpec,
    deadband: float,
) -> StepFlow:
    active, forward = _link_state(soc, deadband)
    emf_source, emf_sink = _source_sink(emf, forward)
    r_source, r_sink = _source_sink(r0, forward)
    ocv_source, ocv_sink = _source_sink(ocv, forward)
    conductance = np.where(active, 0.5 * hardware.duty**2 * inverse_lf[:, None], 0.0)
    i_source = emf_source * conductance / (1.0 + conductance * r_source)
    v_source = emf_source - i_source * r_source
    i_peak = 2.0 * i_source / hardware.duty
    sink_duty = hardware.duty * v_source / emf_sink
    conduction_loss = (
        (hardware.r_inductor_ohm + hardware.r_ds_on_ohm)
        * i_peak**2
        * (hardware.duty + sink_duty)
        / 3.0
    )
    p_out = v_source * i_source - conduction_loss
    i_sink = 2.0 * p_out / (emf_sink + np.sqrt(emf_sink**2 + 4.0 * r_sink * p_out))
    return StepFlow(
        cell_current_a=_scatter_links(i_source, i_sink, forward),
        removed_w=np.einsum("ij,ij->i", ocv_source, i_source),
        delivered_w=np.einsum("ij,ij->i", ocv_sink, i_sink),
        link_source_a=i_source,
        link_sink_a=i_sink,
        link_forward=forward,
    )


def flying_capacitor_currents(
    soc: FloatArray,
    ocv: FloatArray,
    emf: FloatArray,
    r0: FloatArray,
    resistance: FloatArray,
    deadband: float,
) -> StepFlow:
    active, _ = _link_state(soc, deadband)
    r_link = resistance[:, 0]
    i_link = np.where(active, (emf[:, :-1] - emf[:, 1:]) / r_link, 0.0)
    forward = i_link > 0.0
    magnitude = np.abs(i_link)
    ocv_source, ocv_sink = _source_sink(ocv, forward)
    return StepFlow(
        cell_current_a=_scatter_links(magnitude, magnitude, forward),
        removed_w=np.einsum("ij,ij->i", ocv_source, magnitude),
        delivered_w=np.einsum("ij,ij->i", ocv_sink, magnitude),
        link_source_a=magnitude,
        link_sink_a=magnitude,
        link_forward=forward,
    )


def simulate(
    technique: Technique,
    packs: PackBatch,
    component: FloatArray,
    frequency_hz: FloatArray,
    config: ProjectConfig,
    trace: bool = False,
) -> BalanceResult:
    _validate_inputs(technique, packs, component, frequency_hz)
    sim, cell = config.sim, config.cell
    model = _current_model(technique, config)
    element_loss = _element_loss_model(technique, config)
    curve = build_ocv_curve(cell)
    n_steps = round(sim.t_max_s / sim.dt_s)
    trace_every = max(1, round(sim.trace_interval_s / sim.dt_s))

    n = packs.n_packs
    r1_ohm, c1_farad, gain, cross, offset = _rc_branch(packs, config)
    balancing_time = np.full(n, sim.t_max_s)
    balanced = np.zeros(n, dtype=bool)
    totals = {key: np.zeros(n) for key in ("removed", "transferred", *LEDGER_KEYS)}
    v1_full = np.zeros_like(packs.soc0)
    soc_full = packs.soc0.copy()
    trace_time: list[float] = []
    trace_soc: list[FloatArray] = []

    live: dict[str, FloatArray] = {
        "index": np.arange(n),
        "soc": packs.soc0.copy(),
        "v1": np.zeros_like(packs.soc0),
        "r0": packs.r0_ohm,
        "r1": r1_ohm[:, None],
        "gain": gain[:, None],
        "cross": cross[:, None],
        "offset": offset[:, None],
        "soc_step": soc_per_amp_step(sim.dt_s, packs.capacity_ah),
        "parameter": _technique_parameter(technique, component, frequency_hz, packs.r0_ohm, config),
        "removed": np.zeros(n),
        "transferred": np.zeros(n),
        "element_loss": np.zeros(n),
        "r0_loss": np.zeros(n),
        "r1_loss": np.zeros(n),
        "r0_per_link_loss": np.zeros(n),
    }
    if packs.ocv_tilt_v_per_soc is not None:
        live["tilt"] = packs.ocv_tilt_v_per_soc[:, None]

    for step in range(n_steps + 1):
        soc = live["soc"]
        done = soc.max(axis=1) - soc.min(axis=1) < sim.soc_tolerance
        if done.any():
            finished = live["index"][done]
            balancing_time[finished] = step * sim.dt_s
            balanced[finished] = True
            _store(totals, v1_full, soc_full, live, finished, done)
            keep = ~done
            live = {key: live[key][keep] for key in live}
        if trace and step % trace_every == 0:
            soc_full[live["index"]] = live["soc"]
            trace_time.append(step * sim.dt_s)
            trace_soc.append(soc_full.copy())
        if live["index"].size == 0 or step == n_steps:
            break
        soc, v1, r0 = live["soc"], live["v1"], live["r0"]
        ocv = open_circuit_voltage(soc, curve)
        if "tilt" in live:
            ocv = ocv + live["tilt"] * (soc - TILT_PIVOT_SOC)
        emf = ocv - v1
        flow = model(soc, ocv, emf, r0, live["parameter"])
        current = flow.cell_current_a
        live["removed"] += flow.removed_w * sim.dt_s
        live["transferred"] += flow.delivered_w * sim.dt_s
        live["element_loss"] += element_loss(flow, emf, r0, live["parameter"]) * sim.dt_s
        live["r0_loss"] += np.einsum("ij,ij->i", current**2, r0) * sim.dt_s
        live["r0_per_link_loss"] += per_link_r0_loss_w(flow, r0) * sim.dt_s
        v1_target = current * live["r1"]
        weights = RCSquareWeights(sim.dt_s, live["cross"], live["offset"])
        live["r1_loss"] += (
            rc_branch_square_integral(v1, v1_target, weights).sum(axis=1) / live["r1"][:, 0]
        )
        soc -= current * live["soc_step"]
        v1 += live["gain"] * (v1_target - v1)

    _store(totals, v1_full, soc_full, live, live["index"], np.ones(live["index"].size, bool))
    if trace and step % trace_every != 0:
        trace_time.append(step * sim.dt_s)
        trace_soc.append(soc_full.copy())
    return BalanceResult(
        balancing_time_s=balancing_time,
        energy_removed_j=totals["removed"],
        energy_transferred_j=totals["transferred"],
        balanced=balanced,
        trace_time_s=np.array(trace_time),
        trace_soc=np.array(trace_soc).reshape(len(trace_soc), *soc_full.shape),
        final_soc=soc_full,
        element_loss_j=totals["element_loss"],
        r0_loss_j=totals["r0_loss"],
        r1_loss_j=totals["r1_loss"],
        c1_stored_j=0.5 * c1_farad * (v1_full**2).sum(axis=1),
        r0_per_link_loss_j=totals["r0_per_link_loss"],
    )


def _rc_branch(
    packs: PackBatch, config: ProjectConfig
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    cell, dt_s, n = config.cell, config.sim.dt_s, packs.n_packs
    if packs.r1_ohm is None and packs.c1_farad is None:
        weights = rc_square_weights(dt_s, cell.tau_s)
        return (
            np.full(n, cell.r1_ohm),
            np.full(n, cell.c1_farad),
            np.full(n, rc_step_gain(dt_s, cell)),
            np.full(n, weights.cross),
            np.full(n, weights.offset),
        )
    r1_ohm = np.full(n, cell.r1_ohm) if packs.r1_ohm is None else packs.r1_ohm
    c1_farad = np.full(n, cell.c1_farad) if packs.c1_farad is None else packs.c1_farad
    tau_s = r1_ohm * c1_farad
    weights = rc_square_weights(dt_s, tau_s)
    return (
        r1_ohm,
        c1_farad,
        rc_step_gain_per_pack(dt_s, tau_s),
        np.asarray(weights.cross),
        np.asarray(weights.offset),
    )


def tail_slowdown_ratio(result: BalanceResult, tolerance: float, window: float) -> float:
    spread = np.ptp(result.trace_soc[:, 0, :], axis=1)
    head = _first_crossing(result.trace_time_s, spread, spread[0] - window)
    tail_start = _first_crossing(result.trace_time_s, spread, tolerance + window)
    return float((result.balancing_time_s[0] - tail_start) / head)


def _store(
    totals: dict[str, FloatArray],
    v1_full: FloatArray,
    soc_full: FloatArray,
    live: dict[str, FloatArray],
    finished: FloatArray,
    mask: BoolArray,
) -> None:
    for key in totals:
        totals[key][finished] = live[key][mask]
    v1_full[finished] = live["v1"][mask]
    soc_full[finished] = live["soc"][mask]


def _first_crossing(time_s: FloatArray, spread: FloatArray, level: float) -> float:
    return float(time_s[np.argmax(spread <= level)])


def _link_state(soc: FloatArray, deadband: float) -> tuple[BoolArray, BoolArray]:
    difference = soc[:, :-1] - soc[:, 1:]
    return np.abs(difference) > deadband, difference > 0.0


def _source_sink(values: FloatArray, forward: BoolArray) -> tuple[FloatArray, FloatArray]:
    left, right = values[:, :-1], values[:, 1:]
    return np.where(forward, left, right), np.where(forward, right, left)


def _scatter_links(i_source: FloatArray, i_sink: FloatArray, forward: BoolArray) -> FloatArray:
    current = np.zeros((forward.shape[0], forward.shape[1] + 1))
    current[:, :-1] = np.where(forward, i_source, -i_sink)
    current[:, 1:] += np.where(forward, -i_sink, i_source)
    return current


def _current_model(technique: Technique, config: ProjectConfig) -> CurrentModel:
    deadband = config.sim.link_soc_deadband
    match technique:
        case Technique.PASSIVE:
            return passive_currents
        case Technique.BUCK_BOOST:
            return partial(buck_boost_currents, hardware=config.buck_boost, deadband=deadband)
        case Technique.FLYING_CAPACITOR:
            return partial(flying_capacitor_currents, deadband=deadband)


def _element_loss_model(technique: Technique, config: ProjectConfig) -> ElementLossModel:
    match technique:
        case Technique.PASSIVE:
            return passive_element_loss_w
        case Technique.BUCK_BOOST:
            return partial(buck_boost_element_loss_w, hardware=config.buck_boost)
        case Technique.FLYING_CAPACITOR:
            return flying_capacitor_element_loss_w


def _technique_parameter(
    technique: Technique,
    component: FloatArray,
    frequency_hz: FloatArray,
    r0_ohm: FloatArray,
    config: ProjectConfig,
) -> FloatArray:
    if technique is Technique.PASSIVE:
        return component
    inverse = 1.0 / (component * frequency_hz)
    if technique is Technique.FLYING_CAPACITOR:
        return link_resistances_ohm(inverse, r0_ohm, config.flying_capacitor)
    return inverse


def _validate_inputs(
    technique: Technique,
    packs: PackBatch,
    component: FloatArray,
    frequency_hz: FloatArray,
) -> None:
    expected = (packs.n_packs,)
    if component.shape != expected or frequency_hz.shape != expected:
        raise ValueError(
            f"component and frequency_hz must have shape {expected}, got "
            f"{component.shape} and {frequency_hz.shape}"
        )
    if np.any(component <= 0.0):
        raise ValueError("component values must be positive")
    if technique is not Technique.PASSIVE and np.any(frequency_hz <= 0.0):
        raise ValueError(f"{technique.label} requires positive switching frequency")
