from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from src.cell import (
    build_ocv_curve,
    open_circuit_voltage,
    rc_step_gain,
    soc_per_amp_step,
)
from src.config import (
    BoolArray,
    BuckBoostConfig,
    FloatArray,
    FlyingCapacitorConfig,
    ProjectConfig,
    Technique,
)
from src.pack import PackBatch

PowerFlow = tuple[FloatArray, FloatArray, FloatArray]
CurrentModel = Callable[
    [FloatArray, FloatArray, FloatArray, FloatArray, FloatArray], PowerFlow
]


@dataclass(frozen=True)
class BalanceResult:
    balancing_time_s: FloatArray
    energy_removed_j: FloatArray
    energy_transferred_j: FloatArray
    balanced: BoolArray
    trace_time_s: FloatArray
    trace_soc: FloatArray

    @property
    def energy_dissipated_j(self) -> FloatArray:
        return self.energy_removed_j - self.energy_transferred_j

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
) -> PowerFlow:
    bleeding = soc > soc.mean(axis=1, keepdims=True)
    current = np.where(bleeding, emf / (r_bleed[:, None] + r0), 0.0)
    removed = np.einsum("ij,ij->i", ocv, current)
    return current, removed, np.zeros_like(removed)


def buck_boost_currents(
    soc: FloatArray,
    ocv: FloatArray,
    emf: FloatArray,
    r0: FloatArray,
    inverse_lf: FloatArray,
    hardware: BuckBoostConfig,
    deadband: float,
) -> PowerFlow:
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
    return (
        _scatter_links(i_source, i_sink, forward),
        np.einsum("ij,ij->i", ocv_source, i_source),
        np.einsum("ij,ij->i", ocv_sink, i_sink),
    )


def flying_capacitor_currents(
    soc: FloatArray,
    ocv: FloatArray,
    emf: FloatArray,
    r0: FloatArray,
    inverse_fc: FloatArray,
    hardware: FlyingCapacitorConfig,
    deadband: float,
) -> PowerFlow:
    active, _ = _link_state(soc, deadband)
    r_link = (
        inverse_fc[:, None]
        + 2.0 * hardware.r_ds_on_ohm
        + hardware.esr_ohm
        + r0[:, :-1]
        + r0[:, 1:]
    )
    i_link = np.where(active, (emf[:, :-1] - emf[:, 1:]) / r_link, 0.0)
    forward = i_link > 0.0
    magnitude = np.abs(i_link)
    ocv_source, ocv_sink = _source_sink(ocv, forward)
    return (
        _scatter_links(magnitude, magnitude, forward),
        np.einsum("ij,ij->i", ocv_source, magnitude),
        np.einsum("ij,ij->i", ocv_sink, magnitude),
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
    curve = build_ocv_curve(cell)
    gain = rc_step_gain(sim.dt_s, cell)
    n_steps = round(sim.t_max_s / sim.dt_s)
    trace_every = max(1, round(sim.trace_interval_s / sim.dt_s))

    n = packs.n_packs
    balancing_time = np.full(n, sim.t_max_s)
    balanced = np.zeros(n, dtype=bool)
    energy_removed = np.zeros(n)
    energy_transferred = np.zeros(n)
    soc_full = packs.soc0.copy()
    trace_time: list[float] = []
    trace_soc: list[FloatArray] = []

    index = np.arange(n)
    soc = packs.soc0.copy()
    v1 = np.zeros_like(soc)
    r0 = packs.r0_ohm
    soc_step = soc_per_amp_step(sim.dt_s, packs.capacity_ah)
    parameter = _technique_parameter(technique, component, frequency_hz)
    removed = np.zeros(n)
    transferred = np.zeros(n)

    for step in range(n_steps + 1):
        done = soc.max(axis=1) - soc.min(axis=1) < sim.soc_tolerance
        if done.any():
            finished = index[done]
            balancing_time[finished] = step * sim.dt_s
            balanced[finished] = True
            energy_removed[finished] = removed[done]
            energy_transferred[finished] = transferred[done]
            soc_full[finished] = soc[done]
            keep = ~done
            index, soc, v1, r0, soc_step, parameter, removed, transferred = (
                array[keep]
                for array in (
                    index,
                    soc,
                    v1,
                    r0,
                    soc_step,
                    parameter,
                    removed,
                    transferred,
                )
            )
        if trace and step % trace_every == 0:
            soc_full[index] = soc
            trace_time.append(step * sim.dt_s)
            trace_soc.append(soc_full.copy())
        if index.size == 0 or step == n_steps:
            break
        ocv = open_circuit_voltage(soc, curve)
        current, p_removed, p_transferred = model(soc, ocv, ocv - v1, r0, parameter)
        removed += p_removed * sim.dt_s
        transferred += p_transferred * sim.dt_s
        soc -= current * soc_step
        v1 += gain * (current * cell.r1_ohm - v1)

    energy_removed[index] = removed
    energy_transferred[index] = transferred
    soc_full[index] = soc
    if trace and step % trace_every != 0:
        trace_time.append(step * sim.dt_s)
        trace_soc.append(soc_full.copy())
    return BalanceResult(
        balancing_time_s=balancing_time,
        energy_removed_j=energy_removed,
        energy_transferred_j=energy_transferred,
        balanced=balanced,
        trace_time_s=np.array(trace_time),
        trace_soc=np.array(trace_soc).reshape(len(trace_soc), *soc_full.shape),
    )


def tail_slowdown_ratio(
    result: BalanceResult, tolerance: float, window: float
) -> float:
    spread = np.ptp(result.trace_soc[:, 0, :], axis=1)
    head = _first_crossing(result.trace_time_s, spread, spread[0] - window)
    tail_start = _first_crossing(result.trace_time_s, spread, tolerance + window)
    return float((result.balancing_time_s[0] - tail_start) / head)


def _first_crossing(time_s: FloatArray, spread: FloatArray, level: float) -> float:
    return float(time_s[np.argmax(spread <= level)])


def _link_state(soc: FloatArray, deadband: float) -> tuple[BoolArray, BoolArray]:
    difference = soc[:, :-1] - soc[:, 1:]
    return np.abs(difference) > deadband, difference > 0.0


def _source_sink(
    values: FloatArray, forward: BoolArray
) -> tuple[FloatArray, FloatArray]:
    left, right = values[:, :-1], values[:, 1:]
    return np.where(forward, left, right), np.where(forward, right, left)


def _scatter_links(
    i_source: FloatArray, i_sink: FloatArray, forward: BoolArray
) -> FloatArray:
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
            return partial(
                buck_boost_currents, hardware=config.buck_boost, deadband=deadband
            )
        case Technique.FLYING_CAPACITOR:
            return partial(
                flying_capacitor_currents,
                hardware=config.flying_capacitor,
                deadband=deadband,
            )


def _technique_parameter(
    technique: Technique, component: FloatArray, frequency_hz: FloatArray
) -> FloatArray:
    if technique is Technique.PASSIVE:
        return component
    return 1.0 / (component * frequency_hz)


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
