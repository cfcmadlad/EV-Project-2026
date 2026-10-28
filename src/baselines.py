from dataclasses import dataclass

import numpy as np

from src.balancing import buck_boost_currents, buck_boost_parameter
from src.cell import build_ocv_curve, open_circuit_voltage
from src.config import BoolArray, FloatArray, ProjectConfig, Technique
from src.params import SECONDS_PER_HOUR, CellParams, SimSpec
from src.switched_cap import gate_power_w, link_resistance_ohm

DIRECTION_SOC = np.array([[1.0, 0.0]])
SLOPE_WINDOW_SOC = 0.01


@dataclass(frozen=True, slots=True)
class PhysicsPrediction:
    balancing_time_s: FloatArray
    energy_dissipated_j: FloatArray


def inverse_ocv_integral(cell: CellParams) -> tuple[FloatArray, FloatArray]:
    curve = build_ocv_curve(cell)
    inverse = 1.0 / curve.volt
    steps = 0.5 * (inverse[1:] + inverse[:-1]) * np.diff(curve.soc)
    return curve.soc, np.concatenate(([0.0], np.cumsum(steps)))


def ocv_integral(cell: CellParams) -> tuple[FloatArray, FloatArray]:
    curve = build_ocv_curve(cell)
    steps = 0.5 * (curve.volt[1:] + curve.volt[:-1]) * np.diff(curve.soc)
    return curve.soc, np.concatenate(([0.0], np.cumsum(steps)))


def cell_energy_released_j(
    soc_start: FloatArray, soc_end: FloatArray, capacity_ah: FloatArray, cell: CellParams
) -> FloatArray:
    grid, integral = ocv_integral(cell)
    released = np.interp(soc_start, grid, integral) - np.interp(soc_end, grid, integral)
    return SECONDS_PER_HOUR * (capacity_ah * released).sum(axis=1)


def capacity_weighted_mean_soc(soc0: FloatArray, capacity_ah: FloatArray) -> FloatArray:
    return (capacity_ah * soc0).sum(axis=1) / capacity_ah.sum(axis=1)


def ocv_slope_v_per_soc(soc: FloatArray, cell: CellParams) -> FloatArray:
    curve = build_ocv_curve(cell)
    high = open_circuit_voltage(np.clip(soc + SLOPE_WINDOW_SOC, 0.0, 1.0), curve)
    low = open_circuit_voltage(np.clip(soc - SLOPE_WINDOW_SOC, 0.0, 1.0), curve)
    return (high - low) / (
        np.clip(soc + SLOPE_WINDOW_SOC, 0.0, 1.0) - np.clip(soc - SLOPE_WINDOW_SOC, 0.0, 1.0)
    )


def minimal_transport_state(
    soc0: FloatArray, capacity_ah: FloatArray, band: float, steps: int
) -> FloatArray:
    low = soc0.min(axis=1) - band
    high = soc0.max(axis=1)

    def imbalance(floor: FloatArray) -> FloatArray:
        clipped = np.clip(soc0, floor[:, None], (floor + band)[:, None])
        return (capacity_ah * (soc0 - clipped)).sum(axis=1)

    for _ in range(steps):
        mid = 0.5 * (low + high)
        positive = imbalance(mid) > 0.0
        low = np.where(positive, mid, low)
        high = np.where(positive, high, mid)
    floor = 0.5 * (low + high)
    return np.clip(soc0, floor[:, None], (floor + band)[:, None])


def passive_balancing_time_s(
    soc0: FloatArray,
    capacity_ah: FloatArray,
    r0_ohm: FloatArray,
    r_bleed_ohm: FloatArray,
    cell: CellParams,
    sim: SimSpec,
) -> FloatArray:
    grid, integral = inverse_ocv_integral(cell)
    target = soc0.min(axis=1, keepdims=True) + sim.soc_tolerance
    drained = np.interp(soc0, grid, integral) - np.interp(target, grid, integral)
    r_path = r_bleed_ohm[:, None] + r0_ohm + cell.r1_ohm
    cell_time = SECONDS_PER_HOUR * capacity_ah * r_path * np.clip(drained, 0.0, None)
    return cell_time.max(axis=1)


def passive_prediction(
    soc0: FloatArray,
    capacity_ah: FloatArray,
    r0_ohm: FloatArray,
    r_bleed_ohm: FloatArray,
    config: ProjectConfig,
) -> PhysicsPrediction:
    cell, sim = config.cell, config.sim
    time_s = passive_balancing_time_s(soc0, capacity_ah, r0_ohm, r_bleed_ohm, cell, sim)
    floor = soc0.min(axis=1, keepdims=True)
    final = np.minimum(soc0, floor + 0.5 * sim.soc_tolerance)
    last = soc0.argmax(axis=1)
    rows = np.arange(soc0.shape[0])
    final[rows, last] = np.minimum(soc0[rows, last], floor[:, 0] + sim.soc_tolerance)
    return PhysicsPrediction(time_s, cell_energy_released_j(soc0, final, capacity_ah, cell))


def buck_boost_prediction(
    soc0: FloatArray,
    capacity_ah: FloatArray,
    r0_ohm: FloatArray,
    inductance_h: FloatArray,
    frequency_hz: FloatArray,
    config: ProjectConfig,
) -> PhysicsPrediction:
    cell, sim, spec = config.cell, config.sim, config.ml_eval
    n = soc0.shape[0]
    mean_soc = capacity_weighted_mean_soc(soc0, capacity_ah)
    curve = build_ocv_curve(cell)
    voltage = open_circuit_voltage(mean_soc, curve)
    r0_mean = r0_ohm.mean(axis=1)
    flow = buck_boost_currents(
        np.repeat(DIRECTION_SOC, n, axis=0),
        np.column_stack((voltage, voltage)),
        np.column_stack((voltage, voltage)),
        np.column_stack((r0_mean, r0_mean)),
        buck_boost_parameter(
            config, 1.0 / (inductance_h * frequency_hz), inductance_h, frequency_hz
        ),
        config.buck_boost,
        sim.link_soc_deadband,
    )
    i_source, i_sink = flow.link_source_a[:, 0], flow.link_sink_a[:, 0]
    charge_ratio = i_sink / i_source

    band_soc = minimal_transport_state(soc0, capacity_ah, sim.soc_tolerance, spec.bisection_steps)
    crossing_ah = np.cumsum(capacity_ah * (soc0 - band_soc), axis=1)[:, :-1]
    moved_ah = np.abs(crossing_ah)
    time_s = np.maximum(SECONDS_PER_HOUR * moved_ah.max(axis=1) / i_sink, sim.dt_s)

    signed_ah = np.sign(crossing_ah) * moved_ah
    delivered = np.zeros_like(soc0)
    drawn = np.zeros_like(soc0)
    forward = np.clip(signed_ah, 0.0, None)
    backward = np.clip(-signed_ah, 0.0, None)
    delivered[:, 1:] += forward
    drawn[:, :-1] += forward / charge_ratio[:, None]
    delivered[:, :-1] += backward
    drawn[:, 1:] += backward / charge_ratio[:, None]
    final = soc0 + (delivered - drawn) / capacity_ah
    return PhysicsPrediction(time_s, cell_energy_released_j(soc0, final, capacity_ah, cell))


def flying_capacitor_prediction(
    soc0: FloatArray,
    capacity_ah: FloatArray,
    r0_ohm: FloatArray,
    capacitance_f: FloatArray,
    frequency_hz: FloatArray,
    config: ProjectConfig,
) -> PhysicsPrediction:
    cell, sim, spec = config.cell, config.sim, config.ml_eval
    n, cells = soc0.shape
    mean_soc = capacity_weighted_mean_soc(soc0, capacity_ah)
    slope = ocv_slope_v_per_soc(mean_soc, cell)
    r_link = link_resistance_ohm(
        1.0 / (capacitance_f * frequency_hz), r0_ohm, config.flying_capacitor, frequency_hz
    )
    links = cells - 1
    diff = np.zeros((links, cells))
    diff[np.arange(links), np.arange(links)] = 1.0
    diff[np.arange(links), np.arange(links) + 1] = -1.0
    system = (diff @ diff.T)[None, :, :] * cell.r1_ohm
    system = system + r_link[:, :, None] * np.eye(links)[None, :, :]
    conductance = np.linalg.inv(system)
    stiffness = (slope[:, None, None] / SECONDS_PER_HOUR) * np.einsum(
        "jk,njl,lm->nkm", diff, conductance, diff
    )
    scale = 1.0 / np.sqrt(capacity_ah)
    symmetric = scale[:, :, None] * stiffness * scale[:, None, :]
    rates, vectors = np.linalg.eigh(symmetric)
    coefficients = np.einsum("nij,ni->nj", vectors, np.sqrt(capacity_ah) * soc0)

    def spread_at(time_s: FloatArray) -> FloatArray:
        return np.ptp(state_at(time_s), axis=1)

    def state_at(time_s: FloatArray) -> FloatArray:
        decay = np.exp(-np.clip(rates, 0.0, None) * time_s[:, None])
        return scale * np.einsum("nij,nj->ni", vectors, coefficients * decay)

    grid = np.geomspace(spec.time_grid_min_s, sim.t_max_s, spec.time_grid_points)
    spread_grid = np.stack([spread_at(np.full(n, t)) for t in grid], axis=1)
    reached: BoolArray = spread_grid < sim.soc_tolerance
    first = np.argmax(reached, axis=1)
    balanced = reached.any(axis=1)
    high = np.where(balanced, grid[first], sim.t_max_s)
    low = np.where(first > 0, grid[np.maximum(first - 1, 0)], 0.0)
    for _ in range(spec.bisection_steps):
        mid = 0.5 * (low + high)
        below = spread_at(mid) < sim.soc_tolerance
        high = np.where(below, mid, high)
        low = np.where(below, low, mid)
    time_s = np.where(balanced, high, sim.t_max_s)
    final = state_at(time_s)
    gate_w = gate_power_w(
        1.0 / (capacitance_f * frequency_hz), config.flying_capacitor, frequency_hz, (n, links)
    )
    energy = cell_energy_released_j(soc0, final, capacity_ah, cell)
    return PhysicsPrediction(time_s, energy + gate_w.sum(axis=1) * time_s)


def physics_prediction(
    technique: Technique,
    soc0: FloatArray,
    capacity_ah: FloatArray,
    r0_ohm: FloatArray,
    component: FloatArray,
    frequency_hz: FloatArray,
    config: ProjectConfig,
) -> PhysicsPrediction:
    match technique:
        case Technique.PASSIVE:
            return passive_prediction(soc0, capacity_ah, r0_ohm, component, config)
        case Technique.BUCK_BOOST:
            return buck_boost_prediction(soc0, capacity_ah, r0_ohm, component, frequency_hz, config)
        case Technique.FLYING_CAPACITOR:
            return flying_capacitor_prediction(
                soc0, capacity_ah, r0_ohm, component, frequency_hz, config
            )
