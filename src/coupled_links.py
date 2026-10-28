import numpy as np

from src.config import BoolArray, FloatArray
from src.flow import StepFlow
from src.params import BuckBoostSpec

ITERATIONS = 5


def link_state(soc: FloatArray, deadband: float) -> tuple[BoolArray, BoolArray]:
    difference = soc[:, :-1] - soc[:, 1:]
    return np.abs(difference) > deadband, difference > 0.0


def source_sink(values: FloatArray, forward: BoolArray) -> tuple[FloatArray, FloatArray]:
    left, right = values[:, :-1], values[:, 1:]
    return np.where(forward, left, right), np.where(forward, right, left)


def scatter(i_source: FloatArray, i_sink: FloatArray, forward: BoolArray) -> FloatArray:
    current = np.zeros((forward.shape[0], forward.shape[1] + 1))
    current[:, :-1] = np.where(forward, i_source, -i_sink)
    current[:, 1:] += np.where(forward, -i_sink, i_source)
    return current


def solve_tridiagonal(
    lower: FloatArray, diagonal: FloatArray, upper: FloatArray, rhs: FloatArray
) -> FloatArray:
    n, size = diagonal.shape
    c_prime = np.zeros((n, size))
    d_prime = np.zeros((n, size))
    c_prime[:, 0] = upper[:, 0] / diagonal[:, 0]
    d_prime[:, 0] = rhs[:, 0] / diagonal[:, 0]
    for j in range(1, size):
        denominator = diagonal[:, j] - lower[:, j] * c_prime[:, j - 1]
        c_prime[:, j] = upper[:, j] / denominator
        d_prime[:, j] = (rhs[:, j] - lower[:, j] * d_prime[:, j - 1]) / denominator
    solution = np.zeros((n, size))
    solution[:, -1] = d_prime[:, -1]
    for j in range(size - 2, -1, -1):
        solution[:, j] = d_prime[:, j] - c_prime[:, j] * solution[:, j + 1]
    return solution


def bb_parameters(
    parameter: FloatArray,
) -> tuple[FloatArray, FloatArray | float, FloatArray | float, FloatArray | float]:
    if parameter.ndim == 1:
        return parameter, 1.0, 0.0, 0.0
    return parameter[:, 0], parameter[:, 1:2], parameter[:, 2:3], parameter[:, 3:4]


def gate_supply(
    active: BoolArray, gate_w: FloatArray | float, ocv_source: FloatArray, forward: BoolArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    power = np.where(active, gate_w, 0.0) * np.ones_like(ocv_source)
    supply_a = power / ocv_source
    cell_a = scatter(supply_a, np.zeros_like(supply_a), forward)
    return supply_a, cell_a, power.sum(axis=1), supply_a.sum(axis=1)


def flying_gate_w(magnitude: FloatArray, resistance: FloatArray) -> FloatArray | float:
    if resistance.shape[1] < 3:
        return 0.0
    return resistance[:, 2]


def buck_boost_currents_net(
    soc: FloatArray,
    ocv: FloatArray,
    emf: FloatArray,
    r0: FloatArray,
    inverse_lf: FloatArray,
    hardware: BuckBoostSpec,
    deadband: float,
) -> StepFlow:
    active, forward = link_state(soc, deadband)
    inverse_lf, ratio, overhead, gate_w = bb_parameters(inverse_lf)
    ocv_source, ocv_sink = source_sink(ocv, forward)
    conductance = np.where(active, 0.5 * hardware.duty**2 * inverse_lf[:, None], 0.0)
    resistance = hardware.r_inductor_ohm + hardware.r_ds_on_ohm
    supply_a, supply_cell, gate_power, gate_amps = gate_supply(active, gate_w, ocv_source, forward)
    net = np.zeros_like(emf)
    for _ in range(ITERATIONS):
        terminal = emf - r0 * net
        v_source, v_sink = source_sink(terminal, forward)
        i_source = ratio * conductance * v_source
        i_peak = 2.0 * i_source / hardware.duty
        sink_duty = hardware.duty * v_source / v_sink
        loss = resistance * i_peak**2 * (hardware.duty + sink_duty) / 3.0
        i_sink = (v_source * i_source - loss - np.where(active, overhead, 0.0)) / v_sink
        net = scatter(i_source, i_sink, forward) + supply_cell
    return StepFlow(
        cell_current_a=net,
        removed_w=np.einsum("ij,ij->i", ocv_source, i_source + supply_a),
        delivered_w=np.einsum("ij,ij->i", ocv_sink, i_sink),
        link_source_a=i_source,
        link_sink_a=i_sink,
        link_forward=forward,
        terminal_v=emf - r0 * net,
        gate_drive_w=gate_power,
        gate_drive_a=gate_amps,
    )


def flying_capacitor_currents_net(
    soc: FloatArray,
    ocv: FloatArray,
    emf: FloatArray,
    r0: FloatArray,
    resistance: FloatArray,
    deadband: float,
) -> StepFlow:
    active, _ = link_state(soc, deadband)
    element = resistance[:, 1]
    r_left, r_right = r0[:, :-1], r0[:, 1:]
    diagonal = np.where(active, element + r_left + r_right, 1.0)
    lower = np.where(active, -r_left, 0.0)
    upper = np.where(active, -r_right, 0.0)
    rhs = np.where(active, emf[:, :-1] - emf[:, 1:], 0.0)
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0
    link = solve_tridiagonal(lower, diagonal, upper, rhs)
    forward = link > 0.0
    magnitude = np.abs(link)
    ocv_source, ocv_sink = source_sink(ocv, forward)
    net = np.zeros_like(emf)
    net[:, :-1] += link
    net[:, 1:] -= link
    supply_a, supply_cell, gate_power, gate_amps = gate_supply(
        magnitude > 0.0, flying_gate_w(magnitude, resistance), ocv_source, forward
    )
    return StepFlow(
        cell_current_a=net + supply_cell,
        removed_w=np.einsum("ij,ij->i", ocv_source, magnitude + supply_a),
        delivered_w=np.einsum("ij,ij->i", ocv_sink, magnitude),
        link_source_a=magnitude,
        link_sink_a=magnitude,
        link_forward=forward,
        gate_drive_w=gate_power,
        gate_drive_a=gate_amps,
    )
