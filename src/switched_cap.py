import numpy as np

from src.calibration import circuit_maps
from src.config import FloatArray
from src.params import FlyingCapModel, FlyingCapSpec


def two_phase_resistance_ohm(
    inverse_fc: FloatArray, r_phase_one_ohm: FloatArray, r_phase_two_ohm: FloatArray
) -> FloatArray:
    x_one = -np.expm1(-inverse_fc / (2.0 * r_phase_one_ohm))
    x_two = -np.expm1(-inverse_fc / (2.0 * r_phase_two_ohm))
    return inverse_fc * (1.0 / x_one + 1.0 / x_two - 1.0)


def coth_resistance_ohm(inverse_fc: FloatArray, r_phase_ohm: FloatArray) -> FloatArray:
    return inverse_fc / np.tanh(inverse_fc / (4.0 * r_phase_ohm))


def link_resistance_ohm(
    inverse_fc: FloatArray,
    r0: FloatArray,
    hardware: FlyingCapSpec,
    frequency_hz: FloatArray | None = None,
) -> FloatArray:
    if hardware.model is FlyingCapModel.CIRCUIT:
        return circuit_element_ohm(inverse_fc, frequency_hz)[:, None] + r0[:, :-1] + r0[:, 1:]
    if hardware.model is FlyingCapModel.MIDSEM:
        return (
            inverse_fc[:, None]
            + 2.0 * hardware.r_ds_on_ohm
            + hardware.esr_ohm
            + r0[:, :-1]
            + r0[:, 1:]
        )
    r_phase = r0 + 2.0 * hardware.r_ds_on_ohm + hardware.esr_ohm
    if hardware.model is FlyingCapModel.COTH:
        return coth_resistance_ohm(inverse_fc[:, None], 0.5 * (r_phase[:, :-1] + r_phase[:, 1:]))
    return two_phase_resistance_ohm(inverse_fc[:, None], r_phase[:, :-1], r_phase[:, 1:])


def circuit_element_ohm(inverse_fc: FloatArray, frequency_hz: FloatArray | None) -> FloatArray:
    if frequency_hz is None:
        raise ValueError("the circuit-calibrated model needs the switching frequency")
    return circuit_maps().flying_element_ohm(1.0 / (inverse_fc * frequency_hz), frequency_hz)


def element_resistance_ohm(
    inverse_fc: FloatArray,
    r0: FloatArray,
    hardware: FlyingCapSpec,
    total: FloatArray | None = None,
    frequency_hz: FloatArray | None = None,
) -> FloatArray:
    if hardware.model is FlyingCapModel.CIRCUIT:
        return np.broadcast_to(
            circuit_element_ohm(inverse_fc, frequency_hz)[:, None], r0[:, :-1].shape
        )
    if hardware.model is FlyingCapModel.MIDSEM:
        return inverse_fc[:, None] + 2.0 * hardware.r_ds_on_ohm + hardware.esr_ohm
    if total is None:
        total = link_resistance_ohm(inverse_fc, r0, hardware)
    return total - r0[:, :-1] - r0[:, 1:]


def gate_power_w(
    inverse_fc: FloatArray, hardware: FlyingCapSpec, frequency_hz: FloatArray | None, shape: tuple
) -> FloatArray:
    if hardware.model is not FlyingCapModel.CIRCUIT:
        return np.zeros(shape)
    capacitance = 1.0 / (inverse_fc * frequency_hz)
    power = circuit_maps().flying_gate_w(capacitance, frequency_hz)
    return np.broadcast_to(power[:, None], shape)


def link_resistances_ohm(
    inverse_fc: FloatArray,
    r0: FloatArray,
    hardware: FlyingCapSpec,
    frequency_hz: FloatArray | None = None,
) -> FloatArray:
    total = link_resistance_ohm(inverse_fc, r0, hardware, frequency_hz)
    element = element_resistance_ohm(inverse_fc, r0, hardware, total, frequency_hz)
    element = np.broadcast_to(element, total.shape)
    overhead = gate_power_w(inverse_fc, hardware, frequency_hz, total.shape)
    return np.stack((total, element, overhead), axis=1)
