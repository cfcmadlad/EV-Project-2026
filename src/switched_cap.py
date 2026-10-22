import numpy as np

from src.config import FloatArray
from src.params import FlyingCapModel, FlyingCapSpec


def two_phase_resistance_ohm(
    inverse_fc: FloatArray, r_phase_one_ohm: FloatArray, r_phase_two_ohm: FloatArray
) -> FloatArray:
    x_one = -np.expm1(-inverse_fc / (2.0 * r_phase_one_ohm))
    x_two = -np.expm1(-inverse_fc / (2.0 * r_phase_two_ohm))
    return inverse_fc * (1.0 / x_one + 1.0 / x_two - 1.0)


def link_resistance_ohm(
    inverse_fc: FloatArray, r0: FloatArray, hardware: FlyingCapSpec
) -> FloatArray:
    if hardware.model is FlyingCapModel.MIDSEM:
        return (
            inverse_fc[:, None]
            + 2.0 * hardware.r_ds_on_ohm
            + hardware.esr_ohm
            + r0[:, :-1]
            + r0[:, 1:]
        )
    r_phase = r0 + 2.0 * hardware.r_ds_on_ohm + hardware.esr_ohm
    return two_phase_resistance_ohm(inverse_fc[:, None], r_phase[:, :-1], r_phase[:, 1:])


def element_resistance_ohm(
    inverse_fc: FloatArray,
    r0: FloatArray,
    hardware: FlyingCapSpec,
    total: FloatArray | None = None,
) -> FloatArray:
    if hardware.model is FlyingCapModel.MIDSEM:
        return inverse_fc[:, None] + 2.0 * hardware.r_ds_on_ohm + hardware.esr_ohm
    if total is None:
        total = link_resistance_ohm(inverse_fc, r0, hardware)
    return total - r0[:, :-1] - r0[:, 1:]


def link_resistances_ohm(
    inverse_fc: FloatArray, r0: FloatArray, hardware: FlyingCapSpec
) -> FloatArray:
    total = link_resistance_ohm(inverse_fc, r0, hardware)
    element = np.broadcast_to(element_resistance_ohm(inverse_fc, r0, hardware, total), total.shape)
    return np.stack((total, element), axis=1)
