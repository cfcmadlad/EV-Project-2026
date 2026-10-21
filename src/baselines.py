import numpy as np

from src.cell import build_ocv_curve
from src.config import FloatArray
from src.params import SECONDS_PER_HOUR, CellParams, SimSpec


def inverse_ocv_integral(cell: CellParams) -> tuple[FloatArray, FloatArray]:
    curve = build_ocv_curve(cell)
    inverse = 1.0 / curve.volt
    steps = 0.5 * (inverse[1:] + inverse[:-1]) * np.diff(curve.soc)
    return curve.soc, np.concatenate(([0.0], np.cumsum(steps)))


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
