from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.config import SECONDS_PER_HOUR, CellParams, FloatArray


@dataclass(frozen=True)
class OcvCurve:
    soc: FloatArray
    volt: FloatArray


def build_ocv_curve(cell: CellParams) -> OcvCurve:
    soc = np.linspace(0.0, 1.0, cell.ocv_grid_points)
    return OcvCurve(soc=soc, volt=PchipInterpolator(cell.ocv_soc, cell.ocv_volt)(soc))


def open_circuit_voltage(soc: FloatArray, curve: OcvCurve) -> FloatArray:
    return np.interp(soc, curve.soc, curve.volt)


def rc_step_gain(dt_s: float, cell: CellParams) -> float:
    return float(-np.expm1(-dt_s / (cell.r1_ohm * cell.c1_farad)))


def soc_per_amp_step(dt_s: float, capacity_ah: FloatArray) -> FloatArray:
    return dt_s / (SECONDS_PER_HOUR * capacity_ah)
