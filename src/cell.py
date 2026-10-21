from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.config import FloatArray
from src.params import SECONDS_PER_HOUR, CellParams


@dataclass(frozen=True, slots=True)
class OcvCurve:
    soc: FloatArray
    volt: FloatArray


def build_ocv_curve(cell: CellParams) -> OcvCurve:
    soc = np.linspace(0.0, 1.0, cell.ocv.grid_points)
    return OcvCurve(soc=soc, volt=PchipInterpolator(cell.ocv.soc, cell.ocv.volt_v)(soc))


def open_circuit_voltage(soc: FloatArray, curve: OcvCurve) -> FloatArray:
    return np.interp(soc, curve.soc, curve.volt)


def rc_step_gain(dt_s: float, cell: CellParams) -> float:
    return float(-np.expm1(-dt_s / cell.tau_s))


def soc_per_amp_step(dt_s: float, capacity_ah: FloatArray) -> FloatArray:
    return dt_s / (SECONDS_PER_HOUR * capacity_ah)


def rc_step_gain_per_pack(dt_s: float, tau_s: FloatArray) -> FloatArray:
    return -np.expm1(-dt_s / tau_s)
