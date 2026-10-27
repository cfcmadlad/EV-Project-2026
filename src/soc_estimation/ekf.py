import math
from dataclasses import dataclass

import numpy as np

from src.config import FloatArray
from src.params import SECONDS_PER_HOUR

GRID_POINTS = 2001
PROCESS_STD_V1 = 1e-3
INITIAL_STD_V1 = 1e-2
MODE_CONSTANT = "constant"
MODE_TABLE = "table"
MODES = (MODE_CONSTANT, MODE_TABLE)


@dataclass(frozen=True, slots=True)
class EkfCell:
    capacity_ah: float
    ocv_v: list[float]
    docv_v_per_soc: list[float]
    r0_ohm: list[float]
    r1_ohm: list[float]
    tau_s: list[float]


def build_cell(entry: dict, mode: str) -> EkfCell:
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    soc = np.array(entry["ocv"]["soc"])
    volt = np.array(entry["ocv"]["volt_v"])
    ocv = np.interp(grid, soc, volt)
    slope = np.interp(grid, soc, np.gradient(volt, soc))
    if mode == MODE_CONSTANT:
        cell = entry["one_rc_cell"]
        r0 = np.full(GRID_POINTS, cell["r0_ohm"])
        r1 = np.full(GRID_POINTS, cell["r1_ohm"])
        tau = np.full(GRID_POINTS, cell["r1_ohm"] * cell["c1_farad"])
    else:
        table = entry["hppc_tables"]
        order = np.argsort(table["soc"])
        points = np.array(table["soc"])[order]
        r0 = np.interp(grid, points, np.array(table["r0_ohm"])[order])
        r1 = np.interp(grid, points, np.array(table["one_rc_r1_ohm"])[order])
        tau = np.interp(grid, points, np.array(table["one_rc_tau_s"])[order])
    return EkfCell(
        capacity_ah=entry["capacity_ah"],
        ocv_v=ocv.tolist(),
        docv_v_per_soc=slope.tolist(),
        r0_ohm=r0.tolist(),
        r1_ohm=r1.tolist(),
        tau_s=tau.tolist(),
    )


def run_ekf(
    cell: EkfCell,
    current_a: FloatArray,
    voltage_v: FloatArray,
    soc0: float,
    process_std: float,
    measurement_std_v: float,
    initial_std: float,
    step_s: float,
) -> FloatArray:
    n = current_a.size
    estimate = np.empty(n)
    currents, voltages = current_a.tolist(), voltage_v.tolist()
    soc, v1 = soc0, 0.0
    p00, p01, p11 = initial_std**2, 0.0, INITIAL_STD_V1**2
    q00, q11, rr = process_std**2, PROCESS_STD_V1**2, measurement_std_v**2
    per_amp = step_s / (SECONDS_PER_HOUR * cell.capacity_ah)
    scale = GRID_POINTS - 1
    ocv, docv, r0s, r1s, taus = (
        cell.ocv_v,
        cell.docv_v_per_soc,
        cell.r0_ohm,
        cell.r1_ohm,
        cell.tau_s,
    )
    for k in range(n):
        i = currents[k]
        soc -= i * per_amp
        soc = 0.0 if soc < 0.0 else 1.0 if soc > 1.0 else soc
        idx = int(soc * scale + 0.5)
        a = math.exp(-step_s / taus[idx])
        v1 = a * v1 + (1.0 - a) * r1s[idx] * i
        p00 += q00
        p01 *= a
        p11 = a * a * p11 + q11
        slope = docv[idx]
        innovation = voltages[k] - (ocv[idx] - i * r0s[idx] - v1)
        ph0 = slope * p00 - p01
        ph1 = slope * p01 - p11
        s = slope * ph0 - ph1 + rr
        k0, k1 = ph0 / s, ph1 / s
        soc += k0 * innovation
        v1 += k1 * innovation
        p00 -= k0 * ph0
        p01 -= k0 * ph1
        p11 -= k1 * ph1
        soc = 0.0 if soc < 0.0 else 1.0 if soc > 1.0 else soc
        estimate[k] = soc
    return estimate
