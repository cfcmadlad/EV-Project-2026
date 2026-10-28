import math
from dataclasses import dataclass

import numpy as np

from src.config import FloatArray
from src.params import SECONDS_PER_HOUR
from src.soc_estimation.ekf import GRID_POINTS, INITIAL_STD_V1, PROCESS_STD_V1

PROCESS_STD_HYSTERESIS = 1e-3
INITIAL_STD_HYSTERESIS = 0.5
STATES = 4


@dataclass(frozen=True, slots=True)
class Ekf2Cell:
    capacity_ah: float
    ocv_v: list[float]
    docv_v_per_soc: list[float]
    r0_ohm: list[float]
    r1_ohm: list[float]
    tau1_s: list[float]
    r2_ohm: list[float]
    tau2_s: list[float]
    hysteresis_v: list[float]


def build_cell2(entry: dict) -> Ekf2Cell:
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    soc = np.array(entry["ocv"]["soc"])
    volt = np.array(entry["ocv"]["volt_v"])
    table = entry["hppc_tables"]
    order = np.argsort(table["soc"])
    points = np.array(table["soc"])[order]

    def lookup(name: str) -> list[float]:
        return np.interp(grid, points, np.array(table[name])[order]).tolist()

    return Ekf2Cell(
        capacity_ah=entry["capacity_ah"],
        ocv_v=np.interp(grid, soc, volt).tolist(),
        docv_v_per_soc=np.interp(grid, soc, np.gradient(volt, soc)).tolist(),
        r0_ohm=lookup("r0_ohm"),
        r1_ohm=lookup("two_rc_fast_r_ohm"),
        tau1_s=lookup("two_rc_fast_tau_s"),
        r2_ohm=lookup("two_rc_slow_r_ohm"),
        tau2_s=lookup("two_rc_slow_tau_s"),
        hysteresis_v=np.interp(grid, soc, np.array(entry["hysteresis_half_width_v"])).tolist(),
    )


def run_ekf2(
    cell: Ekf2Cell,
    current_a: FloatArray,
    voltage_v: FloatArray,
    soc0: float,
    process_std: float,
    measurement_std_v: float,
    initial_std: float,
    hysteresis_rate: float,
    step_s: float,
) -> FloatArray:
    n = current_a.size
    estimate = np.empty((n, STATES))
    currents, voltages = current_a.tolist(), voltage_v.tolist()
    x = np.array([soc0, 0.0, 0.0, 0.0])
    p = np.diag([initial_std**2, INITIAL_STD_V1**2, INITIAL_STD_V1**2, INITIAL_STD_HYSTERESIS**2])
    q = np.diag([process_std**2, PROCESS_STD_V1**2, PROCESS_STD_V1**2, PROCESS_STD_HYSTERESIS**2])
    rr = measurement_std_v**2
    per_amp = step_s / (SECONDS_PER_HOUR * cell.capacity_ah)
    scale = GRID_POINTS - 1
    f = np.eye(STATES)
    h_vec = np.zeros(STATES)
    for k in range(n):
        i = currents[k]
        soc = min(max(x[0] - i * per_amp, 0.0), 1.0)
        idx = int(soc * scale + 0.5)
        a1 = math.exp(-step_s / cell.tau1_s[idx])
        a2 = math.exp(-step_s / cell.tau2_s[idx])
        decay = math.exp(-hysteresis_rate * abs(i) * per_amp)
        direction = (i > 0.0) - (i < 0.0)
        x = np.array(
            [
                soc,
                a1 * x[1] + (1.0 - a1) * cell.r1_ohm[idx] * i,
                a2 * x[2] + (1.0 - a2) * cell.r2_ohm[idx] * i,
                decay * x[3] + (1.0 - decay) * direction,
            ]
        )
        f[1, 1], f[2, 2], f[3, 3] = a1, a2, decay
        p = f @ p @ f.T + q
        h_vec[0] = cell.docv_v_per_soc[idx]
        h_vec[1] = -1.0
        h_vec[2] = -1.0
        h_vec[3] = -cell.hysteresis_v[idx]
        predicted = (
            cell.ocv_v[idx] - i * cell.r0_ohm[idx] - x[1] - x[2] - cell.hysteresis_v[idx] * x[3]
        )
        ph = p @ h_vec
        gain = ph / (h_vec @ ph + rr)
        x = x + gain * (voltages[k] - predicted)
        p = p - np.outer(gain, ph)
        x[0] = min(max(x[0], 0.0), 1.0)
        x[3] = min(max(x[3], -1.0), 1.0)
        estimate[k] = x
    return estimate
