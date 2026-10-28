from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.config import RESULTS_DIR, FloatArray
from src.table_io import read_rows

SATURATION_ENERGY_J = 50e-6
LTSPICE_DIR = RESULTS_DIR / "ltspice"


def saturation_current_a(inductance_h: FloatArray | float) -> FloatArray | float:
    return np.sqrt(2.0 * SATURATION_ENERGY_J / inductance_h)


def buck_boost_feasible(
    inductance_h: FloatArray, frequency_hz: FloatArray, duty: float, cell_v_max: float
) -> FloatArray:
    peak = cell_v_max * duty / (inductance_h * frequency_hz)
    return peak <= saturation_current_a(inductance_h)


class LogGrid:
    def __init__(self, x: FloatArray, y: FloatArray, values: FloatArray) -> None:
        self.low = np.array([x.min(), y.min()])
        self.high = np.array([x.max(), y.max()])
        self.interpolator = RegularGridInterpolator((np.log(x), np.log(y)), values)

    def __call__(self, x: FloatArray, y: FloatArray) -> FloatArray:
        x = np.clip(np.asarray(x, dtype=np.float64), self.low[0], self.high[0])
        y = np.clip(np.asarray(y, dtype=np.float64), self.low[1], self.high[1])
        x, y = np.broadcast_arrays(x, y)
        points = np.stack((np.log(x), np.log(y)), axis=-1)
        return self.interpolator(points.reshape(-1, 2)).reshape(x.shape)


@dataclass(frozen=True, slots=True)
class CircuitMaps:
    flying_element_ohm: LogGrid
    flying_gate_w: LogGrid
    buck_boost_current_ratio: LogGrid
    buck_boost_overhead_w: LogGrid


def _grid(rows: list[dict], x_key: str, y_key: str, value: str) -> LogGrid:
    xs = np.array(sorted({r[x_key] for r in rows}))
    ys = np.array(sorted({r[y_key] for r in rows}))
    table = np.zeros((xs.size, ys.size))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cell = [r[value] for r in rows if r[x_key] == x and r[y_key] == y]
            table[i, j] = float(np.mean(cell))
    return LogGrid(xs, ys, table)


def flying_rows(rows: list[dict]) -> list[dict]:
    real = [r for r in rows if r["switch"] == "real"]
    for r in real:
        r["element_ohm"] = r["r_eff_ohm"] - 2.0 * r["r0_ohm"]
    return real


def buck_boost_rows(rows: list[dict]) -> list[dict]:
    sync = [r for r in rows if r["switch"] == "real" and r["rectifier"] == "sync"]
    cells = {(r["l_h"], r["f_hz"]) for r in sync}
    real = []
    for cell in cells:
        group = [r for r in sync if (r["l_h"], r["f_hz"]) == cell]
        real += [r for r in group if r["saturates"] == 0.0] or group
    for r in real:
        r["current_ratio"] = r["i_src_a"] / r["py_i_src_mean_a"]
        r["overhead_w"] = max(r["p_loss_total_w"] - r["py_loss_mean_w"], 0.0)
    return real


@cache
def circuit_maps(directory: Path = LTSPICE_DIR) -> CircuitMaps:
    flying = flying_rows(read_rows(directory / "flying_cap.csv"))
    buck = buck_boost_rows(read_rows(directory / "buck_boost.csv"))
    return CircuitMaps(
        flying_element_ohm=_grid(flying, "c_f", "f_hz", "element_ohm"),
        flying_gate_w=_grid(flying, "c_f", "f_hz", "p_gate_w"),
        buck_boost_current_ratio=_grid(buck, "l_h", "f_hz", "current_ratio"),
        buck_boost_overhead_w=_grid(buck, "l_h", "f_hz", "overhead_w"),
    )
