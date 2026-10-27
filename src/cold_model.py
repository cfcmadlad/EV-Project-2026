from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.cell import OcvCurve, open_circuit_voltage
from src.config import FloatArray
from src.hg2 import Recording, charge_throughput_ah
from src.identify import TwoRcFit
from src.validate import Replay

SCALE_BOUNDS = ([0.3, 0.3, 0.3, 0.3, 0.3, 0.5], [3.0, 3.0, 3.0, 10.0, 10.0, 200.0])
INITIAL_SCALES = (1.0, 1.0, 1.0, 1.0, 1.0, 8.0)
DIFF_STEP = 1e-2


@dataclass(frozen=True, slots=True)
class TableModel:
    soc: FloatArray
    r0_ohm: FloatArray
    r1_ohm: FloatArray
    tau1_s: FloatArray
    r2_ohm: FloatArray
    tau2_s: FloatArray


def table_model(fits: list[TwoRcFit]) -> TableModel:
    ordered = sorted(fits, key=lambda fit: fit.soc)
    return TableModel(
        soc=np.array([f.soc for f in ordered]),
        r0_ohm=np.array([f.r0_ohm for f in ordered]),
        r1_ohm=np.array([f.r1_ohm for f in ordered]),
        tau1_s=np.array([f.tau1_s for f in ordered]),
        r2_ohm=np.array([f.r2_ohm for f in ordered]),
        tau2_s=np.array([f.tau2_s for f in ordered]),
    )


def rc_response(time_s: FloatArray, drive: FloatArray, r_ohm: FloatArray, tau_s: FloatArray):
    decay = np.exp(-np.diff(time_s) / tau_s[:-1])
    forcing = ((1.0 - decay) * r_ohm[:-1] * drive[:-1]).tolist()
    state = np.empty(time_s.size)
    state[0] = 0.0
    value = 0.0
    for k, (a, b) in enumerate(zip(decay.tolist(), forcing, strict=True), start=1):
        value = a * value + b
        state[k] = value
    return state


def terminal_voltage(
    time_s: FloatArray,
    current_a: FloatArray,
    soc: FloatArray,
    model: TableModel,
    curve: OcvCurve,
    scales: FloatArray,
) -> FloatArray:
    clipped = np.clip(soc, 0.0, 1.0)

    def lookup(values: FloatArray) -> FloatArray:
        return np.interp(clipped, model.soc, values)

    drive = scales[5] * np.arcsinh(current_a / scales[5])
    voltage = open_circuit_voltage(soc, curve) - scales[0] * lookup(model.r0_ohm) * drive
    voltage -= rc_response(
        time_s, drive, scales[1] * lookup(model.r1_ohm), scales[3] * lookup(model.tau1_s)
    )
    voltage -= rc_response(
        time_s, drive, scales[2] * lookup(model.r2_ohm), scales[4] * lookup(model.tau2_s)
    )
    return voltage


def calibrate(
    time_s: list[FloatArray],
    current_a: list[FloatArray],
    voltage_v: list[FloatArray],
    soc: list[FloatArray],
    model: TableModel,
    curve: OcvCurve,
) -> FloatArray:
    def residual(scales: FloatArray) -> FloatArray:
        return np.concatenate(
            [
                terminal_voltage(t, i, s, model, curve, scales) - v
                for t, i, v, s in zip(time_s, current_a, voltage_v, soc, strict=True)
            ]
        )

    fit = least_squares(residual, INITIAL_SCALES, bounds=SCALE_BOUNDS, diff_step=DIFF_STEP)
    return fit.x


def replay_calibrated(
    recording: Recording,
    sign: float,
    capacity_ah: float,
    model: TableModel,
    curve: OcvCurve,
    scales: FloatArray,
) -> Replay:
    current = sign * recording.current_a
    time_s = recording.time_s
    rest_v = recording.voltage_v[0] + scales[0] * model.r0_ohm[-1] * current[0]
    soc_start = float(np.interp(rest_v, curve.volt, curve.soc))
    soc = soc_start - charge_throughput_ah(time_s, current) / capacity_ah
    return Replay(
        time_s=time_s,
        measured_v=recording.voltage_v,
        model_v=terminal_voltage(time_s, current, soc, model, curve, scales),
        soc=soc,
        current_a=current,
        soc_start=soc_start,
    )
