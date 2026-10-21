from dataclasses import dataclass

import numpy as np

from src.cell import OcvCurve, build_ocv_curve, open_circuit_voltage
from src.config import FloatArray
from src.hg2 import Recording, charge_throughput_ah
from src.params import CellParams, IdentificationSpec

Branch = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Replay:
    time_s: FloatArray
    measured_v: FloatArray
    model_v: FloatArray
    soc: FloatArray
    current_a: FloatArray
    soc_start: float


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    rmse_whole_v: float
    max_abs_whole_v: float
    rmse_window_v: float
    max_abs_window_v: float
    window_fraction: float
    soc_start: float
    soc_end: float


def rc_state(time_s: FloatArray, current_a: FloatArray, r1_ohm: float, tau_s: float) -> FloatArray:
    decay = np.exp(-np.diff(time_s) / tau_s)
    drive = (r1_ohm * (1.0 - decay) * current_a[:-1]).tolist()
    keep = decay.tolist()
    state = np.empty(time_s.size)
    state[0] = 0.0
    value = 0.0
    for k, (a, b) in enumerate(zip(keep, drive, strict=True), start=1):
        value = a * value + b
        state[k] = value
    return state


def replay_cycle(
    recording: Recording,
    sign: float,
    capacity_ah: float,
    r0_ohm: float,
    branches: tuple[Branch, ...],
    curve: OcvCurve,
) -> Replay:
    current = sign * recording.current_a
    time_s = recording.time_s
    rest_v = recording.voltage_v[0] + r0_ohm * current[0]
    soc_start = float(np.interp(rest_v, curve.volt, curve.soc))
    soc = soc_start - charge_throughput_ah(time_s, current) / capacity_ah
    model = open_circuit_voltage(soc, curve) - r0_ohm * current
    for r_ohm, tau_s in branches:
        model = model - rc_state(time_s, current, r_ohm, tau_s)
    return Replay(
        time_s=time_s,
        measured_v=recording.voltage_v,
        model_v=model,
        soc=soc,
        current_a=current,
        soc_start=soc_start,
    )


def replay_cell(recording: Recording, sign: float, cell: CellParams) -> Replay:
    return replay_cycle(
        recording,
        sign,
        cell.capacity_ah,
        cell.r0_ohm,
        ((cell.r1_ohm, cell.tau_s),),
        build_ocv_curve(cell),
    )


def replay_metrics(replay: Replay, spec: IdentificationSpec) -> ReplayMetrics:
    error = replay.model_v - replay.measured_v
    low, high = spec.validation_window
    window = (replay.soc >= low) & (replay.soc <= high)
    return ReplayMetrics(
        rmse_whole_v=float(np.sqrt(np.mean(error**2))),
        max_abs_whole_v=float(np.abs(error).max()),
        rmse_window_v=float(np.sqrt(np.mean(error[window] ** 2))),
        max_abs_window_v=float(np.abs(error[window]).max()),
        window_fraction=float(window.mean()),
        soc_start=replay.soc_start,
        soc_end=float(replay.soc[-1]),
    )
