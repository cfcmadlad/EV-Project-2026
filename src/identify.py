from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.config import FloatArray, IntArray
from src.hg2 import Recording, charge_throughput_ah, segments
from src.params import IdentificationSpec

STRICT_STEP_V = 1e-6
START_TAUS_S = (5.0, 30.0, 150.0)
START_R1_OHM = 0.01
TWO_RC_STARTS = (
    (0.01, 1.0, 0.01, 60.0),
    (0.01, 3.0, 0.01, 200.0),
    (0.01, 0.5, 0.01, 30.0),
)
MV_PER_V = 1e3
PERCENT = 100.0


@dataclass(frozen=True, slots=True)
class CapacityReport:
    c20_discharge_ah: float
    c20_discharge_integral_ah: float
    c20_charge_ah: float
    one_c_discharge_ah: tuple[float, ...]
    one_c_discharge_integral_ah: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OcvResult:
    soc: FloatArray
    ocv_v: FloatArray
    discharge_v: FloatArray
    charge_v: FloatArray
    hysteresis_v: FloatArray
    hysteresis_literal_v: FloatArray
    monotone_adjustment_v: float
    dense_soc: FloatArray
    dense_ocv_v: FloatArray


@dataclass(frozen=True, slots=True)
class PulseBlock:
    pre: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class PulseData:
    time_s: FloatArray
    current_a: FloatArray
    voltage_v: FloatArray
    soc: FloatArray
    block: PulseBlock


@dataclass(frozen=True, slots=True)
class RcFit:
    soc: float
    r0_ohm: float
    r1_ohm: float
    c1_farad: float
    tau_s: float
    rmse_v: float
    at_bound: bool
    sample_interval_s: float
    residual_v: FloatArray


@dataclass(frozen=True, slots=True)
class Reference:
    soc: FloatArray
    volt_v: FloatArray


def discharge_rows(recording: Recording) -> np.ndarray:
    return recording.status == "DCH"


def charge_rows(recording: Recording) -> np.ndarray:
    return recording.status == "CHA"


def counter_capacity_ah(recording: Recording, rows: np.ndarray) -> float:
    return float(np.abs(recording.capacity_ah[rows]).max())


def integral_capacity_ah(recording: Recording, sign: float, rows: np.ndarray) -> float:
    current = sign * recording.current_a[rows]
    return float(charge_throughput_ah(recording.time_s[rows], current)[-1])


def capacity_report(c20: Recording, one_c: list[Recording], sign: float) -> CapacityReport:
    discharge, charge = discharge_rows(c20), charge_rows(c20)
    return CapacityReport(
        c20_discharge_ah=counter_capacity_ah(c20, discharge),
        c20_discharge_integral_ah=integral_capacity_ah(c20, sign, discharge),
        c20_charge_ah=counter_capacity_ah(c20, charge),
        one_c_discharge_ah=tuple(counter_capacity_ah(r, discharge_rows(r)) for r in one_c),
        one_c_discharge_integral_ah=tuple(
            integral_capacity_ah(r, sign, discharge_rows(r)) for r in one_c
        ),
    )


def enforce_strictly_increasing(values: FloatArray) -> tuple[FloatArray, float]:
    running = np.maximum.accumulate(values)
    flat = np.diff(running, prepend=running[0] - 1.0) <= 0.0
    strict = running + STRICT_STEP_V * np.cumsum(flat)
    return strict, float(np.abs(strict - values).max())


def build_ocv(
    c20: Recording,
    sign: float,
    capacity_ah: float,
    r0_ohm: Callable[[FloatArray], FloatArray],
    spec: IdentificationSpec,
) -> OcvResult:
    discharge, charge = discharge_rows(c20), charge_rows(c20)
    soc_d = 1.0 - np.abs(c20.capacity_ah[discharge]) / capacity_ah
    current_d = sign * c20.current_a[discharge]
    volt_d = c20.voltage_v[discharge] + current_d * r0_ohm(soc_d)
    counter_c = c20.capacity_ah[charge] - c20.capacity_ah[charge][0]
    soc_c = counter_c / counter_c[-1]
    soc_c_literal = counter_c / capacity_ah
    current_c = -sign * c20.current_a[charge]
    volt_c = c20.voltage_v[charge] - current_c * r0_ohm(soc_c)
    volt_d_at_c = np.interp(soc_c, soc_d[::-1], volt_d[::-1])
    dense_ocv = 0.5 * (volt_c + volt_d_at_c)
    volt_d_literal = np.interp(soc_c_literal, soc_d[::-1], volt_d[::-1])
    grid = np.linspace(0.0, 1.0, spec.ocv_points)
    ocv, adjustment = enforce_strictly_increasing(np.interp(grid, soc_c, dense_ocv))
    return OcvResult(
        soc=grid,
        ocv_v=ocv,
        discharge_v=np.interp(grid, soc_d[::-1], volt_d[::-1]),
        charge_v=np.interp(grid, soc_c, volt_c),
        hysteresis_v=np.interp(grid, soc_c, 0.5 * (volt_c - volt_d_at_c)),
        hysteresis_literal_v=np.interp(grid, soc_c_literal, 0.5 * (volt_c - volt_d_literal)),
        monotone_adjustment_v=adjustment,
        dense_soc=soc_c,
        dense_ocv_v=dense_ocv,
    )


def ocv_slope_v_per_soc(ocv: OcvResult, soc: float) -> float:
    return float(np.gradient(ocv.ocv_v, ocv.soc)[np.argmin(np.abs(ocv.soc - soc))])


def find_pulse_blocks(
    recording: Recording, sign: float, spec: IdentificationSpec
) -> list[PulseBlock]:
    current = sign * recording.current_a
    parts = segments(recording)
    blocks = []
    for part, following in zip(parts[:-1], parts[1:], strict=True):
        length = part.stop - part.start
        if recording.status[part.start] != "DCH" or length > spec.pulse_max_samples:
            continue
        level_a = float(np.median(current[part]))
        if abs(level_a - spec.pulse_current_a) > spec.pulse_current_tolerance_a:
            continue
        if recording.status[following.start] != "PAU" or part.start == 0:
            continue
        blocks.append(PulseBlock(pre=part.start - 1, start=part.start, stop=following.stop))
    return blocks


def pulse_data(
    recording: Recording, sign: float, capacity_ah: float, block: PulseBlock
) -> PulseData:
    current = sign * recording.current_a
    soc = 1.0 - charge_throughput_ah(recording.time_s, current) / capacity_ah
    window = slice(block.pre, block.stop)
    return PulseData(
        time_s=recording.time_s[window],
        current_a=current[window],
        voltage_v=recording.voltage_v[window],
        soc=soc[window],
        block=PulseBlock(pre=0, start=block.start - block.pre, stop=block.stop - block.pre),
    )


def v1_response(
    time_s: FloatArray, current_a: FloatArray, r1_ohm: float, tau_s: float
) -> FloatArray:
    growth = np.exp((time_s - time_s[0]) / tau_s)
    inner = np.concatenate(([0.0], np.cumsum(current_a[:-1] * np.diff(growth))))
    return r1_ohm * inner / growth


def model_voltage(
    data: PulseData, reference: Reference, r0_ohm: float, r1_ohm: float, tau_s: float
) -> FloatArray:
    block = data.block
    window = slice(block.start, block.stop)
    ocv_here = np.interp(data.soc[window], reference.soc, reference.volt_v)
    ocv_anchor = np.interp(data.soc[block.pre], reference.soc, reference.volt_v)
    drift = ocv_here - ocv_anchor
    v1 = v1_response(data.time_s[window], data.current_a[window], r1_ohm, tau_s)
    return data.voltage_v[block.pre] + drift - r0_ohm * data.current_a[window] - v1


def step_resistance_ohm(data: PulseData) -> float:
    block = data.block
    step_a = data.current_a[block.start] - data.current_a[block.pre]
    return float((data.voltage_v[block.pre] - data.voltage_v[block.start]) / step_a)


def fit_rc(
    data: PulseData,
    reference: Reference,
    spec: IdentificationSpec,
    starts_s: tuple[float, ...] = START_TAUS_S,
) -> RcFit:
    block = data.block
    window = slice(block.start, block.stop)
    r0 = step_resistance_ohm(data)
    measured = data.voltage_v[window]
    low = np.log([spec.r1_bounds_ohm[0], spec.tau_bounds_s[0]])
    high = np.log([spec.r1_bounds_ohm[1], spec.tau_bounds_s[1]])

    def residual(x: FloatArray) -> FloatArray:
        r1, tau = float(np.exp(x[0])), float(np.exp(x[1]))
        return model_voltage(data, reference, r0, r1, tau) - measured

    best = None
    for tau0 in starts_s:
        result = least_squares(residual, np.log([START_R1_OHM, tau0]), bounds=(low, high))
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    r1, tau = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    margin = spec.bound_flag_tolerance
    at_bound = bool(np.any(best.x - low < margin) or np.any(high - best.x < margin))
    final = residual(best.x)
    return RcFit(
        soc=float(data.soc[block.pre]),
        r0_ohm=r0,
        r1_ohm=r1,
        c1_farad=tau / r1,
        tau_s=tau,
        rmse_v=float(np.sqrt(np.mean(final**2))),
        at_bound=at_bound,
        sample_interval_s=float(data.time_s[block.start] - data.time_s[block.pre]),
        residual_v=-final,
    )


def in_window(soc: float, window: tuple[float, float]) -> bool:
    return window[0] <= soc <= window[1]


def window_summary(
    fits: list[RcFit], values: Callable[[RcFit], float], window: tuple[float, float]
) -> dict[str, float]:
    selected = np.array([values(fit) for fit in fits if in_window(fit.soc, window)])
    return {
        "median": float(np.median(selected)),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "n_points": int(selected.size),
    }


def reference_from(ocv: OcvResult) -> Reference:
    return Reference(soc=ocv.soc, volt_v=ocv.ocv_v)


def rest_cross_check(
    recording: Recording,
    sign: float,
    capacity_ah: float,
    blocks: list[PulseBlock],
    reference: Reference,
) -> dict[str, object]:
    current = sign * recording.current_a
    soc = 1.0 - charge_throughput_ah(recording.time_s, current) / capacity_ah
    soc_points = np.array([soc[block.pre] for block in blocks])
    rest_v = np.array([recording.voltage_v[block.pre] for block in blocks])
    difference = rest_v - np.interp(soc_points, reference.soc, reference.volt_v)
    return {
        "soc": soc_points.tolist(),
        "rest_voltage_v": rest_v.tolist(),
        "difference_mv": (difference * MV_PER_V).tolist(),
        "max_abs_mv": float(np.abs(difference).max() * MV_PER_V),
        "rms_mv": float(np.sqrt(np.mean(difference**2)) * MV_PER_V),
    }


def block_bootstrap_indices(
    size: int, block: int, rng: np.random.Generator, n_resamples: int
) -> IntArray:
    n_blocks = -(-size // block)
    starts = rng.integers(0, size, size=(n_resamples, n_blocks))
    indices = (starts[:, :, None] + np.arange(block)) % size
    return indices.reshape(n_resamples, -1)[:, :size]


def rest_noise_pool(recording: Recording, block: PulseBlock, samples: int) -> FloatArray:
    voltage = recording.voltage_v[block.pre - samples : block.pre + 1]
    steps = np.diff(voltage)
    return (steps - steps.mean()) / np.sqrt(2.0)


def slope_bootstrap(
    ocv: OcvResult, spec: IdentificationSpec, rng: np.random.Generator
) -> dict[str, object]:
    low, high = spec.slope_window
    selected = (ocv.dense_soc >= low) & (ocv.dense_soc <= high)
    soc, volt = ocv.dense_soc[selected], ocv.dense_ocv_v[selected]
    coefficients = np.polyfit(soc, volt, 1)
    residual = volt - np.polyval(coefficients, soc)
    indices = block_bootstrap_indices(
        soc.size, spec.slope_block_points, rng, spec.bootstrap_resamples
    )
    synthetic = np.polyval(coefficients, soc)[None, :] + residual[indices]
    centred = soc - soc.mean()
    slopes = (synthetic - synthetic.mean(axis=1, keepdims=True)) @ centred / np.sum(centred**2)
    return {"slope_v_per_soc": float(coefficients[0]), "samples": slopes, "n_points": int(soc.size)}


def interval(samples: FloatArray, percent: float) -> dict[str, float]:
    tail = 0.5 * (100.0 - percent)
    return {
        "median": float(np.median(samples)),
        "low": float(np.percentile(samples, tail)),
        "high": float(np.percentile(samples, 100.0 - tail)),
    }


@dataclass(frozen=True, slots=True)
class TwoRcFit:
    soc: float
    r0_ohm: float
    r1_ohm: float
    tau1_s: float
    r2_ohm: float
    tau2_s: float
    rmse_v: float
    at_bound: bool
    residual_v: FloatArray


def extended_window(
    recording: Recording, block: PulseBlock, spec: IdentificationSpec
) -> PulseBlock:
    parts = segments(recording)
    after = [part for part in parts if part.start >= block.stop]
    stop = block.stop
    for part in after:
        is_pulse = recording.status[part.start] == "DCH" and (
            part.stop - part.start <= spec.pulse_max_samples
        )
        if is_pulse:
            break
        stop = part.stop
    return PulseBlock(pre=block.pre, start=block.start, stop=stop)


def two_rc_state(
    time_s: FloatArray, current_a: FloatArray, r_ohm: float, tau_s: float
) -> FloatArray:
    decay = np.exp(-np.diff(time_s) / tau_s)
    drive = (r_ohm * (1.0 - decay) * current_a[:-1]).tolist()
    state = [0.0]
    value = 0.0
    for a, b in zip(decay.tolist(), drive, strict=True):
        value = a * value + b
        state.append(value)
    return np.array(state)


def model_voltage_two_rc(
    data: PulseData, reference: Reference, r0_ohm: float, branches: tuple[tuple[float, float], ...]
) -> FloatArray:
    block = data.block
    window = slice(block.start, block.stop)
    ocv_here = np.interp(data.soc[window], reference.soc, reference.volt_v)
    ocv_anchor = np.interp(data.soc[block.pre], reference.soc, reference.volt_v)
    voltage = data.voltage_v[block.pre] + ocv_here - ocv_anchor - r0_ohm * data.current_a[window]
    for r_ohm, tau_s in branches:
        voltage = voltage - two_rc_state(data.time_s[window], data.current_a[window], r_ohm, tau_s)
    return voltage


def fit_two_rc(
    data: PulseData,
    reference: Reference,
    spec: IdentificationSpec,
    r0_ohm: float,
    starts: tuple[tuple[float, float, float, float], ...] = TWO_RC_STARTS,
) -> TwoRcFit:
    block = data.block
    window = slice(block.start, block.stop)
    measured = data.voltage_v[window]
    r_low, r_high = spec.r1_bounds_ohm
    low = np.log([r_low, spec.fast_tau_bounds_s[0], r_low, spec.slow_tau_bounds_s[0]])
    high = np.log([r_high, spec.fast_tau_bounds_s[1], r_high, spec.slow_tau_bounds_s[1]])

    def residual(x: FloatArray) -> FloatArray:
        r1, tau1, r2, tau2 = np.exp(x)
        return model_voltage_two_rc(data, reference, r0_ohm, ((r1, tau1), (r2, tau2))) - measured

    best = None
    for start in starts:
        result = least_squares(residual, np.log(start), bounds=(low, high))
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    r1, tau1, r2, tau2 = (float(value) for value in np.exp(best.x))
    margin = spec.bound_flag_tolerance
    at_bound = bool(np.any(best.x - low < margin) or np.any(high - best.x < margin))
    return TwoRcFit(
        soc=float(data.soc[block.pre]),
        r0_ohm=r0_ohm,
        r1_ohm=r1,
        tau1_s=tau1,
        r2_ohm=r2,
        tau2_s=tau2,
        rmse_v=float(np.sqrt(np.mean(residual(best.x) ** 2))),
        at_bound=at_bound,
        residual_v=-residual(best.x),
    )


def bootstrap_pulses(
    data: PulseData,
    reference: Reference,
    fit: TwoRcFit,
    recording: Recording,
    original: PulseBlock,
    spec: IdentificationSpec,
    rng: np.random.Generator,
) -> FloatArray:
    block = data.block
    window = slice(block.start, block.stop)
    n_samples = block.stop - block.start
    fitted = data.voltage_v[window] - fit.residual_v
    resample = block_bootstrap_indices(
        n_samples, spec.bootstrap_block_samples, rng, spec.bootstrap_resamples
    )
    pool = rest_noise_pool(recording, original, spec.rest_noise_samples)
    pre_noise = rng.choice(pool, size=spec.bootstrap_resamples)
    first_noise = rng.choice(pool, size=spec.bootstrap_resamples)
    start = ((fit.r1_ohm, fit.tau1_s, fit.r2_ohm, fit.tau2_s),)
    out = np.empty((spec.bootstrap_resamples, 5))
    for k in range(spec.bootstrap_resamples):
        voltage = data.voltage_v.copy()
        voltage[window] = fitted + fit.residual_v[resample[k]]
        voltage[block.pre] += pre_noise[k]
        voltage[block.start] += first_noise[k]
        synthetic = PulseData(data.time_s, data.current_a, voltage, data.soc, block)
        refit = fit_two_rc(synthetic, reference, spec, step_resistance_ohm(synthetic), start)
        out[k] = (refit.r0_ohm, refit.r1_ohm, refit.tau1_s, refit.r2_ohm, refit.tau2_s)
    return out
