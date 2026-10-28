import hashlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spicelib import RawRead, SimRunner, SpiceEditor
from spicelib.simulators.ltspice_simulator import LTspice

from src.config import FloatArray
from src.lt_circuits import GATE_DRIVE_V, ROFF_OHM, Device

FULL_ON_FRACTION = 0.9
CONDUCTION_THRESHOLD = 0.01
MIN_CURRENT_A = 1e-3
RUN_TIMEOUT_S = 300
RETRIES = 2


@dataclass(frozen=True, slots=True)
class Window:
    start_s: float
    stop_s: float

    @property
    def length_s(self) -> float:
        return self.stop_s - self.start_s


@dataclass(frozen=True, slots=True)
class Trace:
    time_s: FloatArray
    signals: dict[str, FloatArray]


def simulator_available() -> bool:
    return bool(LTspice.spice_exe) and Path(LTspice.spice_exe[0]).is_file()


def simulator_path() -> str:
    return LTspice.spice_exe[0] if LTspice.spice_exe else ""


def netlist_sha256(netlist: str) -> str:
    return hashlib.sha256(netlist.encode("utf-8")).hexdigest()


def read_trace(raw: Path) -> Trace:
    reader = RawRead(str(raw))
    names = reader.get_trace_names()
    time_s = np.zeros(1)
    if "time" in names:
        time_s = np.abs(np.asarray(reader.get_trace("time").get_wave(0), dtype=np.float64))
    signals = {
        name.lower(): np.asarray(reader.get_trace(name).get_wave(0), dtype=np.float64)
        for name in names
        if name != "time"
    }
    return Trace(time_s, signals)


def _simulate(tags: list[str], netlists: dict[str, str], workdir: Path, parallel: int) -> None:
    runner = SimRunner(output_folder=str(workdir), simulator=LTspice, parallel_sims=parallel)
    for tag in tags:
        path = workdir / f"{tag}.net"
        path.write_text(netlists[tag], newline="\n")
        for stale in (".raw", ".log"):
            (workdir / f"{tag}{stale}").unlink(missing_ok=True)
        runner.run(SpiceEditor(str(path)), run_filename=f"{tag}.net", timeout=RUN_TIMEOUT_S)
    runner.wait_completion()


def _finished(workdir: Path, tag: str) -> bool:
    log = workdir / f"{tag}.log"
    raw = workdir / f"{tag}.raw"
    if not (log.is_file() and raw.is_file()):
        return False
    return "Total elapsed time" in log.read_text(errors="replace")


def run_batch[T](
    netlists: dict[str, str],
    workdir: Path,
    parallel: int,
    handle: Callable[[str, Trace], T],
    chunk: int = 32,
) -> dict[str, T]:
    results: dict[str, T] = {}
    tags = list(netlists)
    for start in range(0, len(tags), chunk):
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
        group = tags[start : start + chunk]
        _simulate(group, netlists, workdir, parallel)
        for attempt in range(RETRIES):
            missing = [t for t in group if not _finished(workdir, t)]
            if not missing:
                break
            _simulate(missing, netlists, workdir, max(1, parallel // 2**attempt))
        for tag in group:
            raw = workdir / f"{tag}.raw"
            if not _finished(workdir, tag):
                raise RuntimeError(f"LTspice produced no raw file for {tag}")
            results[tag] = handle(tag, read_trace(raw))
            raw.unlink()
    shutil.rmtree(workdir, ignore_errors=True)
    return results


def _segment_weights(time_s: FloatArray, window: Window) -> tuple[FloatArray, FloatArray]:
    inside = (time_s > window.start_s) & (time_s < window.stop_s)
    grid = np.concatenate(([window.start_s], time_s[inside], [window.stop_s]))
    return grid, np.diff(grid)


def _on_grid(trace: Trace, grid: FloatArray, name: str) -> FloatArray:
    return np.interp(grid, trace.time_s, trace.signals[name.lower()])


def mean_of(trace: Trace, window: Window, expression: dict[str, float]) -> float:
    grid, h = _segment_weights(trace.time_s, window)
    values = sum(w * _on_grid(trace, grid, n) for n, w in expression.items())
    return float(np.sum(h * 0.5 * (values[:-1] + values[1:])) / window.length_s)


def _combine(trace: Trace, grid: FloatArray, expression: dict[str, float]) -> FloatArray:
    return sum(w * _on_grid(trace, grid, n) for n, w in expression.items())


def mean_product(
    trace: Trace,
    window: Window,
    left: dict[str, float],
    right: dict[str, float],
    full_on: dict[str, float] | None = None,
) -> float:
    grid, h = _segment_weights(trace.time_s, window)
    a, b = _combine(trace, grid, left), _combine(trace, grid, right)
    segment = h * (2.0 * a[:-1] * b[:-1] + a[:-1] * b[1:] + a[1:] * b[:-1] + 2.0 * a[1:] * b[1:])
    segment = segment / 6.0
    if full_on is not None:
        on = _combine(trace, grid, full_on) >= FULL_ON_FRACTION * GATE_DRIVE_V
        segment = segment * (on[:-1] & on[1:])
    return float(np.sum(segment) / window.length_s)


def _difference(plus: str, minus: str) -> dict[str, float]:
    expression = {f"V({plus})": 1.0}
    if minus != "0":
        expression[f"V({minus})"] = -1.0
    return expression


def device_terms(trace: Trace, window: Window, device: Device) -> dict[str, float]:
    voltage = _difference(device.plus, device.minus)
    current = {device.current: -1.0 if device.kind == "switch" else 1.0}
    total = mean_product(trace, window, voltage, current)
    if device.kind != "mos":
        return {"conduction": total, "transition": 0.0, "gate": 0.0}
    source = device.minus
    full_on = _difference(device.gate, source)
    conduction = mean_product(trace, window, voltage, current, full_on)
    drive = _difference(device.driver_plus, source)
    supplied = -mean_product(trace, window, drive, {device.driver_current: 1.0})
    leakage = mean_product(trace, window, voltage, voltage) / ROFF_OHM
    return {
        "conduction": conduction,
        "transition": total - conduction + leakage,
        "gate": supplied,
    }


def cell_terms(trace: Trace, window: Window, v_source: float, v_sink: float, r0: float) -> dict:
    source = {"I(VA)": -1.0}
    sink = {"I(VB)": 1.0}
    return {
        "i_src_a": mean_of(trace, window, source),
        "i_snk_a": mean_of(trace, window, sink),
        "i_src_rms_a": float(np.sqrt(mean_product(trace, window, source, source))),
        "i_snk_rms_a": float(np.sqrt(mean_product(trace, window, sink, sink))),
        "p_src_w": v_source * mean_of(trace, window, source),
        "p_snk_w": v_sink * mean_of(trace, window, sink),
        "p_r0_src_w": r0 * mean_product(trace, window, source, source),
        "p_r0_snk_w": r0 * mean_product(trace, window, sink, sink),
    }


def waveform_limits(trace: Trace, window: Window, name: str) -> tuple[float, float]:
    inside = (trace.time_s >= window.start_s) & (trace.time_s <= window.stop_s)
    wave = trace.signals[name.lower()][inside]
    return float(wave.max()), float(wave.min())


def conduction_fractions(
    trace: Trace, window: Window, period_s: float, duty: float
) -> tuple[float, float]:
    inside = (trace.time_s >= window.start_s) & (trace.time_s <= window.stop_s)
    time_s, wave = trace.time_s[inside], trace.signals["i(l1)"][inside]
    threshold = max(CONDUCTION_THRESHOLD * wave.max(), MIN_CURRENT_A)
    conducting = (wave[:-1] > threshold) & (wave[1:] > threshold)
    fraction = float(np.sum(np.diff(time_s) * conducting) / window.length_s)
    second = max(fraction - duty, 0.0)
    return second, 1.0 - duty - second
