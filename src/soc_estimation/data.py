import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, FloatArray
from src.hg2 import discharge_sign, load_recording
from src.params import SECONDS_PER_HOUR, SocEstimationSpec
from src.resample import interval_end, interval_mean

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hg2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_PATH = PROCESSED_DIR / "soc_splits.json"
TEMPERATURE_PATH = PROCESSED_DIR / "hg2_temperature.json"
PROVENANCE_PATH = PROCESSED_DIR / "provenance.json"
CYCLE_PATTERN = re.compile(r"^\d+_(?P<cycle>Mixed\d|UDDS|HWFET|LA92|US06)\.csv$")
DRIVE_STATUS = "TABLE"
ZERO_CURRENT_A = 1e-6


@dataclass(frozen=True, slots=True)
class Cycle:
    name: str
    cycle: str
    temperature_c: int
    capacity_ah: float
    current_a: FloatArray
    voltage_v: FloatArray
    case_temperature_c: FloatArray
    soc: FloatArray

    @property
    def n_steps(self) -> int:
        return self.current_a.size


def raw_path(folder: str, name: str) -> Path:
    return RAW_DIR / name if folder == "25degC" else RAW_DIR / folder / name


def temperature_capacities() -> dict[int, float]:
    payload = json.loads(TEMPERATURE_PATH.read_text())
    return {int(t): v["capacity_ah"] for t, v in payload["per_temperature"].items()}


def build_cycle(
    path: Path, cycle: str, temperature_c: int, capacity_ah: float, sign: float, step_s: float
) -> Cycle:
    recording = load_recording(path)
    rows = recording.status == DRIVE_STATUS
    time_s = recording.time_s[rows]
    current = sign * recording.current_a[rows]
    average = interval_mean(time_s, current, step_s)
    area = np.concatenate(([0.0], np.cumsum(0.5 * (current[1:] + current[:-1]) * np.diff(time_s))))
    edges = time_s[0] + step_s * np.arange(1, average.size + 1)
    discharged_ah = np.interp(edges, time_s, area) / SECONDS_PER_HOUR
    return Cycle(
        name=path.name,
        cycle=cycle,
        temperature_c=temperature_c,
        capacity_ah=capacity_ah,
        current_a=average,
        voltage_v=interval_end(time_s, recording.voltage_v[rows], step_s),
        case_temperature_c=interval_end(time_s, recording.temperature_c[rows], step_s),
        soc=1.0 - discharged_ah / capacity_ah,
    )


def inspect_file(
    folder: str, name: str, temperature_c: int, capacity_ah: float, spec: SocEstimationSpec
) -> dict[str, object]:
    try:
        recording = load_recording(raw_path(folder, name))
    except ValueError as error:
        return {
            "usable": False,
            "reason": str(error),
            "duration_s": 0.0,
            "discharge_ah": 0.0,
            "case_temperature_mean_c": 0.0,
        }
    rows = recording.status == DRIVE_STATUS
    sign = -1.0
    time_s = recording.time_s[rows]
    current = sign * recording.current_a[rows]
    has_current = rows.sum() > 1 and float(np.abs(current).max()) > ZERO_CURRENT_A
    discharged = float(np.trapezoid(current, time_s) / SECONDS_PER_HOUR) if has_current else 0.0
    reason = ""
    if not has_current:
        reason = "no current recorded in the drive part of the file"
    elif discharged < spec.min_discharged_fraction * capacity_ah:
        reason = (
            f"incomplete run: {discharged:.3f} Ah discharged, "
            f"{spec.min_discharged_fraction:.0%} of {capacity_ah:.3f} Ah required"
        )
    return {
        "usable": reason == "",
        "reason": reason,
        "duration_s": float(time_s[-1] - time_s[0]) if rows.sum() > 1 else 0.0,
        "discharge_ah": discharged,
        "case_temperature_mean_c": float(recording.temperature_c[rows].mean())
        if rows.sum() > 1
        else 0.0,
    }


def check_sign() -> float:
    return discharge_sign(load_recording(RAW_DIR / "549_C20DisCh.csv"))


def load_split(entries: list[dict], spec: SocEstimationSpec) -> list[Cycle]:
    capacities = temperature_capacities()
    sign = check_sign()
    return [
        build_cycle(
            raw_path(e["folder"], e["file"]),
            e["cycle"],
            e["temperature_c"],
            capacities[e["temperature_c"]],
            sign,
            spec.step_s,
        )
        for e in entries
    ]


def load_splits() -> dict:
    return json.loads(SPLITS_PATH.read_text())


def mixed_index(cycle: str) -> int | None:
    return int(cycle[5:]) if cycle.startswith("Mixed") else None
