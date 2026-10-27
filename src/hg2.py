import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from src.config import FloatArray, IntArray
from src.params import SECONDS_PER_HOUR

HEADER_START = "Time Stamp,"
COLUMN_COUNT = 14
SECONDS_PER_MINUTE = 60.0
DESCRIPTOR_PATTERN = re.compile(r"^(?P<test>\d+)_(?P<kind>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)\.csv$")


class TestKind(StrEnum):
    __test__ = False

    HPPC = "HPPC"
    C20 = "C20DisCh"
    DISCHARGE_HALF_C = "Dis_0p5C"
    DISCHARGE_2C = "Dis_2C"
    CAPACITY_1C = "Cap_1C"
    UDDS = "UDDS"
    HWFET = "HWFET"
    LA92 = "LA92"
    US06 = "US06"
    MIXED = "Mixed"


DRIVE_CYCLES = (TestKind.UDDS, TestKind.HWFET, TestKind.LA92, TestKind.US06)
MIXED_PATTERN = re.compile(r"^Mixed(?P<index>\d)$")


@dataclass(frozen=True, slots=True)
class Recording:
    name: str
    kind: TestKind
    test_number: int
    time_s: FloatArray
    voltage_v: FloatArray
    current_a: FloatArray
    temperature_c: FloatArray
    capacity_ah: FloatArray
    step: IntArray
    status: np.ndarray


def classify(file_name: str) -> tuple[int, TestKind]:
    match = DESCRIPTOR_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"unrecognised file name {file_name}")
    kinds = {kind.value: kind for kind in TestKind}
    descriptor = match["kind"]
    if MIXED_PATTERN.match(descriptor):
        return int(match["test"]), TestKind.MIXED
    if descriptor not in kinds:
        raise ValueError(f"unrecognised test descriptor {descriptor}")
    return int(match["test"]), kinds[descriptor]


def parse_program_time_s(text: str) -> float:
    hours, minutes, seconds = text.split(":")
    return int(hours) * SECONDS_PER_HOUR + int(minutes) * SECONDS_PER_MINUTE + float(seconds)


def load_recording(path: Path) -> Recording:
    test_number, kind = classify(path.name)
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(HEADER_START)]
    if len(starts) != 1:
        raise ValueError(f"{path.name}: expected one column header line, found {len(starts)}")
    header = [cell.strip() for cell in lines[starts[0]].split(",")][:-1]
    rows = [line.split(",") for line in lines[starts[0] + 2 :] if line.strip()]
    if any(len(row) != COLUMN_COUNT + 1 for row in rows):
        raise ValueError(f"{path.name}: ragged rows")
    column = {name: header.index(name) for name in header}
    time_s = np.array([parse_program_time_s(row[column["Prog Time"]]) for row in rows])
    if np.any(np.diff(time_s) < 0.0):
        raise ValueError(f"{path.name}: program time is not monotonic")

    def floats(name: str) -> FloatArray:
        return np.array([float(row[column[name]]) for row in rows])

    return Recording(
        name=path.name,
        kind=kind,
        test_number=test_number,
        time_s=time_s - time_s[0],
        voltage_v=floats("Voltage"),
        current_a=floats("Current"),
        temperature_c=floats("Temperature"),
        capacity_ah=floats("Capacity"),
        step=np.array([int(row[column["Step"]]) for row in rows]),
        status=np.array([row[column["Status"]] for row in rows]),
    )


def discharge_sign(recording: Recording) -> float:
    discharging = recording.current_a[recording.status == "DCH"]
    if discharging.size == 0:
        raise ValueError(f"{recording.name}: no discharge rows")
    return -1.0 if float(np.median(discharging)) < 0.0 else 1.0


def segments(recording: Recording) -> list[slice]:
    changes = np.flatnonzero(np.diff(recording.step) != 0) + 1
    edges = np.concatenate(([0], changes, [recording.step.size]))
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def charge_throughput_ah(time_s: FloatArray, discharge_a: FloatArray) -> FloatArray:
    increments = 0.5 * (discharge_a[1:] + discharge_a[:-1]) * np.diff(time_s) / SECONDS_PER_HOUR
    return np.concatenate(([0.0], np.cumsum(increments)))


def load_all(
    directory: Path, include_mixed: bool = False, rejected: dict[str, str] | None = None
) -> dict[TestKind, list[Recording]]:
    recordings: dict[TestKind, list[Recording]] = {}
    for path in sorted(directory.glob("*.csv")):
        if not include_mixed and classify(path.name)[1] is TestKind.MIXED:
            continue
        try:
            recording = load_recording(path)
        except ValueError as error:
            if rejected is None:
                raise
            rejected[path.name] = str(error)
            continue
        recordings.setdefault(recording.kind, []).append(recording)
    return recordings
