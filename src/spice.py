import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import FloatArray
from src.params import BuckBoostSpec, ConverterCheckSpec, FlyingCapSpec

BINARY_MARKER = "Binary:\n"
HEADER_ENCODING = "utf-16-le"
DOUBLE_BYTES = 8
SINGLE_BYTES = 4
BUCK_BOOST_SAVES = ("I(VA)", "I(VB)", "I(L1)", "I(S1)", "I(D1)", "V(x)")
FLYING_CAPACITOR_SAVES = (
    "I(VA)",
    "I(VB)",
    "I(C1)",
    "I(S1A)",
    "I(S1B)",
    "I(S2A)",
    "I(S2B)",
    "V(cp)",
)


@dataclass(frozen=True, slots=True)
class LinkCells:
    ocv_source_v: float
    ocv_sink_v: float
    r0_source_ohm: float
    r0_sink_ohm: float


@dataclass(frozen=True, slots=True)
class SpiceCase:
    name: str
    converter: str
    cells: LinkCells
    component: float
    frequency_hz: float
    step_divisor: int

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz


@dataclass(frozen=True, slots=True)
class RawTrace:
    time_s: FloatArray
    signals: dict[str, FloatArray]


@dataclass(frozen=True, slots=True)
class PeriodMeans:
    source_current_a: FloatArray
    sink_current_a: FloatArray
    source_r0_loss_w: FloatArray
    sink_r0_loss_w: FloatArray
    element_loss_w: FloatArray


def _fmt(value: float) -> str:
    return f"{value:.12g}"


def netlist_sha256(netlist: str) -> str:
    return hashlib.sha256(netlist.encode("utf-8")).hexdigest()


def stop_time_s(case: SpiceCase, spec: ConverterCheckSpec) -> float:
    return (spec.settle_periods + spec.measured_periods) * case.period_s


def measure_start_s(case: SpiceCase, spec: ConverterCheckSpec) -> float:
    return spec.settle_periods * case.period_s


def buck_boost_netlist(case: SpiceCase, hardware: BuckBoostSpec, spec: ConverterCheckSpec) -> str:
    period = case.period_s
    edge = spec.gate_edge_fraction * period
    on_time = hardware.duty * period
    lines = [
        f"* buck-boost link {case.name}: DCM, ideal-diode freewheel, DC sources with series R0",
        f"VA ai 0 {_fmt(case.cells.ocv_source_v)}",
        f"RA ai a {_fmt(case.cells.r0_source_ohm)}",
        f"VB 0 bi {_fmt(case.cells.ocv_sink_v)}",
        f"RB bi bm {_fmt(case.cells.r0_sink_ohm)}",
        "S1 a x g 0 SWH",
        "D1 bm x DFW",
        f"L1 x xl {_fmt(case.component)}",
        f"RL xl 0 {_fmt(hardware.r_inductor_ohm)}",
        f"VG g 0 PULSE(0 1 0 {_fmt(edge)} {_fmt(edge)} {_fmt(on_time - edge)} {_fmt(period)})",
        f".model SWH SW(Ron={_fmt(hardware.r_ds_on_ohm)} Roff={_fmt(spec.switch_roff_ohm)} "
        "Vt=0.5 Vh=0)",
        f".model DFW D(Ron={_fmt(hardware.r_ds_on_ohm)} Roff={_fmt(spec.switch_roff_ohm)} Vfwd=0)",
        f".options plotwinsize=0 numdgt=15 reltol={_fmt(spec.relative_tolerance)}",
        f".save {' '.join(BUCK_BOOST_SAVES)}",
        f".tran 0 {_fmt(stop_time_s(case, spec))} 0 {_fmt(period / case.step_divisor)} uic",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def flying_capacitor_netlist(
    case: SpiceCase, hardware: FlyingCapSpec, spec: ConverterCheckSpec
) -> str:
    period = case.period_s
    edge = spec.gate_edge_fraction * period
    half = 0.5 * period
    start_v = case.cells.ocv_source_v
    switch = (
        f".model SWF SW(Ron={_fmt(hardware.r_ds_on_ohm)} Roff={_fmt(spec.switch_roff_ohm)} "
        "Vt=0.5 Vh=0)"
    )
    lines = [
        f"* flying capacitor link {case.name}: two 50 % phases, DC sources with series R0",
        f"VA ai 0 {_fmt(case.cells.ocv_source_v)}",
        f"RA ai n1 {_fmt(case.cells.r0_source_ohm)}",
        f"VB bi n1 {_fmt(case.cells.ocv_sink_v)}",
        f"RB bi n2 {_fmt(case.cells.r0_sink_ohm)}",
        f"C1 cp cx {_fmt(case.component)} ic={_fmt(start_v)}",
        f"RE cx cm {_fmt(hardware.esr_ohm)}",
        "S1A cp n1 g1 0 SWF",
        "S1B cm 0 g1 0 SWF",
        "S2A cp n2 g2 0 SWF",
        "S2B cm n1 g2 0 SWF",
        f"VG1 g1 0 PULSE(0 1 0 {_fmt(edge)} {_fmt(edge)} {_fmt(half - edge)} {_fmt(period)})",
        "BG2 g2 0 V=1-V(g1)",
        switch,
        f".options plotwinsize=0 numdgt=15 reltol={_fmt(spec.relative_tolerance)}",
        f".save {' '.join(FLYING_CAPACITOR_SAVES)}",
        f".tran 0 {_fmt(stop_time_s(case, spec))} 0 {_fmt(period / case.step_divisor)} uic",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def netlist_for(
    case: SpiceCase, buck_boost: BuckBoostSpec, flying: FlyingCapSpec, spec: ConverterCheckSpec
) -> str:
    if case.converter == "buck_boost":
        return buck_boost_netlist(case, buck_boost, spec)
    return flying_capacitor_netlist(case, flying, spec)


def read_raw(path: Path) -> RawTrace:
    blob = path.read_bytes()
    marker = BINARY_MARKER.encode(HEADER_ENCODING)
    cut = blob.find(marker)
    if cut < 0:
        raise ValueError(f"{path.name}: binary marker not found")
    header = blob[:cut].decode(HEADER_ENCODING)
    body = blob[cut + len(marker) :]
    n_variables = int(re.search(r"No\. Variables:\s*(\d+)", header).group(1))
    n_points = int(re.search(r"No\. Points:\s*(\d+)", header).group(1))
    listing = header.split("\nVariables:")[1]
    names = re.findall(r"^\s*\d+\s+(\S+)\s+\S+\s*$", listing, re.MULTILINE)
    if len(names) != n_variables:
        raise ValueError(f"{path.name}: header lists {len(names)} of {n_variables} variables")
    per_point = len(body) / n_points
    if per_point == DOUBLE_BYTES + SINGLE_BYTES * (n_variables - 1):
        widths = [np.dtype("<f8")] + [np.dtype("<f4")] * (n_variables - 1)
    elif per_point == DOUBLE_BYTES * n_variables:
        widths = [np.dtype("<f8")] * n_variables
    else:
        raise ValueError(f"{path.name}: {per_point} bytes per point matches no known layout")
    columns = np.frombuffer(body, dtype=np.dtype([(f"c{i}", w) for i, w in enumerate(widths)]))
    values = [columns[f"c{i}"].astype(np.float64) for i in range(n_variables)]
    return RawTrace(
        time_s=np.abs(values[0]),
        signals={name: values[i] for i, name in enumerate(names) if i > 0},
    )


def _insert_boundaries(
    time_s: FloatArray, signals: list[FloatArray], boundaries: FloatArray
) -> tuple[FloatArray, list[FloatArray]]:
    merged = np.unique(np.concatenate([time_s, boundaries]))
    return merged, [np.interp(merged, time_s, signal) for signal in signals]


def _bilinear_segments(time_s: FloatArray, left: FloatArray, right: FloatArray) -> FloatArray:
    h = np.diff(time_s)
    return (
        h
        * (
            2.0 * left[:-1] * right[:-1]
            + left[:-1] * right[1:]
            + left[1:] * right[:-1]
            + 2.0 * left[1:] * right[1:]
        )
        / 6.0
    )


def period_integrals(
    trace: RawTrace,
    start_s: float,
    period_s: float,
    n_periods: int,
    pairs: list[tuple[FloatArray, FloatArray]],
) -> FloatArray:
    boundaries = start_s + period_s * np.arange(n_periods + 1)
    flat = [signal for pair in pairs for signal in pair]
    time_s, merged = _insert_boundaries(trace.time_s, flat, boundaries)
    inside = (time_s >= boundaries[0] - 1e-18) & (time_s <= boundaries[-1] + 1e-18)
    time_s = time_s[inside]
    merged = [signal[inside] for signal in merged]
    segment_start = time_s[:-1]
    index = np.clip(
        np.floor((segment_start - start_s) / period_s + 1e-9).astype(np.int64), 0, n_periods - 1
    )
    out = np.zeros((len(pairs), n_periods))
    for row, (left, right) in enumerate(zip(merged[0::2], merged[1::2], strict=True)):
        out[row] = np.bincount(
            index, weights=_bilinear_segments(time_s, left, right), minlength=n_periods
        )
    return out


def loss_terms(
    case: SpiceCase, buck_boost: BuckBoostSpec, flying: FlyingCapSpec
) -> dict[str, float]:
    if case.converter == "buck_boost":
        return {
            "I(S1)": buck_boost.r_ds_on_ohm,
            "I(D1)": buck_boost.r_ds_on_ohm,
            "I(L1)": buck_boost.r_inductor_ohm,
        }
    return {
        "I(S1A)": flying.r_ds_on_ohm,
        "I(S1B)": flying.r_ds_on_ohm,
        "I(S2A)": flying.r_ds_on_ohm,
        "I(S2B)": flying.r_ds_on_ohm,
        "I(C1)": flying.esr_ohm,
    }


def measure_periods(
    case: SpiceCase, trace: RawTrace, terms: dict[str, float], spec: ConverterCheckSpec
) -> PeriodMeans:
    period = case.period_s
    start = measure_start_s(case, spec)
    n = spec.measured_periods
    source = -trace.signals["I(VA)"]
    sink = trace.signals["I(VB)"]
    ones = np.ones_like(source)
    element = [(trace.signals[name], trace.signals[name]) for name in terms]
    pairs = [(source, ones), (sink, ones), (source, source), (sink, sink), *element]
    totals = period_integrals(trace, start, period, n, pairs) / period
    resistances = np.array(list(terms.values()))[:, None]
    return PeriodMeans(
        source_current_a=totals[0],
        sink_current_a=totals[1],
        source_r0_loss_w=case.cells.r0_source_ohm * totals[2],
        sink_r0_loss_w=case.cells.r0_sink_ohm * totals[3],
        element_loss_w=(resistances * totals[4:]).sum(axis=0),
    )


def cycle_to_cycle_change(values: FloatArray) -> float:
    return float(np.max(np.abs(np.diff(values))) / abs(values.mean()))
