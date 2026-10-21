import struct
from pathlib import Path

import numpy as np
import pytest

from src.config import CONFIG
from src.params import ConverterCheckSpec
from src.spice import (
    LinkCells,
    RawTrace,
    SpiceCase,
    buck_boost_netlist,
    cycle_to_cycle_change,
    flying_capacitor_netlist,
    netlist_for,
    netlist_sha256,
    period_integrals,
    read_raw,
)
from src.spice_check import spice_cases

CELLS = LinkCells(3.74, 3.70, 0.01158, 0.01158)
SPEC = ConverterCheckSpec()
NETLIST_DIR = Path(__file__).resolve().parents[1] / "spice"


def write_raw(path: Path, time_s: np.ndarray, signals: dict[str, np.ndarray], double: bool) -> None:
    names = list(signals)
    listing = "".join(f"\t{i + 1}\t{name}\tvoltage\n" for i, name in enumerate(names))
    header = (
        "Title: test\nPlotname: Transient Analysis\nFlags: real forward\n"
        f"No. Variables: {len(names) + 1}\nNo. Points: {time_s.size}\n"
        f"Variables:\n\t0\ttime\ttime\n{listing}Binary:\n"
    )
    body = bytearray()
    for index in range(time_s.size):
        body += struct.pack("<d", time_s[index])
        for name in names:
            body += struct.pack("<d" if double else "<f", signals[name][index])
    path.write_bytes(header.encode("utf-16-le") + bytes(body))


@pytest.mark.parametrize("double", [False, True])
def test_raw_reader_handles_both_layouts(tmp_path: Path, double: bool) -> None:
    time_s = np.linspace(0.0, 1.0, 11)
    signals = {"V(a)": time_s * 2.0, "I(V1)": 1.0 - time_s}
    path = tmp_path / "case.raw"
    write_raw(path, time_s, signals, double)
    trace = read_raw(path)
    np.testing.assert_allclose(trace.time_s, time_s)
    np.testing.assert_allclose(trace.signals["V(a)"], signals["V(a)"], rtol=1e-6)
    np.testing.assert_allclose(trace.signals["I(V1)"], signals["I(V1)"], rtol=1e-6)


def test_raw_reader_rejects_an_unknown_layout(tmp_path: Path) -> None:
    path = tmp_path / "bad.raw"
    header = (
        "Flags: real forward\nNo. Variables: 2\nNo. Points: 3\nVariables:\n"
        "\t0\ttime\ttime\n\t1\tV(a)\tvoltage\nBinary:\n"
    )
    path.write_bytes(header.encode("utf-16-le") + b"\x00" * 7)
    with pytest.raises(ValueError, match="no known layout"):
        read_raw(path)


def test_period_integrals_match_calculus_for_a_linear_ramp() -> None:
    time_s = np.linspace(0.0, 4.0, 401)
    signal = 2.0 * time_s + 1.0
    trace = RawTrace(time_s=time_s, signals={})
    pairs = [(signal, np.ones_like(signal)), (signal, signal)]
    totals = period_integrals(trace, 1.0, 1.0, 3, pairs)
    edges = np.array([1.0, 2.0, 3.0, 4.0])
    linear = edges**2 + edges
    cubic = (2.0 * edges + 1.0) ** 3 / 6.0
    np.testing.assert_allclose(totals[0], np.diff(linear), rtol=1e-12)
    np.testing.assert_allclose(totals[1], np.diff(cubic), rtol=1e-12)


def test_period_integrals_split_periods_at_a_point_between_samples() -> None:
    time_s = np.array([0.0, 0.7, 1.9, 3.0])
    signal = np.ones_like(time_s)
    trace = RawTrace(time_s=time_s, signals={})
    totals = period_integrals(trace, 0.0, 1.0, 3, [(signal, signal)])
    np.testing.assert_allclose(totals[0], [1.0, 1.0, 1.0], rtol=1e-12)


def test_cycle_to_cycle_change() -> None:
    assert cycle_to_cycle_change(np.array([1.0, 1.001, 1.002])) == pytest.approx(0.001 / 1.001)


def buck_case(step_divisor: int = 400) -> SpiceCase:
    return SpiceCase("bb", "buck_boost", CELLS, 55e-6, 30e3, step_divisor)


def test_buck_boost_netlist_matches_the_specification() -> None:
    netlist = buck_boost_netlist(buck_case(), CONFIG.buck_boost, SPEC)
    period = 1.0 / 30e3
    stop = (SPEC.settle_periods + SPEC.measured_periods) * period
    assert "L1 x xl 5.5e-05" in netlist
    assert "RL xl 0 0.08" in netlist
    assert "Ron=0.02 Roff=1000000" in netlist
    assert f".tran 0 {stop:.12g} 0 {period / 400:.12g} uic" in netlist
    assert f" {0.4 * period - SPEC.gate_edge_fraction * period:.12g} " in netlist
    assert netlist.endswith(".end\n")


def test_flying_capacitor_netlist_matches_the_specification() -> None:
    case = SpiceCase("fc", "flying_capacitor", CELLS, 5.5e-3, 5.5e3, 400)
    netlist = flying_capacitor_netlist(case, CONFIG.flying_capacitor, SPEC)
    assert netlist.count("\nS") == 4
    assert "Ron=0.01 Roff=1000000" in netlist
    assert "RE cx cm 0.005" in netlist
    assert "BG2 g2 0 V=1-V(g1)" in netlist
    assert "C1 cp cx 0.0055" in netlist


def test_netlists_are_deterministic() -> None:
    first = buck_boost_netlist(buck_case(), CONFIG.buck_boost, SPEC)
    second = buck_boost_netlist(buck_case(), CONFIG.buck_boost, SPEC)
    finer = buck_boost_netlist(buck_case(800), CONFIG.buck_boost, SPEC)
    assert netlist_sha256(first) == netlist_sha256(second)
    assert netlist_sha256(first) != netlist_sha256(finer)


def test_committed_netlists_match_the_generator() -> None:
    if not NETLIST_DIR.exists():
        pytest.skip("spice netlists not generated")
    for case in spice_cases(CONFIG):
        expected = netlist_for(case, CONFIG.buck_boost, CONFIG.flying_capacitor, SPEC)
        assert (NETLIST_DIR / f"{case.name}.cir").read_text() == expected


def test_case_list_covers_the_specified_operating_points() -> None:
    cases = spice_cases(CONFIG)
    assert len(cases) == 12
    buck = {(c.component, c.frequency_hz) for c in cases if c.converter == "buck_boost"}
    flying = {(c.component, c.frequency_hz) for c in cases if c.converter == "flying_capacitor"}
    assert buck == {(10e-6, 50e3), (55e-6, 30e3), (100e-6, 10e3)}
    assert flying == {(1e-3, 1e3), (5.5e-3, 5.5e3), (10e-3, 10e3)}
    assert all(c.step_divisor >= 400 for c in cases)
