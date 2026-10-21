import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import CONFIG, RESULTS_DIR, ProjectConfig
from src.spice import (
    RawTrace,
    SpiceCase,
    cycle_to_cycle_change,
    loss_terms,
    measure_periods,
    measure_start_s,
    netlist_for,
    netlist_sha256,
    read_raw,
)
from src.spice_check import CACHE_PATH, case_inputs, spice_cases, waveform_diagnostics
from src.spice_plots import plot_buck_boost_waveform, plot_flying_capacitor_waveform

ROOT = Path(__file__).resolve().parents[1]
NETLIST_DIR = ROOT / "spice"
RUN_DIR = ROOT / "data" / "raw" / "spice"
TIMEOUT_S = 1800
CANDIDATES = (
    Path(os.environ.get("PROGRAMFILES", "")) / "ADI" / "LTspice" / "LTspice.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ADI" / "LTspice" / "LTspice.exe",
    Path(os.environ.get("PROGRAMFILES", "")) / "LTC" / "LTspiceXVII" / "XVIIx64.exe",
)
INSTALL_STEPS = (
    "No SPICE simulator found. Install one of:\n"
    "  ngspice: https://ngspice.sourceforge.io/download.html, extract to C:\\Spice64 and add "
    "C:\\Spice64\\bin to PATH\n"
    "  LTspice: https://www.analog.com/en/resources/design-tools-and-calculators/"
    "ltspice-simulator.html"
)


def find_simulator() -> Path | None:
    found = shutil.which("LTspice")
    if found:
        return Path(found)
    return next((path for path in CANDIDATES if path.is_file()), None)


def run_netlist(simulator: Path, netlist: str, name: str) -> tuple[RawTrace, str]:
    folder = RUN_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.cir"
    path.write_text(netlist, newline="\n")
    for stale in folder.glob(f"{name}.*"):
        if stale.suffix != ".cir":
            stale.unlink()
    subprocess.run([str(simulator), "-b", str(path)], check=True, timeout=TIMEOUT_S, cwd=folder)
    log = (folder / f"{name}.log").read_text(errors="replace")
    version = re.search(r"LTspice\s+(\S+)", log)
    return read_raw(folder / f"{name}.raw"), version.group(1) if version else "unknown"


def measure_case(
    simulator: Path, case: SpiceCase, config: ProjectConfig
) -> tuple[dict[str, object], RawTrace, str]:
    spec = config.converter_check
    netlist = netlist_for(case, config.buck_boost, config.flying_capacitor, spec)
    (NETLIST_DIR / f"{case.name}.cir").write_text(netlist, newline="\n")
    trace, version = run_netlist(simulator, netlist, case.name)
    means = measure_periods(
        case, trace, loss_terms(case, config.buck_boost, config.flying_capacitor), spec
    )
    record: dict[str, object] = {
        "name": case.name,
        "netlist_sha256": netlist_sha256(netlist),
        "n_periods_measured": spec.measured_periods,
        "n_raw_points": int(trace.time_s.size),
        "source_current_a": float(means.source_current_a.mean()),
        "sink_current_a": float(means.sink_current_a.mean()),
        "source_r0_loss_w": float(means.source_r0_loss_w.mean()),
        "sink_r0_loss_w": float(means.sink_r0_loss_w.mean()),
        "switch_loss_direct_w": float(means.element_loss_w.mean()),
        "cycle_change_source": cycle_to_cycle_change(means.source_current_a),
        "cycle_change_sink": cycle_to_cycle_change(means.sink_current_a),
        **waveform_diagnostics(case, trace, measure_start_s(case, spec), spec),
    }
    return record, trace, version


def refinement_change(
    simulator: Path, case: SpiceCase, config: ProjectConfig, reference: dict[str, object]
) -> float:
    spec = config.converter_check
    finer = SpiceCase(
        name=f"{case.name}_fine",
        converter=case.converter,
        cells=case.cells,
        component=case.component,
        frequency_hz=case.frequency_hz,
        step_divisor=spec.refined_step_divisor,
    )
    netlist = netlist_for(finer, config.buck_boost, config.flying_capacitor, spec)
    trace, _ = run_netlist(simulator, netlist, finer.name)
    means = measure_periods(
        finer, trace, loss_terms(finer, config.buck_boost, config.flying_capacitor), spec
    )
    source = float(means.source_current_a.mean())
    sink = float(means.sink_current_a.mean())
    return max(
        abs(source / float(reference["source_current_a"]) - 1.0),
        abs(sink / float(reference["sink_current_a"]) - 1.0),
    )


def main() -> int:
    simulator = find_simulator()
    if simulator is None:
        print(INSTALL_STEPS)
        return 1
    config = CONFIG
    NETLIST_DIR.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    traces: dict[str, RawTrace] = {}
    version = "unknown"
    for case in spice_cases(config):
        record, trace, version = measure_case(simulator, case, config)
        if case.name.split("_")[-2] == "s0":
            record["step_refinement_change"] = refinement_change(simulator, case, config, record)
            traces[case.name] = trace
        records.append(record)
        print(
            f"{case.name}: source {record['source_current_a']:.6f} A "
            f"sink {record['sink_current_a']:.6f} A "
            f"cycle change {max(record['cycle_change_source'], record['cycle_change_sink']):.2e}"
        )
    payload = {
        "status": "ok",
        "tool": f"LTspice {version}",
        "parameter_set": config.params.name,
        "inputs": case_inputs(config),
        "cases": records,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
    plot_buck_boost_waveform(
        {k: v for k, v in traces.items() if k.startswith("buck_boost")},
        spice_cases(config),
        config,
        RESULTS_DIR / "spice_buckboost_waveform",
    )
    plot_flying_capacitor_waveform(
        {k: v for k, v in traces.items() if k.startswith("flying_capacitor")},
        spice_cases(config),
        config,
        RESULTS_DIR / "spice_flyingcap_waveform",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
