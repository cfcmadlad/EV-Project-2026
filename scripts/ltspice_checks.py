import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.ltspice_buck_boost import cells_for
from scripts.ltspice_schematics import schematics
from src.config import CONFIG, RESULTS_DIR
from src.lt_asc import write_asc
from src.lt_circuits import Rectifier, Switch, buck_boost_netlist, flying_capacitor_netlist
from src.lt_measure import Window, mean_of, run_batch, simulator_available, simulator_path
from src.lt_runs import buck_boost_row, flying_row, ideal_ron, timing
from src.table_io import write_rows

OUT = RESULTS_DIR / "ltspice" / "checks.csv"
WORK = Path(__file__).resolve().parents[1] / "data" / "raw" / "ltspice" / "checks"
FC_POINTS = ((1e-3, 1e3), (5.5e-3, 5.5e3), (10e-3, 10e3))
BB_POINTS = ((10e-6, 50e3), (55e-6, 30e3), (100e-6, 10e3))
WAVEFORM_POINTS = 801
WAVEFORMS = {
    "bb_base_soc55_50_55uH_30kHz": {
        "inductor_a": ("I(L1)", 1.0),
        "switch_node_v": ("V(x)", 1.0),
        "source_a": ("I(VA)", -1.0),
        "sink_a": ("I(VB)", 1.0),
    },
    "fc_base_5.5mF_5.5kHz": {
        "capacitor_a": ("I(RE)", 1.0),
        "capacitor_top_v": ("V(cp)", 1.0),
        "source_a": ("I(VA)", -1.0),
        "sink_a": ("I(VB)", 1.0),
    },
}
VARIANTS = (
    ("base", 200, 3e-4),
    ("fine_step", 400, 3e-4),
    ("tight_tol", 200, 1e-4),
    ("loose_tol", 200, 1e-3),
)


def schematic_matches_netlist() -> bool:
    scratch = WORK.parent / "asc_check"
    scratch.mkdir(parents=True, exist_ok=True)
    exe = simulator_path()
    matched = True
    for name, netlist in schematics().items():
        asc = scratch / f"{name}.asc"
        write_asc(netlist, asc)
        subprocess.run([exe, "-netlist", str(asc)], check=True, timeout=120)
        produced = (scratch / f"{name}.net").read_text(errors="replace").splitlines()
        wanted = [
            line
            for line in netlist.splitlines()
            if line and line[0].isalpha() and not line.startswith("*")
        ]
        matched &= all(line in produced for line in wanted)
    return matched


def main() -> int:
    if not simulator_available():
        print("LTspice not found; results/ltspice/checks.csv is the cached result")
        return 1
    fc, bb = CONFIG.flying_capacitor, CONFIG.buck_boost
    plan = {}
    for variant, divisor, reltol in VARIANTS:
        for capacitance, frequency in FC_POINTS:
            cells = cells_for("soc55_50", 0.55, 0.50)
            clock = timing(frequency, divisor, reltol)
            built = flying_capacitor_netlist(
                cells, capacitance, frequency, Switch.REAL, fc.r_ds_on_ohm, fc.esr_ohm, clock.solver
            )
            plan[f"fc_{variant}_{capacitance * 1e3:g}mF_{frequency / 1e3:g}kHz"] = (
                "flying_capacitor",
                variant,
                built,
                cells,
                clock,
                frequency,
                capacitance,
            )
        for label, a, b in (("soc55_50", 0.55, 0.50), ("v4p2_3p0", 4.2, 3.0)):
            for inductance, frequency in BB_POINTS:
                cells = cells_for(label, a, b)
                clock = timing(frequency, divisor, reltol)
                built = buck_boost_netlist(
                    cells,
                    inductance,
                    frequency,
                    bb.duty,
                    Switch.REAL,
                    Rectifier.SYNC,
                    ideal_ron("buck_boost"),
                    bb.r_inductor_ohm,
                    clock.solver,
                )
                plan[f"bb_{variant}_{label}_{inductance * 1e6:g}uH_{frequency / 1e3:g}kHz"] = (
                    "buck_boost",
                    variant,
                    built,
                    cells,
                    clock,
                    frequency,
                    inductance,
                )

    waveforms: dict[str, dict[str, np.ndarray]] = {}

    def handle(tag, trace):
        converter, _, built, cells, clock, frequency, _ = plan[tag]
        if converter == "flying_capacitor":
            row = flying_row(trace, built, cells, clock, fc.esr_ohm)
        else:
            row = buck_boost_row(trace, built, cells, clock, frequency, bb.duty, bb.r_inductor_ohm)
        period = 1.0 / frequency
        last = clock.last
        cycle = Window(last.stop_s - period, last.stop_s)
        row["cycle_charge_src_c"] = mean_of(trace, cycle, {"I(VA)": -1.0}) * period
        row["cycle_charge_snk_c"] = mean_of(trace, cycle, {"I(VB)": 1.0}) * period
        if tag in WAVEFORMS:
            grid = (
                last.stop_s - 2.0 * period + 2.0 * period * np.linspace(0.0, 1.0, WAVEFORM_POINTS)
            )
            waveforms[tag] = {
                "t_us": (grid - grid[0]) * 1e6,
                **{
                    name: np.interp(grid, trace.time_s, trace.signals[key.lower()]) * sign
                    for name, (key, sign) in WAVEFORMS[tag].items()
                },
            }
        return row

    measured = run_batch({tag: item[2].netlist for tag, item in plan.items()}, WORK, 1, handle)
    rows = []
    for tag, item in plan.items():
        converter, variant, _, cells, _, frequency, component = item
        rows.append(
            {
                "tag": tag,
                "converter": converter,
                "variant": variant,
                "component": component,
                "f_hz": frequency,
                "v_source": cells.v_source,
                "v_sink": cells.v_sink,
                **measured[tag],
            }
        )
    write_rows(rows, OUT)
    for tag, columns in waveforms.items():
        kind = "buck_boost" if tag.startswith("bb") else "flying_capacitor"
        count = len(columns["t_us"])
        write_rows(
            [{k: float(v[i]) for k, v in columns.items()} for i in range(count)],
            OUT.parent / f"waveform_{kind}.csv",
        )
    matched = schematic_matches_netlist()
    print(f"wrote {len(rows)} rows to {OUT}; schematics match generated netlists: {matched}")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
