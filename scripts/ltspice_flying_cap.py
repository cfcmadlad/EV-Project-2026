import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.cell import build_ocv_curve, open_circuit_voltage
from src.config import CONFIG, RESULTS_DIR
from src.lt_circuits import Cells, Switch, flying_capacitor_netlist
from src.lt_measure import netlist_sha256, run_batch, simulator_available
from src.lt_runs import flying_row, timing
from src.params import FlyingCapModel
from src.switched_cap import link_resistance_ohm
from src.table_io import write_rows

OUT = RESULTS_DIR / "ltspice" / "flying_cap.csv"
WORK = Path(__file__).resolve().parents[1] / "data" / "raw" / "ltspice" / "flying"
CAPACITANCE_F = (1e-3, 3.25e-3, 5.5e-3, 7.75e-3, 10e-3)
FREQUENCY_HZ = (1e3, 3.25e3, 5.5e3, 7.75e3, 10e3)
SOC_SPREAD = (0.01, 0.03, 0.05, 0.10)
BASE_SOC = 0.50
PARALLEL = 4


def python_current(
    model: FlyingCapModel, capacitance_f: float, frequency_hz: float, cells: Cells
) -> float:
    spec = replace(CONFIG.flying_capacitor, model=model)
    r0 = np.full((1, 2), cells.r0_ohm)
    resistance = link_resistance_ohm(np.array([1.0 / (frequency_hz * capacitance_f)]), r0, spec)
    return float((cells.v_source - cells.v_sink) / resistance[0, 0])


def cases() -> list[tuple[str, Switch, float, float, float, Cells]]:
    curve = build_ocv_curve(CONFIG.cell)
    out = []
    for capacitance in CAPACITANCE_F:
        for frequency in FREQUENCY_HZ:
            for spread in SOC_SPREAD:
                v_source, v_sink = open_circuit_voltage(
                    np.array([BASE_SOC + spread, BASE_SOC]), curve
                )
                cells = Cells(float(v_source), float(v_sink), CONFIG.cell.r0_ohm)
                for switch in Switch:
                    tag = f"fc_{switch}_{capacitance * 1e3:g}mF_{frequency / 1e3:g}kHz_{spread:g}"
                    out.append((tag, switch, capacitance, frequency, spread, cells))
    return out


def main() -> int:
    if not simulator_available():
        print("LTspice not found; results/ltspice/flying_cap.csv is the cached result")
        return 1
    spec = CONFIG.flying_capacitor
    plan = {tag: rest for tag, *rest in cases()}
    builts, netlists = {}, {}
    for tag, (switch, capacitance, frequency, _, cells) in plan.items():
        clock = timing(frequency)
        built = flying_capacitor_netlist(
            cells, capacitance, frequency, switch, spec.r_ds_on_ohm, spec.esr_ohm, clock.solver
        )
        builts[tag], netlists[tag] = built, built.netlist

    def handle(tag: str, trace):
        switch, capacitance, frequency, _, cells = plan[tag]
        return flying_row(trace, builts[tag], cells, timing(frequency), spec.esr_ohm)

    measured = run_batch(netlists, WORK, PARALLEL, handle)
    rows = []
    for tag, (switch, capacitance, frequency, spread, cells) in plan.items():
        row = {
            "tag": tag,
            "switch": str(switch),
            "c_f": capacitance,
            "f_hz": frequency,
            "soc_spread": spread,
            "v_source": cells.v_source,
            "v_sink": cells.v_sink,
            "r0_ohm": cells.r0_ohm,
            "py_i_series_a": python_current(FlyingCapModel.MIDSEM, capacitance, frequency, cells),
            "py_i_coth_a": python_current(FlyingCapModel.COTH, capacitance, frequency, cells),
            "py_efficiency": cells.v_sink / cells.v_source,
            **measured[tag],
            "netlist_sha256": netlist_sha256(netlists[tag]),
        }
        rows.append(row)
    write_rows(rows, OUT)
    print(f"wrote {len(rows)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
