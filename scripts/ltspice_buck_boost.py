import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.balancing import buck_boost_currents
from src.calibration import saturation_current_a
from src.cell import build_ocv_curve, open_circuit_voltage
from src.config import CONFIG, RESULTS_DIR
from src.ledger import buck_boost_element_loss_w
from src.lt_circuits import Cells, Rectifier, Switch, buck_boost_netlist
from src.lt_measure import netlist_sha256, run_batch, simulator_available
from src.lt_runs import buck_boost_row, ideal_ron, timing
from src.table_io import write_rows

OUT = RESULTS_DIR / "ltspice" / "buck_boost.csv"
WORK = Path(__file__).resolve().parents[1] / "data" / "raw" / "ltspice" / "buck_boost"
INDUCTANCE_H = (10e-6, 22e-6, 55e-6, 100e-6)
FREQUENCY_HZ = (10e3, 20e3, 30e3, 50e3)
OPERATING_POINTS = (
    ("soc55_50", 0.55, 0.50),
    ("soc60_50", 0.60, 0.50),
    ("v4p2_3p0", 4.2, 3.0),
)
PARALLEL = 4


def cells_for(label: str, a: float, b: float) -> Cells:
    curve = build_ocv_curve(CONFIG.cell)
    if label.startswith("v"):
        return Cells(a, b, CONFIG.cell.r0_ohm)
    v_source, v_sink = open_circuit_voltage(np.array([a, b]), curve)
    return Cells(float(v_source), float(v_sink), CONFIG.cell.r0_ohm)


def python_reference(inductance_h: float, frequency_hz: float, cells: Cells) -> dict[str, float]:
    out = {}
    emf = np.array([[cells.v_source, cells.v_sink]])
    r0 = np.full((1, 2), cells.r0_ohm)
    inverse = np.array([1.0 / (inductance_h * frequency_hz)])
    for name, ripple in (("mean", False), ("rms", True)):
        hardware = replace(CONFIG.buck_boost, ripple_r0_loss=ripple)
        flow = buck_boost_currents(np.array([[1.0, 0.0]]), emf, emf, r0, inverse, hardware, 0.0)
        out[f"py_i_src_{name}_a"] = float(flow.link_source_a[0, 0])
        out[f"py_i_snk_{name}_a"] = float(flow.link_sink_a[0, 0])
        out[f"py_efficiency_{name}"] = float(flow.delivered_w[0] / flow.removed_w[0])
        out[f"py_loss_{name}_w"] = float(flow.removed_w[0] - flow.delivered_w[0])
        if not ripple:
            out["py_i_pk_a"] = 2.0 * out["py_i_src_mean_a"] / hardware.duty
            out["py_element_loss_w"] = float(
                buck_boost_element_loss_w(flow, emf, r0, inverse, hardware)[0]
            )
    return out


def plan() -> dict[str, tuple]:
    out = {}
    for inductance in INDUCTANCE_H:
        for frequency in FREQUENCY_HZ:
            for label, a, b in OPERATING_POINTS:
                for switch in Switch:
                    for rectifier in Rectifier:
                        tag = (
                            f"bb_{switch}_{rectifier}_{inductance * 1e6:g}uH_"
                            f"{frequency / 1e3:g}kHz_{label}"
                        )
                        out[tag] = (
                            switch,
                            rectifier,
                            inductance,
                            frequency,
                            label,
                            cells_for(label, a, b),
                        )
    return out


def main() -> int:
    if not simulator_available():
        print("LTspice not found; results/ltspice/buck_boost.csv is the cached result")
        return 1
    spec = CONFIG.buck_boost
    runs = plan()
    builts, netlists = {}, {}
    for tag, (switch, rectifier, inductance, frequency, _, cells) in runs.items():
        clock = timing(frequency)
        built = buck_boost_netlist(
            cells,
            inductance,
            frequency,
            spec.duty,
            switch,
            rectifier,
            ideal_ron("buck_boost"),
            spec.r_inductor_ohm,
            clock.solver,
        )
        builts[tag], netlists[tag] = built, built.netlist

    def handle(tag: str, trace):
        _, _, _, frequency, _, cells = runs[tag]
        return buck_boost_row(
            trace, builts[tag], cells, timing(frequency), frequency, spec.duty, spec.r_inductor_ohm
        )

    measured = run_batch(netlists, WORK, PARALLEL, handle)
    rows = []
    for tag, (switch, rectifier, inductance, frequency, label, cells) in runs.items():
        isat = float(saturation_current_a(inductance))
        row = {
            "tag": tag,
            "switch": str(switch),
            "rectifier": str(rectifier),
            "l_h": inductance,
            "f_hz": frequency,
            "point": label,
            "v_source": cells.v_source,
            "v_sink": cells.v_sink,
            "r0_ohm": cells.r0_ohm,
            **python_reference(inductance, frequency, cells),
            **measured[tag],
            "isat_a": isat,
            "saturates": float(measured[tag]["i_pk_a"] > isat),
            "netlist_sha256": netlist_sha256(netlists[tag]),
        }
        rows.append(row)
    write_rows(rows, OUT)
    print(f"wrote {len(rows)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
