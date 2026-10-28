import sys
from pathlib import Path

import numpy as np

from src.cell import build_ocv_curve, open_circuit_voltage
from src.config import CONFIG, RESULTS_DIR
from src.lt_circuits import Switch, shunt_netlist
from src.lt_measure import netlist_sha256, run_batch, simulator_available
from src.table_io import write_rows

OUT = RESULTS_DIR / "ltspice" / "shunt.csv"
WORK = Path(__file__).resolve().parents[1] / "data" / "raw" / "ltspice" / "shunt"
BLEED_OHM = (20.0, 60.0, 100.0)
SOC = 0.55


def main() -> int:
    if not simulator_available():
        print("LTspice not found; results/ltspice/shunt.csv is the cached result")
        return 1
    voltage = float(open_circuit_voltage(np.array([SOC]), build_ocv_curve(CONFIG.cell))[0])
    r0 = CONFIG.cell.r0_ohm
    plan = {f"shunt_{switch}_{int(r)}ohm": (switch, r) for switch in Switch for r in BLEED_OHM}
    netlists = {tag: shunt_netlist(voltage, r0, r, switch) for tag, (switch, r) in plan.items()}
    measured = run_batch(netlists, WORK, 1, lambda _tag, trace: float(-trace.signals["i(va)"][0]))
    rows = []
    for tag, (switch, r) in plan.items():
        python_current = voltage / (r + r0)
        current = measured[tag]
        rows.append(
            {
                "tag": tag,
                "switch": str(switch),
                "r_bleed_ohm": r,
                "v_cell": voltage,
                "i_bleed_a": current,
                "p_resistor_w": current**2 * r,
                "py_i_bleed_a": python_current,
                "py_p_resistor_w": python_current**2 * r,
                "current_error": current / python_current - 1.0,
                "netlist_sha256": netlist_sha256(netlists[tag]),
            }
        )
    write_rows(rows, OUT)
    print(f"wrote {len(rows)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
