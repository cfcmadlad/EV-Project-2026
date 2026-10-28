import sys
from pathlib import Path

import numpy as np

from src.cell import build_ocv_curve, open_circuit_voltage
from src.config import CONFIG
from src.lt_asc import write_asc
from src.lt_circuits import (
    Cells,
    Rectifier,
    Switch,
    buck_boost_netlist,
    flying_capacitor_netlist,
    shunt_netlist,
)
from src.lt_runs import ideal_ron, timing

OUT = Path(__file__).resolve().parents[1] / "ltspice"


def schematics() -> dict[str, str]:
    curve = build_ocv_curve(CONFIG.cell)
    v_source, v_sink = (float(v) for v in open_circuit_voltage(np.array([0.55, 0.50]), curve))
    cells = Cells(v_source, v_sink, CONFIG.cell.r0_ohm)
    fc, bb = CONFIG.flying_capacitor, CONFIG.buck_boost
    capacitance, f_fc = fc.component.mid, fc.frequency_hz.mid
    inductance, f_bb = bb.component.mid, bb.frequency_hz.mid
    out = {}
    for switch in Switch:
        out[f"flying_cap_{switch}"] = flying_capacitor_netlist(
            cells,
            capacitance,
            f_fc,
            switch,
            ideal_ron("flying_capacitor"),
            fc.esr_ohm,
            timing(f_fc).solver,
        ).netlist
        for rectifier in Rectifier:
            out[f"buck_boost_{rectifier}_{switch}"] = buck_boost_netlist(
                cells,
                inductance,
                f_bb,
                bb.duty,
                switch,
                rectifier,
                ideal_ron("buck_boost"),
                bb.r_inductor_ohm,
                timing(f_bb).solver,
            ).netlist
    out["shunt_real"] = shunt_netlist(
        v_source, cells.r0_ohm, CONFIG.passive.component.mid, Switch.REAL
    )
    return out


def main() -> int:
    for name, netlist in schematics().items():
        write_asc(netlist, OUT / f"{name}.asc")
    print(f"wrote {len(schematics())} schematics to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
