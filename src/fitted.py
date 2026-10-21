import json
from pathlib import Path

from src.params import (
    BuckBoostSpec,
    CellParams,
    FlyingCapSpec,
    OCVTable,
    PackSpec,
    ParameterSet,
    PassiveSpec,
)

PARAMS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "hg2_25c_params.json"


def load_parameter_set(path: Path) -> ParameterSet:
    payload = json.loads(path.read_text())
    if payload["soc_dependent_tables"]["enabled"]:
        raise NotImplementedError("SOC-dependent parameter tables are stored but not simulated")
    cell = payload["cell"]
    return ParameterSet(
        name=payload["name"],
        cell=CellParams(
            capacity_ah=cell["capacity_ah"],
            r0_ohm=cell["r0_ohm"],
            r1_ohm=cell["r1_ohm"],
            c1_farad=cell["c1_farad"],
            ocv=OCVTable(soc=tuple(cell["ocv"]["soc"]), volt_v=tuple(cell["ocv"]["volt_v"])),
        ),
        pack=PackSpec(),
        passive=PassiveSpec(),
        buck_boost=BuckBoostSpec(),
        flying_capacitor=FlyingCapSpec(),
    )


FITTED_HG2_25C = load_parameter_set(PARAMS_PATH)
