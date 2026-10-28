import json
from dataclasses import replace

import numpy as np

from src.fitted import FITTED_HG2_25C, PARAMS_PATH, load_parameter_set
from src.params import ASSUMED


def test_default_set_is_loaded_from_json() -> None:
    payload = json.loads(PARAMS_PATH.read_text())
    assert FITTED_HG2_25C.name == payload["name"] == "fitted_hg2_25c"
    assert FITTED_HG2_25C.cell.capacity_ah == payload["cell"]["capacity_ah"]
    assert FITTED_HG2_25C.cell.r0_ohm == payload["cell"]["r0_ohm"]


def test_soc_dependent_tables_are_stored_but_disabled() -> None:
    tables = json.loads(PARAMS_PATH.read_text())["soc_dependent_tables"]
    assert tables["enabled"] is False
    assert len(tables["soc"]) == len(tables["r0_ohm"]) == 12


def test_fitted_ocv_is_strictly_increasing_and_measured() -> None:
    volt = np.array(FITTED_HG2_25C.cell.ocv.volt_v)
    soc = np.array(FITTED_HG2_25C.cell.ocv.soc)
    assert volt.size == 101
    assert np.all(np.diff(volt) > 0.0)
    assert soc[0] == 0.0 and soc[-1] == 1.0
    assert volt[0] != 3.0


def test_cell_to_cell_spread_stays_assumed() -> None:
    assert FITTED_HG2_25C.pack == ASSUMED.pack
    buck_boost = load_parameter_set(PARAMS_PATH).buck_boost
    assert buck_boost.circuit_calibrated
    assert replace(buck_boost, circuit_calibrated=False) == ASSUMED.buck_boost
