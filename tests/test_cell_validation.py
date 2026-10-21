import json

from src.config import CONFIG
from src.uncertainty import IDENTIFICATION_PATH

SPEC = CONFIG.identification
CACHED = json.loads(IDENTIFICATION_PATH.read_text())


def test_kept_model_meets_the_external_check() -> None:
    selection = CACHED["selection"]
    assert selection["status"] == "PASS"
    kept = selection["kept_model"]
    for cycle in ("UDDS", "LA92"):
        metrics = CACHED["validation"][cycle][kept]
        assert metrics["rmse_whole_v"] <= SPEC.rmse_whole_cycle_v
        assert metrics["rmse_window_v"] <= SPEC.rmse_window_v


def test_fitted_model_beats_assumed_parameters_on_every_usable_cycle() -> None:
    kept = CACHED["selection"]["kept_model"]
    for cycle, models in CACHED["validation"].items():
        assert models[kept]["rmse_whole_v"] < models["assumed"]["rmse_whole_v"], cycle


def test_hwfet_is_excluded_because_the_file_has_no_current() -> None:
    assert CACHED["usable_drive_cycles"]["HWFET"] is False
    assert "HWFET" not in CACHED["validation"]
