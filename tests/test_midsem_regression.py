import json
from dataclasses import replace

from run_all import baseline_table, run_sweeps
from src.checks import midsem_regression_check
from src.config import CONFIG, RESULTS_DIR
from src.experiments import baseline_pack, run_baseline
from src.params import ASSUMED

ASSUMED_CONFIG = replace(CONFIG, params=ASSUMED)
REFERENCE = json.loads((RESULTS_DIR / "midsem_reference" / "metrics.json").read_text())


def test_baseline_matches_pinned_midsem_values() -> None:
    assert midsem_regression_check(CONFIG)["pass"]


def test_baseline_table_matches_reference_exactly() -> None:
    pack = baseline_pack(ASSUMED_CONFIG)
    table = baseline_table(run_baseline(pack, ASSUMED_CONFIG, trace=False))
    assert table == REFERENCE["baseline"]


def test_sweeps_match_reference_exactly() -> None:
    sweeps = run_sweeps(baseline_pack(ASSUMED_CONFIG), ASSUMED_CONFIG)
    for name, (values, times) in sweeps.items():
        assert values.tolist() == REFERENCE["sweeps"][name]["values"]
        assert times.tolist() == REFERENCE["sweeps"][name]["balancing_time_s"]
