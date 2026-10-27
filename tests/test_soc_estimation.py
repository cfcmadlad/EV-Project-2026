import numpy as np
import pytest

from src.resample import interval_end, interval_mean
from src.soc_estimation.coulomb import coulomb_count
from src.soc_estimation.data import SPLITS_PATH, load_splits
from src.soc_estimation.ekf import MODE_CONSTANT, build_cell, run_ekf
from src.soc_estimation.gru import macs_per_estimate, parameter_count, window_ends
from src.soc_estimation.report import compare

CAPACITY_AH = 3.0
ENTRY = {
    "capacity_ah": CAPACITY_AH,
    "ocv": {"soc": [0.0, 1.0], "volt_v": [3.0, 4.2]},
    "one_rc_cell": {"r0_ohm": 0.02, "r1_ohm": 0.02, "c1_farad": 1000.0},
}


def synthetic_discharge(n_steps: int, current_a: float) -> tuple[np.ndarray, np.ndarray]:
    soc = 1.0 - np.arange(1, n_steps + 1) * current_a / (3600.0 * CAPACITY_AH)
    v1 = 0.02 * current_a * (1.0 - np.exp(-np.arange(1, n_steps + 1) / 20.0))
    return soc, 3.0 + 1.2 * soc - current_a * 0.02 - v1


def test_coulomb_counting_matches_the_integral_label() -> None:
    current = np.tile([3.0, -1.0, 6.0, 0.0], 100)
    soc = coulomb_count(current, 1.0, CAPACITY_AH, 1.0, 0.0)
    assert soc[-1] == pytest.approx(1.0 - current.sum() / 3600.0 / CAPACITY_AH, abs=1e-12)


def test_coulomb_offset_drifts_linearly() -> None:
    clean = coulomb_count(np.full(600, 2.0), 1.0, CAPACITY_AH, 1.0, 0.0)
    offset = coulomb_count(np.full(600, 2.0), 1.0, CAPACITY_AH, 1.0, 0.375)
    assert clean[-1] - offset[-1] == pytest.approx(0.375 * 600 / 3600.0 / CAPACITY_AH)


def test_ekf_recovers_from_a_wrong_start_with_correct_parameters() -> None:
    soc, voltage = synthetic_discharge(1200, 3.0)
    cell = build_cell(ENTRY, MODE_CONSTANT)
    estimate = run_ekf(cell, np.full(1200, 3.0), voltage, 0.8, 1e-5, 0.01, 0.2, 1.0)
    assert np.abs(estimate[300:] - soc[300:]).max() < 0.02


def test_interval_mean_preserves_charge() -> None:
    time_s = np.cumsum(np.full(1001, 0.1)) - 0.1
    current = np.sin(time_s)
    averaged = interval_mean(time_s, current, 1.0)
    assert averaged.size == 100
    assert averaged.sum() == pytest.approx(np.trapezoid(current, time_s)[()], abs=0.01)
    assert interval_end(time_s, time_s, 1.0)[0] == pytest.approx(1.0)


def test_gru_cost_formulas() -> None:
    assert parameter_count(32) == 3 * (3 * 32 + 32 * 32 + 64) + 33
    assert macs_per_estimate(32, 300) == 300 * 3 * (96 + 1024) + 32
    assert window_ends(1000, 300, 2)[0] == 299


def test_splits_are_disjoint_and_test_b_is_the_unseen_temperature() -> None:
    assert SPLITS_PATH.exists()
    splits = load_splits()["splits"]
    files = {name: {(e["folder"], e["file"]) for e in entries} for name, entries in splits.items()}
    names = list(files)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            assert files[first].isdisjoint(files[second])
    assert {e["temperature_c"] for e in splits["test_b"]} == {10}
    assert 10 not in {
        e["temperature_c"] for k in ("train", "validation", "test_a") for e in splits[k]
    }
    assert all(
        e["cycle"] in ("Mixed1", "Mixed2", "Mixed3", "Mixed4", "Mixed5", "Mixed6")
        for e in splits["train"]
    )


def leaf(rmse: float, std: float = 0.0) -> dict[str, dict[str, float]]:
    return {"all": {"rmse_percent": rmse, "rmse_percent_std": std}}


def test_verdict_needs_a_win_on_both_sets_beyond_the_seed_spread() -> None:
    scenario = {"cc": "s", "ekf": "s", "gru": "s"}
    results = {
        "cc": {"s": {"test_a": leaf(9.0), "test_b": leaf(9.0)}},
        "ekf": {"s": {"test_a": leaf(4.0), "test_b": leaf(5.0)}},
        "gru": {"s": {"test_a": leaf(3.9, 0.3), "test_b": leaf(3.0, 0.3)}},
    }
    verdicts = compare(results, scenario)
    assert verdicts["gru_vs_ekf"]["verdict"] == "no_verdict"
    assert verdicts["ekf_vs_cc"]["verdict"] == "ekf_beats_cc"
    assert verdicts["gru_vs_cc"]["verdict"] == "gru_beats_cc"
