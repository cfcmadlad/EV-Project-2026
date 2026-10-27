from dataclasses import replace

import numpy as np

from src.config import CONFIG
from src.params import VoltageBalancingSpec
from src.selector import choose, policy_row
from src.voltage_balancing import required_sigma_mv, run_voltage_study, setting_key


def test_choose_takes_the_cheapest_feasible_and_falls_back_to_the_fastest() -> None:
    time_s = np.array([[100.0, 50.0, 80.0], [400.0, 300.0, 200.0]])
    energy_j = np.array([[1.0, 5.0, 3.0], [1.0, 5.0, 3.0]])
    assert choose(time_s, energy_j, 90.0).tolist() == [2, 2]
    assert choose(time_s, energy_j, 500.0).tolist() == [0, 0]


def test_policy_row_counts_violations_only_where_a_feasible_choice_exists() -> None:
    time_s = np.array([[100.0, 50.0, 80.0], [400.0, 300.0, 200.0]])
    energy_j = np.array([[1.0, 5.0, 3.0], [1.0, 5.0, 3.0]])
    oracle = choose(time_s, energy_j, 90.0)
    row = policy_row(np.array([0, 0]), time_s, energy_j, oracle, 90.0)
    assert row["budget_violation_fraction_among_feasible"] == 1.0
    assert row["agrees_with_oracle"] == 0.0


def test_required_sigma_interpolates_between_grid_points() -> None:
    result = required_sigma_mv([1.0, 2.0, 4.0], [1.0, 0.9, 0.5], 0.95)
    assert result["largest_grid_sigma_mv"] == 1.0
    assert 1.0 < result["interpolated_sigma_mv"] < 2.0


def test_small_sensor_error_balances_and_large_error_does_not() -> None:
    spec = VoltageBalancingSpec(
        sigma_mv=(0.5, 20.0), stop_thresholds=(0.005,), packs=8, max_balancing_h=6.0
    )
    study = run_voltage_study(replace(CONFIG, voltage_balancing=spec), np.random.default_rng(1))
    rows = study["results"]["passive"]
    small = rows[setting_key(0.005, 0.5)]["fraction_true_spread_below_target"]
    large = rows[setting_key(0.005, 20.0)]["fraction_true_spread_below_target"]
    assert small > large
