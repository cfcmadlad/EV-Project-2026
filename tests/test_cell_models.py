from dataclasses import replace

import numpy as np
import pytest

from src.cell import OcvCurve
from src.cold_model import TableModel, calibrate, terminal_voltage
from src.config import CONFIG
from src.drive_cycle import load_profiles, population_block
from src.params import SECONDS_PER_HOUR
from src.soc_estimation.ekf2 import build_cell2, run_ekf2
from src.temperature import ArrheniusFit, r0_forms

SOC_GRID = np.linspace(0.0, 1.0, 5)
MODEL = TableModel(
    soc=SOC_GRID,
    r0_ohm=np.full(5, 0.02),
    r1_ohm=np.full(5, 0.01),
    tau1_s=np.full(5, 20.0),
    r2_ohm=np.full(5, 0.015),
    tau2_s=np.full(5, 200.0),
)
CURVE = OcvCurve(soc=np.linspace(0.0, 1.0, 11), volt=3.0 + 1.2 * np.linspace(0.0, 1.0, 11))


def drive_series(n: int = 1500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.arange(1, n + 1, dtype=float)
    current = 2.0 + 3.0 * np.sin(time_s / 40.0)
    soc = 1.0 - np.cumsum(current) / (SECONDS_PER_HOUR * 2.8)
    return time_s, current, soc


def test_linear_limit_of_the_nonlinear_model_is_a_plain_two_rc_cell() -> None:
    time_s, current, soc = drive_series()
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e6])
    voltage = terminal_voltage(time_s, current, soc, MODEL, CURVE, scales)
    first = 3.0 + 1.2 * soc[0] - 0.02 * current[0]
    assert voltage[0] == pytest.approx(first, abs=1e-6)
    assert np.all(np.isfinite(voltage))


def test_calibration_recovers_known_scale_factors() -> None:
    time_s, current, soc = drive_series()
    truth = np.array([1.3, 0.8, 1.2, 1.5, 0.9, 8.0])
    voltage = terminal_voltage(time_s, current, soc, MODEL, CURVE, truth)
    fitted = calibrate([time_s], [current], [voltage], [soc], MODEL, CURVE)
    rebuilt = terminal_voltage(time_s, current, soc, MODEL, CURVE, fitted)
    assert np.sqrt(np.mean((rebuilt - voltage) ** 2)) < 1e-3


def test_two_rc_ekf_recovers_from_a_wrong_start() -> None:
    soc_grid = np.linspace(0.0, 1.0, 21)
    entry = {
        "capacity_ah": 2.8,
        "ocv": {"soc": soc_grid.tolist(), "volt_v": (3.0 + 1.2 * soc_grid).tolist()},
        "hppc_tables": {
            "soc": soc_grid.tolist(),
            "r0_ohm": [0.02] * 21,
            "two_rc_fast_r_ohm": [0.01] * 21,
            "two_rc_fast_tau_s": [20.0] * 21,
            "two_rc_slow_r_ohm": [0.015] * 21,
            "two_rc_slow_tau_s": [200.0] * 21,
        },
        "hysteresis_half_width_v": [0.0] * 21,
    }
    time_s, current, soc = drive_series()
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e6])
    voltage = terminal_voltage(time_s, current, soc, MODEL, CURVE, scales)
    estimate = run_ekf2(build_cell2(entry), current, voltage, 0.8, 1e-5, 0.01, 0.2, 0.0, 1.0)
    assert np.abs(estimate[400:] - soc[400:]).max() < 0.02


def test_log_quadratic_form_beats_arrhenius_on_curved_data() -> None:
    temperature = np.array([0.0, 10.0, 25.0, 40.0])
    base = ArrheniusFit(0.2, float(np.log(0.012)), 25.0).r0_ohm(temperature)
    curved = base * np.array([1.0, 1.0, 1.0, 1.0]) + np.array([0.0, 0.0, 0.0, 0.004])
    forms = r0_forms(temperature, curved, 25.0)
    assert (
        forms["log_quadratic_in_inverse_temperature"]["max_relative_residual"]
        < forms["arrhenius"]["max_relative_residual"]
    )


def test_population_block_has_a_gain_for_active_balancing() -> None:
    config = replace(CONFIG, drive_cycle=replace(CONFIG.drive_cycle, population_packs=6))
    block = population_block(config, load_profiles(config.drive_cycle), np.random.default_rng(3))
    assert block["driving"]["buck_boost"]["fraction_packs_with_gain"] > 0.5
    assert block["driving"]["none"]["gain_vs_none_ah_median"] == 0.0
