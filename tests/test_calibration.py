import csv

import numpy as np
import pytest
import torch

from src.calibration import LogGrid, circuit_maps, saturation_current_a
from src.quantize import quantize_state, quantize_tensor


def write(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def synthetic_directory(tmp_path):
    flying_header = ["switch", "c_f", "f_hz", "r0_ohm", "r_eff_ohm", "p_gate_w"]
    flying = [
        ["real", c, f, 0.01, 0.2 + c * 10 + f * 1e-5, 1e-4 * f / 1e3]
        for c in (1e-3, 1e-2)
        for f in (1e3, 1e4)
    ]
    write(tmp_path / "flying_cap.csv", flying_header, flying)
    buck_header = [
        "switch",
        "rectifier",
        "point",
        "l_h",
        "f_hz",
        "i_src_a",
        "py_i_src_mean_a",
        "p_loss_total_w",
        "py_loss_mean_w",
        "saturates",
    ]
    buck = [
        ["real", "sync", "soc55_50", ind, f, 0.18, 0.2, 0.03, 0.02, 0.0]
        for ind in (1e-5, 1e-4)
        for f in (1e4, 5e4)
    ]
    write(tmp_path / "buck_boost.csv", buck_header, buck)
    return tmp_path


def test_log_grid_hits_nodes_and_clamps():
    grid = LogGrid(
        np.array([1.0, 10.0]), np.array([1.0, 100.0]), np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    assert grid(np.array(1.0), np.array(1.0)) == pytest.approx(1.0)
    assert grid(np.array(10.0), np.array(100.0)) == pytest.approx(4.0)
    assert grid(np.array(1e6), np.array(1e6)) == pytest.approx(4.0)
    assert grid(np.array(np.sqrt(10.0)), np.array(1.0)) == pytest.approx(2.0)


def test_circuit_maps_from_csv(tmp_path):
    maps = circuit_maps(synthetic_directory(tmp_path))
    element = maps.flying_element_ohm(np.array(1e-3), np.array(1e3))
    assert element == pytest.approx(0.2 + 1e-3 * 10 + 1e3 * 1e-5 - 0.02)
    assert maps.buck_boost_current_ratio(np.array(5e-5), np.array(2e4)) == pytest.approx(0.9)
    assert maps.buck_boost_overhead_w(np.array(5e-5), np.array(2e4)) == pytest.approx(0.01)


def test_saturation_rule_scales_with_inverse_root_l():
    assert saturation_current_a(100e-6) * 10.0 == pytest.approx(saturation_current_a(1e-6))
    assert saturation_current_a(10e-6) == pytest.approx(np.sqrt(2 * 50e-6 / 10e-6))


def test_quantisation_levels_and_size():
    weight = torch.linspace(-1.0, 1.0, 101).reshape(1, -1)
    assert quantize_tensor(weight, 8).unique().numel() <= 255
    assert quantize_tensor(weight, 1).unique().numel() == 2
    state = {"w": weight, "b": torch.zeros(3)}
    _, size8 = quantize_state(state, 8)
    _, size1 = quantize_state(state, 1)
    assert size8 == 101 + 4 + 12
    assert size1 == 13 + 4 + 12


def test_default_models_are_circuit_calibrated():
    from src.config import CONFIG
    from src.params import FlyingCapModel

    assert CONFIG.buck_boost.circuit_calibrated
    assert CONFIG.flying_capacitor.model is FlyingCapModel.CIRCUIT


def test_circuit_flying_resistance_adds_cells_and_overhead():
    from src.params import FlyingCapModel, FlyingCapSpec
    from src.switched_cap import link_resistances_ohm

    r0 = np.array([[0.010, 0.012, 0.014, 0.011]])
    frequency = np.array([5.5e3])
    inverse = 1.0 / (5.5e-3 * frequency)
    resistance = link_resistances_ohm(
        inverse, r0, FlyingCapSpec(model=FlyingCapModel.CIRCUIT), frequency
    )
    assert resistance.shape == (1, 3, 3)
    np.testing.assert_allclose(
        resistance[:, 1] + r0[:, :-1] + r0[:, 1:], resistance[:, 0], rtol=1e-12
    )
    assert np.all(resistance[:, 2] > 0.0)


def test_circuit_model_needs_frequency():
    from src.params import FlyingCapModel, FlyingCapSpec
    from src.switched_cap import link_resistance_ohm

    with pytest.raises(ValueError):
        link_resistance_ohm(
            np.array([0.04]), np.full((1, 4), 0.01), FlyingCapSpec(model=FlyingCapModel.CIRCUIT)
        )


def test_calibrated_buck_boost_matches_ltspice_at_the_baseline_link():
    from src.calibration import LTSPICE_DIR
    from src.lt_report import calibrated_prediction
    from src.table_io import read_rows

    rows = read_rows(LTSPICE_DIR / "buck_boost.csv")
    row = next(
        r
        for r in rows
        if r["switch"] == "real"
        and r["rectifier"] == "sync"
        and r["point"] == "soc55_50"
        and r["l_h"] == 55e-6
        and r["f_hz"] == 30e3
    )
    calibrated = calibrated_prediction(row, True)
    averaged = calibrated_prediction(row, False)
    assert abs(calibrated["i_src"] / row["i_src_a"] - 1.0) < 0.03
    assert abs(calibrated["i_src"] / row["i_src_a"] - 1.0) <= abs(
        averaged["i_src"] / row["i_src_a"] - 1.0
    )
    assert calibrated["loss"] > averaged["loss"]


def test_pruned_buck_boost_range_respects_the_saturation_rule():
    from src.calibration import buck_boost_feasible, saturation_current_a

    inductance = np.array([10e-6, 100e-6])
    frequency = np.array([50e3, 10e3])
    feasible = buck_boost_feasible(inductance, frequency, 0.4, 4.19)
    assert not feasible[0]
    assert feasible[1] == (saturation_current_a(100e-6) >= 4.19 * 0.4 / (100e-6 * 10e3))
