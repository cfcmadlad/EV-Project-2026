import numpy as np
import pytest
import torch

from src.config import Technique
from src.deep_surrogate import (
    MODELS,
    VALIDATION_FRACTION,
    fit_scaler,
    make_split,
    scores,
    summarise,
)
from src.soc_estimation.hybrid import HYBRID_INPUTS, macs_per_estimate, parameter_count
from src.trajectory import (
    DECODERS,
    GRID,
    OUTPUT_CHANNELS,
    crossing_norm_time,
    trajectory_scores,
)

N_FEATURES = 19


def test_split_is_disjoint_stratified_and_validation_comes_from_train():
    technique = np.repeat(np.arange(3), 400)
    split = make_split(technique, 2026, 0.2)
    sets = [set(split.train), set(split.validation), set(split.test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(split.test) == 240
    assert len(split.validation) == pytest.approx(VALIDATION_FRACTION * 960, abs=2)
    for t in range(3):
        assert np.sum(technique[split.test] == t) == 80


def test_scaler_uses_only_the_given_rows():
    train = np.array([[0.0, 1.0], [2.0, 3.0]])
    scaler = fit_scaler(train)
    np.testing.assert_allclose(scaler(train).mean(axis=0), 0.0, atol=1e-6)
    assert scaler(np.array([[100.0, 100.0]])).max() > 10.0


@pytest.mark.parametrize("kind", list(MODELS))
def test_surrogate_networks_map_features_to_two_targets(kind):
    model = MODELS[kind](N_FEATURES, 32, 2)
    assert model(torch.zeros(5, N_FEATURES)).shape == (5, 2)
    wider = MODELS[kind](N_FEATURES + 2, 32, 2)
    assert wider(torch.zeros(5, N_FEATURES + 2)).shape == (5, 2)


def test_scores_are_perfect_for_perfect_predictions():
    technique = np.repeat(np.arange(3), 10)
    values = np.linspace(1.0, 2.0, 30)
    result = scores(values, values, technique)
    assert result["overall"]["r2"] == pytest.approx(1.0)
    assert set(result) == {"overall", *(t.key for t in Technique)}


def test_summarise_reports_mean_std_and_range_over_seeds():
    runs = [{"t": {"overall": {"r2": v}}} for v in (0.9, 0.95, 1.0)]
    leaf = summarise(runs)["t"]["overall"]["r2"]
    assert leaf["mean"] == pytest.approx(0.95)
    assert leaf["min"] == 0.9 and leaf["max"] == 1.0


def test_crossing_time_interpolates_between_grid_points():
    curve = np.linspace(1.0, 0.0, GRID)[None, :]
    assert crossing_norm_time(curve, 0.5)[0] == pytest.approx(0.5, abs=1e-6)
    assert crossing_norm_time(np.ones((1, GRID)), 0.5)[0] == 1.0


@pytest.mark.parametrize("kind", list(DECODERS))
def test_trajectory_decoders_emit_a_curve_and_a_time(kind):
    model = DECODERS[kind](N_FEATURES, 32)
    curves, time = model(torch.zeros(3, N_FEATURES))
    assert curves.shape == (3, GRID, OUTPUT_CHANNELS)
    assert time.shape == (3,)


def test_trajectory_scores_vanish_for_identical_curves():
    technique = np.repeat(np.arange(3), 2)
    curves = np.tile(np.linspace(0.1, 0.0, GRID)[None, :, None], (6, 1, OUTPUT_CHANNELS))
    result = trajectory_scores(curves, curves, np.ones(6), np.ones(6), technique)
    assert result["overall"]["spread_rmse_percent"] == 0.0
    assert result["overall"]["total_time_median_rel_error_percent"] == 0.0


def test_gru_cost_formulas():
    assert parameter_count(3, 32) == 3585
    assert macs_per_estimate(3, 32, 300) == 1008032
    assert parameter_count(HYBRID_INPUTS, 32) > parameter_count(3, 32)
