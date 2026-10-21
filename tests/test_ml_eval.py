from dataclasses import replace

import numpy as np
import pytest

from src.config import CONFIG, Technique
from src.dataset import generate_dataset
from src.experiments import random_streams
from src.ml_eval import (
    GROUPS,
    MODELS,
    beats,
    build_eval_data,
    column_indices,
    conformal,
    cross_validate,
    linear_predict,
    mean_std,
    score,
    summarise_cv,
)
from src.params import DatasetSpec, MLEvalSpec, MLSpec

SMALL = replace(
    CONFIG,
    dataset=DatasetSpec(runs_per_technique=60),
    ml=MLSpec(max_iter=15, permutation_repeats=1, timing_repeats=1),
    ml_eval=MLEvalSpec(n_folds=3, time_grid_points=24, bisection_steps=12),
)


@pytest.fixture(scope="module")
def data():
    dataset = generate_dataset(SMALL, random_streams(SMALL.seed)["dataset"])
    return build_eval_data(dataset, SMALL)


def test_beats_needs_every_fold_and_a_margin() -> None:
    assert beats([1.0, 1.1, 0.9, 1.0, 1.0], [3.0, 3.1, 2.9, 3.0, 3.0])["beats"]
    assert not beats([1.0, 1.1, 0.9, 1.0, 3.5], [3.0, 3.1, 2.9, 3.0, 3.0])["beats"]
    assert not beats([1.0, 1.2, 0.8, 1.1, 0.9], [1.05, 1.25, 0.85, 1.15, 0.95])["beats"]


def test_mean_std_uses_the_sample_standard_deviation() -> None:
    result = mean_std([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["std"] == pytest.approx(1.0)


def test_score_reports_each_technique_and_the_overall_group() -> None:
    technique = np.repeat(np.arange(3), 5)
    truth = np.linspace(1.0, 10.0, 15)
    scores = score(truth, truth * 1.1, technique)
    assert set(scores) == set(GROUPS)
    assert scores["overall"]["mdape"] == pytest.approx(10.0)
    assert scores["passive"]["mae"] == pytest.approx(np.mean(truth[:5] * 0.1))


def test_linear_regression_recovers_an_exact_log_linear_law(data) -> None:
    rows = np.arange(data.n_rows)
    train, test = rows[::2], rows[1::2]
    exact = data.physics[:, 0] ** 1.1
    shifted = replace(data, targets=np.column_stack((exact, exact)), physics=data.physics.copy())
    prediction = linear_predict(shifted, train, test, 0)
    np.testing.assert_allclose(prediction, exact[test], rtol=1e-9)


def test_physics_baselines_are_positive_for_every_row(data) -> None:
    assert np.all(data.physics > 0.0)
    assert data.physics.shape == (data.n_rows, 2)


def test_cross_validation_structure_and_folds_are_stratified(data) -> None:
    columns = column_indices(data.feature_names, drop_soc=False)
    raw = cross_validate(data, SMALL, SMALL.seed, columns)
    summary = summarise_cv(raw)
    assert summary["n_folds"] == 3
    for name in ("balancing_time_s", "energy_dissipated_j"):
        for model in MODELS:
            entry = summary["scores"][name][model]["overall"]["mae"]
            assert entry["mean"] > 0.0
        verdict = summary["verdicts"][name]["passive"]["hybrid_vs_hgb"]
        assert set(verdict) == {"lower_in_all_folds", "mean_improvement", "fold_std", "beats"}


def test_conformal_intervals_cover_near_the_nominal_level(data) -> None:
    columns = column_indices(data.feature_names, drop_soc=False)
    best = {"balancing_time_s": "physics", "energy_dissipated_j": "linear"}
    block, fitted = conformal(data, SMALL, SMALL.seed, columns, best)
    assert block["balancing_time_s"]["model"] == "physics"
    for name in best:
        for technique in Technique:
            entry = block[name]["techniques"][technique.key]
            assert 0.5 <= entry["coverage"] <= 1.0
            assert entry["half_width_factor"] >= 1.0
            assert entry["upper_over_lower"] == pytest.approx(entry["half_width_factor"] ** 2)
    assert set(fitted) == {("hgb", 0), ("hgb", 1), ("hybrid", 0), ("hybrid", 1)}
