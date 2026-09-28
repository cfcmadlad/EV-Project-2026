from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import FloatArray, IntArray, ModelConfig, Technique
from src.dataset import TARGET_NAMES, Dataset

Scores = dict[str, float]


@dataclass(frozen=True)
class PredictorReport:
    metrics: dict[str, dict[str, Scores]]
    y_test: FloatArray
    y_pred: FloatArray
    technique_test: IntArray
    importance_mean: FloatArray
    importance_std: FloatArray
    prediction_time_per_pack_s: float


def train_and_evaluate(
    dataset: Dataset, config: ModelConfig, seed: int
) -> PredictorReport:
    if dataset.targets.shape[1] != len(TARGET_NAMES):
        raise ValueError(f"dataset must provide targets {TARGET_NAMES}")
    if np.any(dataset.targets <= 0.0):
        raise ValueError("targets must be strictly positive for the log transform")
    train, test = train_test_split(
        np.arange(dataset.features.shape[0]),
        test_size=config.test_fraction,
        random_state=seed,
        stratify=dataset.technique,
    )
    x_train, x_test = dataset.features[train], dataset.features[test]
    y_test = dataset.targets[test]
    technique_test = dataset.technique[test]
    models = [
        _build_model(config, seed).fit(x_train, dataset.targets[train, j])
        for j in range(len(TARGET_NAMES))
    ]

    y_pred = np.column_stack([model.predict(x_test) for model in models])
    prediction_time = (
        min(_prediction_time(models, x_test) for _ in range(config.timing_repeats))
        / test.size
    )

    importances = [
        permutation_importance(
            model,
            x_test,
            y_test[:, j],
            n_repeats=config.permutation_repeats,
            random_state=seed,
        )
        for j, model in enumerate(models)
    ]
    return PredictorReport(
        metrics={
            name: _target_metrics(y_test[:, j], y_pred[:, j], technique_test)
            for j, name in enumerate(TARGET_NAMES)
        },
        y_test=y_test,
        y_pred=y_pred,
        technique_test=technique_test,
        importance_mean=np.array([imp.importances_mean for imp in importances]),
        importance_std=np.array([imp.importances_std for imp in importances]),
        prediction_time_per_pack_s=prediction_time,
    )


def _prediction_time(
    models: list[TransformedTargetRegressor], features: FloatArray
) -> float:
    start = perf_counter()
    for model in models:
        model.predict(features)
    return perf_counter() - start


def _build_model(config: ModelConfig, seed: int) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes,
            early_stopping=False,
            random_state=seed,
        ),
        func=np.log,
        inverse_func=np.exp,
    )


def _target_metrics(
    y_true: FloatArray, y_pred: FloatArray, technique: IntArray
) -> dict[str, Scores]:
    groups = {"overall": np.ones_like(technique, dtype=bool)} | {
        t.key: technique == t for t in Technique
    }
    return {
        name: {
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        }
        for name, mask in groups.items()
    }
