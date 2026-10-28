import json
import os
import sys
from time import perf_counter

import numpy as np
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from src.baselines import physics_prediction
from src.config import CONFIG, PROJECT_ROOT, RESULTS_DIR, Technique
from src.dataset import TARGET_NAMES
from src.deep_surrogate import (
    CAPACITY_COLUMNS,
    CONFIGS,
    MODELS,
    R0_COLUMNS,
    SEEDS,
    SOC_COLUMNS,
    build,
    fit,
    fit_scaler,
    make_split,
    median_time_per_pack,
    parameter_count,
    predict,
    scores,
    summarise,
)
from src.fingerprint import text_sha256

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
OUT = PROJECT_ROOT / "data" / "processed" / "deep_surrogates.json"
HGB_CONFIGS = ((0.05, 31), (0.1, 31), (0.05, 15), (0.1, 63))
TIMING_REPEATS = 20
PHYSICS_REPEATS = 5


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.genfromtxt(RESULTS_DIR / "dataset.csv", delimiter=",", names=True)
    names = table.dtype.names
    n_features = names.index(TARGET_NAMES[0])
    features = np.column_stack([table[n] for n in names[:n_features]])
    targets = np.column_stack([table[n] for n in TARGET_NAMES])
    return features, targets, table["technique"].astype(int)


def hgb(learning_rate: float, leaves: int, seed: int) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=CONFIG.ml.max_iter,
            max_leaf_nodes=leaves,
            early_stopping=False,
            random_state=seed,
        ),
        func=np.log,
        inverse_func=np.exp,
    )


def run_hgb(features, targets, technique, split, physics=False) -> dict:
    scale = np.ones_like(targets)
    extra_s = physics_seconds_per_pack(features, technique, split.test) if physics else 0.0
    if physics:
        scale = np.exp(physics_logs(features, technique))
        features = np.column_stack((features, np.log(scale)))
    ratio = targets / scale

    def fit_models(config, seed):
        return [
            hgb(*config, seed).fit(features[split.train], ratio[split.train, j]) for j in range(2)
        ]

    def predict_rows(models, rows):
        return np.column_stack([m.predict(features[rows]) for m in models]) * scale[rows]

    best, best_config = float("inf"), HGB_CONFIGS[0]
    for config in HGB_CONFIGS:
        pred = predict_rows(fit_models(config, SEEDS[0]), split.validation)
        score = float(np.mean((np.log(pred) - np.log(targets[split.validation])) ** 2))
        if score < best:
            best, best_config = score, config
    per_seed, times, start = [], [], perf_counter()
    for seed in SEEDS:
        models = fit_models(best_config, seed)
        pred = predict_rows(models, split.test)
        per_seed.append(
            {
                name: scores(targets[split.test, j], pred[:, j], technique[split.test])
                for j, name in enumerate(TARGET_NAMES)
            }
        )
    train_s = (perf_counter() - start) / len(SEEDS)
    for _ in range(TIMING_REPEATS):
        t0 = perf_counter()
        predict_rows(models, split.test)
        times.append(perf_counter() - t0)
    return {
        "metrics": summarise(per_seed),
        "chosen_config": {"learning_rate": best_config[0], "max_leaf_nodes": best_config[1]},
        "tuning_trials": len(HGB_CONFIGS),
        "parameters": int(
            sum(m.regressor_.n_iter_ * m.regressor_.max_leaf_nodes * 2 for m in models)
        ),
        "seeds": list(SEEDS),
        "timing": {
            "inference_s_per_pack": float(np.median(times)) / split.test.size + extra_s,
            "train_s_per_seed": train_s,
        },
    }


def physics_seconds_per_pack(features, technique, rows) -> float:
    times = []
    for _ in range(PHYSICS_REPEATS):
        start = perf_counter()
        physics_logs(features[rows], technique[rows])
        times.append(perf_counter() - start)
    return float(np.median(times)) / rows.size


def physics_logs(features: np.ndarray, technique: np.ndarray) -> np.ndarray:
    out = np.zeros((features.shape[0], 2))
    for tech in Technique:
        rows = technique == tech
        x = features[rows]
        pred = physics_prediction(
            tech,
            x[:, SOC_COLUMNS],
            x[:, CAPACITY_COLUMNS],
            x[:, R0_COLUMNS],
            x[:, -2],
            x[:, -1],
            CONFIG,
        )
        out[rows] = np.log(np.column_stack((pred.balancing_time_s, pred.energy_dissipated_j)))
    return out


def run_deep(kind, features, targets, technique, split, device, physics=False) -> dict:
    base_kind = kind.removesuffix("_physics")
    extra_s = physics_seconds_per_pack(features, technique, split.test) if physics else 0.0
    reference = physics_logs(features, technique) if physics else np.zeros((len(features), 2))
    if physics:
        features = np.column_stack((features, reference))
    scaler = fit_scaler(features[split.train])
    y_log = np.log(targets) - reference
    y_scaler = fit_scaler(y_log[split.train])

    def tensors(rows):
        return (torch.tensor(scaler(features[rows])), torch.tensor(y_scaler(y_log[rows])))

    x_train, y_train = tensors(split.train)
    x_val, y_val = tensors(split.validation)
    x_test, _ = tensors(split.test)
    best, best_config = float("inf"), CONFIGS[0]
    for width, lr in CONFIGS:
        trial = fit(base_kind, width, lr, SEEDS[0], x_train, y_train, x_val, y_val, device)
        if trial.validation_loss < best:
            best, best_config = trial.validation_loss, (width, lr)
    width, lr = best_config
    per_seed, train_times, epochs = [], [], []
    for seed in SEEDS:
        result = fit(base_kind, width, lr, seed, x_train, y_train, x_val, y_val, device)
        model = build(base_kind, x_train.shape[1], 2, width, result.state)
        pred = np.exp(predict(model, x_test) * y_scaler.std + y_scaler.mean + reference[split.test])
        per_seed.append(
            {
                name: scores(targets[split.test, j], pred[:, j], technique[split.test])
                for j, name in enumerate(TARGET_NAMES)
            }
        )
        train_times.append(result.seconds)
        epochs.append(result.epochs)
    return {
        "metrics": summarise(per_seed),
        "chosen_config": {"width": width, "learning_rate": lr},
        "tuning_trials": len(CONFIGS),
        "parameters": parameter_count(model),
        "epochs": epochs,
        "seeds": list(SEEDS),
        "timing": {
            "inference_s_per_pack": median_time_per_pack(model, x_test, TIMING_REPEATS) + extra_s,
            "train_s_per_seed": float(np.mean(train_times)),
        },
    }


def main() -> int:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, targets, technique = load()
    split = make_split(technique, CONFIG.seed, CONFIG.ml.test_fraction)
    results = {
        "hgb": run_hgb(features, targets, technique, split),
        "hgb_physics": run_hgb(features, targets, technique, split, physics=True),
    }
    for kind in (*MODELS, "mlp_physics", "cell_gru_physics"):
        results[kind] = run_deep(
            kind, features, targets, technique, split, device, kind.endswith("_physics")
        )
        print(kind, results[kind]["parameters"], results[kind]["chosen_config"], flush=True)
    payload = {
        "dataset_sha256": text_sha256(RESULTS_DIR / "dataset.csv"),
        "split": {
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "test": int(split.test.size),
            "seed": CONFIG.seed,
        },
        "models": results,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
