import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from src.balancing import simulate
from src.baselines import physics_prediction
from src.config import FloatArray, IntArray, ProjectConfig, Technique
from src.dataset import TARGET_NAMES, Dataset
from src.experiments import baseline_pack
from src.pack import PackBatch
from src.params import MLSpec
from src.predictor import build_hgb

MODELS = ("physics", "linear", "hgb", "hybrid")
LEARNED = ("hgb", "hybrid")
GROUPS = ("overall", *(t.key for t in Technique))
METRICS = ("r2", "mae", "mdape")
FIT_THREADS = 2
N_JOBS = 8
FLOOR = 1e-9
PERCENT = 100.0
COMPARISONS = (
    ("hgb", "physics"),
    ("hybrid", "hgb"),
    ("hybrid", "physics"),
    ("hybrid", "linear"),
    ("linear", "physics"),
    ("hgb", "linear"),
)
SOC_FEATURE_PREFIX = "soc0_"
CELL_SUFFIX = {"soc0_": "", "capacity_": "_ah", "r0_": "_ohm"}
SCENARIOS = ("component", "spread")


@dataclass(frozen=True, slots=True)
class EvalData:
    features: FloatArray
    feature_names: tuple[str, ...]
    targets: FloatArray
    technique: IntArray
    physics: FloatArray
    spread: FloatArray
    component: FloatArray
    frequency: FloatArray
    soc0: FloatArray
    capacity_ah: FloatArray
    r0_ohm: FloatArray

    @property
    def n_rows(self) -> int:
        return self.features.shape[0]


@dataclass(frozen=True, slots=True)
class FittedModel:
    kind: str
    estimator: TransformedTargetRegressor | HistGradientBoostingRegressor
    columns: tuple[int, ...]

    def predict(self, data: EvalData, rows: IntArray, target: int) -> FloatArray:
        x = data.features[rows][:, list(self.columns)]
        if self.kind == "hgb":
            return self.estimator.predict(x)
        phys = data.physics[rows, target]
        stacked = np.column_stack((x, np.log(phys)))
        return phys * np.exp(self.estimator.predict(stacked))


def column_indices(names: tuple[str, ...], drop_soc: bool) -> tuple[int, ...]:
    return tuple(
        i for i, name in enumerate(names) if not (drop_soc and name.startswith(SOC_FEATURE_PREFIX))
    )


def indices_of(names: tuple[str, ...], prefix: str, n_cells: int) -> list[int]:
    return [names.index(f"{prefix}{k}{CELL_SUFFIX[prefix]}") for k in range(1, n_cells + 1)]


def build_eval_data(dataset: Dataset, config: ProjectConfig) -> EvalData:
    names, f = dataset.feature_names, dataset.features
    n_cells = config.pack.n_cells
    soc = f[:, indices_of(names, "soc0_", n_cells)]
    cap = f[:, indices_of(names, "capacity_", n_cells)]
    r0 = f[:, indices_of(names, "r0_", n_cells)]
    component = f[:, names.index("component_value")]
    frequency = f[:, names.index("switching_frequency_hz")]
    physics = np.zeros_like(dataset.targets)
    for technique in Technique:
        mask = dataset.technique == technique
        pred = physics_prediction(
            technique, soc[mask], cap[mask], r0[mask], component[mask], frequency[mask], config
        )
        physics[mask, 0] = np.maximum(pred.balancing_time_s, FLOOR)
        physics[mask, 1] = np.maximum(pred.energy_dissipated_j, FLOOR)
    return EvalData(
        features=f,
        feature_names=names,
        targets=dataset.targets,
        technique=dataset.technique,
        physics=physics,
        spread=f[:, names.index("soc_spread")],
        component=component,
        frequency=frequency,
        soc0=soc,
        capacity_ah=cap,
        r0_ohm=r0,
    )


def fit_task(
    kind: str,
    target: int,
    train: IntArray,
    data: EvalData,
    ml: MLSpec,
    seed: int,
    columns: tuple[int, ...],
) -> FittedModel:
    x = data.features[train][:, list(columns)]
    y = data.targets[train, target]
    with threadpool_limits(limits=FIT_THREADS):
        if kind == "hgb":
            estimator = TransformedTargetRegressor(
                regressor=build_hgb(ml, seed), func=np.log, inverse_func=np.exp
            ).fit(x, y)
        else:
            phys = data.physics[train, target]
            stacked = np.column_stack((x, np.log(phys)))
            estimator = build_hgb(ml, seed).fit(stacked, np.log(y / phys))
    return FittedModel(kind, estimator, columns)


def fit_many(
    jobs: list[tuple[str, int, IntArray]],
    data: EvalData,
    ml: MLSpec,
    seed: int,
    columns: tuple[int, ...],
) -> list[FittedModel]:
    return Parallel(n_jobs=N_JOBS)(
        delayed(fit_task)(kind, target, train, data, ml, seed, columns)
        for kind, target, train in jobs
    )


def linear_design(data: EvalData, rows: IntArray, technique: Technique, target: int) -> FloatArray:
    columns = [np.ones(rows.size), np.log(data.spread[rows]), np.log(data.component[rows])]
    if technique != Technique.PASSIVE:
        columns.append(np.log(data.frequency[rows]))
    columns.append(np.log(data.physics[rows, target]))
    return np.column_stack(columns)


def linear_predict(data: EvalData, train: IntArray, test: IntArray, target: int) -> FloatArray:
    out = np.zeros(test.size)
    for technique in Technique:
        tr = train[data.technique[train] == technique]
        te_mask = data.technique[test] == technique
        coefficients, *_ = np.linalg.lstsq(
            linear_design(data, tr, technique, target), np.log(data.targets[tr, target]), rcond=None
        )
        out[te_mask] = np.exp(linear_design(data, test[te_mask], technique, target) @ coefficients)
    return out


def group_masks(technique: IntArray) -> dict[str, np.ndarray]:
    return {"overall": np.ones(technique.size, dtype=bool)} | {
        t.key: technique == t for t in Technique
    }


def score(
    y_true: FloatArray, y_pred: FloatArray, technique: IntArray
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, mask in group_masks(technique).items():
        if mask.sum() < 2:
            out[name] = {metric: float("nan") for metric in METRICS}
            continue
        out[name] = {
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "mdape": float(np.median(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * PERCENT),
        }
    return out


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.array(values)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def beats(better: list[float], worse: list[float]) -> dict[str, object]:
    gaps = np.array(worse) - np.array(better)
    spread = max(float(np.std(better, ddof=1)), float(np.std(worse, ddof=1)))
    return {
        "lower_in_all_folds": bool(np.all(gaps > 0.0)),
        "mean_improvement": float(gaps.mean()),
        "fold_std": spread,
        "beats": bool(np.all(gaps > 0.0) and gaps.mean() > spread),
    }


def predictions_for(
    model: str,
    fitted: dict[tuple[str, int], FittedModel],
    data: EvalData,
    train: IntArray,
    test: IntArray,
    target: int,
) -> FloatArray:
    if model == "physics":
        return data.physics[test, target]
    if model == "linear":
        return linear_predict(data, train, test, target)
    return fitted[(model, target)].predict(data, test, target)


def fit_for_split(
    data: EvalData, train: IntArray, ml: MLSpec, seed: int, columns: tuple[int, ...]
) -> dict[tuple[str, int], FittedModel]:
    keys = [(kind, target) for kind in LEARNED for target in range(len(TARGET_NAMES))]
    models = fit_many([(kind, target, train) for kind, target in keys], data, ml, seed, columns)
    return dict(zip(keys, models, strict=True))


def cross_validate(
    data: EvalData,
    config: ProjectConfig,
    seed: int,
    columns: tuple[int, ...],
    models: tuple[str, ...] = MODELS,
) -> dict[str, object]:
    folds = list(
        StratifiedKFold(config.ml_eval.n_folds, shuffle=True, random_state=seed).split(
            np.zeros(data.n_rows), data.technique
        )
    )
    learned = [m for m in models if m in LEARNED]
    keys = [
        (kind, target, fold)
        for fold in range(len(folds))
        for kind in learned
        for target in range(len(TARGET_NAMES))
    ]
    fits = fit_many([(k, t, folds[f][0]) for k, t, f in keys], data, config.ml, seed, columns)
    fitted = dict(zip(keys, fits, strict=True))
    per_fold: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        model: {
            name: {group: {metric: [] for metric in METRICS} for group in GROUPS}
            for name in TARGET_NAMES
        }
        for model in models
    }
    for fold, (train, test) in enumerate(folds):
        for model in models:
            for target, name in enumerate(TARGET_NAMES):
                lookup = (
                    {(model, target): fitted[(model, target, fold)]} if model in LEARNED else {}
                )
                pred = predictions_for(model, lookup, data, train, test, target)
                scores = score(data.targets[test, target], pred, data.technique[test])
                for group in GROUPS:
                    for metric in METRICS:
                        per_fold[model][name][group][metric].append(scores[group][metric])
    return {"n_folds": len(folds), "per_fold": per_fold}


def summarise_cv(raw: dict[str, object]) -> dict[str, object]:
    per_fold = raw["per_fold"]
    summary: dict[str, object] = {}
    verdicts: dict[str, object] = {}
    for name in TARGET_NAMES:
        summary[name] = {
            model: {
                group: {
                    metric: mean_std(per_fold[model][name][group][metric]) for metric in METRICS
                }
                for group in GROUPS
            }
            for model in MODELS
        }
        verdicts[name] = {
            group: {
                f"{a}_vs_{b}": beats(
                    per_fold[a][name][group]["mae"], per_fold[b][name][group]["mae"]
                )
                for a, b in COMPARISONS
            }
            for group in GROUPS
        }
    return {"n_folds": raw["n_folds"], "scores": summary, "verdicts": verdicts}


def best_models(summary: dict[str, object]) -> dict[str, str]:
    return {
        name: min(MODELS, key=lambda m: summary["scores"][name][m]["overall"]["mae"]["mean"])
        for name in TARGET_NAMES
    }


def out_of_range(
    data: EvalData, config: ProjectConfig, seed: int, columns: tuple[int, ...]
) -> dict[str, object]:
    spec = config.ml_eval
    thresholds = {
        t: float(np.quantile(data.component[data.technique == t], spec.oor_component_quantile))
        for t in Technique
    }
    limit = np.array([thresholds[Technique(t)] for t in data.technique])
    splits = {
        "component": data.component <= limit,
        "spread": data.spread <= spec.oor_spread_limit,
    }
    result: dict[str, object] = {
        "component_quantile": spec.oor_component_quantile,
        "spread_limit": spec.oor_spread_limit,
        "component_thresholds": {t.key: thresholds[t] for t in Technique},
    }
    for scenario in SCENARIOS:
        train = np.flatnonzero(splits[scenario])
        test = np.flatnonzero(~splits[scenario])
        fitted = fit_for_split(data, train, config.ml, seed, columns)
        block: dict[str, object] = {"n_train": int(train.size), "n_test": int(test.size)}
        for target, name in enumerate(TARGET_NAMES):
            predictions = {
                model: predictions_for(model, fitted, data, train, test, target) for model in MODELS
            }
            block[name] = {
                model: {
                    **score(data.targets[test, target], predictions[model], data.technique[test]),
                    "plateau": plateau_statistics(
                        data, train, test, target, predictions[model], scenario
                    ),
                }
                for model in MODELS
            }
            block[f"{name}_arrays"] = {
                "test_index": test,
                "true": data.targets[test, target],
                **predictions,
            }
        result[scenario] = block
    return result


def plateau_statistics(
    data: EvalData,
    train: IntArray,
    test: IntArray,
    target: int,
    prediction: FloatArray,
    scenario: str,
) -> dict[str, dict[str, float]]:
    driver = data.component if scenario == "component" else data.spread
    out: dict[str, dict[str, float]] = {}
    for technique in Technique:
        tr = train[data.technique[train] == technique]
        te = data.technique[test] == technique
        true = data.targets[test[te], target]
        pred = prediction[te]
        order = np.argsort(driver[test[te]])
        third = max(1, order.size // 3)
        ratio = pred / true
        train_max = float(data.targets[tr, target].max())
        out[technique.key] = {
            "fraction_true_above_train_max": float(np.mean(true > train_max)),
            "fraction_pred_above_train_max": float(np.mean(pred > train_max)),
            "median_ratio_near_edge": float(np.median(ratio[order[:third]])),
            "median_ratio_far_from_edge": float(np.median(ratio[order[-third:]])),
        }
    return out


def conformal(
    data: EvalData,
    config: ProjectConfig,
    seed: int,
    columns: tuple[int, ...],
    best: dict[str, str],
) -> tuple[dict[str, object], dict[tuple[str, int], FittedModel]]:
    spec = config.ml_eval
    everything = np.arange(data.n_rows)
    train, test = train_test_split(
        everything, test_size=config.ml.test_fraction, random_state=seed, stratify=data.technique
    )
    fit_rows, calibration = train_test_split(
        train,
        train_size=spec.conformal_fit_fraction,
        random_state=seed,
        stratify=data.technique[train],
    )
    fitted = fit_for_split(data, fit_rows, config.ml, seed, columns)
    result: dict[str, object] = {
        "nominal": spec.conformal_level,
        "n_fit": int(fit_rows.size),
        "n_calibration": int(calibration.size),
        "n_test": int(test.size),
    }
    for target, name in enumerate(TARGET_NAMES):
        model = best[name]
        pred_cal = predictions_for(model, fitted, data, fit_rows, calibration, target)
        pred_test = predictions_for(model, fitted, data, fit_rows, test, target)
        residual = np.abs(np.log(pred_cal) - np.log(data.targets[calibration, target]))
        block: dict[str, object] = {"model": model, "techniques": {}}
        for technique in Technique:
            in_cal = data.technique[calibration] == technique
            in_test = data.technique[test] == technique
            n_cal = int(in_cal.sum())
            rank = min(math.ceil((n_cal + 1) * spec.conformal_level), n_cal)
            quantile = float(np.sort(residual[in_cal])[rank - 1])
            true = data.targets[test[in_test], target]
            lower, upper = (
                pred_test[in_test] * math.exp(-quantile),
                pred_test[in_test] * math.exp(quantile),
            )
            coverage = float(np.mean((true >= lower) & (true <= upper)))
            block["techniques"][technique.key] = {
                "n_calibration": n_cal,
                "n_test": int(in_test.sum()),
                "log_quantile": quantile,
                "half_width_factor": math.exp(quantile),
                "upper_over_lower": math.exp(2.0 * quantile),
                "coverage": coverage,
                "pass": bool(spec.coverage_low <= coverage <= spec.coverage_high),
            }
        block["pass"] = all(item["pass"] for item in block["techniques"].values())
        result[name] = block
    return result, fitted


def conformal_repeats(
    data: EvalData, config: ProjectConfig, seed: int, columns: tuple[int, ...], best: dict[str, str]
) -> dict[str, object]:
    spec = config.ml_eval
    plans = []
    for repeat in range(spec.conformal_repeats):
        state = seed + 1 + repeat
        train, test = train_test_split(
            np.arange(data.n_rows),
            test_size=config.ml.test_fraction,
            random_state=state,
            stratify=data.technique,
        )
        fit_rows, calibration = train_test_split(
            train,
            train_size=spec.conformal_fit_fraction,
            random_state=state,
            stratify=data.technique[train],
        )
        plans.append((fit_rows, calibration, test))
    jobs = [
        (best[name], target, fit_rows)
        for fit_rows, _, _ in plans
        for target, name in enumerate(TARGET_NAMES)
        if best[name] in LEARNED
    ]
    models = iter(fit_many(jobs, data, config.ml, seed, columns))
    coverage = {name: {t.key: [] for t in Technique} for name in TARGET_NAMES}
    for fit_rows, calibration, test in plans:
        for target, name in enumerate(TARGET_NAMES):
            kind = best[name]
            lookup = {(kind, target): next(models)} if kind in LEARNED else {}
            pred_cal = predictions_for(kind, lookup, data, fit_rows, calibration, target)
            pred_test = predictions_for(kind, lookup, data, fit_rows, test, target)
            residual = np.abs(np.log(pred_cal) - np.log(data.targets[calibration, target]))
            for technique in Technique:
                in_cal = data.technique[calibration] == technique
                in_test = data.technique[test] == technique
                n_cal = int(in_cal.sum())
                rank = min(math.ceil((n_cal + 1) * spec.conformal_level), n_cal)
                quantile = float(np.sort(residual[in_cal])[rank - 1])
                log_error = np.abs(
                    np.log(pred_test[in_test]) - np.log(data.targets[test[in_test], target])
                )
                coverage[name][technique.key].append(float(np.mean(log_error <= quantile)))
    result: dict[str, object] = {"n_repeats": spec.conformal_repeats}
    for name in TARGET_NAMES:
        result[name] = {
            key: {
                "mean_coverage": float(np.mean(values)),
                "std_coverage": float(np.std(values, ddof=1)),
                "fraction_in_band": float(
                    np.mean([spec.coverage_low <= v <= spec.coverage_high for v in values])
                ),
            }
            for key, values in coverage[name].items()
        }
    return result


def feature_ablation(
    data: EvalData,
    config: ProjectConfig,
    seed: int,
    cv_summary: dict[str, object],
    full_columns: tuple[int, ...],
) -> dict[str, object]:
    reduced = column_indices(data.feature_names, drop_soc=True)
    raw = cross_validate(data, config, seed, reduced, models=("hgb",))["per_fold"]["hgb"]
    train, test = train_test_split(
        np.arange(data.n_rows),
        test_size=config.ml.test_fraction,
        random_state=seed,
        stratify=data.technique,
    )
    fitted_full = fit_many(
        [("hgb", t, train) for t in range(len(TARGET_NAMES))], data, config.ml, seed, full_columns
    )
    fitted_reduced = fit_many(
        [("hgb", t, train) for t in range(len(TARGET_NAMES))], data, config.ml, seed, reduced
    )
    result: dict[str, object] = {
        "dropped_features": [n for n in data.feature_names if n.startswith(SOC_FEATURE_PREFIX)],
        "n_features_full": len(full_columns),
        "n_features_reduced": len(reduced),
    }
    for target, name in enumerate(TARGET_NAMES):
        cv_full = cv_summary["scores"][name]["hgb"]
        cv_reduced = {
            group: {metric: mean_std(raw[name][group][metric]) for metric in METRICS}
            for group in GROUPS
        }
        hold = {
            label: score(
                data.targets[test, target],
                model[target].predict(data, test, target),
                data.technique[test],
            )
            for label, model in (("full", fitted_full), ("reduced", fitted_reduced))
        }
        result[name] = {
            "cross_validation": {
                "full": cv_full,
                "reduced": cv_reduced,
                "mae_change_percent": {
                    group: (cv_reduced[group]["mae"]["mean"] / cv_full[group]["mae"]["mean"] - 1.0)
                    * PERCENT
                    for group in GROUPS
                },
                "r2_change": {
                    group: cv_reduced[group]["r2"]["mean"] - cv_full[group]["r2"]["mean"]
                    for group in GROUPS
                },
            },
            "holdout": {
                **hold,
                "mae_change_percent": {
                    group: (hold["reduced"][group]["mae"] / hold["full"][group]["mae"] - 1.0)
                    * PERCENT
                    for group in GROUPS
                },
            },
        }
    return result


def physics_baseline_check(config: ProjectConfig) -> dict[str, object]:
    pack = baseline_pack(config)
    result: dict[str, object] = {}
    for technique in Technique:
        hardware = config.technique(technique)
        component = np.array([hardware.component.mid])
        frequency = np.array([hardware.frequency_hz.mid])
        sim = simulate(technique, pack, component, frequency, config)
        pred = physics_prediction(
            technique, pack.soc0, pack.capacity_ah, pack.r0_ohm, component, frequency, config
        )
        time_error = float(pred.balancing_time_s[0] / sim.balancing_time_s[0] - 1.0)
        energy_error = float(pred.energy_dissipated_j[0] / sim.energy_dissipated_j[0] - 1.0)
        result[technique.key] = {
            "simulated_time_s": float(sim.balancing_time_s[0]),
            "predicted_time_s": float(pred.balancing_time_s[0]),
            "time_error": time_error,
            "simulated_energy_j": float(sim.energy_dissipated_j[0]),
            "predicted_energy_j": float(pred.energy_dissipated_j[0]),
            "energy_error": energy_error,
        }
    tolerance = config.ml_eval.passive_tolerance
    passive = result[Technique.PASSIVE.key]
    result["passive_tolerance"] = tolerance
    result["passive_time_pass"] = bool(abs(passive["time_error"]) <= tolerance)
    result["passive_energy_pass"] = bool(abs(passive["energy_error"]) <= tolerance)
    return result


def dataset_baseline_errors(data: EvalData) -> dict[str, object]:
    out: dict[str, object] = {}
    for technique in Technique:
        mask = data.technique == technique
        entry = {}
        for target, name in enumerate(TARGET_NAMES):
            ratio = data.physics[mask, target] / data.targets[mask, target]
            error = np.abs(ratio - 1.0) * PERCENT
            entry[name] = {
                "median_abs_percent_error": float(np.median(error)),
                "p95_abs_percent_error": float(np.quantile(error, 0.95)),
                "median_ratio": float(np.median(ratio)),
            }
        out[technique.key] = entry
    return out


def _median_seconds(function: Callable[[], None], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return float(np.median(samples))


def _rows_for_batch(data: EvalData, size: int) -> dict[Technique, IntArray]:
    per_technique = max(1, size // len(Technique))
    return {t: np.flatnonzero(data.technique == t)[:per_technique] for t in Technique}


def _simulate_rows(data: EvalData, rows: dict[Technique, IntArray], config: ProjectConfig) -> None:
    for technique, index in rows.items():
        packs = PackBatch(
            capacity_ah=data.capacity_ah[index],
            r0_ohm=data.r0_ohm[index],
            soc0=data.soc0[index],
        )
        simulate(technique, packs, data.component[index], data.frequency[index], config)


def _physics_rows(data: EvalData, rows: dict[Technique, IntArray], config: ProjectConfig) -> None:
    for technique, index in rows.items():
        physics_prediction(
            technique,
            data.soc0[index],
            data.capacity_ah[index],
            data.r0_ohm[index],
            data.component[index],
            data.frequency[index],
            config,
        )


def timing_report(
    data: EvalData,
    fitted: dict[tuple[str, int], FittedModel],
    config: ProjectConfig,
) -> dict[str, object]:
    spec = config.ml_eval
    result: dict[str, object] = {"label": "host CPU, not target hardware"}
    for size in spec.timing_batch_sizes:
        rows = _rows_for_batch(data, size)
        count = sum(index.size for index in rows.values())
        flat = np.concatenate(list(rows.values()))

        def learned(kind: str, flat=flat) -> None:
            for target in range(len(TARGET_NAMES)):
                fitted[(kind, target)].predict(data, flat, target)

        def hybrid(flat=flat, rows=rows) -> None:
            _physics_rows(data, rows, config)
            learned("hybrid")

        methods = {
            "simulator": (
                lambda rows=rows: _simulate_rows(data, rows, config),
                spec.simulator_timing_repeats,
            ),
            "physics": (lambda rows=rows: _physics_rows(data, rows, config), spec.timing_repeats),
            "hgb": (lambda: learned("hgb"), spec.timing_repeats),
            "hybrid": (hybrid, spec.timing_repeats),
        }
        entry = {
            name: _median_seconds(function, repeats) / count
            for name, (function, repeats) in methods.items()
        }
        result[f"batch_{size}"] = {
            "per_pack_s": entry,
            "speedup_over_simulator": {
                name: entry["simulator"] / seconds for name, seconds in entry.items()
            },
            "n_packs": count,
        }
    return result


def evaluate(
    dataset: Dataset, config: ProjectConfig, seed: int
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    data = build_eval_data(dataset, config)
    columns = column_indices(data.feature_names, drop_soc=False)
    cv = summarise_cv(cross_validate(data, config, seed, columns))
    best = best_models(cv)
    intervals, fitted = conformal(data, config, seed, columns, best)
    intervals["repeated_splits"] = conformal_repeats(data, config, seed, columns, best)
    ranged = out_of_range(data, config, seed, columns)
    ablation = feature_ablation(data, config, seed, cv, columns)
    arrays = {
        scenario: {name: ranged[scenario].pop(f"{name}_arrays") for name in TARGET_NAMES}
        for scenario in SCENARIOS
    }
    metrics: dict[str, object] = {
        "status": "ok",
        "models": list(MODELS),
        "n_rows": data.n_rows,
        "physics_baseline_check": physics_baseline_check(config),
        "physics_baseline_dataset_errors": dataset_baseline_errors(data),
        "cross_validation": cv,
        "best_model": best,
        "out_of_range": ranged,
        "conformal": intervals,
        "feature_ablation": ablation,
    }
    timing = timing_report(data, fitted, config)
    return metrics, {"data": data, "out_of_range_arrays": arrays}, timing
