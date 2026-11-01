from dataclasses import dataclass

import numpy as np

from src.config import FloatArray, IntArray, ProjectConfig, Technique
from src.dataset import TARGET_NAMES, pack_features
from src.ml_eval import (
    FLOOR,
    LEARNED,
    ConformalSplit,
    EvalData,
    FittedModel,
    calibration_quantile,
    predictions_for,
)
from src.pack import PackBatch
from src.params import SECONDS_PER_HOUR
from src.selector import SelectorInputs, budget_key, choose, mid_values, policy_row

TIME, ENERGY = 0, 1


@dataclass(frozen=True, slots=True)
class Normaliser:
    quantile: float
    offset: float
    abstain_width: float


def difficulty(pred: FloatArray, physics: FloatArray, offset: float) -> FloatArray:
    return np.abs(np.log(pred) - np.log(physics)) + offset


def fit_normalisers(
    data: EvalData,
    config: ProjectConfig,
    split: ConformalSplit,
    fitted: dict[tuple[str, int], FittedModel],
    best: dict[str, str],
) -> dict[tuple[int, int], Normaliser]:
    spec = config.abstain
    level = config.ml_eval.conformal_level
    out: dict[tuple[int, int], Normaliser] = {}
    for target, name in enumerate(TARGET_NAMES):
        pred = predictions_for(best[name], fitted, data, split.fit_rows, split.calibration, target)
        physics = data.physics[split.calibration, target]
        error = np.abs(np.log(pred) - np.log(data.targets[split.calibration, target]))
        for technique in Technique:
            mask = data.technique[split.calibration] == technique
            gap = np.abs(np.log(pred[mask]) - np.log(physics[mask]))
            offset = max(float(np.quantile(gap, spec.scale_quantile)), FLOOR)
            scale = gap + offset
            quantile = calibration_quantile(error[mask] / scale, level)
            out[(target, int(technique))] = Normaliser(
                quantile, offset, float(np.quantile(quantile * scale, spec.abstain_quantile))
            )
    return out


def calibration_block(
    data: EvalData,
    config: ProjectConfig,
    split: ConformalSplit,
    fitted: dict[tuple[str, int], FittedModel],
    best: dict[str, str],
    normalisers: dict[tuple[int, int], Normaliser],
    fixed: dict[str, object],
) -> dict[str, object]:
    spec = config.abstain
    block: dict[str, object] = {}
    for target, name in enumerate(TARGET_NAMES):
        pred = predictions_for(best[name], fitted, data, split.fit_rows, split.test, target)
        physics = data.physics[split.test, target]
        error = np.abs(np.log(pred) - np.log(data.targets[split.test, target]))
        techniques: dict[str, object] = {}
        for technique in Technique:
            mask = data.technique[split.test] == technique
            norm = normalisers[(target, int(technique))]
            half = norm.quantile * difficulty(pred[mask], physics[mask], norm.offset)
            coverage = float(np.mean(error[mask] <= half))
            fixed_quantile = fixed[name]["techniques"][technique.key]["log_quantile"]
            techniques[technique.key] = {
                "coverage": coverage,
                "pass": bool(spec.min_coverage <= coverage <= spec.max_coverage),
                "mean_half_width_log": float(half.mean()),
                "fixed_half_width_log": float(fixed_quantile),
                "log_quantile": norm.quantile,
                "scale_offset": norm.offset,
                "abstain_width_log": norm.abstain_width,
                "fixed_coverage": fixed[name]["techniques"][technique.key]["coverage"],
            }
        block[name] = {
            "techniques": techniques,
            "pass": all(item["pass"] for item in techniques.values()),
        }
    return block


def selector_data(
    template: EvalData,
    technique: Technique,
    packs: PackBatch,
    config: ProjectConfig,
    physics_time: FloatArray,
    physics_energy: FloatArray,
) -> EvalData:
    components, frequencies = mid_values(config, packs.n_packs)
    features = pack_features(technique, packs, components[technique], frequencies[technique])
    physics = np.column_stack(
        (
            np.maximum(physics_time[:, technique], FLOOR),
            np.maximum(physics_energy[:, technique], FLOOR),
        )
    )
    n = packs.n_packs
    return EvalData(
        features=features,
        feature_names=template.feature_names,
        targets=np.zeros((n, 2)),
        technique=np.full(n, int(technique)),
        physics=physics,
        spread=np.ptp(packs.soc0, axis=1),
        component=components[technique],
        frequency=frequencies[technique],
        soc0=packs.soc0,
        capacity_ah=packs.capacity_ah,
        r0_ohm=packs.r0_ohm,
    )


@dataclass(frozen=True, slots=True)
class Predictions:
    time_s: FloatArray
    energy_j: FloatArray
    time_upper_s: FloatArray
    time_half_log: FloatArray


def interval_predictions(
    data: EvalData,
    config: ProjectConfig,
    packs: PackBatch,
    fitted: dict[tuple[str, int], FittedModel],
    best: dict[str, str],
    normalisers: dict[tuple[int, int], Normaliser],
    physics_time: FloatArray,
    physics_energy: FloatArray,
) -> Predictions:
    rows = np.arange(packs.n_packs)
    time_cols, energy_cols, upper_cols, half_cols = [], [], [], []
    for technique in Technique:
        pack_data = selector_data(data, technique, packs, config, physics_time, physics_energy)
        time_pred = fitted[(best[TARGET_NAMES[TIME]], TIME)].predict(pack_data, rows, TIME)
        energy_pred = fitted[(best[TARGET_NAMES[ENERGY]], ENERGY)].predict(pack_data, rows, ENERGY)
        norm = normalisers[(TIME, int(technique))]
        half = norm.quantile * difficulty(time_pred, pack_data.physics[:, TIME], norm.offset)
        time_cols.append(time_pred)
        energy_cols.append(energy_pred)
        upper_cols.append(time_pred * np.exp(half))
        half_cols.append(half)
    return Predictions(
        np.column_stack(time_cols),
        np.column_stack(energy_cols),
        np.column_stack(upper_cols),
        np.column_stack(half_cols),
    )


def abstain_choice(
    predictions: Predictions,
    fallback: IntArray,
    thresholds: FloatArray,
    budget_s: float,
) -> tuple[IntArray, float]:
    upper = choose(predictions.time_upper_s, predictions.energy_j, budget_s)
    rows = np.arange(upper.size)
    wide = predictions.time_half_log[rows, upper] > thresholds[upper]
    return np.where(wide, fallback, upper), float(np.mean(wide))


def abstaining_selector_block(
    data: EvalData,
    config: ProjectConfig,
    split: ConformalSplit,
    fitted: dict[tuple[str, int], FittedModel],
    best: dict[str, str],
    fixed: dict[str, object],
    inputs: SelectorInputs,
) -> dict[str, object]:
    spec = config.abstain
    best = {name: kind if kind in LEARNED else "hybrid" for name, kind in best.items()}
    packs = inputs.packs
    true_time, true_energy = inputs.true_time, inputs.true_energy
    physics_time, physics_energy = inputs.physics_time, inputs.physics_energy
    normalisers = fit_normalisers(data, config, split, fitted, best)
    predictions = interval_predictions(
        data, config, packs, fitted, best, normalisers, physics_time, physics_energy
    )
    thresholds = np.array([normalisers[(TIME, int(t))].abstain_width for t in Technique])
    budgets: dict[str, object] = {}
    n = packs.n_packs
    for hours in config.selector.budgets_h:
        budget_s = hours * SECONDS_PER_HOUR
        oracle = choose(true_time, true_energy, budget_s)
        fallback = choose(physics_time, physics_energy, budget_s)
        abstained, rate = abstain_choice(predictions, fallback, thresholds, budget_s)
        choices = {
            "conformal_point": choose(predictions.time_s, predictions.energy_j, budget_s),
            "conformal_upper": choose(predictions.time_upper_s, predictions.energy_j, budget_s),
            "conformal_abstain": abstained,
        }
        budgets[budget_key(hours)] = {
            "abstention_rate": rate,
            "policies": {
                key: policy_row(choice, true_time, true_energy, oracle, budget_s)
                for key, choice in choices.items()
            },
        }
    true_log = np.log(true_time)
    half_error = np.abs(np.log(predictions.time_s) - true_log)
    return {
        "status": "ok",
        "n_packs": n,
        "time_interval_coverage": float(np.mean(half_error <= predictions.time_half_log)),
        "time_mdape": float(np.median(np.abs(predictions.time_s - true_time) / true_time)),
        "energy_mdape": float(np.median(np.abs(predictions.energy_j - true_energy) / true_energy)),
        "predictor": dict(best),
        "calibration_rows": int(split.calibration.size),
        "test_rows": int(split.test.size),
        "calibration": calibration_block(data, config, split, fitted, best, normalisers, fixed),
        "abstain_quantile": spec.abstain_quantile,
        "abstain_width_log": {t.key: float(thresholds[int(t)]) for t in Technique},
        "min_accuracy": spec.min_accuracy,
        "max_violation": spec.max_violation,
        "by_budget": budgets,
    }
