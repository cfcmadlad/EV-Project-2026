import json
from dataclasses import replace

from src.config import PERCENT, PROJECT_ROOT, ProjectConfig
from src.experiments import baseline_pack, run_baseline
from src.params import ASSUMED

REFERENCE_PATH = PROJECT_ROOT / "results" / "midsem_reference" / "metrics.json"
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "midsem_consistency.json"


def cache_inputs(config: ProjectConfig) -> dict[str, object]:
    return {
        "seed": config.seed,
        "runs_per_technique": config.dataset.runs_per_technique,
        "parameter_set": ASSUMED.name,
        "ml": {
            "max_iter": config.ml.max_iter,
            "learning_rate": config.ml.learning_rate,
            "max_leaf_nodes": config.ml.max_leaf_nodes,
            "test_fraction": config.ml.test_fraction,
        },
    }


def baseline_rows(config: ProjectConfig) -> dict[str, dict[str, float]]:
    results = run_baseline(baseline_pack(config), config, trace=False)
    return {
        technique.key: {
            "balancing_time_s": float(result.balancing_time_s[0]),
            "energy_dissipated_j": float(result.energy_dissipated_j[0]),
            "efficiency_pct": float(result.efficiency[0] * PERCENT),
        }
        for technique, result in results.items()
    }


def midsem_block(
    config: ProjectConfig, fitted_ml: dict[str, dict[str, dict[str, float]]]
) -> dict[str, object]:
    if not REFERENCE_PATH.exists():
        return {"status": "not_run"}
    reference = json.loads(REFERENCE_PATH.read_text())
    assumed_config = replace(config, params=ASSUMED)
    assumed = baseline_rows(assumed_config)
    fitted = baseline_rows(config)
    pack = baseline_pack(assumed_config)
    rows: dict[str, object] = {}
    for row in reference["baseline"]:
        key = row["technique"]
        pinned = {
            "balancing_time_s": row["balancing_time_s"],
            "energy_dissipated_j": row["energy_dissipated_j"],
            "efficiency_pct": row["efficiency_pct"],
        }
        rows[key] = {
            "midsem_reference": pinned,
            "reproduced_assumed_model": assumed[key],
            "fitted_hg2_model": fitted[key],
            "max_relative_difference_reproduced_vs_midsem": max(
                abs(assumed[key][name] / value - 1.0)
                for name, value in pinned.items()
                if value != 0.0
            ),
            "time_ratio_fitted_over_midsem": fitted[key]["balancing_time_s"]
            / pinned["balancing_time_s"],
            "energy_ratio_fitted_over_midsem": fitted[key]["energy_dissipated_j"]
            / pinned["energy_dissipated_j"],
        }
    block: dict[str, object] = {
        "status": "ok",
        "midsem_cell_model": "assumed generic 3 Ah NMC cell, midsem flying-capacitor formula",
        "baseline_soc0_percent": [float(v) * PERCENT for v in pack.soc0[0]],
        "baseline": rows,
        "surrogate_protocol": {
            "runs_per_technique": config.dataset.runs_per_technique,
            "features": 19,
            "model": "HistGradientBoostingRegressor",
            "max_iter": config.ml.max_iter,
            "learning_rate": config.ml.learning_rate,
            "log_targets": True,
            "stratified_split": True,
            "test_fraction": config.ml.test_fraction,
        },
        "surrogate_midsem_reference": {
            name: {"r2": v["overall"]["r2"], "mae": v["overall"]["mae"]}
            for name, v in reference["ml"].items()
        },
        "surrogate_trained_on_fitted_hg2_model": {
            name: {"r2": v["overall"]["r2"], "mae": v["overall"]["mae"]}
            for name, v in fitted_ml.items()
        },
    }
    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text())
        if cached["inputs"] == json.loads(json.dumps(cache_inputs(config))):
            block["surrogate_retrained_on_assumed_model"] = cached["ml"]
        else:
            block["surrogate_retrained_on_assumed_model"] = {"status": "stale_cache"}
    else:
        block["surrogate_retrained_on_assumed_model"] = {"status": "not_run"}
    return block
