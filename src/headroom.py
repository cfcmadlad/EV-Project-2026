import json
from dataclasses import replace

import numpy as np

from src.config import PROJECT_ROOT, FloatArray, ProjectConfig, Technique
from src.drive_cycle import Mode, charge_equalised, program_from_profiles, run_scenario
from src.fingerprint import maps_sha256, text_sha256
from src.pack import PackBatch, sample_packs
from src.scaling import string_config

CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "headroom_results.json"
SOC_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "soc_deep_results.json"
CELLS = (4, 8, 16, 32, 96)
CYCLES = ("Mixed1", "Mixed7")
PACKS = 50
NO_HEADROOM_MEDIAN = 0.01
HEADROOM_P90 = 0.02
STRATEGIES = ("rule", "greedy", "rule_with_estimation_error")
PERCENTILES = (10.0, 50.0, 90.0)
NOT_RUN = {"status": "not_run"}
STALE = {"status": "stale_cache"}


def gap(reference_ah: FloatArray, delivered_ah: FloatArray) -> FloatArray:
    return (reference_ah - delivered_ah) / reference_ah


def gap_stats(values: FloatArray) -> dict[str, float]:
    low, median, high = np.percentile(values, PERCENTILES)
    return {
        "p10": float(low),
        "median": float(median),
        "p90": float(high),
        "max": float(values.max()),
        "n": int(values.size),
    }


def verdict(stats: dict[str, float]) -> dict[str, bool]:
    return {
        "no_meaningful_headroom": bool(stats["median"] < NO_HEADROOM_MEDIAN),
        "headroom_exists": bool(stats["p90"] > HEADROOM_P90),
    }


def estimation_sigma() -> float:
    payload = json.loads(SOC_RESULTS_PATH.read_text())
    return payload["results"]["ekf_gru_hybrid"]["nominal"]["test_a"]["rmse_percent"] / 100.0


def cache_inputs(config: ProjectConfig) -> dict[str, object]:
    return {
        "maps_sha256": maps_sha256(),
        "soc_results_sha256": text_sha256(SOC_RESULTS_PATH),
        "profiles_sha256": text_sha256(
            PROJECT_ROOT / "data" / "processed" / "drive_profiles_25c.json"
        ),
        "seed": config.seed,
        "cells": list(CELLS),
        "cycles": list(CYCLES),
        "packs": PACKS,
        "parameter_set": config.params.name,
        "flying_capacitor_model": config.flying_capacitor.model.value,
        "buck_boost_calibrated": config.buck_boost.circuit_calibrated,
    }


def draw_packs(config: ProjectConfig, n_cells: int, rng: np.random.Generator) -> PackBatch:
    local = string_config(config, n_cells)
    return sample_packs(PACKS, rng, local.cell, local.pack)


def run_cell(
    config: ProjectConfig,
    profiles: dict,
    n_cells: int,
    cycle: str,
    packs: PackBatch,
    bias: FloatArray,
) -> dict[str, FloatArray]:
    local = string_config(config, n_cells)
    program = program_from_profiles([np.array(profiles["profiles"][cycle]["current_a"])] * PACKS)
    out = {
        "none": run_scenario(None, packs, program, Mode.DISCHARGE, local).usable_ah,
        "reference": run_scenario(
            None, charge_equalised(packs, Mode.DISCHARGE), program, Mode.DISCHARGE, local
        ).usable_ah,
    }
    for technique in Technique:
        key = technique.key
        out[f"{key}.rule"] = run_scenario(
            technique, packs, program, Mode.DISCHARGE, local
        ).usable_ah
        out[f"{key}.greedy"] = run_scenario(
            technique, packs, program, Mode.DISCHARGE, local, gate_balancing=False
        ).usable_ah
        out[f"{key}.rule_with_estimation_error"] = run_scenario(
            technique, packs, program, Mode.DISCHARGE, local, soc_bias=bias
        ).usable_ah
    return out


def summarise(raw: dict[tuple[int, str], dict[str, FloatArray]]) -> dict[str, object]:
    results: dict[str, object] = {}
    for technique in Technique:
        by_n: dict[str, object] = {}
        for n_cells in CELLS:
            pooled = {s: [] for s in STRATEGIES}
            by_cycle = {}
            for cycle in CYCLES:
                arrays = raw[(n_cells, cycle)]
                reference = arrays["reference"]
                by_cycle[cycle] = {
                    "delivered_ah_mean": {
                        "none": float(arrays["none"].mean()),
                        **{s: float(arrays[f"{technique.key}.{s}"].mean()) for s in STRATEGIES},
                        "reference": float(reference.mean()),
                    },
                    "gap_none": gap_stats(gap(reference, arrays["none"])),
                    **{
                        f"gap_{s}": gap_stats(gap(reference, arrays[f"{technique.key}.{s}"]))
                        for s in STRATEGIES
                    },
                }
                for s in STRATEGIES:
                    pooled[s].append(gap(reference, arrays[f"{technique.key}.{s}"]))
            pooled_stats = {s: gap_stats(np.concatenate(v)) for s, v in pooled.items()}
            by_n[str(n_cells)] = {
                "pooled": pooled_stats,
                "verdict": verdict(pooled_stats["rule"]),
                "estimation_penalty_median": pooled_stats["rule_with_estimation_error"]["median"]
                - pooled_stats["rule"]["median"],
                "by_cycle": by_cycle,
            }
        results[technique.key] = by_n
    return results


def load_cache(config: ProjectConfig) -> dict[str, object]:
    if not CACHE_PATH.exists():
        return dict(NOT_RUN)
    payload = json.loads(CACHE_PATH.read_text())
    if payload["inputs"] != json.loads(json.dumps(cache_inputs(config))):
        raise RuntimeError(
            "headroom cache is stale; rerun python -m scripts.run_headroom before run_all.py"
        )
    return {"status": "ok", **payload}


def headroom_config(config: ProjectConfig) -> ProjectConfig:
    return replace(config)
