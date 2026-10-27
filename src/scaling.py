import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.balancing import simulate
from src.baselines import physics_prediction
from src.config import FloatArray, ProjectConfig, Technique
from src.pack import PackBatch, sample_packs
from src.params import SECONDS_PER_HOUR

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "scaling_results.json"
NOT_RUN = {"status": "not_run"}
STALE = {"status": "stale_cache"}


def link_deadband(config: ProjectConfig, n_cells: int) -> float:
    limit = config.scaling.deadband_fraction * config.sim.soc_tolerance / (n_cells - 1)
    return min(config.sim.link_soc_deadband, limit)


def string_config(config: ProjectConfig, n_cells: int) -> ProjectConfig:
    spec = config.scaling
    pack = replace(config.pack, n_cells=n_cells)
    sim = replace(
        config.sim,
        t_max_s=spec.max_balancing_h * SECONDS_PER_HOUR,
        link_soc_deadband=link_deadband(config, n_cells),
    )
    return replace(config, params=replace(config.params, pack=pack), sim=sim)


def cache_inputs(config: ProjectConfig) -> dict[str, object]:
    spec = config.scaling
    return {
        "parameter_set": config.params.name,
        "seed": config.seed,
        "cells": list(spec.cells),
        "packs": spec.packs,
        "max_balancing_h": spec.max_balancing_h,
        "deadband_fraction": spec.deadband_fraction,
        "pack": {
            "capacity_tolerance": config.pack.capacity_tolerance,
            "r0_tolerance": config.pack.r0_tolerance,
            "soc_spread": [config.pack.soc_spread_min, config.pack.soc_spread_max],
        },
        "flying_capacitor_model": config.flying_capacitor.model.value,
    }


def run_string(
    config: ProjectConfig, n_cells: int, packs: PackBatch, technique: Technique
) -> dict[str, FloatArray]:
    local = string_config(config, n_cells)
    n = packs.n_packs
    hardware = config.technique(technique)
    component = np.full(n, hardware.component.mid)
    frequency = np.full(n, hardware.frequency_hz.mid)
    result = simulate(technique, packs, component, frequency, local)
    physics = physics_prediction(
        technique, packs.soc0, packs.capacity_ah, packs.r0_ohm, component, frequency, local
    )
    return {
        "time_s": result.balancing_time_s,
        "energy_j": result.energy_dissipated_j,
        "balanced": result.balanced.astype(float),
        "physics_time_s": physics.balancing_time_s,
        "physics_energy_j": physics.energy_dissipated_j,
    }


def sample_string_packs(config: ProjectConfig, n_cells: int, rng: np.random.Generator) -> PackBatch:
    local = string_config(config, n_cells)
    return sample_packs(config.scaling.packs, rng, local.cell, local.pack)


PROFILES = ("random", "step")


def step_profile(packs: PackBatch) -> PackBatch:
    n_cells = packs.soc0.shape[1]
    spread = np.ptp(packs.soc0, axis=1, keepdims=True)
    centre = packs.soc0.mean(axis=1, keepdims=True)
    high = (np.arange(n_cells) < n_cells // 2)[None, :]
    soc0 = centre + spread * np.where(high, 0.5, -0.5)
    return replace(packs, soc0=soc0 - soc0.mean(axis=1, keepdims=True) + centre)


def summarise_arrays(arrays: dict[str, FloatArray]) -> dict[str, object]:
    balanced = arrays["balanced"].astype(bool)
    time, energy = arrays["time_s"], arrays["energy_j"]
    physics_time, physics_energy = arrays["physics_time_s"], arrays["physics_energy_j"]
    return {
        "balanced_fraction": float(balanced.mean()),
        "time_s_median": float(np.median(time[balanced])),
        "time_s_p05": float(np.percentile(time[balanced], 5.0)),
        "time_s_p95": float(np.percentile(time[balanced], 95.0)),
        "energy_j_median": float(np.median(energy[balanced])),
        "physics_time_s_median": float(np.median(physics_time[balanced])),
        "physics_time_median_abs_pct_error": float(
            np.median(np.abs(physics_time[balanced] / time[balanced] - 1.0)) * 100.0
        ),
        "physics_time_p95_abs_pct_error": float(
            np.percentile(np.abs(physics_time[balanced] / time[balanced] - 1.0), 95.0) * 100.0
        ),
        "physics_energy_median_abs_pct_error": float(
            np.median(np.abs(physics_energy[balanced] / energy[balanced] - 1.0)) * 100.0
        ),
        "physics_energy_p95_abs_pct_error": float(
            np.percentile(np.abs(physics_energy[balanced] / energy[balanced] - 1.0), 95.0) * 100.0
        ),
    }


def load_cache(config: ProjectConfig) -> dict | None:
    if not CACHE_PATH.exists():
        return None
    payload = json.loads(CACHE_PATH.read_text())
    if payload["inputs"] != json.loads(json.dumps(cache_inputs(config))):
        return dict(STALE)
    return payload


def path_laplacian_smallest(n_cells: int) -> float:
    return 2.0 - 2.0 * np.cos(np.pi / n_cells)


def loglog_slope(cells: list[int], values: list[float]) -> float:
    return float(np.polyfit(np.log(cells), np.log(values), 1)[0])


def theory_for(profile: str, cells: list[int]) -> dict[str, float | None]:
    if profile == "step":
        return {
            "passive": 0.0,
            "buck_boost": 1.0,
            "flying_capacitor": loglog_slope(
                cells, [1.0 / path_laplacian_smallest(n) for n in cells]
            ),
        }
    return {"passive": 0.0, "buck_boost": 0.5, "flying_capacitor": None}


def profile_block(
    payload: dict, profile: str, config: ProjectConfig, large: list[int], cells: list[int]
) -> dict[str, object]:
    spec = config.scaling
    theory = theory_for(profile, large)
    techniques: dict[str, object] = {}
    for technique in Technique:
        stats = payload["results"][profile][technique.key]
        times = [stats[str(n)]["time_s_median"] for n in cells]
        slope_large = loglog_slope(large, [stats[str(n)]["time_s_median"] for n in large])
        expected = theory[technique.key]
        techniques[technique.key] = {
            "median_time_s": dict(zip(map(str, cells), times, strict=True)),
            "fitted_slope_all_n": loglog_slope(cells, times),
            "fitted_slope_large_n": slope_large,
            "physics_baseline_slope_large_n": loglog_slope(
                large, [stats[str(n)]["physics_time_s_median"] for n in large]
            ),
            "theory_slope_large_n": expected,
            "slope_difference_large_n": None if expected is None else slope_large - expected,
            "agrees_with_theory": None
            if expected is None
            else bool(abs(slope_large - expected) <= spec.slope_tolerance),
            "physics_time_median_abs_pct_error_by_n": {
                str(n): stats[str(n)]["physics_time_median_abs_pct_error"] for n in cells
            },
            "physics_time_p95_abs_pct_error_by_n": {
                str(n): stats[str(n)]["physics_time_p95_abs_pct_error"] for n in cells
            },
            "physics_energy_median_abs_pct_error_by_n": {
                str(n): stats[str(n)]["physics_energy_median_abs_pct_error"] for n in cells
            },
            "balanced_fraction_by_n": {str(n): stats[str(n)]["balanced_fraction"] for n in cells},
            "energy_j_median_by_n": {str(n): stats[str(n)]["energy_j_median"] for n in cells},
        }
    return techniques


def scaling_block(config: ProjectConfig) -> dict[str, object]:
    payload = load_cache(config)
    if payload is None:
        return dict(NOT_RUN)
    if payload.get("status") == "stale_cache":
        return dict(STALE)
    spec = config.scaling
    cells = payload["inputs"]["cells"]
    large = [n for n in cells if n >= spec.large_n_min]
    return {
        "status": "ok",
        "cells": cells,
        "large_n_min": spec.large_n_min,
        "slope_tolerance": spec.slope_tolerance,
        "packs_per_n": payload["inputs"]["packs"],
        "max_balancing_h": payload["inputs"]["max_balancing_h"],
        "link_deadband_by_n": {str(n): link_deadband(config, n) for n in cells},
        "profiles": {p: profile_block(payload, p, config, large, cells) for p in PROFILES},
        "ml_model_note": "the 4-cell surrogate takes 4 SOCs, 4 capacities and 4 resistances as "
        "inputs and cannot represent N != 4",
    }
