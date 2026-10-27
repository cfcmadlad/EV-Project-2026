import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.balancing import simulate
from src.baselines import ocv_slope_v_per_soc
from src.config import FloatArray, ProjectConfig, Technique
from src.pack import PackBatch, sample_packs
from src.params import SECONDS_PER_HOUR

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "voltage_balancing.json"
MV_PER_V = 1000.0


def threshold_key(threshold: float) -> str:
    return f"stop_{threshold * 100:g}pct".replace(".", "p")


def setting_key(threshold: float, sigma_mv: float) -> str:
    return f"{threshold_key(threshold)}_sigma_{sigma_mv:g}mv".replace(".", "p")


def cache_inputs(config: ProjectConfig) -> dict[str, object]:
    spec = config.voltage_balancing
    return {
        "seed": config.seed,
        "parameter_set": config.params.name,
        "sigma_mv": list(spec.sigma_mv),
        "stop_thresholds": list(spec.stop_thresholds),
        "packs": spec.packs,
        "noise_averaging_samples": spec.noise_averaging_samples,
        "max_balancing_h": spec.max_balancing_h,
        "target_spread": spec.target_spread,
        "soc_tolerance": config.sim.soc_tolerance,
    }


def sensor_arrays(
    config: ProjectConfig, packs: PackBatch, unit_offsets: FloatArray, sigma_mv: float
) -> tuple[FloatArray, FloatArray]:
    slope = ocv_slope_v_per_soc(packs.soc0.mean(axis=1), config.cell)[:, None]
    offset_soc = sigma_mv / MV_PER_V * unit_offsets / slope
    noise_soc = np.broadcast_to(
        sigma_mv / MV_PER_V / np.sqrt(config.voltage_balancing.noise_averaging_samples) / slope,
        offset_soc.shape,
    ).copy()
    return offset_soc, noise_soc


def run_technique(
    technique: Technique,
    config: ProjectConfig,
    packs: PackBatch,
    unit_offsets: FloatArray,
    seeds: list[np.random.SeedSequence],
) -> dict[str, dict[str, object]]:
    spec = config.voltage_balancing
    n = packs.n_packs
    hardware = config.technique(technique)
    component = np.full(n, hardware.component.mid)
    frequency = np.full(n, hardware.frequency_hz.mid)
    out: dict[str, dict[str, object]] = {}
    index = 0
    for threshold in spec.stop_thresholds:
        local = replace(
            config,
            sim=replace(
                config.sim,
                soc_tolerance=threshold,
                t_max_s=spec.max_balancing_h * SECONDS_PER_HOUR,
            ),
        )
        for sigma in spec.sigma_mv:
            offset, noise = sensor_arrays(config, packs, unit_offsets, sigma)
            rng = np.random.default_rng(seeds[index])
            index += 1
            result = simulate(
                technique,
                packs,
                component,
                frequency,
                local,
                soc_bias=offset,
                noise_std=noise,
                noise_rng=rng,
            )
            true_spread = np.ptp(result.final_soc, axis=1)
            out[setting_key(threshold, sigma)] = {
                "fraction_true_spread_below_target": float(
                    np.mean(true_spread < spec.target_spread)
                ),
                "true_spread_median": float(np.median(true_spread)),
                "true_spread_p95": float(np.percentile(true_spread, 95.0)),
                "estimated_criterion_reached_fraction": float(np.mean(result.balanced)),
                "balancing_time_s_median": float(np.median(result.balancing_time_s)),
                "energy_dissipated_j_median": float(np.median(result.energy_dissipated_j)),
            }
    return out


def run_voltage_study(config: ProjectConfig, rng: np.random.Generator) -> dict[str, object]:
    spec = config.voltage_balancing
    packs = sample_packs(spec.packs, rng, config.cell, config.pack)
    unit_offsets = rng.standard_normal(packs.soc0.shape)
    count = len(spec.stop_thresholds) * len(spec.sigma_mv)
    seeds = [list(rng.spawn(1)[0].bit_generator.seed_seq.spawn(count)) for _ in Technique]
    outputs = Parallel(n_jobs=len(Technique))(
        delayed(run_technique)(t, config, packs, unit_offsets, seeds[i])
        for i, t in enumerate(Technique)
    )
    return {
        "inputs": json.loads(json.dumps(cache_inputs(config))),
        "results": {t.key: out for t, out in zip(Technique, outputs, strict=True)},
    }


def required_sigma_mv(
    sigmas: list[float], fractions: list[float], required: float
) -> dict[str, float | None]:
    ok = [s for s, f in zip(sigmas, fractions, strict=True) if f >= required]
    grid = max(ok) if ok else None
    interpolated = None
    if fractions[0] >= required:
        interpolated = sigmas[-1] if fractions[-1] >= required else None
        for k in range(len(sigmas) - 1):
            if fractions[k] >= required > fractions[k + 1]:
                share = (fractions[k] - required) / (fractions[k] - fractions[k + 1])
                interpolated = float(
                    np.exp(np.log(sigmas[k]) + share * (np.log(sigmas[k + 1]) - np.log(sigmas[k])))
                )
                break
    return {"largest_grid_sigma_mv": grid, "interpolated_sigma_mv": interpolated}


def voltage_balancing_block(config: ProjectConfig) -> dict[str, object]:
    if not CACHE_PATH.exists():
        return {"status": "not_run"}
    payload = json.loads(CACHE_PATH.read_text())
    if payload["inputs"] != json.loads(json.dumps(cache_inputs(config))):
        return {"status": "stale_cache"}
    spec = config.voltage_balancing
    sigmas = list(spec.sigma_mv)
    required: dict[str, object] = {}
    for key, rows in payload["results"].items():
        required[key] = {
            threshold_key(threshold): required_sigma_mv(
                sigmas,
                [
                    rows[setting_key(threshold, s)]["fraction_true_spread_below_target"]
                    for s in sigmas
                ],
                spec.required_fraction,
            )
            for threshold in spec.stop_thresholds
        }
    slope = ocv_slope_v_per_soc(np.array([0.5]), config.cell)[0] / 100.0 * MV_PER_V
    return {
        "status": "ok",
        "n_packs": spec.packs,
        "sigma_mv": sigmas,
        "stop_thresholds": list(spec.stop_thresholds),
        "required_fraction": spec.required_fraction,
        "ocv_slope_at_50_percent_mv_per_percent": float(slope),
        "sensor_model": "fixed offset N(0, sigma) per cell plus white noise sigma / sqrt(60) per "
        "second (60 s average); rest voltage mapped to SOC with the fitted OCV curve",
        "results": payload["results"],
        "required_voltage_accuracy": required,
    }
