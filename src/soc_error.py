from dataclasses import dataclass, replace

import numpy as np
from joblib import Parallel, delayed

from src.balancing import BalanceResult, simulate
from src.checks import first_packs
from src.config import PERCENT, FloatArray, ProjectConfig, Technique
from src.identification import table_slope_mv_per_percent
from src.pack import PackBatch, sample_packs
from src.params import SECONDS_PER_HOUR


@dataclass(frozen=True, slots=True)
class BiasStudy:
    labels: tuple[str, ...]
    final_spread: dict[Technique, dict[str, FloatArray]]
    results: dict[Technique, dict[str, BalanceResult]]
    deterministic: dict[Technique, BalanceResult]


def sigma_label(sigma_percent: float) -> str:
    return f"sigma_{sigma_percent:g}pct"


def bias_draws(
    sigma_percent: tuple[float, ...], packs: PackBatch, rng: np.random.Generator
) -> dict[str, FloatArray]:
    unit = rng.standard_normal(packs.soc0.shape)
    return {sigma_label(s): s / PERCENT * unit for s in sigma_percent}


def _run_technique(
    technique: Technique, packs: PackBatch, biases: dict[str, FloatArray], config: ProjectConfig
) -> tuple[dict[str, BalanceResult], BalanceResult]:
    n = packs.n_packs
    hardware = config.technique(technique)
    component = np.full(n, hardware.component.mid)
    frequency = np.full(n, hardware.frequency_hz.mid)
    results = {
        label: simulate(technique, packs, component, frequency, config, soc_bias=bias)
        for label, bias in biases.items()
    }
    check = config.soc_error.deterministic_check_packs
    return results, simulate(
        technique, first_packs(packs, check), component[:check], frequency[:check], config
    )


def study_config(config: ProjectConfig) -> ProjectConfig:
    cap_s = config.soc_error.max_balancing_h * SECONDS_PER_HOUR
    return replace(config, sim=replace(config.sim, t_max_s=cap_s))


def run_bias_study(
    packs: PackBatch, biases: dict[str, FloatArray], config: ProjectConfig
) -> BiasStudy:
    config = study_config(config)
    outputs = Parallel(n_jobs=len(Technique))(
        delayed(_run_technique)(technique, packs, biases, config) for technique in Technique
    )
    results = {t: out[0] for t, out in zip(Technique, outputs, strict=True)}
    return BiasStudy(
        labels=tuple(biases),
        final_spread={
            t: {label: np.ptp(r.final_soc, axis=1) for label, r in per_label.items()}
            for t, per_label in results.items()
        },
        results=results,
        deterministic={t: out[1] for t, out in zip(Technique, outputs, strict=True)},
    )


def summarise_study(
    study: BiasStudy, reference_label: str, config: ProjectConfig
) -> dict[str, dict[str, object]]:
    tail = config.soc_error.spread_tail_percent
    tolerance = config.sim.soc_tolerance
    block: dict[str, dict[str, object]] = {}
    for technique in Technique:
        reference = study.results[technique][reference_label]
        per_label: dict[str, object] = {}
        for label in study.labels:
            result = study.results[technique][label]
            spread = study.final_spread[technique][label]
            extra_energy = result.energy_dissipated_j - reference.energy_dissipated_j
            time_change = result.balancing_time_s - reference.balancing_time_s
            per_label[label] = {
                "true_final_spread_median": float(np.median(spread)),
                "true_final_spread_p95": float(np.percentile(spread, tail)),
                "fraction_true_spread_above_tolerance": float(np.mean(spread >= tolerance)),
                "extra_energy_dissipated_j_median": float(np.median(extra_energy)),
                "energy_ratio_median": float(
                    np.median(result.energy_dissipated_j / reference.energy_dissipated_j)
                ),
                "balancing_time_change_s_median": float(np.median(time_change)),
                "balancing_time_ratio_median": float(
                    np.median(result.balancing_time_s / reference.balancing_time_s)
                ),
                "estimated_criterion_reached_fraction": float(np.mean(result.balanced)),
            }
        block[technique.key] = per_label
    return block


def soc_error_block(
    config: ProjectConfig, rng: np.random.Generator
) -> tuple[dict[str, object], BiasStudy]:
    spec = config.soc_error
    packs = sample_packs(spec.packs, rng, config.cell, config.pack)
    biases = bias_draws(spec.sigma_percent, packs, rng)
    study = run_bias_study(packs, biases, config)
    zero = sigma_label(0.0)
    check = spec.deterministic_check_packs
    matches = all(
        np.array_equal(study.results[t][zero].final_soc[:check], study.deterministic[t].final_soc)
        and np.array_equal(
            study.results[t][zero].energy_removed_j[:check], study.deterministic[t].energy_removed_j
        )
        and np.array_equal(
            study.results[t][zero].balancing_time_s[:check],
            study.deterministic[t].balancing_time_s,
        )
        for t in Technique
    )
    slope = table_slope_mv_per_percent(config.cell)
    block: dict[str, object] = {
        "status": "ok",
        "n_packs": spec.packs,
        "max_balancing_h": spec.max_balancing_h,
        "deterministic_check_packs": check,
        "sigma_percent": list(spec.sigma_percent),
        "reference": zero,
        "sigma_zero_reproduces_deterministic": bool(matches),
        "techniques": summarise_study(study, zero, config),
        "voltage_equivalent": {
            "ocv_slope_at_50_percent_mv_per_percent": slope,
            "voltage_error_mv": spec.voltage_error_mv,
            "soc_error_percent": spec.voltage_error_mv / slope,
        },
    }
    return block, study
