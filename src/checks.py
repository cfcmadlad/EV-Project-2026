from dataclasses import replace

import numpy as np
from joblib import Parallel, delayed

from src.balancing import BalanceResult, simulate
from src.baselines import passive_balancing_time_s
from src.config import PERCENT, FloatArray, ProjectConfig, Technique
from src.experiments import baseline_pack, run_baseline
from src.pack import PackBatch, sample_packs
from src.params import ASSUMED

MIDSEM_PINNED = {
    "passive": {
        "balancing_time_s": (16400.0, 0),
        "energy_dissipated_j": (7680.0, 0),
        "efficiency_pct": (0.0, 1),
    },
    "buck_boost": {
        "balancing_time_s": (4203.0, 0),
        "energy_dissipated_j": (220.8, 1),
        "efficiency_pct": (96.4, 1),
    },
    "flying_capacitor": {
        "balancing_time_s": (9140.0, 0),
        "energy_dissipated_j": (20.5, 1),
        "efficiency_pct": (99.6, 1),
    },
}
ScalarMap = dict[str, float | bool | str]


def random_case(
    technique: Technique, n_packs: int, config: ProjectConfig, rng: np.random.Generator
) -> tuple[PackBatch, FloatArray, FloatArray]:
    hardware = config.technique(technique)
    packs = sample_packs(n_packs, rng, config.cell, config.pack)
    component = rng.uniform(hardware.component.low, hardware.component.high, n_packs)
    frequency = rng.uniform(hardware.frequency_hz.low, hardware.frequency_hz.high, n_packs)
    return packs, component, frequency


def ledger_discrepancy(result: BalanceResult) -> dict[str, FloatArray]:
    dissipated = result.energy_dissipated_j
    net_gap = np.abs(dissipated - result.ledger_j)
    per_link_gap = np.abs(dissipated - result.per_link_ledger_j)
    return {
        "net_vs_removed": net_gap / result.energy_removed_j,
        "net_vs_dissipated": net_gap / dissipated,
        "per_link_vs_removed": per_link_gap / result.energy_removed_j,
        "per_link_vs_dissipated": per_link_gap / dissipated,
    }


def first_packs(packs: PackBatch, n_packs: int) -> PackBatch:
    return PackBatch(
        capacity_ah=packs.capacity_ah[:n_packs],
        r0_ohm=packs.r0_ohm[:n_packs],
        soc0=packs.soc0[:n_packs],
    )


def dt_scaling(
    technique: Technique,
    packs: PackBatch,
    component: FloatArray,
    frequency: FloatArray,
    config: ProjectConfig,
) -> dict[str, dict[str, float]]:
    n_packs = config.checks.dt_scaling_packs
    small = first_packs(packs, n_packs)
    scaling = {}
    for dt_s in config.checks.dt_scaling_steps_s:
        stepped = replace(config, sim=replace(config.sim, dt_s=dt_s))
        result = simulate(technique, small, component[:n_packs], frequency[:n_packs], stepped)
        gaps = ledger_discrepancy(result)
        scaling[f"dt_{dt_s:g}_s"] = {
            "max_net_vs_removed": float(gaps["net_vs_removed"].max()),
            "max_net_vs_dissipated": float(gaps["net_vs_dissipated"].max()),
            "max_per_link_vs_removed": float(gaps["per_link_vs_removed"].max()),
            "max_per_link_vs_dissipated": float(gaps["per_link_vs_dissipated"].max()),
        }
    return scaling


def technique_ledger(
    technique: Technique,
    case: tuple[PackBatch, FloatArray, FloatArray],
    config: ProjectConfig,
) -> dict[str, object]:
    checks = config.checks
    packs, component, frequency = case
    gaps = ledger_discrepancy(simulate(technique, packs, component, frequency, config))
    worst = float(gaps["net_vs_removed"].max())
    return {
        "n_packs": checks.packs_per_technique,
        "max_net_vs_removed": worst,
        "median_net_vs_removed": float(np.median(gaps["net_vs_removed"])),
        "max_net_vs_dissipated": float(gaps["net_vs_dissipated"].max()),
        "median_net_vs_dissipated": float(np.median(gaps["net_vs_dissipated"])),
        "max_per_link_vs_removed": float(gaps["per_link_vs_removed"].max()),
        "max_per_link_vs_dissipated": float(gaps["per_link_vs_dissipated"].max()),
        "threshold_vs_removed": checks.ledger_tolerance,
        "pass": worst <= checks.ledger_tolerance,
        "dt_scaling": dt_scaling(technique, packs, component, frequency, config),
    }


def baseline_ledger(results: dict[Technique, BalanceResult]) -> dict[str, ScalarMap]:
    return {
        technique.key: {
            "dissipated_j": float(result.energy_dissipated_j[0]),
            "ledger_net_j": float(result.ledger_j[0]),
            "ledger_per_link_j": float(result.per_link_ledger_j[0]),
            "element_loss_j": float(result.element_loss_j[0]),
            "cell_r0_net_loss_j": float(result.r0_loss_j[0]),
            "cell_r0_per_link_loss_j": float(result.r0_per_link_loss_j[0]),
            "r1_branch_loss_j": float(result.r1_loss_j[0]),
            "c1_stored_j": float(result.c1_stored_j[0]),
        }
        for technique, result in results.items()
    }


def baseline_ledger_check(pack: PackBatch, config: ProjectConfig) -> dict[str, ScalarMap]:
    return baseline_ledger(run_baseline(pack, config, trace=False))


def passive_closed_form_check(
    config: ProjectConfig, rng: np.random.Generator, pack: PackBatch
) -> dict[str, object]:
    checks = config.checks
    r_mid = np.array([config.passive.component.mid])
    baseline_sim = simulate(Technique.PASSIVE, pack, r_mid, np.zeros(1), config)
    baseline_formula = passive_balancing_time_s(
        pack.soc0, pack.capacity_ah, pack.r0_ohm, r_mid, config.cell, config.sim
    )
    packs, r_bleed, _ = random_case(Technique.PASSIVE, checks.packs_per_technique, config, rng)
    simulated = simulate(Technique.PASSIVE, packs, r_bleed, np.zeros_like(r_bleed), config)
    formula = passive_balancing_time_s(
        packs.soc0, packs.capacity_ah, packs.r0_ohm, r_bleed, config.cell, config.sim
    )
    error = np.abs(formula - simulated.balancing_time_s) / simulated.balancing_time_s
    baseline_error = float(
        abs(baseline_formula[0] - baseline_sim.balancing_time_s[0])
        / baseline_sim.balancing_time_s[0]
    )
    worst = max(baseline_error, float(error.max()))
    return {
        "baseline_simulated_s": float(baseline_sim.balancing_time_s[0]),
        "baseline_closed_form_s": float(baseline_formula[0]),
        "baseline_relative_error": baseline_error,
        "n_packs": checks.packs_per_technique,
        "max_relative_error": float(error.max()),
        "median_relative_error": float(np.median(error)),
        "threshold": checks.passive_closed_form_tolerance,
        "pass": worst <= checks.passive_closed_form_tolerance,
    }


def flying_capacitor_charge_check(
    config: ProjectConfig, rng: np.random.Generator
) -> dict[str, object]:
    checks = config.checks
    technique = Technique.FLYING_CAPACITOR
    packs, component, frequency = random_case(technique, checks.packs_per_technique, config, rng)
    result = simulate(technique, packs, component, frequency, config)
    charge_change_ah = packs.capacity_ah * (packs.soc0 - result.final_soc)
    charge_out_ah = np.clip(charge_change_ah, 0.0, None).sum(axis=1)
    charge_in_ah = np.clip(-charge_change_ah, 0.0, None).sum(axis=1)
    imbalance = np.abs(charge_out_ah - charge_in_ah) / charge_out_ah
    return {
        "n_packs": checks.packs_per_technique,
        "max_relative_imbalance": float(imbalance.max()),
        "threshold": checks.charge_conservation_tolerance,
        "pass": bool(imbalance.max() <= checks.charge_conservation_tolerance),
    }


def midsem_regression_check(config: ProjectConfig) -> dict[str, object]:
    assumed = replace(config, params=ASSUMED)
    results = run_baseline(baseline_pack(assumed), assumed, trace=False)
    rows: dict[str, ScalarMap] = {}
    for technique, result in results.items():
        measured = {
            "balancing_time_s": float(result.balancing_time_s[0]),
            "energy_dissipated_j": float(result.energy_dissipated_j[0]),
            "efficiency_pct": float(result.efficiency[0] * PERCENT),
        }
        pinned = MIDSEM_PINNED[technique.key]
        matches = all(
            round(measured[name], decimals) == value for name, (value, decimals) in pinned.items()
        )
        rows[technique.key] = {**measured, "matches_pinned": matches}
    return {
        "parameter_set": ASSUMED.name,
        "techniques": rows,
        "pass": all(bool(row["matches_pinned"]) for row in rows.values()),
    }


def physics_checks(config: ProjectConfig, rng: np.random.Generator) -> dict[str, object]:
    ledger_rng, closed_form_rng, charge_rng = rng.spawn(3)
    pack = baseline_pack(config)
    cases = [
        random_case(technique, config.checks.packs_per_technique, config, ledger_rng)
        for technique in Technique
    ]
    results = Parallel(n_jobs=len(Technique) + 4)(
        [
            *(
                delayed(technique_ledger)(t, case, config)
                for t, case in zip(Technique, cases, strict=True)
            ),
            delayed(baseline_ledger_check)(pack, config),
            delayed(passive_closed_form_check)(config, closed_form_rng, pack),
            delayed(flying_capacitor_charge_check)(config, charge_rng),
            delayed(midsem_regression_check)(config),
        ]
    )
    ledgers, baseline, closed_form, charge, regression = (
        results[: len(Technique)],
        *results[len(Technique) :],
    )
    return {
        "energy_ledger": {t.key: ledger for t, ledger in zip(Technique, ledgers, strict=True)},
        "baseline_ledger": baseline,
        "passive_closed_form": closed_form,
        "flying_capacitor_charge": charge,
        "midsem_regression": regression,
    }
