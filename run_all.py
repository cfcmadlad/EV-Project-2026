import csv
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from src.balancing import BalanceResult, simulate, tail_slowdown_ratio
from src.cell_plots import plot_baseline_comparison
from src.checks import physics_checks
from src.config import (
    CONFIG,
    HZ_PER_KHZ,
    MF_PER_FARAD,
    PERCENT,
    RESULTS_DIR,
    FloatArray,
    ProjectConfig,
    Technique,
)
from src.dataset import TARGET_NAMES, Dataset, generate_dataset
from src.experiments import baseline_pack, random_streams, run_baseline
from src.identification import table_slope_mv_per_percent
from src.pack import PackBatch, tile_pack
from src.params import ASSUMED, SECONDS_PER_HOUR
from src.plots import (
    plot_importance,
    plot_parity,
    plot_soc_trace,
    plot_spread_comparison,
    plot_sweep,
)
from src.predictor import PredictorReport, train_and_evaluate
from src.uncertainty import (
    BOOTSTRAP_PATH,
    IDENTIFICATION_PATH,
    NOT_RUN,
    load_cached,
    propagate_baseline,
)

TARGET_AXES = (
    ("Balancing time", "h", 1.0 / SECONDS_PER_HOUR),
    ("Energy dissipated", "J", 1.0),
)
Row = dict[str, str | float]


def baseline_table(results: dict[Technique, BalanceResult]) -> list[Row]:
    return [
        {
            "technique": technique.key,
            "balancing_time_s": float(result.balancing_time_s[0]),
            "balancing_time_h": float(result.balancing_time_s[0] / SECONDS_PER_HOUR),
            "energy_dissipated_j": float(result.energy_dissipated_j[0]),
            "efficiency_pct": float(result.efficiency[0] * PERCENT),
        }
        for technique, result in results.items()
    ]


def run_sweeps(pack: PackBatch, config: ProjectConfig) -> dict[str, tuple[FloatArray, FloatArray]]:
    n = config.sweep.n_points
    bleed = config.passive.component
    capacitor = config.flying_capacitor
    r_bleed = np.linspace(bleed.low, bleed.high, n)
    frequency = np.linspace(capacitor.frequency_hz.low, capacitor.frequency_hz.high, n)
    passive = simulate(Technique.PASSIVE, tile_pack(pack, n), r_bleed, np.zeros(n), config)
    flying = simulate(
        Technique.FLYING_CAPACITOR,
        tile_pack(pack, n),
        np.full(n, capacitor.component.mid),
        frequency,
        config,
    )
    return {
        "passive_r_bleed_ohm": (r_bleed, passive.balancing_time_s),
        "flying_capacitor_frequency_hz": (frequency, flying.balancing_time_s),
    }


def sanity_checks(
    results: dict[Technique, BalanceResult], config: ProjectConfig
) -> dict[str, bool | dict[str, float]]:
    times = np.array([results[t].balancing_time_s[0] for t in Technique])
    energy = np.array([results[t].energy_dissipated_j[0] for t in Technique])
    slowdown = {
        t.key: tail_slowdown_ratio(
            results[t], config.sim.soc_tolerance, config.sanity.slowdown_window
        )
        for t in Technique
    }
    capacitor_slowdown = slowdown[Technique.FLYING_CAPACITOR.key]
    return {
        "times_minutes_to_hours": bool(
            all(results[t].balanced[0] for t in Technique)
            and np.all(times >= config.sanity.min_balancing_time_s)
        ),
        "passive_dissipates_most": bool(np.argmax(energy) == Technique.PASSIVE),
        "capacitor_slows_near_end": capacitor_slowdown >= config.sanity.min_capacitor_slowdown
        and capacitor_slowdown > slowdown[Technique.PASSIVE.key],
        "tail_to_head_time_ratio": slowdown,
    }


def save_figures(
    baseline: dict[Technique, BalanceResult],
    sweeps: dict[str, tuple[FloatArray, FloatArray]],
    dataset: Dataset,
    report: PredictorReport,
    config: ProjectConfig,
    results_dir: Path,
) -> None:
    style = config.plot
    spread = f"{config.pack.soc_spread_max * PERCENT:.0f}% initial SOC spread"
    for technique, result in baseline.items():
        plot_soc_trace(
            result,
            technique,
            config.sim.soc_tolerance,
            results_dir / f"soc_{technique.key}",
            style,
        )
    plot_spread_comparison(
        baseline,
        config.sim.soc_tolerance,
        results_dir / "soc_spread_comparison",
        style,
    )
    r_bleed, passive_time = sweeps["passive_r_bleed_ohm"]
    plot_sweep(
        r_bleed,
        passive_time,
        "Bleed resistance R_b (Ω)",
        f"Passive shunt: balancing time vs bleed resistance\n({spread})",
        style.technique_colors[Technique.PASSIVE],
        results_dir / "sweep_passive_r_bleed",
        style,
    )
    frequency, capacitor_time = sweeps["flying_capacitor_frequency_hz"]
    plot_sweep(
        frequency / HZ_PER_KHZ,
        capacitor_time,
        "Switching frequency (kHz)",
        "Flying capacitor: balancing time vs switching frequency\n"
        f"(C = {config.flying_capacitor.component.mid * MF_PER_FARAD:.1f} mF, "
        f"{spread})",
        style.technique_colors[Technique.FLYING_CAPACITOR],
        results_dir / "sweep_flying_capacitor_frequency",
        style,
    )
    for j, (name, (quantity, unit, scale)) in enumerate(
        zip(TARGET_NAMES, TARGET_AXES, strict=True)
    ):
        plot_parity(
            report.y_test[:, j],
            report.y_pred[:, j],
            report.technique_test,
            quantity,
            unit,
            scale,
            report.metrics[name]["overall"]["r2"],
            results_dir / f"ml_parity_{name}",
            style,
        )
        plot_importance(
            dataset.feature_labels,
            report.importance_mean[j],
            report.importance_std[j],
            quantity,
            results_dir / f"ml_importance_{name}",
            style,
        )


def write_baseline_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_csv(dataset: Dataset, path: Path) -> None:
    header = (
        *dataset.feature_names,
        *TARGET_NAMES,
        "technique",
        "efficiency",
        "balanced",
    )
    columns = (
        dataset.features,
        dataset.targets,
        dataset.technique,
        dataset.efficiency,
        dataset.balanced,
    )
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.10g",
    )


def sweep_summary(
    sweeps: dict[str, tuple[FloatArray, FloatArray]],
) -> dict[str, dict[str, list[float] | float]]:
    return {
        name: {
            "values": values.tolist(),
            "balancing_time_s": times.tolist(),
            "min_time_s": float(times.min()),
            "max_time_s": float(times.max()),
        }
        for name, (values, times) in sweeps.items()
    }


def baseline_comparison(
    fitted: dict[Technique, BalanceResult], config: ProjectConfig
) -> list[dict[str, float | str]]:
    assumed_config = replace(config, params=ASSUMED)
    assumed = run_baseline(baseline_pack(assumed_config), assumed_config, trace=False)
    rows: list[dict[str, float | str]] = []
    for technique in Technique:
        old, new = assumed[technique], fitted[technique]
        old_time, new_time = float(old.balancing_time_s[0]), float(new.balancing_time_s[0])
        old_energy = float(old.energy_dissipated_j[0])
        new_energy = float(new.energy_dissipated_j[0])
        rows.append(
            {
                "technique": technique.key,
                "time_s_assumed": old_time,
                "time_s_fitted": new_time,
                "time_ratio": new_time / old_time,
                "time_h_assumed": old_time / SECONDS_PER_HOUR,
                "time_h_fitted": new_time / SECONDS_PER_HOUR,
                "energy_j_assumed": old_energy,
                "energy_j_fitted": new_energy,
                "energy_ratio": new_energy / old_energy,
                "efficiency_pct_assumed": float(old.efficiency[0] * PERCENT),
                "efficiency_pct_fitted": float(new.efficiency[0] * PERCENT),
            }
        )
    return rows


def cell_stages(
    config: ProjectConfig, pack: PackBatch, baseline: dict[Technique, BalanceResult]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, float | str]]]:
    identification = load_cached(IDENTIFICATION_PATH)
    bootstrap = load_cached(BOOTSTRAP_PATH)
    comparison = baseline_comparison(baseline, config)
    slopes = {
        "assumed_mv_per_percent": table_slope_mv_per_percent(ASSUMED.cell),
        "active_mv_per_percent": table_slope_mv_per_percent(config.cell),
    }
    if identification is None:
        ident_block: dict[str, object] = dict(NOT_RUN)
        valid_block: dict[str, object] = dict(NOT_RUN)
    else:
        ident_block = {
            "status": "ok",
            **{k: identification[k] for k in ("capacity", "ocv", "hppc", "intervals")},
        }
        valid_block = {
            "status": identification["selection"]["status"],
            **{k: identification[k] for k in ("selection", "usable_drive_cycles", "validation")},
        }
    if bootstrap is None or config.params.name != "fitted_hg2_25c":
        uncertainty: dict[str, object] = dict(NOT_RUN)
    else:
        uncertainty = {"status": "ok", **propagate_baseline(pack, bootstrap, config)}
    return (
        ident_block,
        valid_block,
        {
            "status": "ok",
            "rows": comparison,
            "ocv_slope_at_50_percent": slopes,
            "parameter_set_active": config.params.name,
            "uncertainty": uncertainty,
        },
        comparison,
    )


def summary_lines(
    table: list[Row],
    sanity: dict[str, bool | dict[str, float]],
    report: PredictorReport,
    speedup: float,
    n_runs: int,
    runtime_s: float,
) -> list[str]:
    lines = ["Baseline (10% SOC spread, mid-range components)"]
    lines += [
        f"  {row['technique']:<17} time {row['balancing_time_h']:6.2f} h  "
        f"dissipated {row['energy_dissipated_j']:9.1f} J  "
        f"efficiency {row['efficiency_pct']:5.1f} %"
        for row in table
    ]
    lines.append(
        "Sanity: " + ", ".join(f"{k}={v}" for k, v in sanity.items() if isinstance(v, bool))
    )
    lines.append(f"ML on {n_runs} simulated runs (test R2 / MAE)")
    lines += [
        f"  {name:<20} R2 {scores['overall']['r2']:.4f}  MAE {scores['overall']['mae']:.2f}"
        for name, scores in report.metrics.items()
    ]
    lines.append(f"Speedup per pack: {speedup:,.0f}x")
    lines.append(f"Total runtime: {runtime_s:.1f} s")
    return lines


def run(config: ProjectConfig, results_dir: Path) -> list[str]:
    start = perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)
    streams = random_streams(config.seed)

    pack = baseline_pack(config)
    baseline = run_baseline(pack, config)
    table = baseline_table(baseline)
    sanity = sanity_checks(baseline, config)
    sweeps = run_sweeps(pack, config)

    generation_start = perf_counter()
    dataset = generate_dataset(config, streams["dataset"])
    n_runs = dataset.features.shape[0]
    simulation_time = (perf_counter() - generation_start) / n_runs
    report = train_and_evaluate(dataset, config.ml, config.seed)
    speedup = simulation_time / report.prediction_time_per_pack_s

    checks_start = perf_counter()
    checks = physics_checks(config, streams["checks"])
    checks_time = perf_counter() - checks_start

    ident_block, valid_block, comparison_block, comparison = cell_stages(config, pack, baseline)
    write_baseline_csv(comparison, results_dir / "baseline_assumed_vs_fitted.csv")
    plot_baseline_comparison(comparison, results_dir / "baseline_assumed_vs_fitted", config.plot)
    write_baseline_csv(table, results_dir / "baseline_table.csv")
    write_dataset_csv(dataset, results_dir / "dataset.csv")
    save_figures(baseline, sweeps, dataset, report, config, results_dir)
    metrics = {
        "seed": config.seed,
        "parameter_set": config.params.name,
        "baseline": table,
        "sanity_checks": sanity,
        "sweeps": sweep_summary(sweeps),
        "dataset": {
            "runs": n_runs,
            "unbalanced_runs": int(n_runs - dataset.balanced.sum()),
        },
        "ml": report.metrics,
        "speedup": {"status": "moved", "location": "timing.json"},
        "physics_checks": checks,
        "cell_identification": ident_block,
        "cell_validation": valid_block,
        "baseline_assumed_vs_fitted": comparison_block,
    }
    write_json(metrics, results_dir / "metrics.json")
    runtime = perf_counter() - start
    timing = {
        "speedup": {
            "simulation_time_per_pack_s": simulation_time,
            "prediction_time_per_pack_s": report.prediction_time_per_pack_s,
            "speedup": speedup,
        },
        "physics_checks_s": checks_time,
        "run_all_s": runtime,
    }
    write_json(timing, results_dir / "timing.json")
    return summary_lines(table, sanity, report, speedup, n_runs, runtime)


def write_json(payload: dict[str, object], path: Path) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    print("\n".join(run(CONFIG, RESULTS_DIR)))


if __name__ == "__main__":
    main()
