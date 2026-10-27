import csv
import json
from concurrent.futures import ProcessPoolExecutor
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
from src.drive_cycle import drive_cycle_stage
from src.drive_plots import plot_drive_capacity, plot_drive_spread
from src.experiments import baseline_pack, random_streams, run_baseline
from src.identification import table_slope_mv_per_percent
from src.ml_eval import evaluate
from src.ml_plots import plot_conformal_coverage, plot_cv_comparison, plot_out_of_range
from src.pack import PackBatch, tile_pack
from src.params import ASSUMED, SECONDS_PER_HOUR, FlyingCapModel
from src.plots import (
    plot_importance,
    plot_parity,
    plot_soc_trace,
    plot_spread_comparison,
    plot_sweep,
)
from src.predictor import PredictorReport, train_and_evaluate
from src.rl_plots import plot_rl_vs_baselines, plot_training_curves
from src.rl_report import RESULTS_PATH as RL_RESULTS_PATH
from src.rl_report import rl_block
from src.rl_report import timing_block as rl_timing_block
from src.scaling import scaling_block
from src.scaling_plots import plot_scaling
from src.soc_error import soc_error_block
from src.soc_estimation.plots import plot_error_summary, plot_traces, plot_wrong_start
from src.soc_estimation.report import RESULTS_PATH as SOC_RESULTS_PATH
from src.soc_estimation.report import soc_estimation_block, timing_block
from src.spice_check import converter_check
from src.spice_plots import plot_current_parity
from src.temperature import temperature_block
from src.temperature_plots import plot_params_vs_temperature, plot_soc_error
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


def passive_sweep(pack: PackBatch, config: ProjectConfig) -> tuple[FloatArray, FloatArray]:
    n = config.sweep.n_points
    bleed = config.passive.component
    r_bleed = np.linspace(bleed.low, bleed.high, n)
    passive = simulate(Technique.PASSIVE, tile_pack(pack, n), r_bleed, np.zeros(n), config)
    return r_bleed, passive.balancing_time_s


def flying_sweep(pack: PackBatch, config: ProjectConfig) -> tuple[FloatArray, FloatArray]:
    n = config.sweep.n_points
    capacitor = config.flying_capacitor
    frequency = np.linspace(capacitor.frequency_hz.low, capacitor.frequency_hz.high, n)
    flying = simulate(
        Technique.FLYING_CAPACITOR,
        tile_pack(pack, n),
        np.full(n, capacitor.component.mid),
        frequency,
        config,
    )
    return frequency, flying.balancing_time_s


def run_sweeps(pack: PackBatch, config: ProjectConfig) -> dict[str, tuple[FloatArray, FloatArray]]:
    return {
        "passive_r_bleed_ohm": passive_sweep(pack, config),
        "flying_capacitor_frequency_hz": flying_sweep(pack, config),
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


def with_flying_model(config: ProjectConfig, model: FlyingCapModel) -> ProjectConfig:
    spec = replace(config.flying_capacitor, model=model)
    return replace(config, params=replace(config.params, flying_capacitor=spec))


def flying_model_comparison(
    pack: PackBatch,
    config: ProjectConfig,
    active_result: BalanceResult,
    active_sweep: tuple[FloatArray, FloatArray],
) -> dict[str, object]:
    baseline: dict[str, dict[str, float]] = {}
    sweep: dict[str, list[float]] = {}
    for model in FlyingCapModel:
        if model is config.flying_capacitor.model:
            result, (values, times) = active_result, active_sweep
        else:
            variant = with_flying_model(config, model)
            hardware = variant.flying_capacitor
            result = simulate(
                Technique.FLYING_CAPACITOR,
                pack,
                np.array([hardware.component.mid]),
                np.array([hardware.frequency_hz.mid]),
                variant,
            )
            values, times = flying_sweep(pack, variant)
        baseline[model.value] = {
            "balancing_time_s": float(result.balancing_time_s[0]),
            "energy_dissipated_j": float(result.energy_dissipated_j[0]),
            "efficiency_pct": float(result.efficiency[0] * PERCENT),
        }
        sweep["frequency_hz"] = values.tolist()
        sweep[f"balancing_time_s_{model.value}"] = times.tolist()
    old, new = baseline[FlyingCapModel.MIDSEM.value], baseline[FlyingCapModel.TWO_PHASE.value]
    return {
        "status": "ok",
        "parameter_set": config.params.name,
        "baseline": baseline,
        "time_ratio_two_phase_over_midsem": new["balancing_time_s"] / old["balancing_time_s"],
        "energy_ratio_two_phase_over_midsem": new["energy_dissipated_j"]
        / old["energy_dissipated_j"],
        "frequency_sweep": sweep,
        "active_model": config.flying_capacitor.model.value,
    }


def write_comparison_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()}
            for row in rows
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


def timed_physics_checks(
    config: ProjectConfig, rng: np.random.Generator
) -> tuple[dict[str, object], float]:
    start = perf_counter()
    checks = physics_checks(config, rng)
    return checks, perf_counter() - start


def write_ml_eval_figures(
    metrics: dict[str, object], extra: dict[str, object], config: ProjectConfig, results_dir: Path
) -> None:
    spec = config.ml_eval
    plot_cv_comparison(metrics["cross_validation"], config.plot, results_dir / "ml_cv_comparison")
    plot_out_of_range(
        extra["out_of_range_arrays"], extra["data"], config.plot, results_dir / "ml_out_of_range"
    )
    plot_conformal_coverage(
        metrics["conformal"],
        config.plot,
        spec.coverage_low,
        spec.coverage_high,
        results_dir / "ml_conformal_coverage",
    )


def run(config: ProjectConfig, results_dir: Path) -> list[str]:
    start = perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)
    streams = random_streams(config.seed)

    with ProcessPoolExecutor(max_workers=1) as pool:
        checks_future = pool.submit(timed_physics_checks, config, streams["checks"])
        return _run_stages(config, results_dir, streams, start, checks_future)


def _run_stages(
    config: ProjectConfig,
    results_dir: Path,
    streams: dict[str, np.random.Generator],
    start: float,
    checks_future,
) -> list[str]:
    pack = baseline_pack(config)
    baseline = run_baseline(pack, config)
    table = baseline_table(baseline)
    sanity = sanity_checks(baseline, config)
    sweeps = run_sweeps(pack, config)

    dataset = generate_dataset(config, streams["dataset"])
    n_runs = dataset.features.shape[0]
    simulation_time = dataset.simulation_s / n_runs
    report = train_and_evaluate(dataset, config.ml, config.seed)
    speedup = simulation_time / report.prediction_time_per_pack_s

    ml_eval_block, ml_extra, ml_timing = evaluate(dataset, config, config.seed)
    write_ml_eval_figures(ml_eval_block, ml_extra, config, results_dir)
    checks, checks_time = checks_future.result()

    ident_block, valid_block, comparison_block, comparison = cell_stages(config, pack, baseline)
    write_baseline_csv(comparison, results_dir / "baseline_assumed_vs_fitted.csv")
    plot_baseline_comparison(comparison, results_dir / "baseline_assumed_vs_fitted", config.plot)
    converter_block, converter_rows = converter_check(config)
    if converter_rows:
        write_comparison_csv(converter_rows, results_dir / "spice_vs_averaged.csv")
        plot_current_parity(converter_rows, config, results_dir / "spice_vs_averaged")
    drive_block, drive_traces = drive_cycle_stage(config, pack, baseline)
    if drive_traces is not None:
        plot_drive_spread(
            drive_traces, config.sim.soc_tolerance, results_dir / "drive_cycle_spread", config.plot
        )
        plot_drive_capacity(drive_block, results_dir / "drive_cycle_usable_capacity", config.plot)
    temperature_result = temperature_block(config)
    if temperature_result["status"] == "ok":
        plot_params_vs_temperature(
            temperature_result, results_dir / "params_vs_temperature", config.plot
        )
    soc_error_result, soc_error_study = soc_error_block(config, streams["soc_error"])
    plot_soc_error(
        soc_error_study,
        config.soc_error.sigma_percent,
        config.sim.soc_tolerance,
        results_dir / "soc_error_true_spread",
        config.plot,
    )
    soc_estimation_result = soc_estimation_block(config)
    if soc_estimation_result["status"] == "ok":
        soc_cache = json.loads(SOC_RESULTS_PATH.read_text())
        plot_traces(soc_cache["traces"], results_dir / "soc_est_test_traces", config.plot)
        plot_error_summary(
            soc_estimation_result["results"], results_dir / "soc_est_error_summary", config.plot
        )
        plot_wrong_start(
            soc_cache["wrong_start_curves"], results_dir / "soc_est_wrong_init", config.plot
        )
    scaling_result = scaling_block(config)
    if scaling_result["status"] == "ok":
        plot_scaling(scaling_result, results_dir / "scaling_time_vs_n", config.plot)
    rl_result = rl_block(config)
    if rl_result["status"] == "ok":
        rl_cache = json.loads(RL_RESULTS_PATH.read_text())
        plot_training_curves(
            rl_cache["training"]["curves"], results_dir / "rl_training_curves", config.plot
        )
        plot_rl_vs_baselines(rl_cache["test"], results_dir / "rl_vs_baselines", config.plot)
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
        "converter_check": converter_block,
        "flying_capacitor_model_comparison": flying_model_comparison(
            pack,
            config,
            baseline[Technique.FLYING_CAPACITOR],
            sweeps["flying_capacitor_frequency_hz"],
        ),
        "ml_eval": ml_eval_block,
        "drive_cycle": drive_block,
        "temperature": temperature_result,
        "soc_error": soc_error_result,
        "soc_estimation": soc_estimation_result,
        "scaling": scaling_result,
        "rl_control": rl_result,
    }
    write_json(metrics, results_dir / "metrics.json")
    runtime = perf_counter() - start
    timing = {
        "speedup": {
            "simulation_time_per_pack_s": simulation_time,
            "prediction_time_per_pack_s": report.prediction_time_per_pack_s,
            "speedup": speedup,
        },
        "ml_eval": ml_timing,
        "soc_estimation": timing_block(),
        "rl_control": rl_timing_block(),
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
