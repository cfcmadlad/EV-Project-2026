import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.balancing import BalanceResult, simulate, tail_slowdown_ratio
from src.config import (
    CONFIG,
    HZ_PER_KHZ,
    MF_PER_FARAD,
    PERCENT,
    RESULTS_DIR,
    SECONDS_PER_HOUR,
    FloatArray,
    ProjectConfig,
    Technique,
)
from src.dataset import TARGET_NAMES, Dataset, generate_dataset
from src.pack import PackBatch, sample_packs, tile_pack
from src.plots import (
    plot_importance,
    plot_parity,
    plot_soc_trace,
    plot_spread_comparison,
    plot_sweep,
)
from src.predictor import PredictorReport, train_and_evaluate

TARGET_AXES = (
    ("Balancing time", "h", 1.0 / SECONDS_PER_HOUR),
    ("Energy dissipated", "J", 1.0),
)
Row = dict[str, str | float]


def run_baseline(
    pack: PackBatch, config: ProjectConfig
) -> dict[Technique, BalanceResult]:
    return {
        technique: simulate(
            technique,
            pack,
            np.array([config.technique(technique).component.mid]),
            np.array([config.technique(technique).frequency_hz.mid]),
            config,
            trace=True,
        )
        for technique in Technique
    }


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


def run_sweeps(
    pack: PackBatch, config: ProjectConfig
) -> dict[str, tuple[FloatArray, FloatArray]]:
    n = config.sweep.n_points
    bleed = config.passive.component
    capacitor = config.flying_capacitor
    r_bleed = np.linspace(bleed.low, bleed.high, n)
    frequency = np.linspace(capacitor.frequency_hz.low, capacitor.frequency_hz.high, n)
    passive = simulate(
        Technique.PASSIVE, tile_pack(pack, n), r_bleed, np.zeros(n), config
    )
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
        "capacitor_slows_near_end": capacitor_slowdown
        >= config.sanity.min_capacitor_slowdown
        and capacitor_slowdown > slowdown[Technique.PASSIVE.key],
        "tail_to_head_time_ratio": slowdown,
    }


def save_figures(
    baseline: dict[Technique, BalanceResult],
    sweeps: dict[str, tuple[FloatArray, FloatArray]],
    dataset: Dataset,
    report: PredictorReport,
    config: ProjectConfig,
) -> None:
    style = config.plot
    spread = f"{config.pack.soc_spread_max * PERCENT:.0f}% initial SOC spread"
    for technique, result in baseline.items():
        plot_soc_trace(
            result,
            technique,
            config.sim.soc_tolerance,
            RESULTS_DIR / f"soc_{technique.key}",
            style,
        )
    plot_spread_comparison(
        baseline,
        config.sim.soc_tolerance,
        RESULTS_DIR / "soc_spread_comparison",
        style,
    )
    r_bleed, passive_time = sweeps["passive_r_bleed_ohm"]
    plot_sweep(
        r_bleed,
        passive_time,
        "Bleed resistance R_b (Ω)",
        f"Passive shunt: balancing time vs bleed resistance\n({spread})",
        style.technique_colors[Technique.PASSIVE],
        RESULTS_DIR / "sweep_passive_r_bleed",
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
        RESULTS_DIR / "sweep_flying_capacitor_frequency",
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
            RESULTS_DIR / f"ml_parity_{name}",
            style,
        )
        plot_importance(
            dataset.feature_labels,
            report.importance_mean[j],
            report.importance_std[j],
            quantity,
            RESULTS_DIR / f"ml_importance_{name}",
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
        "Sanity: "
        + ", ".join(f"{k}={v}" for k, v in sanity.items() if isinstance(v, bool))
    )
    lines.append(f"ML on {n_runs} simulated runs (test R2 / MAE)")
    lines += [
        f"  {name:<20} R2 {scores['overall']['r2']:.4f}  "
        f"MAE {scores['overall']['mae']:.2f}"
        for name, scores in report.metrics.items()
    ]
    lines.append(f"Speedup per pack: {speedup:,.0f}x")
    lines.append(f"Total runtime: {runtime_s:.1f} s")
    return lines


def main() -> None:
    start = perf_counter()
    config = CONFIG
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_rng, dataset_rng = np.random.default_rng(config.seed).spawn(2)

    pack = sample_packs(
        1,
        baseline_rng,
        config.cell,
        config.pack,
        np.array([config.pack.soc_spread_max]),
    )
    baseline = run_baseline(pack, config)
    table = baseline_table(baseline)
    sanity = sanity_checks(baseline, config)
    sweeps = run_sweeps(pack, config)

    generation_start = perf_counter()
    dataset = generate_dataset(config, dataset_rng)
    n_runs = dataset.features.shape[0]
    simulation_time = (perf_counter() - generation_start) / n_runs
    report = train_and_evaluate(dataset, config.model, config.seed)
    speedup = simulation_time / report.prediction_time_per_pack_s

    write_baseline_csv(table, RESULTS_DIR / "baseline_table.csv")
    write_dataset_csv(dataset, RESULTS_DIR / "dataset.csv")
    save_figures(baseline, sweeps, dataset, report, config)
    metrics = {
        "seed": config.seed,
        "baseline": table,
        "sanity_checks": sanity,
        "sweeps": sweep_summary(sweeps),
        "dataset": {
            "runs": n_runs,
            "unbalanced_runs": int(n_runs - dataset.balanced.sum()),
        },
        "ml": report.metrics,
        "speedup": {
            "simulation_time_per_pack_s": simulation_time,
            "prediction_time_per_pack_s": report.prediction_time_per_pack_s,
            "speedup": speedup,
        },
    }
    with (RESULTS_DIR / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    runtime = perf_counter() - start
    print("\n".join(summary_lines(table, sanity, report, speedup, n_runs, runtime)))


if __name__ == "__main__":
    main()
