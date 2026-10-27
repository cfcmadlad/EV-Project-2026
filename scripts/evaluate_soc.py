import json
import sys
from time import perf_counter

import numpy as np
import torch

from src.config import CONFIG, PERCENT
from src.experiments import random_streams
from src.soc_estimation.data import (
    PROCESSED_DIR,
    SPLITS_PATH,
    TEMPERATURE_PATH,
    Cycle,
    load_split,
    load_splits,
)
from src.soc_estimation.ekf import MODES, build_cell
from src.soc_estimation.evaluate import (
    ErrorSums,
    cycle_rng,
    perturbed_inputs,
    run_coulomb,
    run_ekf_scenario,
    scenarios,
    start_index,
)
from src.soc_estimation.feedthrough import feedthrough_block
from src.soc_estimation.gru import (
    Normaliser,
    SocGru,
    feature_matrix,
    normalise,
    predict,
)
from src.soc_estimation.report import current_inputs

GRU_DIR = PROCESSED_DIR / "soc_gru"
OUTPUT_PATH = PROCESSED_DIR / "soc_estimation_results.json"
GRU_SCENARIOS = ("nominal", "noise_5mv", "offset_tester_accuracy", "offset_injected")
EKF_SCENARIOS = (
    "nominal",
    "wrong_start",
    "noise_5mv",
    "offset_tester_accuracy",
    "offset_injected",
)
CC_SCENARIOS = (
    "offset_tester_accuracy",
    "offset_injected",
    "wrong_start_offset_injected",
)
TUNING_SCENARIOS = ("nominal", "wrong_start", "offset_injected")
NOMINAL_STRIDE = 2
VARIANT_STRIDE = 6
TRACE_STEP = 10
CURVE_SECONDS = 1200
CURVE_STEP = 5
TRACE_CASES = (("test_a", 25, "UDDS"), ("test_b", 10, "UDDS"))


def tune_ekf(entries: dict, validation: list[Cycle], spec) -> dict[str, object]:
    cases = scenarios(spec)
    first = start_index(spec)
    best: dict[str, object] = {"score": float("inf")}
    table: list[dict[str, object]] = []
    for mode in MODES:
        cells = {
            t: build_cell(entries[str(t)], mode) for t in {c.temperature_c for c in validation}
        }
        for q in spec.ekf_process_std_grid:
            for r in spec.ekf_measurement_std_v_grid:
                scores = []
                for name in TUNING_SCENARIOS:
                    errors = []
                    for index, cycle in enumerate(validation):
                        estimate = run_ekf_scenario(
                            cells[cycle.temperature_c],
                            cycle,
                            cases[name],
                            q,
                            r,
                            spec,
                            cycle_rng(CONFIG.seed, index),
                        )
                        errors.append((estimate - cycle.soc)[first:])
                    scores.append(float(np.sqrt(np.mean(np.concatenate(errors) ** 2))))
                score = float(np.mean(scores))
                table.append(
                    {
                        "mode": mode,
                        "process_std": q,
                        "measurement_std_v": r,
                        "validation_rmse_percent": [s * PERCENT for s in scores],
                    }
                )
                if score < best["score"]:
                    best = {"score": score, "mode": mode, "process_std": q, "measurement_std_v": r}
    return {"best": best, "grid": table}


def load_gru_models(log: dict, window_s: int) -> list[tuple[int, SocGru]]:
    models = []
    for run in log["runs"]:
        if run["window_s"] != window_s:
            continue
        model = SocGru(log["hidden_size"])
        model.load_state_dict(torch.load(GRU_DIR / run["file"]))
        models.append((run["seed"], model))
    return models


def gru_estimate(
    model: SocGru,
    normaliser: Normaliser,
    window_s: int,
    cycle: Cycle,
    voltage: np.ndarray,
    current: np.ndarray,
    stride: int,
    first: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = normalise(feature_matrix(voltage, current, cycle.case_temperature_c), normaliser)
    ends = np.arange(first, cycle.n_steps, stride)
    return ends, predict(model, features, ends, window_s)


def downsample(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else float(v) for v in values[TRACE_STEP - 1 :: TRACE_STEP]]


def main() -> int:
    if not SPLITS_PATH.exists() or not (GRU_DIR / "training_log.json").exists():
        print("run scripts/make_soc_splits.py and scripts/train_gru.py first")
        return 1
    start = perf_counter()
    spec = CONFIG.soc_estimation
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    split_file = load_splits()
    splits = split_file["splits"]
    entries = json.loads(TEMPERATURE_PATH.read_text())["per_temperature"]
    validation = load_split(splits["validation"], spec)
    log = json.loads((GRU_DIR / "training_log.json").read_text())
    cases = scenarios(spec)
    first = start_index(spec)

    tuning = tune_ekf(entries, validation, spec)
    best = tuning["best"]
    cells = {int(t): build_cell(e, best["mode"]) for t, e in entries.items()}

    by_window = {
        w: [r["best_validation_rmse_percent"] for r in log["runs"] if r["window_s"] == w]
        for w in spec.windows_s
    }
    window_s = min(by_window, key=lambda w: float(np.mean(by_window[w])))
    models = load_gru_models(log, window_s)
    normaliser = Normaliser(tuple(log["normaliser"]["mean"]), tuple(log["normaliser"]["std"]))
    best_seed = min(
        (r for r in log["runs"] if r["window_s"] == window_s),
        key=lambda r: r["best_validation_rmse_percent"],
    )["seed"]

    sums: dict[tuple, ErrorSums] = {}

    def accumulate(key: tuple, name: str, tail: np.ndarray, first_error: float) -> None:
        sums.setdefault(key, ErrorSums()).add(name, tail, first_error)

    traces: dict[str, object] = {}
    curves: dict[tuple, list[np.ndarray]] = {}
    pools: dict[str, list[np.ndarray]] = {"ekf": [], "gru": []}
    test_cycles = [(s, c) for s in ("test_a", "test_b") for c in load_split(splits[s], spec)]
    for index, (split, cycle) in enumerate(test_cycles):
        label = f"{cycle.temperature_c}_{cycle.cycle}"
        truth = cycle.soc
        keys = ((split, cycle.temperature_c), (split, "all"))
        for name in CC_SCENARIOS:
            error = run_coulomb(cycle, cases[name], spec) - truth
            for key in keys:
                accumulate(("cc", name, *key), label, error[first:], error[first])
            if name == "wrong_start_offset_injected":
                curves.setdefault(("cc", split), []).append(
                    np.abs(error[:CURVE_SECONDS:CURVE_STEP])
                )
        estimates: dict[str, np.ndarray] = {}
        for name in EKF_SCENARIOS:
            estimate = run_ekf_scenario(
                cells[cycle.temperature_c],
                cycle,
                cases[name],
                best["process_std"],
                best["measurement_std_v"],
                spec,
                cycle_rng(CONFIG.seed, index),
            )
            error = estimate - truth
            estimates[name] = estimate
            for key in keys:
                accumulate(("ekf", name, *key), label, error[first:], error[first])
            if name == "wrong_start":
                curves.setdefault(("ekf", split), []).append(
                    np.abs(error[:CURVE_SECONDS:CURVE_STEP])
                )
            if name == "nominal":
                window = (truth >= spec.pool_window[0]) & (truth <= spec.pool_window[1])
                window[:first] = False
                pools["ekf"].append(error[window])
        gru_estimates: dict[int, np.ndarray] = {}
        for name in GRU_SCENARIOS:
            current, voltage = perturbed_inputs(cycle, cases[name], cycle_rng(CONFIG.seed, index))
            stride = NOMINAL_STRIDE if name == "nominal" else VARIANT_STRIDE
            for seed, model in models:
                ends, prediction = gru_estimate(
                    model, normaliser, window_s, cycle, voltage, current, stride, first
                )
                error = prediction - truth[ends]
                for key in keys:
                    accumulate(("gru", name, seed, *key), label, error, error[0])
                if name == "nominal":
                    full = np.full(cycle.n_steps, np.nan)
                    full[ends] = prediction
                    gru_estimates[seed] = full
                    if seed == best_seed:
                        window = (truth[ends] >= spec.pool_window[0]) & (
                            truth[ends] <= spec.pool_window[1]
                        )
                        pools["gru"].append(error[window])
        if (split, cycle.temperature_c, cycle.cycle) in TRACE_CASES:
            traces[f"{split}_{label}"] = {
                "step_s": TRACE_STEP,
                "true": downsample(truth),
                "coulomb_offset_injected": downsample(
                    run_coulomb(cycle, cases["offset_injected"], spec)
                ),
                "ekf": downsample(estimates["nominal"]),
                "gru": downsample(gru_estimates[best_seed]),
            }

    def leaf(key: tuple) -> dict[str, object]:
        return sums[key].leaf()

    results: dict[str, object] = {"cc": {}, "ekf": {}, "gru": {}}
    temperatures = {"test_a": (*spec.train_temperatures_c, "all"), "test_b": (10, "all")}
    for name in CC_SCENARIOS:
        results["cc"][name] = {
            s: {str(t): leaf(("cc", name, s, t)) for t in ts} for s, ts in temperatures.items()
        }
    for name in EKF_SCENARIOS:
        results["ekf"][name] = {
            s: {str(t): leaf(("ekf", name, s, t)) for t in ts if ("ekf", name, s, t) in sums}
            for s, ts in temperatures.items()
        }
    for name in GRU_SCENARIOS:
        results["gru"][name] = {}
        for s, ts in temperatures.items():
            results["gru"][name][s] = {}
            for t in ts:
                if ("gru", name, models[0][0], s, t) not in sums:
                    continue
                per_seed = [leaf(("gru", name, seed, s, t)) for seed, _ in models]
                aggregated = {"n_seeds": len(per_seed), "per_seed": per_seed}
                for field in (
                    "rmse_percent",
                    "mae_percent",
                    "max_abs_percent",
                    "abs_error_at_300s_percent_mean",
                ):
                    values = np.array([p[field] for p in per_seed])
                    aggregated[field] = float(values.mean())
                    aggregated[f"{field}_std"] = float(values.std(ddof=0))
                results["gru"][name][s][str(t)] = aggregated

    pool_arrays = {k: np.concatenate(v) for k, v in pools.items()}
    stream = random_streams(CONFIG.seed)["soc_feedthrough"]
    through = feedthrough_block(CONFIG, pool_arrays, stream)
    curve_block = {
        f"{est}_{split}": {
            "step_s": CURVE_STEP,
            "mean_abs_error_percent": (np.mean(rows, axis=0) * PERCENT).tolist(),
        }
        for (est, split), rows in curves.items()
    }
    runs = log["runs"]
    payload = {
        "inputs": current_inputs(),
        "splits_summary": {k: len(v) for k, v in splits.items()},
        "ekf": {
            "mode": best["mode"],
            "process_std": best["process_std"],
            "measurement_std_v": best["measurement_std_v"],
            "tuning_score_percent": best["score"] * PERCENT,
            "tuning_set": [c.name for c in validation],
            "grid_size": len(tuning["grid"]),
        },
        "gru": {
            "selected_window_s": window_s,
            "validation_rmse_by_window_percent": {
                str(w): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for w, v in by_window.items()
            },
            "hidden_size": log["hidden_size"],
            "parameters": runs[0]["parameters"],
            "macs_per_estimate": {
                str(w): next(r["macs_per_estimate"] for r in runs if r["window_s"] == w)
                for w in spec.windows_s
            },
            "weight_bytes_max": max(r["weight_bytes"] for r in runs),
            "seeds": list(spec.seeds),
            "best_seed": best_seed,
            "training_time_s": {
                f"w{r['window_s']}_s{r['seed']}": r["training_time_s"] for r in runs
            },
            "epochs_run": {f"w{r['window_s']}_s{r['seed']}": r["epochs_run"] for r in runs},
        },
        "results": results,
        "feedthrough": through,
        "traces": traces,
        "wrong_start_curves": curve_block,
        "label_note": split_file["definition"]["label_accuracy_note"],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(f"wrote {OUTPUT_PATH.name} in {perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
