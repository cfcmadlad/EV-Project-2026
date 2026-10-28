import json
import os
import sys
from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.config import CONFIG, PERCENT
from src.experiments import random_streams
from src.quantize import quantize_state
from src.soc_estimation.data import PROCESSED_DIR, TEMPERATURE_PATH, load_split, load_splits
from src.soc_estimation.ekf2 import build_cell2
from src.soc_estimation.evaluate import ErrorSums, cycle_rng, scenarios, start_index
from src.soc_estimation.feedthrough import feedthrough_block
from src.soc_estimation.gru import gather
from src.soc_estimation.hybrid import (
    HYBRID_INPUTS,
    PLAIN_INPUTS,
    Ekf2Params,
    WindowGru,
    augmentation_scenarios,
    build_sequence,
    estimate,
    macs_per_estimate,
    parameter_count,
    stack,
    to_device,
)
from src.soc_estimation.report import current_inputs

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
OUT = PROCESSED_DIR / "soc_deep_results.json"
CHECKPOINTS = PROCESSED_DIR / "soc_hybrid"
QUANT_BITS = (8, 4, 1)
SCENARIOS = ("nominal", "noise_5mv", "offset_tester_accuracy", "offset_injected")
NOMINAL_STRIDE = 2
VARIANT_STRIDE = 6
VALIDATION_STRIDE = 5
PATIENCE = 4
DECAY = 0.8
KINDS = {"gru_augmented": PLAIN_INPUTS, "ekf_gru_hybrid": HYBRID_INPUTS}


def train(kind, window_s, seed, data, device, spec):
    features, base, target, ends = data["train"]
    v_features, v_base, v_target, v_ends = data["validation"]
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = WindowGru(KINDS[kind], spec.hidden_size).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, DECAY)
    x = to_device(features, device)
    y = to_device(target - base, device)
    xv = to_device(v_features, device)
    best, state, stale, history = float("inf"), None, 0, []
    start = perf_counter()
    for _ in range(spec.epochs):
        model.train()
        order = generator.permutation(ends.size)
        for first in range(0, order.size, spec.batch_size):
            batch = ends[order[first : first + spec.batch_size]]
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(gather(x, batch, window_s)), y[batch])
            loss.backward()
            optimiser.step()
        scheduler.step()
        prediction = estimate(model, xv, v_base, v_ends, window_s)
        score = float(np.sqrt(np.mean((prediction - v_target[v_ends]) ** 2)))
        history.append(score)
        if score < best:
            best, stale = score, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return state, best, len(history), perf_counter() - start


def prepare(kind, sequences, normaliser, window_s, stride):
    n = KINDS[kind]
    features, base, target, ends = stack(sequences, window_s, stride)
    scaled = ((features - normaliser[0]) / normaliser[1])[:, :n]
    if kind != "ekf_gru_hybrid":
        base = np.zeros_like(base)
    return scaled, base, target, ends


def main() -> int:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = CONFIG.soc_estimation
    splits = load_splits()["splits"]
    entries = json.loads(TEMPERATURE_PATH.read_text())["per_temperature"]
    cells = {int(t): build_cell2(e) for t, e in entries.items()}
    reference = json.loads((PROCESSED_DIR / "soc_estimation_results.json").read_text())
    params = Ekf2Params(
        reference["ekf2"]["process_std"],
        reference["ekf2"]["measurement_std_v"],
        reference["ekf2"]["hysteresis_rate"],
    )
    cases = scenarios(spec)
    first = start_index(spec)
    train_cycles = load_split(splits["train"], spec)
    val_cycles = load_split(splits["validation"], spec)
    aug_rng = np.random.default_rng(np.random.SeedSequence([CONFIG.seed, 7]))
    augmented = augmentation_scenarios(spec, aug_rng)
    train_sequences, val_sequences = [], []
    for index, cycle in enumerate(train_cycles):
        for scenario in [cases["nominal"], *augmented]:
            train_sequences.append(
                build_sequence(
                    cycle,
                    scenario,
                    cycle_rng(CONFIG.seed, 100 + index),
                    cells[cycle.temperature_c],
                    params,
                    spec,
                    True,
                )
            )
    for index, cycle in enumerate(val_cycles):
        val_sequences.append(
            build_sequence(
                cycle,
                cases["nominal"],
                cycle_rng(CONFIG.seed, 200 + index),
                cells[cycle.temperature_c],
                params,
                spec,
                True,
            )
        )
    stacked = np.concatenate([s.features for s in train_sequences])
    normaliser = (stacked.mean(axis=0), stacked.std(axis=0))

    models, runs = {}, []
    for kind in KINDS:
        for window_s in spec.windows_s:
            data = {
                "train": prepare(kind, train_sequences, normaliser, window_s, spec.train_stride_s),
                "validation": prepare(kind, val_sequences, normaliser, window_s, VALIDATION_STRIDE),
            }
            for seed in spec.seeds:
                state, score, epochs, seconds = train(kind, window_s, seed, data, device, spec)
                model = WindowGru(KINDS[kind], spec.hidden_size)
                model.load_state_dict(state)
                models[(kind, window_s, seed)] = model.eval()
                runs.append(
                    {
                        "kind": kind,
                        "window_s": window_s,
                        "seed": seed,
                        "validation_rmse_percent": score * PERCENT,
                        "epochs": epochs,
                        "training_time_s": seconds,
                    }
                )
                print(kind, window_s, seed, f"{score * PERCENT:.3f}", flush=True)

    selected = {}
    for kind in KINDS:
        by_window = {
            w: np.mean(
                [
                    r["validation_rmse_percent"]
                    for r in runs
                    if r["kind"] == kind and r["window_s"] == w
                ]
            )
            for w in spec.windows_s
        }
        selected[kind] = int(min(by_window, key=by_window.get))

    sums: dict[tuple, ErrorSums] = {}
    nominal_cache = []
    pools = {"ekf2": [], **{kind: [] for kind in KINDS}}
    best_seed = {
        kind: min(
            (r for r in runs if r["kind"] == kind and r["window_s"] == selected[kind]),
            key=lambda r: r["validation_rmse_percent"],
        )["seed"]
        for kind in KINDS
    }
    for index, (split, cycle) in enumerate(
        [(s, c) for s in ("test_a", "test_b") for c in load_split(splits[s], spec)]
    ):
        truth = cycle.soc
        for name in SCENARIOS:
            sequence = build_sequence(
                cycle,
                cases[name],
                cycle_rng(CONFIG.seed, index),
                cells[cycle.temperature_c],
                params,
                spec,
                True,
            )
            stride = NOMINAL_STRIDE if name == "nominal" else VARIANT_STRIDE
            if name == "nominal":
                nominal_cache.append((cycle, sequence))
            label = f"{cycle.temperature_c}_{cycle.cycle}"
            ends = np.arange(first, cycle.n_steps, stride)
            ekf_error = sequence.base - truth
            sums.setdefault(("ekf2", name, split), ErrorSums()).add(
                label, ekf_error[ends], ekf_error[first]
            )
            for kind in KINDS:
                window_s = selected[kind]
                scaled = to_device(
                    ((sequence.features - normaliser[0]) / normaliser[1])[:, : KINDS[kind]], device
                )
                base = sequence.base if kind == "ekf_gru_hybrid" else np.zeros_like(sequence.base)
                ends = np.arange(first, cycle.n_steps, stride)
                for seed in CONFIG.soc_estimation.seeds:
                    model = models[(kind, window_s, seed)].to(device)
                    prediction = estimate(model, scaled, base, ends, window_s)
                    error = prediction - truth[ends]
                    sums.setdefault((kind, name, seed, split), ErrorSums()).add(
                        label, error, error[0]
                    )
                    if name == "nominal" and seed == best_seed[kind]:
                        window = (truth[ends] >= spec.pool_window[0]) & (
                            truth[ends] <= spec.pool_window[1]
                        )
                        pools[kind].append(error[window])
            if name == "nominal":
                window = (truth >= spec.pool_window[0]) & (truth <= spec.pool_window[1])
                window[:first] = False
                pools["ekf2"].append(ekf_error[window])

    results = {"ekf2": {}, **{kind: {} for kind in KINDS}}
    for name in SCENARIOS:
        results["ekf2"][name] = {s: sums[("ekf2", name, s)].leaf() for s in ("test_a", "test_b")}
        for kind in KINDS:
            results[kind][name] = {}
            for s in ("test_a", "test_b"):
                per_seed = [sums[(kind, name, seed, s)].leaf() for seed in spec.seeds]
                block = {"per_seed": per_seed}
                for field in ("rmse_percent", "mae_percent", "max_abs_percent"):
                    values = np.array([p[field] for p in per_seed])
                    block[field] = float(values.mean())
                    block[f"{field}_std"] = float(values.std())
                results[kind][name][s] = block
    stream = random_streams(CONFIG.seed)["soc_feedthrough"]
    through = feedthrough_block(CONFIG, {k: np.concatenate(v) for k, v in pools.items()}, stream)

    quantised = {}
    for kind in KINDS:
        window_s = selected[kind]
        state = models[(kind, window_s, best_seed[kind])].cpu().state_dict()
        quantised[kind] = {}
        for bits in (32, *QUANT_BITS):
            deq, size = (
                (state, sum(t.numel() for t in state.values()) * 4)
                if bits == 32
                else quantize_state(state, bits)
            )
            model = WindowGru(KINDS[kind], spec.hidden_size)
            model.load_state_dict(deq)
            model.eval()
            sq, count = 0.0, 0
            for cycle, sequence in nominal_cache:
                scaled = to_device(
                    ((sequence.features - normaliser[0]) / normaliser[1])[:, : KINDS[kind]],
                    torch.device("cpu"),
                )
                base = sequence.base if kind == "ekf_gru_hybrid" else np.zeros_like(sequence.base)
                ends = np.arange(first, cycle.n_steps, NOMINAL_STRIDE)
                error = estimate(model, scaled, base, ends, window_s) - cycle.soc[ends]
                sq += float(np.sum(error**2))
                count += error.size
            quantised[kind][str(bits)] = {
                "weight_bytes": size,
                "test_rmse_percent": float(np.sqrt(sq / count)) * PERCENT,
            }

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for (kind, window_s, seed), model in models.items():
        if window_s == selected[kind]:
            torch.save(model.cpu().state_dict(), CHECKPOINTS / f"{kind}_w{window_s}_s{seed}.pt")
    payload = {
        "inputs": current_inputs(),
        "ekf2_params": asdict(params),
        "augmentation": {
            "copies": len(augmented),
            "offsets_a": [s.offset_a for s in augmented],
            "noise_v": spec.voltage_noise_v,
        },
        "selected_window_s": selected,
        "best_seed": best_seed,
        "parameters": {k: parameter_count(n, spec.hidden_size) for k, n in KINDS.items()},
        "macs_per_estimate": {
            k: macs_per_estimate(n, spec.hidden_size, selected[k]) for k, n in KINDS.items()
        },
        "runs": runs,
        "quantisation": quantised,
        "results": results,
        "feedthrough": through,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
