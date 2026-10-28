import json
import os
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.config import CONFIG, PROJECT_ROOT, Technique
from src.deep_surrogate import (
    BATCH_SIZE,
    CONFIGS,
    MAX_EPOCHS,
    PATIENCE,
    SEEDS,
    WEIGHT_DECAY,
    fit_scaler,
    make_split,
    parameter_count,
    seed_everything,
    summarise,
)
from src.experiments import random_streams
from src.fingerprint import maps_sha256
from src.trajectory import DECODERS, GRID, Trajectories, trajectory_dataset, trajectory_scores

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CACHE = PROJECT_ROOT / "data" / "raw" / "trajectories.npz"
OUT = PROJECT_ROOT / "data" / "processed" / "trajectory_results.json"
RUNS_PER_TECHNIQUE = 1000
EXAMPLES_PER_TECHNIQUE = 3
TIMING_REPEATS = 20


def load_or_build() -> Trajectories:
    if CACHE.is_file():
        data = np.load(CACHE)
        return Trajectories(data["features"], data["technique"], data["time_s"], data["curves"])
    rng = random_streams(CONFIG.seed)["trajectory"]
    data = trajectory_dataset(CONFIG, rng, RUNS_PER_TECHNIQUE)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        features=data.features,
        technique=data.technique,
        time_s=data.time_s,
        curves=data.curves,
    )
    return data


def train(kind, width, lr, seed, x, y, t, xv, yv, tv, device):
    seed_everything(seed)
    model = DECODERS[kind](x.shape[1], width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    x, y, t, xv, yv, tv = (a.to(device) for a in (x, y, t, xv, yv, tv))
    best, state, stale = float("inf"), None, 0
    start, epochs = perf_counter(), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        epochs = epoch
        model.train()
        order = torch.randperm(x.shape[0], generator=generator).to(device)
        for first in range(0, order.numel(), BATCH_SIZE):
            b = order[first : first + BATCH_SIZE]
            optimiser.zero_grad()
            curves, time = model(x[b])
            loss = nn.functional.mse_loss(curves, y[b]) + nn.functional.mse_loss(time, t[b])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            curves, time = model(xv)
            score = float(nn.functional.mse_loss(curves, yv) + nn.functional.mse_loss(time, tv))
        if score < best:
            best, stale = score, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return state, best, epochs, perf_counter() - start


def main() -> int:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_or_build()
    split = make_split(data.technique, CONFIG.seed, CONFIG.ml.test_fraction)
    x_scaler = fit_scaler(data.features[split.train])
    channel = data.curves[split.train].reshape(-1, data.curves.shape[2])
    c_scaler = fit_scaler(channel)
    log_t = np.log(data.time_s)
    t_scaler = fit_scaler(log_t[split.train][:, None])

    def prep(rows):
        return (
            torch.tensor(x_scaler(data.features[rows])),
            torch.tensor(c_scaler(data.curves[rows])),
            torch.tensor(t_scaler(log_t[rows][:, None])[:, 0]),
        )

    train_set, val_set, test_set = prep(split.train), prep(split.validation), prep(split.test)
    results = {}
    examples = {}
    for kind in DECODERS:
        best, best_config = float("inf"), CONFIGS[0]
        for width, lr in CONFIGS:
            _, score, _, _ = train(kind, width, lr, SEEDS[0], *train_set, *val_set, device)
            if score < best:
                best, best_config = score, (width, lr)
        width, lr = best_config
        per_seed, seconds = [], []
        for seed in SEEDS:
            state, _, _, took = train(kind, width, lr, seed, *train_set, *val_set, device)
            model = DECODERS[kind](train_set[0].shape[1], width)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                curves, time = model(test_set[0])
            pred_curves = curves.numpy() * c_scaler.std + c_scaler.mean
            pred_time = np.exp(time.numpy() * t_scaler.std[0] + t_scaler.mean[0])
            per_seed.append(
                trajectory_scores(
                    data.curves[split.test],
                    pred_curves,
                    data.time_s[split.test],
                    pred_time,
                    data.technique[split.test],
                )
            )
            seconds.append(took)
            if seed == SEEDS[0]:
                examples[kind] = pick_examples(data, split.test, pred_curves, pred_time)
        times = []
        for _ in range(TIMING_REPEATS):
            start = perf_counter()
            with torch.no_grad():
                model(test_set[0])
            times.append(perf_counter() - start)
        results[kind] = {
            "metrics": summarise([{"all": s} for s in per_seed])["all"],
            "chosen_config": {"width": width, "learning_rate": lr},
            "parameters": parameter_count(model),
            "tuning_trials": len(CONFIGS),
            "seeds": list(SEEDS),
            "timing": {
                "inference_s_per_pack": float(np.median(times)) / split.test.size,
                "train_s_per_seed": float(np.mean(seconds)),
            },
        }
        print(kind, results[kind]["parameters"], flush=True)
    payload = {
        "maps_sha256": maps_sha256(),
        "n_trajectories": int(data.time_s.size),
        "grid_points": GRID,
        "split": {k: int(getattr(split, k).size) for k in ("train", "validation", "test")},
        "models": results,
        "examples": examples,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    return 0


def pick_examples(data, test_rows, pred_curves, pred_time) -> dict:
    out = {}
    for tech in Technique:
        local = np.nonzero(data.technique[test_rows] == tech)[0][:EXAMPLES_PER_TECHNIQUE]
        out[tech.key] = [
            {
                "true_spread": data.curves[test_rows][k, :, 0].tolist(),
                "pred_spread": pred_curves[k, :, 0].tolist(),
                "true_time_s": float(data.time_s[test_rows][k]),
                "pred_time_s": float(pred_time[k]),
            }
            for k in local
        ]
    return out


if __name__ == "__main__":
    sys.exit(main())
