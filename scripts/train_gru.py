import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.config import CONFIG, PROJECT_ROOT
from src.params import SocEstimationSpec
from src.soc_estimation.data import SPLITS_PATH, Cycle, load_split, load_splits
from src.soc_estimation.gru import (
    Normaliser,
    SocGru,
    cycle_features,
    fit_normaliser,
    gather,
    macs_per_estimate,
    normalise,
    parameter_count,
    predict,
    window_ends,
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "soc_gru"
LOG_PATH = OUTPUT_DIR / "training_log.json"
PATIENCE = 4
LEARNING_RATE_DECAY = 0.8
VALIDATION_STRIDE = 5
MAX_WEIGHT_BYTES = 5_000_000


def build(
    cycles: list[Cycle], window_s: int, stride: int, normaliser: Normaliser | None
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    features, targets, ends = [], [], []
    offset = 0
    for cycle in cycles:
        features.append(cycle_features(cycle))
        targets.append(cycle.soc)
        ends.append(window_ends(cycle.n_steps, window_s, stride) + offset)
        offset += cycle.n_steps
    stacked = np.concatenate(features)
    scaled = normalise(stacked, normaliser) if normaliser is not None else None
    return (
        scaled,
        torch.tensor(np.concatenate(targets), dtype=torch.float32),
        np.concatenate(ends),
        stacked,
    )


def validation_rmse(
    model: SocGru, features: torch.Tensor, targets: torch.Tensor, ends: np.ndarray, window_s: int
) -> float:
    prediction = predict(model, features, ends, window_s)
    return float(np.sqrt(np.mean((prediction - targets.numpy()[ends]) ** 2)))


def train_one(
    window_s: int,
    seed: int,
    train: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    validation: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    spec: SocEstimationSpec,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SocGru(spec.hidden_size)
    optimiser = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, LEARNING_RATE_DECAY)
    features, targets, ends = train
    loss_fn = nn.MSELoss()
    best, best_state, stale = float("inf"), None, 0
    history: list[float] = []
    start = perf_counter()
    for _ in range(spec.epochs):
        model.train()
        order = rng.permutation(ends.size)
        for first in range(0, order.size, spec.batch_size):
            batch = ends[order[first : first + spec.batch_size]]
            optimiser.zero_grad()
            loss = loss_fn(model(gather(features, batch, window_s)), targets[batch])
            loss.backward()
            optimiser.step()
        scheduler.step()
        score = validation_rmse(model, *validation[:2], validation[2], window_s)
        history.append(score)
        if score < best:
            best, stale = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    elapsed = perf_counter() - start
    assert best_state is not None
    return best_state, {
        "window_s": window_s,
        "seed": seed,
        "best_validation_rmse_percent": best * 100.0,
        "validation_rmse_percent_by_epoch": [h * 100.0 for h in history],
        "epochs_run": len(history),
        "training_time_s": elapsed,
    }


def main() -> int:
    if not SPLITS_PATH.exists():
        print("soc_splits.json missing; run scripts/make_soc_splits.py first")
        return 1
    spec = CONFIG.soc_estimation
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    splits = load_splits()["splits"]
    train_cycles = load_split(splits["train"], spec)
    validation_cycles = load_split(splits["validation"], spec)
    normaliser = fit_normaliser(train_cycles)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for window_s in spec.windows_s:
        train_features, train_targets, train_ends, _ = build(
            train_cycles, window_s, spec.train_stride_s, normaliser
        )
        val_features, val_targets, val_ends, _ = build(
            validation_cycles, window_s, VALIDATION_STRIDE, normaliser
        )
        for seed in spec.seeds:
            state, record = train_one(
                window_s,
                seed,
                (train_features, train_targets, train_ends),
                (val_features, val_targets, val_ends),
                spec,
            )
            path = OUTPUT_DIR / f"gru_w{window_s}_s{seed}.pt"
            torch.save(state, path)
            size = Path(path).stat().st_size
            assert size < MAX_WEIGHT_BYTES
            record |= {
                "file": path.name,
                "weight_bytes": size,
                "parameters": parameter_count(spec.hidden_size),
                "macs_per_estimate": macs_per_estimate(spec.hidden_size, window_s),
                "n_train_windows": int(train_ends.size),
            }
            runs.append(record)
            print(
                f"W={window_s} seed={seed} val RMSE {record['best_validation_rmse_percent']:.3f} %"
                f" in {record['training_time_s']:.0f} s",
                flush=True,
            )
    LOG_PATH.write_text(
        json.dumps(
            {
                "hidden_size": spec.hidden_size,
                "normaliser": {"mean": list(normaliser.mean), "std": list(normaliser.std)},
                "runs": runs,
            },
            indent=1,
        )
        + "\n",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
