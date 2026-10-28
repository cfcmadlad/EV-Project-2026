import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.config import CONFIG, PROJECT_ROOT
from src.drive_cycle import PROFILES_PATH, load_profiles
from src.rl_residual import VARIANTS, ResidualEnv, RlVariant, with_cells

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "rl2_policy"
LOG_PATH = OUTPUT_DIR / "training_log.json"
CELL_COUNTS = (4, 8)
TIMESTEPS = 150_000
IMITATION_EPISODES = 12
IMITATION_EPOCHS = 300
IMITATION_LR = 1e-3
FINE_TUNE_LOG_STD = -2.0
MAX_POLICY_BYTES = 5_000_000


def collect_expert(env: ResidualEnv, links: int) -> tuple[np.ndarray, np.ndarray]:
    observations, actions = [], []
    for episode in range(IMITATION_EPISODES):
        obs, _ = env.reset(seed=1000 + episode)
        done = False
        while not done:
            action = obs[-links:].copy()
            observations.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(observations), np.array(actions)


def clone(model: PPO, observations: np.ndarray, actions: np.ndarray) -> float:
    policy = model.policy
    optimiser = torch.optim.Adam(policy.parameters(), lr=IMITATION_LR)
    x, y = torch.tensor(observations), torch.tensor(actions, dtype=torch.float32)
    loss = torch.tensor(0.0)
    for _ in range(IMITATION_EPOCHS):
        optimiser.zero_grad()
        mean = policy.get_distribution(x).distribution.mean
        loss = torch.nn.functional.mse_loss(mean, y)
        loss.backward()
        optimiser.step()
    policy.log_std.data.fill_(FINE_TUNE_LOG_STD)
    return float(loss)


def train_one(job: tuple[RlVariant, int, int]) -> dict:
    variant, n_cells, seed = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = with_cells(CONFIG, n_cells)
    spec = config.rl
    profiles = load_profiles(config.drive_cycle)
    env = Monitor(ResidualEnv(config, profiles, spec.train_cycles, variant))
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=spec.n_steps,
        batch_size=spec.batch_size,
        learning_rate=variant.fine_tune_lr,
        gamma=spec.gamma,
        gae_lambda=spec.gae_lambda,
        policy_kwargs={
            "net_arch": [spec.hidden_units, spec.hidden_units],
            "log_std_init": spec.log_std_init,
        },
        device="cpu",
        verbose=0,
    )
    clone_loss = None
    start = perf_counter()
    if variant.method == "imitation":
        observations, actions = collect_expert(env.unwrapped, n_cells - 1)
        clone_loss = clone(model, observations, actions)
    model.learn(total_timesteps=TIMESTEPS)
    name = f"{variant.name}_n{n_cells}_s{seed}"
    path = OUTPUT_DIR / f"{name}.zip"
    model.save(path)
    assert path.stat().st_size < MAX_POLICY_BYTES
    return {
        "variant": variant.name,
        "n_cells": n_cells,
        "seed": seed,
        "file": path.name,
        "bytes": path.stat().st_size,
        "clone_loss": clone_loss,
        "training_time_s": perf_counter() - start,
    }


def main() -> int:
    if not PROFILES_PATH.exists():
        print("drive profiles missing; run scripts/prepare_drive_profiles.py")
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (variant, n, seed) for variant in VARIANTS for n in CELL_COUNTS for seed in CONFIG.rl.seeds
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=8, mp_context=context) as pool:
        runs = list(pool.map(train_one, jobs))
    LOG_PATH.write_text(
        json.dumps({"timesteps": TIMESTEPS, "runs": runs}, indent=1) + "\n", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
