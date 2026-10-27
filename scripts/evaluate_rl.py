import hashlib
import json
import sys
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO

from src.config import CONFIG, PROJECT_ROOT
from src.drive_cycle import PROFILES_PATH, Mode, charge_equalised, load_profiles
from src.experiments import random_streams
from src.rl_env import (
    greedy_max_transfer,
    max_control_steps,
    no_balancing,
    rollout,
    rule_based,
    test_batch,
)
from src.rl_sim import build_sim

POLICY_DIR = PROJECT_ROOT / "data" / "processed" / "rl_policy"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rl_results.json"
FIELDS = ("pack_ah", "spread", "loss_j", "reward", "outside_s", "finished")


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ppo_controller(model: PPO):
    def act(sim, obs):
        action, _ = model.predict(obs.astype(np.float32), deterministic=True)
        return action

    return act


def with_reward(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    weight = CONFIG.rl.loss_weight_ah_per_j
    return {**out, "reward": out["pack_ah"] - weight * out["loss_j"]}


def main() -> int:
    log_path = POLICY_DIR / "training_log.json"
    if not log_path.exists():
        print("run scripts/train_rl.py first")
        return 1
    start = perf_counter()
    spec = CONFIG.rl
    torch.set_num_threads(1)
    profiles = load_profiles(CONFIG.drive_cycle)
    log = json.loads(log_path.read_text())
    models = {r["seed"]: PPO.load(POLICY_DIR / r["file"], device="cpu") for r in log["runs"]}
    rngs = random_streams(CONFIG.seed)["rl_eval"].spawn(len(spec.test_cycles))
    per_pack: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for cycle, rng in zip(spec.test_cycles, rngs, strict=True):
        packs, program, bias = test_batch(CONFIG, profiles, cycle, rng)
        steps = max_control_steps(program, CONFIG)
        controllers = {
            "none": no_balancing,
            "rule_based": rule_based,
            "greedy": greedy_max_transfer,
            **{f"ppo_s{seed}": ppo_controller(model) for seed, model in models.items()},
        }
        results = {}
        for name, controller in controllers.items():
            sim = build_sim(packs, program, bias, CONFIG)
            results[name] = with_reward(rollout(sim, controller, steps))
        ceiling = build_sim(charge_equalised(packs, Mode.DISCHARGE), program, bias, CONFIG)
        results["ideal_reference"] = with_reward(rollout(ceiling, no_balancing, steps))
        per_pack[cycle] = results

    seed_names = [f"ppo_s{seed}" for seed in models]
    summary: dict[str, object] = {}
    for cycle, results in per_pack.items():
        block: dict[str, object] = {}
        for name, out in results.items():
            block[name] = {
                f: {"mean": float(out[f].mean()), "std": float(out[f].std(ddof=1))} for f in FIELDS
            }
        ppo_mean = {f: np.mean([results[s][f] for s in seed_names], axis=0) for f in FIELDS}
        block["ppo_mean_of_seeds"] = {
            f: {
                "mean": float(ppo_mean[f].mean()),
                "std_over_seeds": float(np.std([results[s][f].mean() for s in seed_names], ddof=1)),
            }
            for f in FIELDS
        }
        comparisons = {}
        for other in ("none", "rule_based", "greedy"):
            row = {}
            for field in ("reward", "pack_ah"):
                diff = ppo_mean[field] - results[other][field]
                se = float(diff.std(ddof=1) / np.sqrt(diff.size))
                row[field] = {
                    "mean_improvement": float(diff.mean()),
                    "standard_error": se,
                    "exceeds_standard_errors": bool(diff.mean() > spec.standard_errors * se),
                }
            comparisons[other] = row
        block["ppo_vs"] = comparisons
        summary[cycle] = block
    payload = {
        "inputs": {
            "drive_profiles_sha256": digest(PROFILES_PATH),
            "policies_sha256": {r["file"]: digest(POLICY_DIR / r["file"]) for r in log["runs"]},
        },
        "training": {
            "seeds": [r["seed"] for r in log["runs"]],
            "total_timesteps": spec.total_timesteps,
            "policy_bytes_max": max(r["bytes"] for r in log["runs"]),
            "training_time_s": {str(r["seed"]): r["training_time_s"] for r in log["runs"]},
            "curves": {str(r["seed"]): r["curve"] for r in log["runs"]},
        },
        "test": summary,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(f"wrote {OUTPUT_PATH.name} in {perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
