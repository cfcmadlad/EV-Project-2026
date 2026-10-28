import json
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

from src.config import CONFIG, PROJECT_ROOT
from src.drive_cycle import Mode, charge_equalised, load_profiles
from src.experiments import random_streams
from src.rl_env import (
    greedy_max_transfer,
    max_control_steps,
    no_balancing,
    rollout,
    rule_based,
    test_batch,
)
from src.rl_residual import VARIANTS, policy_controller, with_cells
from src.rl_sim import build_sim

POLICY_DIR = PROJECT_ROOT / "data" / "processed" / "rl2_policy"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rl2_results.json"
FIELDS = ("pack_ah", "spread", "loss_j", "reward", "outside_s", "finished")


def with_reward(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    weight = CONFIG.rl.loss_weight_ah_per_j
    return {**out, "reward": out["pack_ah"] - weight * out["loss_j"]}


def main() -> int:
    log_path = POLICY_DIR / "training_log.json"
    if not log_path.exists():
        print("run scripts/train_rl2.py first")
        return 1
    torch.set_num_threads(1)
    spec = CONFIG.rl
    log = json.loads(log_path.read_text())
    by_name = {v.name: v for v in VARIANTS}
    summary: dict[str, object] = {}
    for n_cells in sorted({r["n_cells"] for r in log["runs"]}):
        config = with_cells(CONFIG, n_cells)
        profiles = load_profiles(config.drive_cycle)
        rngs = random_streams(CONFIG.seed)["rl_eval"].spawn(len(spec.test_cycles))
        runs = [r for r in log["runs"] if r["n_cells"] == n_cells]
        models = {r["file"]: PPO.load(POLICY_DIR / r["file"], device="cpu") for r in runs}
        block: dict[str, object] = {}
        for cycle, rng in zip(spec.test_cycles, rngs, strict=True):
            packs, program, bias = test_batch(config, profiles, cycle, rng)
            steps = max_control_steps(program, config)
            controllers = {
                "none": no_balancing,
                "rule_based": rule_based,
                "greedy": greedy_max_transfer,
            }
            for r in runs:
                variant = by_name[r["variant"]]
                controllers[f"{r['variant']}_s{r['seed']}"] = policy_controller(
                    models[r["file"]], variant
                )
            results = {
                name: with_reward(rollout(build_sim(packs, program, bias, config), ctrl, steps))
                for name, ctrl in controllers.items()
            }
            ceiling = build_sim(charge_equalised(packs, Mode.DISCHARGE), program, bias, config)
            results["ideal_reference"] = with_reward(rollout(ceiling, no_balancing, steps))
            table: dict[str, object] = {
                name: {f: float(out[f].mean()) for f in FIELDS} for name, out in results.items()
            }
            comparisons = {}
            for variant in VARIANTS:
                names = [f"{variant.name}_s{s}" for s in spec.seeds]
                mean = np.mean([results[n]["reward"] for n in names], axis=0)
                diff = mean - results["rule_based"]["reward"]
                se = float(diff.std(ddof=1) / np.sqrt(diff.size))
                comparisons[variant.name] = {
                    "mean_improvement_over_rule": float(diff.mean()),
                    "standard_error": se,
                    "exceeds_standard_errors": bool(diff.mean() > spec.standard_errors * se),
                    "seed_means": [float(results[n]["reward"].mean()) for n in names],
                }
            block[cycle] = {"table": table, "vs_rule": comparisons}
        summary[f"n{n_cells}"] = block
    payload = {
        "timesteps": log["timesteps"],
        "configurations_per_method": 2,
        "training_time_s": {r["file"]: r["training_time_s"] for r in log["runs"]},
        "test": summary,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
