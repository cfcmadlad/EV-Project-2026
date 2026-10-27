import json
import sys
from time import perf_counter

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.config import CONFIG, PROJECT_ROOT
from src.drive_cycle import PROFILES_PATH, load_profiles
from src.rl_env import BalancingEnv

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "rl_policy"
LOG_PATH = OUTPUT_DIR / "training_log.json"
MAX_POLICY_BYTES = 5_000_000


class CurveCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.points: list[tuple[int, float]] = []

    def _on_rollout_end(self) -> None:
        returns = [info["r"] for info in self.model.ep_info_buffer]
        if returns:
            self.points.append((self.num_timesteps, float(sum(returns) / len(returns))))

    def _on_step(self) -> bool:
        return True


def main() -> int:
    if not PROFILES_PATH.exists():
        print("drive profiles missing; run scripts/prepare_drive_profiles.py")
        return 1
    spec = CONFIG.rl
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    profiles = load_profiles(CONFIG.drive_cycle)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in spec.seeds:
        env = Monitor(BalancingEnv(CONFIG, profiles, spec.train_cycles))
        callback = CurveCallback()
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            n_steps=spec.n_steps,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            gamma=spec.gamma,
            gae_lambda=spec.gae_lambda,
            policy_kwargs={
                "net_arch": [spec.hidden_units, spec.hidden_units],
                "log_std_init": spec.log_std_init,
            },
            device="cpu",
            verbose=0,
        )
        start = perf_counter()
        model.learn(total_timesteps=spec.total_timesteps, callback=callback)
        elapsed = perf_counter() - start
        path = OUTPUT_DIR / f"ppo_s{seed}.zip"
        model.save(path)
        size = path.stat().st_size
        assert size < MAX_POLICY_BYTES
        runs.append(
            {
                "seed": seed,
                "file": path.name,
                "bytes": size,
                "training_time_s": elapsed,
                "curve": callback.points,
            }
        )
        print(f"seed {seed}: {elapsed:.0f} s, {size} bytes", flush=True)
    LOG_PATH.write_text(json.dumps({"runs": runs}, indent=1) + "\n", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
