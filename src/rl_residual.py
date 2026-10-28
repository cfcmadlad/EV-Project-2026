from dataclasses import dataclass, replace

import numpy as np
from gymnasium import spaces

from src.config import FloatArray, ProjectConfig
from src.rl_env import BalancingEnv, rule_based
from src.rl_sim import observe

SHAPING_GAIN_AH = 2.0


@dataclass(frozen=True, slots=True)
class RlVariant:
    name: str
    method: str
    residual_scale: float
    shaping: bool
    fine_tune_lr: float


VARIANTS = (
    RlVariant("residual_plain", "residual", 0.25, False, 3e-4),
    RlVariant("residual_shaped", "residual", 0.25, True, 3e-4),
    RlVariant("imitation_plain", "imitation", 1.0, False, 1e-4),
    RlVariant("imitation_shaped", "imitation", 1.0, True, 1e-4),
)


def with_cells(config: ProjectConfig, n_cells: int) -> ProjectConfig:
    pack = replace(config.pack, n_cells=n_cells)
    return replace(config, params=replace(config.params, pack=pack))


def true_spread(env: BalancingEnv) -> float:
    return float(np.ptp(env.sim.state.soc[0]))


class ResidualEnv(BalancingEnv):
    def __init__(
        self,
        config: ProjectConfig,
        profiles: dict,
        cycles: tuple[str, ...],
        variant: RlVariant,
    ) -> None:
        super().__init__(config, profiles, cycles)
        self.variant = variant
        links = config.pack.n_cells - 1
        width = 2 * config.pack.n_cells + 3 + links
        self.observation_space = spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
        self.potential = 0.0
        self.gamma = config.rl.gamma

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        base = rule_based(self.sim, obs[None, :])[0]
        return np.concatenate((obs, base)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.potential = -SHAPING_GAIN_AH * true_spread(self)
        return self._augment(obs), info

    def step(self, action):
        assert self.sim is not None
        raw = observe(self.sim)[0]
        base = rule_based(self.sim, raw[None, :])[0]
        if self.variant.method == "residual":
            command = np.clip(base + self.variant.residual_scale * np.asarray(action), -1.0, 1.0)
        else:
            command = np.clip(np.asarray(action), -1.0, 1.0)
        obs, reward, terminated, truncated, info = super().step(command.astype(np.float32))
        if self.variant.shaping:
            potential = -SHAPING_GAIN_AH * true_spread(self)
            reward += self.gamma * potential - self.potential
            self.potential = potential
        return self._augment(obs), reward, terminated, truncated, info


def policy_controller(model, variant: RlVariant):
    def act(sim, obs: FloatArray) -> FloatArray:
        base = rule_based(sim, obs)
        features = np.concatenate((obs, base), axis=1).astype(np.float32)
        action, _ = model.predict(features, deterministic=True)
        if variant.method == "residual":
            return np.clip(base + variant.residual_scale * action, -1.0, 1.0)
        return np.clip(action, -1.0, 1.0)

    return act
