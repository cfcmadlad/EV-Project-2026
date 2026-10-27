from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import FloatArray, ProjectConfig
from src.drive_cycle import CurrentProgram, program_from_profiles
from src.pack import PackBatch, sample_packs
from src.rl_sim import PackSim, advance, build_sim, estimated_soc, observe

Controller = Callable[[PackSim, FloatArray], FloatArray]


def link_difference(sim: PackSim) -> FloatArray:
    estimate = estimated_soc(sim)
    return estimate[:, :-1] - estimate[:, 1:]


def no_balancing(sim: PackSim, obs: FloatArray) -> FloatArray:
    return np.zeros((obs.shape[0], sim.holder["command"].shape[1]))


def rule_based(sim: PackSim, obs: FloatArray) -> FloatArray:
    config = sim.config
    estimate = estimated_soc(sim)
    difference = link_difference(sim)
    active = np.abs(difference) > config.sim.link_soc_deadband
    unbalanced = np.ptp(estimate, axis=1) >= config.sim.soc_tolerance
    scale = config.buck_boost.duty / config.rl.d_max
    return np.sign(difference) * active * unbalanced[:, None] * scale


def greedy_max_transfer(sim: PackSim, obs: FloatArray) -> FloatArray:
    return np.sign(link_difference(sim))


def rollout(sim: PackSim, controller: Controller, max_steps: int) -> dict[str, FloatArray]:
    for _ in range(max_steps):
        obs = observe(sim)
        advance(sim, controller(sim, obs))
        if sim.done.all():
            break
    final = {key: value.copy() for key, value in sim.final.items()}
    final["outside_s"] = sim.outside_s.copy()
    final["finished"] = sim.done.astype(float)
    return final


def max_control_steps(program: CurrentProgram, config: ProjectConfig) -> int:
    horizon_s = config.drive_cycle.max_cycle_repeats * int(program.length.max())
    return horizon_s // config.rl.control_step_s + 1


def test_batch(
    config: ProjectConfig, profiles: dict, cycle: str, rng: np.random.Generator
) -> tuple[PackBatch, CurrentProgram, FloatArray]:
    n = config.rl.test_packs
    packs = sample_packs(n, rng, config.cell, config.pack)
    bias = rng.normal(0.0, config.rl.estimate_sigma, packs.soc0.shape)
    profile = np.array(profiles["profiles"][cycle]["current_a"])
    return packs, program_from_profiles([profile] * n), bias


class BalancingEnv(gym.Env):
    def __init__(self, config: ProjectConfig, profiles: dict, cycles: tuple[str, ...]) -> None:
        super().__init__()
        self.config = config
        self.programs = {
            name: program_from_profiles([np.array(profiles["profiles"][name]["current_a"])])
            for name in cycles
        }
        links = config.pack.n_cells - 1
        width = 2 * config.pack.n_cells + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (links,), dtype=np.float32)
        self.sim: PackSim | None = None
        self.limit = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        config = self.config
        packs = sample_packs(1, rng, config.cell, config.pack)
        bias = rng.normal(0.0, config.rl.estimate_sigma, packs.soc0.shape)
        name = list(self.programs)[int(rng.integers(len(self.programs)))]
        program = self.programs[name]
        self.sim = build_sim(packs, program, bias, config)
        self.limit = max_control_steps(program, config)
        return observe(self.sim)[0].astype(np.float32), {"cycle": name}

    def step(self, action):
        assert self.sim is not None
        reward, done = advance(self.sim, np.asarray(action, dtype=float)[None, :])
        self.limit -= 1
        info = {
            "pack_ah": float(self.sim.final["pack_ah"][0]),
            "loss_j": float(self.sim.final["loss_j"][0]),
            "spread": float(self.sim.final["spread"][0]),
        }
        return (
            observe(self.sim)[0].astype(np.float32),
            float(reward[0]),
            bool(done[0]),
            bool(self.limit <= 0 and not done[0]),
            info,
        )
