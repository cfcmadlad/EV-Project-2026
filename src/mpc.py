import json
from dataclasses import dataclass, replace

import numpy as np
import torch

from src.balancing import build_control
from src.config import PROJECT_ROOT, FloatArray, ProjectConfig, Technique
from src.diff_sim import DTYPE, Twin, TwinState, build_twin, initial_state, tensor, twin_step
from src.drive_cycle import Mode, charge_equalised, program_from_profiles, run_scenario
from src.fingerprint import maps_sha256, text_sha256
from src.pack import PackBatch, sample_packs
from src.scaling import string_config

CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "mpc_results.json"
PROFILES_PATH = PROJECT_ROOT / "data" / "processed" / "drive_profiles_25c.json"
CYCLES = ("Mixed1", "Mixed7")
TECHNIQUES = (Technique.BUCK_BOOST, Technique.FLYING_CAPACITOR)
NOT_RUN = {"status": "not_run"}
SOFTMIN_V = 0.005
SOURCES = ("src/mpc.py", "src/diff_sim.py")


@dataclass(frozen=True, slots=True)
class Drive:
    usable_ah: FloatArray
    stop_time_s: FloatArray
    element_loss_j: FloatArray
    gate_drive_j: FloatArray
    solver_iterations: int


def cache_inputs(config: ProjectConfig) -> dict[str, object]:
    spec = config.mpc
    return {
        "maps_sha256": maps_sha256(),
        "profiles_sha256": text_sha256(PROFILES_PATH),
        "sources_sha256": {name: text_sha256(PROJECT_ROOT / name) for name in SOURCES},
        "seed": config.seed,
        "parameter_set": config.params.name,
        "spec": {key: getattr(spec, key) for key in spec.__slots__},
        "softmin_v": SOFTMIN_V,
        "cycles": list(CYCLES),
        "torch": torch.__version__,
    }


def draw_packs(config: ProjectConfig, n_cells: int) -> PackBatch:
    local = string_config(config, n_cells)
    batches = [
        sample_packs(
            config.mpc.packs,
            np.random.default_rng([config.seed, seed, n_cells]),
            local.cell,
            local.pack,
        )
        for seed in config.mpc.seeds
    ]
    return PackBatch(
        capacity_ah=np.concatenate([b.capacity_ah for b in batches]),
        r0_ohm=np.concatenate([b.r0_ohm for b in batches]),
        soc0=np.concatenate([b.soc0 for b in batches]),
    )


class Mpc:
    def __init__(
        self, planner: Twin, profile: torch.Tensor, length: torch.Tensor, config: ProjectConfig
    ) -> None:
        spec = config.mpc
        self.planner = planner
        self.profile = profile
        self.length = length
        self.spec = spec
        self.interval_steps = round(spec.interval_s / config.sim.dt_s)
        self.replan_steps = self.interval_steps * spec.replan_intervals
        self.plan = torch.ones(planner.n_packs, spec.horizon_intervals, planner.n_cells - 1)
        self.plan = self.plan.to(DTYPE)
        self.planned_at = 0
        self.iterations = 0

    def preview(self, step: int) -> torch.Tensor:
        horizon = self.spec.horizon_intervals * self.interval_steps
        index = (step + torch.arange(horizon)) % self.length[:, None]
        values = self.profile.gather(1, index)
        return values.reshape(values.shape[0], self.spec.horizon_intervals, -1).mean(dim=2)

    def cost(self, state: TwinState, current: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        for interval in range(self.spec.horizon_intervals):
            state = twin_step(self.planner, state, current[:, interval], plan[:, interval])
        margin = state.terminal_v - self.planner.lower_voltage_v
        return SOFTMIN_V * torch.logsumexp(-margin / SOFTMIN_V, dim=1)

    def replan(self, step: int, state: TwinState) -> None:
        spec = self.spec
        warm = torch.roll(self.plan, -spec.replan_intervals, dims=1)
        warm[:, -spec.replan_intervals :] = 1.0
        plan = warm.clone().requires_grad_(True)
        optimiser = torch.optim.Adam([plan], lr=spec.learning_rate)
        current = self.preview(step)
        start = TwinState(
            soc=state.soc.detach(),
            v1=state.v1.detach(),
            removed_j=torch.zeros_like(state.removed_j),
            transferred_j=torch.zeros_like(state.removed_j),
            element_loss_j=torch.zeros_like(state.removed_j),
            gate_drive_j=torch.zeros_like(state.removed_j),
            pack_charge_ah=torch.zeros_like(state.removed_j),
            terminal_v=state.terminal_v.detach(),
        )
        for _ in range(spec.iterations):
            optimiser.zero_grad()
            self.cost(start, current, plan).sum().backward()
            optimiser.step()
            with torch.no_grad():
                plan.clamp_(0.0, 1.0)
        self.plan = plan.detach()
        self.planned_at = step
        self.iterations += spec.iterations

    def activity(self, step: int, state: TwinState) -> torch.Tensor:
        if step % self.replan_steps == 0:
            self.replan(step, state)
        return self.plan[:, (step - self.planned_at) // self.interval_steps]


def drive(
    twin: Twin,
    soc0: FloatArray,
    profile: torch.Tensor,
    length: torch.Tensor,
    config: ProjectConfig,
    controller: Mpc | None,
) -> Drive:
    n = twin.n_packs
    state = initial_state(twin, tensor(soc0))
    alive = torch.ones(n, dtype=torch.bool)
    usable = torch.zeros(n, dtype=DTYPE)
    stop = torch.zeros(n, dtype=DTYPE)
    element = torch.zeros(n, dtype=DTYPE)
    gate = torch.zeros(n, dtype=DTYPE)
    rows = torch.arange(n)
    max_steps = config.drive_cycle.max_cycle_repeats * int(length.max())
    for step in range(max_steps):
        current = profile[rows, step % length]
        current = torch.where(alive, current, torch.zeros_like(current))
        activity = None
        if controller is not None:
            with torch.enable_grad():
                activity = controller.activity(step, state)
        with torch.no_grad():
            state = twin_step(twin, state, current, activity, halt=~alive)
            hit = (state.terminal_v.min(dim=1).values <= twin.lower_voltage_v) | (
                state.soc.min(dim=1).values <= 0.0
            )
            last = step + 1 == max_steps
            done = alive & (hit | torch.full_like(hit, last))
            usable = torch.where(done, state.pack_charge_ah, usable)
            stop = torch.where(done, torch.full_like(stop, (step + 1) * twin.dt_s), stop)
            element = torch.where(done, state.element_loss_j, element)
            gate = torch.where(done, state.gate_drive_j, gate)
            alive = alive & ~done
        if not alive.any():
            break
    return Drive(
        usable.numpy(),
        stop.numpy(),
        element.numpy(),
        gate.numpy(),
        0 if controller is None else controller.iterations,
    )


def run_study(
    config: ProjectConfig,
    profiles: dict,
    n_cells: int,
    cycle: str,
    technique: Technique,
    packs: PackBatch,
) -> dict[str, object]:
    local = string_config(config, n_cells)
    n = packs.n_packs
    series = np.array(profiles["profiles"][cycle]["current_a"])
    program = program_from_profiles([series] * n)
    hardware = local.technique(technique)
    component = np.full(n, hardware.component.mid)
    frequency = np.full(n, hardware.frequency_hz.mid)
    none = run_scenario(None, packs, program, Mode.DISCHARGE, local).usable_ah
    reference = run_scenario(
        None, charge_equalised(packs, Mode.DISCHARGE), program, Mode.DISCHARGE, local
    ).usable_ah
    rule_numpy = run_scenario(technique, packs, program, Mode.DISCHARGE, local).usable_ah
    control = build_control(
        technique, packs, component, frequency, local, track_load=True, gate_balancing=True
    )
    twin = build_twin(control, technique, local)
    planner_config = replace(local, sim=replace(local.sim, dt_s=config.mpc.interval_s))
    planner_control = build_control(
        technique, packs, component, frequency, planner_config, track_load=True
    )
    planner = build_twin(planner_control, technique, planner_config)
    profile = tensor(program.current_a)
    length = torch.as_tensor(program.length)
    rule = drive(twin, packs.soc0, profile, length, local, None)
    controller = Mpc(planner, profile, length, local)
    mpc = drive(twin, packs.soc0, profile, length, local, controller)
    return {
        "none": none,
        "reference": reference,
        "rule_numpy": rule_numpy,
        "rule": rule.usable_ah,
        "mpc": mpc.usable_ah,
        "mpc_element_loss_j": mpc.element_loss_j,
        "rule_element_loss_j": rule.element_loss_j,
        "mpc_gate_drive_j": mpc.gate_drive_j,
        "rule_gate_drive_j": rule.gate_drive_j,
        "solver_iterations": mpc.solver_iterations,
    }


def summarise(raw: dict[tuple[str, int, str], dict[str, object]], config: ProjectConfig) -> dict:
    spec = config.mpc
    results: dict[str, object] = {}
    for technique in TECHNIQUES:
        by_n: dict[str, object] = {}
        for n_cells in spec.cells:
            pooled = {
                key: np.concatenate([raw[(technique.key, n_cells, c)][key] for c in CYCLES])
                for key in ("none", "reference", "rule_numpy", "rule", "mpc")
            }
            gain = pooled["mpc"] - pooled["rule"]
            denominator = pooled["reference"].mean() - pooled["rule"].mean()
            by_cycle = {}
            for cycle in CYCLES:
                arrays = raw[(technique.key, n_cells, cycle)]
                by_cycle[cycle] = {
                    "mean_gain_ah": float((arrays["mpc"] - arrays["rule"]).mean()),
                    "rule_ah": float(arrays["rule"].mean()),
                    "mpc_ah": float(arrays["mpc"].mean()),
                }
            by_n[str(n_cells)] = {
                "n_packs": int(gain.size),
                "delivered_ah_mean": {k: float(v.mean()) for k, v in pooled.items()},
                "mean_gain_ah": float(gain.mean()),
                "median_gain_ah": float(np.median(gain)),
                "win_fraction": float(np.mean(gain >= -spec.win_tolerance_ah)),
                "strictly_better_fraction": float(np.mean(gain > spec.win_tolerance_ah)),
                "gap_to_reference_ah": float(denominator),
                "gap_closed": float(gain.mean() / denominator) if denominator > 0.0 else None,
                "plant_consistency_max_relative": float(
                    np.max(np.abs(pooled["rule"] - pooled["rule_numpy"]) / pooled["rule_numpy"])
                ),
                "solver_iterations_per_run": int(
                    raw[(technique.key, n_cells, CYCLES[0])]["solver_iterations"]
                ),
                "mean_element_loss_change_j": float(
                    np.mean(
                        np.concatenate(
                            [
                                raw[(technique.key, n_cells, c)]["mpc_element_loss_j"]
                                - raw[(technique.key, n_cells, c)]["rule_element_loss_j"]
                                for c in CYCLES
                            ]
                        )
                    )
                ),
                "by_cycle": by_cycle,
            }
        results[technique.key] = by_n
    return results


def load_cache(config: ProjectConfig) -> dict[str, object]:
    if not CACHE_PATH.exists():
        return dict(NOT_RUN)
    payload = json.loads(CACHE_PATH.read_text())
    if payload["inputs"] != json.loads(json.dumps(cache_inputs(config))):
        raise RuntimeError("mpc cache is stale; rerun python -m scripts.run_mpc before run_all.py")
    return {"status": "ok", **payload}
