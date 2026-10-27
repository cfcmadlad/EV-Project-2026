import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.balancing import BalanceResult, build_control
from src.config import BoolArray, FloatArray, IntArray, ProjectConfig, Technique
from src.pack import PackBatch, tile_pack
from src.params import DriveCycleSpec
from src.stepper import PackState, cell_ocv, step

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROFILES_PATH = PROCESSED_DIR / "drive_profiles_25c.json"
PROVENANCE_PATH = PROCESSED_DIR / "provenance.json"
NOT_RUN = {"status": "not_run"}
STALE = {"status": "stale_cache"}
STRATEGIES: tuple[Technique | None, ...] = (None, *Technique)
STRATEGY_KEYS = ("none", *(t.key for t in Technique))


class Mode(StrEnum):
    DISCHARGE = "discharge"
    CHARGE = "charge"


@dataclass(frozen=True, slots=True)
class CurrentProgram:
    current_a: FloatArray
    length: IntArray

    @property
    def n_programs(self) -> int:
        return self.current_a.shape[0]

    def select(self, mask: BoolArray) -> "CurrentProgram":
        return CurrentProgram(self.current_a[mask], self.length[mask])


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    stop_time_s: FloatArray
    usable_ah: FloatArray
    limit_reached: BoolArray
    soc0: FloatArray
    final_soc: FloatArray
    pack_charge_ah: FloatArray
    removed_j: FloatArray
    transferred_j: FloatArray
    element_loss_j: FloatArray
    balancing_r0_j: FloatArray
    balancing_v1_j: FloatArray
    trace_time_s: FloatArray
    trace_spread: FloatArray

    @property
    def final_spread(self) -> FloatArray:
        return np.ptp(self.final_soc, axis=1)

    @property
    def energy_dissipated_j(self) -> FloatArray:
        return self.removed_j - self.transferred_j

    @property
    def ledger_j(self) -> FloatArray:
        return self.element_loss_j + self.balancing_r0_j + self.balancing_v1_j


def load_profiles(spec: DriveCycleSpec) -> dict | None:
    if not PROFILES_PATH.exists():
        return None
    payload = json.loads(PROFILES_PATH.read_text())
    hashes = {r["name"]: r["sha256"] for r in json.loads(PROVENANCE_PATH.read_text())["files"]}
    fresh = all(
        hashes.get(profile["source_file"]) == profile["source_sha256"]
        for profile in payload["profiles"].values()
    )
    if not fresh or payload["averaging_step_s"] != spec.profile_step_s:
        return dict(STALE)
    return payload


def program_from_profiles(profiles: list[FloatArray]) -> CurrentProgram:
    length = np.array([profile.size for profile in profiles])
    padded = np.zeros((len(profiles), int(length.max())))
    for row, profile in enumerate(profiles):
        padded[row, : profile.size] = profile
    return CurrentProgram(padded, length)


def constant_program(current_a: float, n_programs: int, n_steps: int) -> CurrentProgram:
    return CurrentProgram(
        np.full((n_programs, n_steps), current_a), np.full(n_programs, n_steps, dtype=int)
    )


def run_scenario(
    technique: Technique | None,
    packs: PackBatch,
    program: CurrentProgram,
    mode: Mode,
    config: ProjectConfig,
) -> ScenarioRun:
    spec, n = config.drive_cycle, packs.n_packs
    if program.n_programs != n:
        raise ValueError(f"program has {program.n_programs} rows for {n} packs")
    if spec.profile_step_s != config.sim.dt_s:
        raise ValueError("profile_step_s must equal the simulation step")
    hardware = None if technique is None else config.technique(technique)
    component = np.full(n, 1.0 if hardware is None else hardware.component.mid)
    frequency = np.full(n, 1.0 if hardware is None else hardware.frequency_hz.mid)
    control = build_control(
        technique, packs, component, frequency, config, track_load=True, gate_balancing=True
    )
    state = PackState.initial(packs.soc0, cell_ocv(packs.soc0, control))
    sign = 1.0 if mode is Mode.DISCHARGE else -1.0
    dt_s = config.sim.dt_s
    max_steps = spec.max_cycle_repeats * int(program.length.max())
    trace_every = max(1, round(spec.trace_interval_s / dt_s))
    index = np.arange(n)
    full = {
        key: np.zeros(n)
        for key in (
            "stop_time_s",
            "usable_ah",
            "pack_charge_ah",
            "removed_j",
            "transferred_j",
            "element_loss_j",
            "balancing_r0_j",
            "balancing_v1_j",
        )
    }
    limit_reached = np.zeros(n, dtype=bool)
    final_soc = packs.soc0.copy()
    spread_full = np.ptp(packs.soc0, axis=1)
    trace_time: list[float] = [0.0]
    trace_spread: list[FloatArray] = [spread_full.copy()]
    live = program
    for k in range(max_steps):
        current = live.current_a[np.arange(index.size), k % live.length]
        state = step(state, current, control)
        if mode is Mode.DISCHARGE:
            hit = (state.terminal_v.min(axis=1) <= spec.lower_voltage_v) | (
                state.soc.min(axis=1) <= 0.0
            )
        else:
            hit = (state.terminal_v.max(axis=1) >= spec.upper_voltage_v) | (
                state.soc.max(axis=1) >= 1.0
            )
        last = k + 1 == max_steps
        spread_full[index] = state.spread
        if (k + 1) % trace_every == 0:
            trace_time.append((k + 1) * dt_s)
            trace_spread.append(spread_full.copy())
        done = np.ones_like(hit) if last else hit
        if done.any():
            finished = index[done]
            limit_reached[finished] = hit[done]
            final_soc[finished] = state.soc[done]
            full["stop_time_s"][finished] = (k + 1) * dt_s
            full["usable_ah"][finished] = sign * state.pack_charge_ah[done]
            for key in full:
                if key not in ("stop_time_s", "usable_ah"):
                    full[key][finished] = getattr(state, key)[done]
            keep = ~done
            state, index, live = state.select(keep), index[keep], live.select(keep)
            control = control.with_dynamics(control.dynamics.select(keep))
        if index.size == 0:
            break
    return ScenarioRun(
        stop_time_s=full["stop_time_s"],
        usable_ah=full["usable_ah"],
        limit_reached=limit_reached,
        soc0=packs.soc0,
        final_soc=final_soc,
        pack_charge_ah=full["pack_charge_ah"],
        removed_j=full["removed_j"],
        transferred_j=full["transferred_j"],
        element_loss_j=full["element_loss_j"],
        balancing_r0_j=full["balancing_r0_j"],
        balancing_v1_j=full["balancing_v1_j"],
        trace_time_s=np.array(trace_time),
        trace_spread=np.array(trace_spread),
    )


def _strategy_block(run: ScenarioRun, index: int, tolerance: float) -> dict[str, object]:
    return {
        "usable_ah": float(run.usable_ah[index]),
        "stop_time_s": float(run.stop_time_s[index]),
        "final_spread": float(run.final_spread[index]),
        "energy_dissipated_j": float(run.energy_dissipated_j[index]),
        "element_loss_j": float(run.element_loss_j[index]),
        "criterion_met": bool(run.final_spread[index] < tolerance),
        "limit_reached": bool(run.limit_reached[index]),
    }


def ledger_gap(run: ScenarioRun) -> dict[str, float]:
    gap = np.abs(run.energy_dissipated_j - run.ledger_j)
    return {
        "max_vs_removed": float(np.max(gap / run.removed_j)),
        "max_vs_dissipated": float(np.max(gap / np.abs(run.energy_dissipated_j))),
    }


def soc_conservation_error(run: ScenarioRun, capacity_ah: FloatArray) -> float:
    change = run.final_soc - run.soc0
    expected = -run.pack_charge_ah[:, None] / capacity_ah
    return float(np.max(np.abs(change - expected) / np.abs(expected)))


def _soc_equalised(pack: PackBatch) -> PackBatch:
    mean = pack.soc0.mean(axis=1, keepdims=True)
    return replace(pack, soc0=np.broadcast_to(mean, pack.soc0.shape).copy())


def charge_equalised(pack: PackBatch, mode: Mode) -> PackBatch:
    stored = pack.capacity_ah * pack.soc0
    if mode is Mode.DISCHARGE:
        soc0 = stored.mean(axis=1, keepdims=True) / pack.capacity_ah
    else:
        headroom = pack.capacity_ah * (1.0 - pack.soc0)
        soc0 = 1.0 - headroom.mean(axis=1, keepdims=True) / pack.capacity_ah
    return replace(pack, soc0=soc0)


def _drive_after_rest(pack: PackBatch, rest: BalanceResult, n_cycles: int) -> PackBatch:
    return replace(tile_pack(pack, n_cycles), soc0=np.repeat(rest.final_soc, n_cycles, axis=0))


def _strategy_runs(
    technique: Technique | None,
    pack: PackBatch,
    program: CurrentProgram,
    charge: CurrentProgram,
    rest: BalanceResult | None,
    config: ProjectConfig,
    modes: tuple[Mode, ...] = (Mode.DISCHARGE, Mode.CHARGE),
) -> dict[str, ScenarioRun]:
    n_cycles = program.n_programs
    runs: dict[str, ScenarioRun] = {}
    if Mode.DISCHARGE in modes:
        runs["driving"] = run_scenario(
            technique, tile_pack(pack, n_cycles), program, Mode.DISCHARGE, config
        )
    if Mode.CHARGE in modes:
        runs["charging"] = run_scenario(technique, pack, charge, Mode.CHARGE, config)
    if rest is not None:
        runs["rest_then_driving"] = run_scenario(
            None, _drive_after_rest(pack, rest, n_cycles), program, Mode.DISCHARGE, config
        )
    return runs


def _trace_block(
    runs: dict[str, dict[str, ScenarioRun]], scenario: str, index: int
) -> dict[str, tuple[FloatArray, FloatArray]]:
    block = {}
    for key in STRATEGY_KEYS:
        run = runs[key][scenario]
        keep = run.trace_time_s <= run.stop_time_s[index]
        block[key] = (run.trace_time_s[keep], run.trace_spread[keep, index])
    return block


def run_drive_cycle(
    pack: PackBatch,
    rest_results: dict[Technique, BalanceResult],
    profiles: dict,
    config: ProjectConfig,
) -> dict[str, object]:
    spec = config.drive_cycle
    names = [n for n in profiles["profiles"] if n.startswith("Mixed")]
    program = program_from_profiles(
        [np.array(profiles["profiles"][name]["current_a"]) for name in names]
    )
    primary = names.index(spec.primary_cycle)
    charge_current_a = -spec.charge_c_rate * config.cell.capacity_ah
    charge = constant_program(charge_current_a, 1, round(spec.charge_horizon_s / config.sim.dt_s))
    outputs = Parallel(n_jobs=len(STRATEGIES) + 2)(
        [
            *(
                delayed(_strategy_runs)(t, pack, program, charge, rest_results.get(t), config)
                for t in STRATEGIES
            ),
            delayed(_strategy_runs)(None, _soc_equalised(pack), program, charge, None, config),
            delayed(_strategy_runs)(
                None,
                charge_equalised(pack, Mode.DISCHARGE),
                program,
                charge,
                None,
                config,
                (Mode.DISCHARGE,),
            ),
            delayed(_strategy_runs)(
                None,
                charge_equalised(pack, Mode.CHARGE),
                program,
                charge,
                None,
                config,
                (Mode.CHARGE,),
            ),
        ]
    )
    runs = dict(zip(STRATEGY_KEYS, outputs[: len(STRATEGIES)], strict=True))
    soc_equalised, charge_equalised_driving, charge_equalised_charging = outputs[len(STRATEGIES) :]
    tolerance = config.sim.soc_tolerance
    scenarios: dict[str, object] = {"rest": {}, "driving": {}, "charging": {}}
    for key, technique in zip(STRATEGY_KEYS, STRATEGIES, strict=True):
        driving = runs[key]["driving"]
        scenarios["driving"][key] = _strategy_block(driving, primary, tolerance)
        scenarios["charging"][key] = _strategy_block(runs[key]["charging"], 0, tolerance)
        if technique is None:
            continue
        rest = rest_results[technique]
        after = runs[key]["rest_then_driving"]
        scenarios["rest"][key] = {
            "rest_time_s": float(rest.balancing_time_s[0]),
            "rest_energy_dissipated_j": float(rest.energy_dissipated_j[0]),
            "spread_after_rest": float(np.ptp(rest.final_soc[0])),
            "criterion_met": bool(rest.balanced[0]),
            **{
                f"followup_{name}": value
                for name, value in _strategy_block(after, primary, tolerance).items()
                if name in ("usable_ah", "stop_time_s", "final_spread")
            },
        }
    none = scenarios["driving"]["none"]["usable_ah"]
    none_charge = scenarios["charging"]["none"]["usable_ah"]
    for key in STRATEGY_KEYS[1:]:
        scenarios["driving"][key]["gain_vs_none_ah"] = scenarios["driving"][key]["usable_ah"] - none
        scenarios["charging"][key]["gain_vs_none_ah"] = (
            scenarios["charging"][key]["usable_ah"] - none_charge
        )
        scenarios["rest"][key]["followup_gain_vs_none_ah"] = (
            scenarios["rest"][key]["followup_usable_ah"] - none
        )
    capacity = pack.capacity_ah
    conservation = {
        "driving": soc_conservation_error(
            runs["none"]["driving"], np.repeat(capacity, program.n_programs, axis=0)
        ),
        "charging": soc_conservation_error(runs["none"]["charging"], capacity),
    }
    conservation["max_relative_error"] = max(conservation["driving"], conservation["charging"])
    conservation["threshold"] = spec.soc_conservation_tolerance
    conservation["pass"] = bool(
        conservation["max_relative_error"] <= spec.soc_conservation_tolerance
    )
    ledger = {
        scenario: {
            key: {
                **ledger_gap(runs[key][scenario]),
            }
            for key in STRATEGY_KEYS[1:]
        }
        for scenario in ("driving", "charging")
    }
    worst = max(
        value["max_vs_removed"] for scenario in ledger.values() for value in scenario.values()
    )
    ledger_block = {
        "by_scenario": ledger,
        "max_vs_removed": worst,
        "threshold": spec.ledger_tolerance,
        "pass": bool(worst <= spec.ledger_tolerance),
    }
    sensitivity = {
        name: {key: float(runs[key]["driving"].usable_ah[i]) for key in STRATEGY_KEYS}
        | {
            "soc_equalised": float(soc_equalised["driving"].usable_ah[i]),
            "charge_equalised": float(charge_equalised_driving["driving"].usable_ah[i]),
        }
        for i, name in enumerate(names)
    }
    traces = {
        "driving": _trace_block(runs, "driving", primary),
        "charging": _trace_block(runs, "charging", 0),
    }
    primary_profile = profiles["profiles"][spec.primary_cycle]
    block: dict[str, object] = {
        "status": "ok",
        "cycle": {
            "primary": spec.primary_cycle,
            "source_file": primary_profile["source_file"],
            "duration_s": primary_profile["duration_s"],
            "discharge_ah": primary_profile["discharge_ah"],
            "peak_discharge_a": primary_profile["peak_discharge_a"],
            "peak_regen_a": primary_profile["peak_regen_a"],
            "profile_step_s": profiles["averaging_step_s"],
            "cycles_available": names,
        },
        "limits": {
            "lower_voltage_v": spec.lower_voltage_v,
            "upper_voltage_v": spec.upper_voltage_v,
            "charge_current_a": -charge_current_a,
            "charge_c_rate": spec.charge_c_rate,
        },
        "scenarios": scenarios,
        "reference_packs": {
            "soc_equalised": {
                "driving_usable_ah": float(soc_equalised["driving"].usable_ah[primary]),
                "charging_usable_ah": float(soc_equalised["charging"].usable_ah[0]),
            },
            "charge_equalised": {
                "driving_usable_ah": float(charge_equalised_driving["driving"].usable_ah[primary]),
                "charging_usable_ah": float(charge_equalised_charging["charging"].usable_ah[0]),
            },
        },
        "cycle_sensitivity_driving_usable_ah": sensitivity,
        "soc_conservation": conservation,
        "energy_ledger_under_load": ledger_block,
    }
    return {"block": block, "traces": traces}


def drive_cycle_stage(
    config: ProjectConfig, pack: PackBatch, rest_results: dict[Technique, BalanceResult]
) -> tuple[dict[str, object], dict[str, dict[str, tuple[FloatArray, FloatArray]]] | None]:
    profiles = load_profiles(config.drive_cycle)
    if profiles is None:
        return dict(NOT_RUN), None
    if profiles.get("status") == "stale_cache":
        return dict(STALE), None
    if config.params.name != "fitted_hg2_25c":
        return dict(NOT_RUN), None
    output = run_drive_cycle(pack, rest_results, profiles, config)
    return output["block"], output["traces"]
