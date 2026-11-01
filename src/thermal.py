from dataclasses import dataclass, replace

import numpy as np

from src.balancing import build_control
from src.cell import build_ocv_curve, open_circuit_voltage
from src.config import FloatArray, ProjectConfig, Technique
from src.flow import StepFlow
from src.pack import PackBatch
from src.params import SECONDS_PER_HOUR
from src.selector import SelectorInputs, budget_key, choose, mid_values, policy_row
from src.stepper import PackState, step

TRACE_INTERVAL_S = 60.0


@dataclass(frozen=True, slots=True)
class ThermalRun:
    balancing_time_s: FloatArray
    removed_j: FloatArray
    resistor_heat_j: FloatArray
    cell_heat_j: FloatArray
    ledger_resistor_j: FloatArray
    ledger_cell_j: FloatArray
    peak_resistor_rise_k: FloatArray
    peak_cell_rise_k: FloatArray
    storage_balance_error: FloatArray
    trace_rise_k: FloatArray


def lumped_step(
    rise: FloatArray, power_w: FloatArray, rth: float, capacity: float, dt_s: float
) -> tuple[FloatArray, FloatArray]:
    tau = rth * capacity
    decay = np.exp(-dt_s / tau)
    steady = power_w * rth
    next_rise = steady + (rise - steady) * decay
    integral = steady * dt_s + (rise - steady) * tau * (1.0 - decay)
    return next_rise, integral


def gated_currents(base, holder: dict[str, FloatArray]):
    def currents(
        soc: FloatArray, ocv: FloatArray, emf: FloatArray, r0: FloatArray, parameter: FloatArray
    ) -> StepFlow:
        flow = base(soc, ocv, emf, r0, parameter)
        current = flow.cell_current_a * holder["on"]
        return replace(flow, cell_current_a=current, removed_w=np.einsum("ij,ij->i", ocv, current))

    return currents


def thermal_run(
    technique: Technique,
    packs: PackBatch,
    component: FloatArray,
    frequency_hz: FloatArray,
    config: ProjectConfig,
    ceiling_k: float | None = None,
) -> ThermalRun:
    spec = config.thermal
    sim = config.sim
    control = build_control(technique, packs, component, frequency_hz, config, gate_balancing=True)
    n, cells = packs.soc0.shape
    holder = {"on": np.ones((n, cells))}
    if ceiling_k is not None:
        control = replace(control, currents=gated_currents(control.currents, holder))
    passive = technique is Technique.PASSIVE
    r_bleed = component[:, None] if passive else np.zeros((n, 1))
    per_amp = control.dynamics.soc_per_amp_step
    state = PackState.initial(packs.soc0)
    resistor_rise = np.zeros((n, cells))
    cell_rise = np.zeros((n, cells))
    resistor_heat = np.zeros(n)
    cell_heat = np.zeros(n)
    resistor_rejected = np.zeros(n)
    cell_rejected = np.zeros(n)
    peak_resistor = np.zeros(n)
    peak_cell = np.zeros(n)
    done_at = np.full(n, np.nan)
    trace = []
    trace_every = round(TRACE_INTERVAL_S / sim.dt_s)
    n_steps = round(sim.t_max_s / sim.dt_s)
    for step_number in range(n_steps + 1):
        newly = (state.spread < sim.soc_tolerance) & np.isnan(done_at)
        done_at[newly] = step_number * sim.dt_s
        if not np.isnan(done_at).any() or step_number == n_steps:
            break
        if step_number % trace_every == 0:
            trace.append(resistor_rise[0].max())
        if ceiling_k is not None:
            holder["on"] = (resistor_rise < ceiling_k).astype(float)
        following = step(state, None, control)
        current = (state.soc - following.soc) / per_amp
        resistor_rise, resistor_integral = lumped_step(
            resistor_rise,
            current**2 * r_bleed,
            spec.resistor_rth_k_per_w,
            spec.resistor_heat_capacity_j_per_k,
            sim.dt_s,
        )
        cell_rise, cell_integral = lumped_step(
            cell_rise,
            current**2 * packs.r0_ohm,
            spec.cell_rth_k_per_w,
            spec.cell_heat_capacity_j_per_k,
            sim.dt_s,
        )
        resistor_heat += (current**2 * r_bleed).sum(axis=1) * sim.dt_s
        cell_heat += (current**2 * packs.r0_ohm).sum(axis=1) * sim.dt_s
        resistor_rejected += resistor_integral.sum(axis=1) / spec.resistor_rth_k_per_w
        cell_rejected += cell_integral.sum(axis=1) / spec.cell_rth_k_per_w
        peak_resistor = np.maximum(peak_resistor, resistor_rise.max(axis=1))
        peak_cell = np.maximum(peak_cell, cell_rise.max(axis=1))
        state = following
    stored = spec.resistor_heat_capacity_j_per_k * resistor_rise.sum(
        axis=1
    ) + spec.cell_heat_capacity_j_per_k * cell_rise.sum(axis=1)
    heat = resistor_heat + cell_heat
    balance = np.abs(stored + resistor_rejected + cell_rejected - heat) / np.maximum(heat, 1e-12)
    return ThermalRun(
        balancing_time_s=np.where(np.isnan(done_at), sim.t_max_s, done_at),
        removed_j=state.removed_j,
        resistor_heat_j=resistor_heat,
        cell_heat_j=cell_heat,
        ledger_resistor_j=state.element_loss_j if passive else np.zeros(n),
        ledger_cell_j=state.r0_loss_j,
        peak_resistor_rise_k=peak_resistor,
        peak_cell_rise_k=peak_cell,
        storage_balance_error=balance,
        trace_rise_k=np.array(trace),
    )


def baseline_row(run: ThermalRun) -> dict[str, float]:
    return {
        "balancing_time_s": float(run.balancing_time_s[0]),
        "peak_resistor_rise_k": float(run.peak_resistor_rise_k[0]),
        "peak_cell_rise_k": float(run.peak_cell_rise_k[0]),
        "resistor_heat_j": float(run.resistor_heat_j[0]),
        "ledger_resistor_j": float(run.ledger_resistor_j[0]),
        "cell_heat_j": float(run.cell_heat_j[0]),
        "ledger_cell_j": float(run.ledger_cell_j[0]),
        "storage_balance_error": float(run.storage_balance_error[0]),
    }


def peak_bleed_power_w(packs: PackBatch, r_bleed: FloatArray, config: ProjectConfig) -> FloatArray:
    ocv = open_circuit_voltage(packs.soc0, build_ocv_curve(config.cell))
    current = ocv / (r_bleed[:, None] + packs.r0_ohm)
    return (current**2 * r_bleed[:, None]).max(axis=1)


def thermal_block(
    config: ProjectConfig, baseline: PackBatch, inputs: SelectorInputs
) -> tuple[dict[str, object], dict[str, FloatArray]]:
    spec = config.thermal
    ceiling = spec.resistor_ceiling_rise_k
    components, frequencies = mid_values(config, baseline.n_packs)
    plain = {t: thermal_run(t, baseline, components[t], frequencies[t], config) for t in Technique}
    limited = thermal_run(
        Technique.PASSIVE,
        baseline,
        components[Technique.PASSIVE],
        frequencies[Technique.PASSIVE],
        config,
        ceiling,
    )
    unconstrained = plain[Technique.PASSIVE]
    pass_row = baseline_row(unconstrained)
    limited_row = baseline_row(limited)
    ledger_gap = max(
        abs(row["resistor_heat_j"] - row["ledger_resistor_j"]) / row["ledger_resistor_j"]
        for row in (pass_row, limited_row)
    )
    cell_gap = max(
        abs(baseline_row(run)["cell_heat_j"] - baseline_row(run)["ledger_cell_j"])
        / baseline_row(run)["ledger_cell_j"]
        for run in (*plain.values(), limited)
    )
    packs = inputs.packs
    sel_components, sel_frequencies = mid_values(config, packs.n_packs)
    thermal_pass = thermal_run(
        Technique.PASSIVE,
        packs,
        sel_components[Technique.PASSIVE],
        sel_frequencies[Technique.PASSIVE],
        config,
        ceiling,
    )
    true_time = inputs.true_time.copy()
    true_energy = inputs.true_energy.copy()
    true_time[:, Technique.PASSIVE] = thermal_pass.balancing_time_s
    true_energy[:, Technique.PASSIVE] = thermal_pass.removed_j
    allowed_w = ceiling / spec.resistor_rth_k_per_w
    factor = np.maximum(
        1.0,
        peak_bleed_power_w(packs, sel_components[Technique.PASSIVE], config) / allowed_w,
    )
    aware_time = inputs.surrogate_time.copy()
    aware_time[:, Technique.PASSIVE] *= factor
    budgets: dict[str, object] = {}
    for hours in config.selector.budgets_h:
        budget_s = hours * SECONDS_PER_HOUR
        oracle = choose(true_time, true_energy, budget_s)
        naive = choose(inputs.surrogate_time, inputs.surrogate_energy, budget_s)
        aware = choose(aware_time, inputs.surrogate_energy, budget_s)
        budgets[budget_key(hours)] = {
            "passive_chosen_fraction": float(np.mean(aware == Technique.PASSIVE)),
            "decision_changed_fraction": float(np.mean(aware != naive)),
            "policies": {
                "selector": policy_row(naive, true_time, true_energy, oracle, budget_s),
                "thermal_selector": policy_row(aware, true_time, true_energy, oracle, budget_s),
                "oracle": policy_row(oracle, true_time, true_energy, oracle, budget_s),
            },
        }
    block = {
        "status": "ok",
        "parameters": {
            "resistor_rth_k_per_w": spec.resistor_rth_k_per_w,
            "resistor_heat_capacity_j_per_k": spec.resistor_heat_capacity_j_per_k,
            "cell_rth_k_per_w": spec.cell_rth_k_per_w,
            "cell_heat_capacity_j_per_k": spec.cell_heat_capacity_j_per_k,
            "resistor_ceiling_rise_k": ceiling,
            "source": "none in the repository: every thermal parameter is ASSUMED",
        },
        "baseline": {
            "passive": pass_row,
            "buck_boost": baseline_row(plain[Technique.BUCK_BOOST]),
            "flying_capacitor": baseline_row(plain[Technique.FLYING_CAPACITOR]),
        },
        "passive_limited": limited_row
        | {
            "time_ratio_to_unconstrained": float(
                limited.balancing_time_s[0] / unconstrained.balancing_time_s[0]
            ),
            "ceiling_rise_k": ceiling,
            "unconstrained_exceeds_ceiling": bool(pass_row["peak_resistor_rise_k"] > ceiling),
        },
        "heat_vs_ledger": {
            "resistor_max_relative_gap": float(ledger_gap),
            "cell_max_relative_gap": float(cell_gap),
            "tolerance": spec.conservation_tolerance,
            "max_storage_balance_error": float(
                max(
                    run.storage_balance_error.max()
                    for run in (*plain.values(), limited, thermal_pass)
                )
            ),
        },
        "selector": {
            "n_packs": packs.n_packs,
            "passive_limited_time_ratio_median": float(
                np.median(thermal_pass.balancing_time_s / inputs.true_time[:, Technique.PASSIVE])
            ),
            "max_violation": config.selector.max_violation,
            "by_budget": budgets,
        },
    }
    traces = {
        "unconstrained": plain[Technique.PASSIVE].trace_rise_k,
        "limited": limited.trace_rise_k,
    }
    return block, traces
