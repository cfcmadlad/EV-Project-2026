from dataclasses import dataclass, replace

import numpy as np

from src.balancing import _scatter_links, _source_sink, build_control
from src.config import BoolArray, FloatArray, ProjectConfig
from src.drive_cycle import CurrentProgram
from src.flow import StepFlow
from src.pack import PackBatch
from src.params import BuckBoostSpec, RlSpec
from src.stepper import PackState, StepControl, cell_ocv, step


def commanded_currents(
    ocv: FloatArray,
    emf: FloatArray,
    r0: FloatArray,
    command: FloatArray,
    inverse_lf: FloatArray,
    hardware: BuckBoostSpec,
    spec: RlSpec,
) -> tuple[StepFlow, FloatArray]:
    forward = command > 0.0
    emf_source, emf_sink = _source_sink(emf, forward)
    r_source, r_sink = _source_sink(r0, forward)
    ocv_source, ocv_sink = _source_sink(ocv, forward)
    duty = np.minimum(np.abs(command) * spec.d_max, spec.dcm_margin / (1.0 + emf_source / emf_sink))
    conductance = 0.5 * duty**2 * inverse_lf[:, None]
    i_source = emf_source * conductance / (1.0 + conductance * r_source)
    v_source = emf_source - i_source * r_source
    i_peak = 2.0 * i_source / np.where(duty > 0.0, duty, 1.0)
    sink_duty = duty * v_source / emf_sink
    conduction_loss = (
        (hardware.r_inductor_ohm + hardware.r_ds_on_ohm) * i_peak**2 * (duty + sink_duty) / 3.0
    )
    p_out = v_source * i_source - conduction_loss
    i_sink = 2.0 * p_out / (emf_sink + np.sqrt(emf_sink**2 + 4.0 * r_sink * p_out))
    flow = StepFlow(
        cell_current_a=_scatter_links(i_source, i_sink, forward),
        removed_w=np.einsum("ij,ij->i", ocv_source, i_source),
        delivered_w=np.einsum("ij,ij->i", ocv_sink, i_sink),
        link_source_a=i_source,
        link_sink_a=i_sink,
        link_forward=forward,
    )
    return flow, duty


def commanded_loss_w(
    flow: StepFlow,
    emf: FloatArray,
    r0: FloatArray,
    duty: FloatArray,
    inverse_lf: FloatArray,
    hardware: BuckBoostSpec,
) -> FloatArray:
    emf_source, emf_sink = _source_sink(emf, flow.link_forward)
    r_source, _ = _source_sink(r0, flow.link_forward)
    terminal_v = emf_source - flow.link_source_a * r_source
    peak_a = terminal_v * duty * inverse_lf[:, None]
    demagnetising = peak_a / (emf_sink * inverse_lf[:, None])
    resistance = hardware.r_inductor_ohm + hardware.r_ds_on_ohm
    loss = resistance * peak_a**2 * (duty + demagnetising) / 3.0
    return np.where(flow.link_source_a > 0.0, loss, 0.0).sum(axis=1)


@dataclass(slots=True)
class PackSim:
    control: StepControl
    program: CurrentProgram
    soc_bias: FloatArray
    config: ProjectConfig
    holder: dict[str, FloatArray]
    state: PackState
    time_s: int
    done: BoolArray
    outside_s: FloatArray
    final: dict[str, FloatArray]


def build_sim(
    packs: PackBatch, program: CurrentProgram, soc_bias: FloatArray, config: ProjectConfig
) -> PackSim:
    spec, hardware = config.rl, config.buck_boost
    base = build_control(None, packs, np.ones(packs.n_packs), np.ones(packs.n_packs), config, True)
    n, links = packs.n_packs, packs.soc0.shape[1] - 1
    inverse_lf = np.full(n, 1.0 / (hardware.component.mid * hardware.frequency_hz.mid))
    holder = {"command": np.zeros((n, links)), "duty": np.zeros((n, links))}

    def currents(soc, ocv, emf, r0, parameter):
        flow, holder["duty"] = commanded_currents(
            ocv, emf, r0, holder["command"][: soc.shape[0]], parameter, hardware, spec
        )
        return flow

    def losses(flow, emf, r0, parameter):
        return commanded_loss_w(flow, emf, r0, holder["duty"], parameter, hardware)

    control = replace(
        base,
        currents=currents,
        element_loss=losses,
        dynamics=replace(base.dynamics, technique_parameter=inverse_lf),
    )
    state = PackState.initial(packs.soc0, cell_ocv(packs.soc0, control))
    zeros = np.zeros(n)
    return PackSim(
        control=control,
        program=program,
        soc_bias=soc_bias,
        config=config,
        holder=holder,
        state=state,
        time_s=0,
        done=np.zeros(n, dtype=bool),
        outside_s=zeros.copy(),
        final={
            key: zeros.copy()
            for key in ("pack_ah", "spread", "loss_j", "stop_time_s", "element_loss_j")
        },
    )


def program_current(sim: PackSim, offset_s: int) -> FloatArray:
    program = sim.program
    index = (sim.time_s + offset_s) % program.length
    return program.current_a[np.arange(program.n_programs), index]


def window_mean(sim: PackSim) -> FloatArray:
    window = sim.config.rl.current_window_s
    available = min(window, sim.time_s)
    if available == 0:
        return np.zeros(sim.program.n_programs)
    return np.mean([program_current(sim, -(j + 1)) for j in range(available)], axis=0)


def observe(sim: PackSim) -> FloatArray:
    spec = sim.config.rl
    estimate = sim.state.soc + sim.soc_bias
    return np.concatenate(
        (
            (estimate - 0.5) * 5.0,
            sim.state.terminal_v - 3.6,
            program_current(sim, 0)[:, None] / 10.0,
            window_mean(sim)[:, None] / 10.0,
            np.full((estimate.shape[0], 1), min(sim.time_s / spec.elapsed_reference_s, 1.0)),
        ),
        axis=1,
    )


def estimated_soc(sim: PackSim) -> FloatArray:
    return sim.state.soc + sim.soc_bias


def advance(sim: PackSim, command: FloatArray) -> tuple[FloatArray, BoolArray]:
    config = sim.config
    spec, dt_s = config.rl, config.sim.dt_s
    alive = ~sim.done
    sim.holder["command"] = command * alive[:, None]
    before = sim.state
    start_ah = before.pack_charge_ah.copy()
    start_loss = before.removed_j - before.transferred_j
    for _ in range(round(spec.control_step_s / dt_s)):
        current = program_current(sim, 0) * (~sim.done)
        sim.state = step(sim.state, current, sim.control)
        sim.time_s += 1
        voltage = sim.state.terminal_v
        newly = ~sim.done & (
            (voltage.min(axis=1) <= config.drive_cycle.lower_voltage_v)
            | (sim.state.soc.min(axis=1) <= 0.0)
        )
        sim.outside_s += (
            ~sim.done & ~newly & (voltage.max(axis=1) > config.drive_cycle.upper_voltage_v)
        )
        if newly.any():
            state = sim.state
            sim.final["pack_ah"][newly] = state.pack_charge_ah[newly]
            sim.final["spread"][newly] = state.spread[newly]
            sim.final["loss_j"][newly] = (state.removed_j - state.transferred_j)[newly]
            sim.final["element_loss_j"][newly] = state.element_loss_j[newly]
            sim.final["stop_time_s"][newly] = sim.time_s
            sim.done |= newly
    after = sim.state
    reward = (after.pack_charge_ah - start_ah) - spec.loss_weight_ah_per_j * (
        (after.removed_j - after.transferred_j) - start_loss
    )
    return reward * alive, sim.done.copy()
