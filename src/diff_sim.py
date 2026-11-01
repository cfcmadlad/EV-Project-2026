from dataclasses import dataclass

import numpy as np
import torch

from src.config import ProjectConfig, Technique
from src.params import SECONDS_PER_HOUR, LinkCoupling
from src.stepper import StepControl

DTYPE = torch.float64
SUPPLY_FLOOR = 1e-9


def tensor(values: np.ndarray | float) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=DTYPE)


@dataclass(frozen=True, slots=True)
class Twin:
    technique: Technique
    ocv_soc: torch.Tensor
    ocv_volt: torch.Tensor
    r0: torch.Tensor
    r1: torch.Tensor
    gain: torch.Tensor
    per_amp: torch.Tensor
    parameter: torch.Tensor
    deadband: float
    duty: float
    series_ohm: float
    dt_s: float
    soc_tolerance: float
    lower_voltage_v: float

    @property
    def n_packs(self) -> int:
        return self.r0.shape[0]

    @property
    def n_cells(self) -> int:
        return self.r0.shape[1]


@dataclass(frozen=True, slots=True)
class TwinState:
    soc: torch.Tensor
    v1: torch.Tensor
    removed_j: torch.Tensor
    transferred_j: torch.Tensor
    element_loss_j: torch.Tensor
    gate_drive_j: torch.Tensor
    pack_charge_ah: torch.Tensor
    terminal_v: torch.Tensor

    @property
    def spread(self) -> torch.Tensor:
        return self.soc.max(dim=1).values - self.soc.min(dim=1).values


def build_twin(control: StepControl, technique: Technique, config: ProjectConfig) -> Twin:
    if config.sim.link_coupling is not LinkCoupling.PER_LINK:
        raise ValueError("the differentiable twin implements the per-link current models")
    dynamics = control.dynamics
    if dynamics.ocv_tilt_v_per_soc is not None or dynamics.soc_bias is not None:
        raise ValueError("the twin does not model OCV tilt or SOC bias")
    hardware = config.buck_boost
    if hardware.ripple_r0_loss:
        raise ValueError("the twin does not model the ripple R0 loss")
    return Twin(
        technique=technique,
        ocv_soc=tensor(control.curve.soc),
        ocv_volt=tensor(control.curve.volt),
        r0=tensor(dynamics.r0_ohm),
        r1=tensor(dynamics.r1_ohm),
        gain=tensor(dynamics.rc_gain),
        per_amp=tensor(dynamics.soc_per_amp_step),
        parameter=tensor(dynamics.technique_parameter),
        deadband=config.sim.link_soc_deadband,
        duty=hardware.duty,
        series_ohm=hardware.r_inductor_ohm + hardware.r_ds_on_ohm,
        dt_s=control.dt_s,
        soc_tolerance=config.sim.soc_tolerance,
        lower_voltage_v=config.drive_cycle.lower_voltage_v,
    )


def interpolate(x: torch.Tensor, grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    x = x.clamp(grid[0].item(), grid[-1].item())
    upper = torch.bucketize(x.detach(), grid).clamp(1, grid.numel() - 1)
    lower = upper - 1
    span = grid[upper] - grid[lower]
    weight = (x - grid[lower]) / span
    return values[lower] + weight * (values[upper] - values[lower])


def initial_state(twin: Twin, soc0: torch.Tensor) -> TwinState:
    zeros = torch.zeros(soc0.shape[0], dtype=DTYPE)
    return TwinState(
        soc=soc0.clone(),
        v1=torch.zeros_like(soc0),
        removed_j=zeros.clone(),
        transferred_j=zeros.clone(),
        element_loss_j=zeros.clone(),
        gate_drive_j=zeros.clone(),
        pack_charge_ah=zeros.clone(),
        terminal_v=interpolate(soc0, twin.ocv_soc, twin.ocv_volt),
    )


def pick(values: torch.Tensor, forward: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = values[:, :-1], values[:, 1:]
    return torch.where(forward, left, right), torch.where(forward, right, left)


def scatter(source: torch.Tensor, sink: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
    left = torch.where(forward, source, -sink)
    right = torch.where(forward, -sink, source)
    return torch.nn.functional.pad(left, (0, 1)) + torch.nn.functional.pad(right, (1, 0))


def buck_boost_flow(
    twin: Twin,
    soc: torch.Tensor,
    ocv: torch.Tensor,
    emf: torch.Tensor,
    activity: torch.Tensor,
) -> dict[str, torch.Tensor]:
    parameter = twin.parameter
    if parameter.ndim == 1:
        inverse_lf, ratio, overhead, gate_w = parameter, 1.0, 0.0, 0.0
    else:
        inverse_lf = parameter[:, 0]
        ratio, overhead, gate_w = parameter[:, 1:2], parameter[:, 2:3], parameter[:, 3:4]
    difference = soc[:, :-1] - soc[:, 1:]
    active = difference.abs() > twin.deadband
    forward = difference > 0.0
    emf_source, emf_sink = pick(emf, forward)
    r_source, r_sink = pick(twin.r0, forward)
    ocv_source, ocv_sink = pick(ocv, forward)
    conductance = torch.where(
        active, 0.5 * twin.duty**2 * inverse_lf[:, None], torch.zeros_like(difference)
    )
    conductance = conductance * activity
    i_source = ratio * emf_source * conductance / (1.0 + conductance * r_source)
    v_source = emf_source - i_source * r_source
    i_peak = 2.0 * i_source / twin.duty
    sink_duty = twin.duty * v_source / emf_sink
    conduction = twin.series_ohm * i_peak**2 * (twin.duty + sink_duty) / 3.0
    link_overhead = torch.where(active, overhead * activity, torch.zeros_like(difference))
    p_out = torch.clamp(v_source * i_source - conduction - link_overhead, min=0.0)
    i_sink = 2.0 * p_out / (emf_sink + torch.sqrt(emf_sink**2 + 4.0 * r_sink * p_out))
    power = torch.where(active, gate_w * activity, torch.zeros_like(difference))
    supply_a = power / ocv_source
    cell = scatter(i_source, i_sink, forward) + scatter(
        supply_a, torch.zeros_like(supply_a), forward
    )
    on_loss = twin.series_ohm * i_peak**2 * twin.duty
    emf_source_term = emf_source - i_source * r_source
    off_loss = twin.series_ohm * i_peak**2 * (twin.duty * emf_source_term / emf_sink)
    link_loss = (on_loss + off_loss) / 3.0 + overhead * activity
    element_loss = torch.where(i_source > 0.0, link_loss, torch.zeros_like(link_loss)).sum(dim=1)
    return {
        "cell": cell,
        "removed_w": (ocv_source * (i_source + supply_a)).sum(dim=1),
        "delivered_w": (ocv_sink * i_sink).sum(dim=1),
        "element_loss_w": element_loss,
        "gate_w": power.sum(dim=1),
    }


def flying_capacitor_flow(
    twin: Twin,
    soc: torch.Tensor,
    ocv: torch.Tensor,
    emf: torch.Tensor,
    activity: torch.Tensor,
) -> dict[str, torch.Tensor]:
    resistance = twin.parameter
    r_link, element = resistance[:, 0], resistance[:, 1]
    gate_w = resistance[:, 2] if resistance.shape[1] >= 3 else 0.0
    difference = soc[:, :-1] - soc[:, 1:]
    active = difference.abs() > twin.deadband
    link = torch.where(active, (emf[:, :-1] - emf[:, 1:]) / r_link, torch.zeros_like(difference))
    link = link * activity
    forward = link > 0.0
    magnitude = link.abs()
    ocv_source, ocv_sink = pick(ocv, forward)
    on = magnitude > 0.0
    power = torch.where(on, gate_w * activity, torch.zeros_like(magnitude))
    supply_a = power / ocv_source
    cell = scatter(magnitude, magnitude, forward) + scatter(
        supply_a, torch.zeros_like(supply_a), forward
    )
    return {
        "cell": cell,
        "removed_w": (ocv_source * (magnitude + supply_a)).sum(dim=1),
        "delivered_w": (ocv_sink * magnitude).sum(dim=1),
        "element_loss_w": (magnitude**2 * element).sum(dim=1),
        "gate_w": power.sum(dim=1),
    }


def twin_step(
    twin: Twin,
    state: TwinState,
    pack_current: torch.Tensor | None,
    activity: torch.Tensor | None = None,
    halt: torch.Tensor | None = None,
) -> TwinState:
    soc, v1 = state.soc, state.v1
    ocv = interpolate(soc, twin.ocv_soc, twin.ocv_volt)
    emf = ocv - v1
    if pack_current is not None:
        emf = emf - pack_current[:, None] * twin.r0
    if activity is None:
        activity = torch.ones(soc.shape[0], soc.shape[1] - 1, dtype=DTYPE)
    model = buck_boost_flow if twin.technique is Technique.BUCK_BOOST else flying_capacitor_flow
    flow = model(twin, soc, ocv, emf, activity)
    gate = state.spread >= twin.soc_tolerance
    if halt is not None:
        gate = gate & ~halt
    column = gate.to(DTYPE)
    cell = flow["cell"] * column[:, None]
    current = cell if pack_current is None else cell + pack_current[:, None]
    v1_target = current * twin.r1
    soc_next = soc - current * twin.per_amp
    v1_next = v1 + twin.gain * (v1_target - v1)
    charge = state.pack_charge_ah
    if pack_current is not None:
        charge = charge + pack_current * twin.dt_s / SECONDS_PER_HOUR
    terminal = interpolate(soc_next, twin.ocv_soc, twin.ocv_volt) - current * twin.r0 - v1_next
    return TwinState(
        soc=soc_next,
        v1=v1_next,
        removed_j=state.removed_j + flow["removed_w"] * column * twin.dt_s,
        transferred_j=state.transferred_j + flow["delivered_w"] * column * twin.dt_s,
        element_loss_j=state.element_loss_j + flow["element_loss_w"] * column * twin.dt_s,
        gate_drive_j=state.gate_drive_j + flow["gate_w"] * column * twin.dt_s,
        pack_charge_ah=charge,
        terminal_v=terminal,
    )
