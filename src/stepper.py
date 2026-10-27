from collections.abc import Callable
from dataclasses import dataclass, fields, replace

import numpy as np

from src.cell import OcvCurve, open_circuit_voltage
from src.config import BoolArray, FloatArray
from src.flow import StepFlow
from src.ledger import (
    ElementLossModel,
    RCSquareWeights,
    per_link_r0_loss_w,
    rc_branch_square_integral,
)
from src.params import SECONDS_PER_HOUR

CurrentModel = Callable[[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray], StepFlow]
TILT_PIVOT_SOC = 0.5


def _select[T](obj: T, mask: BoolArray) -> T:
    return replace(
        obj,
        **{
            f.name: None if getattr(obj, f.name) is None else getattr(obj, f.name)[mask]
            for f in fields(obj)
        },
    )


@dataclass(frozen=True, slots=True)
class PackDynamics:
    r0_ohm: FloatArray
    r1_ohm: FloatArray
    rc_gain: FloatArray
    rc_cross: FloatArray
    rc_offset: FloatArray
    soc_per_amp_step: FloatArray
    technique_parameter: FloatArray
    ocv_tilt_v_per_soc: FloatArray | None = None
    soc_bias: FloatArray | None = None
    noise_std: FloatArray | None = None

    def select(self, mask: BoolArray) -> "PackDynamics":
        return _select(self, mask)


@dataclass(frozen=True, slots=True)
class PackState:
    soc: FloatArray
    v1_volt: FloatArray
    removed_j: FloatArray
    transferred_j: FloatArray
    element_loss_j: FloatArray
    r0_loss_j: FloatArray
    r1_loss_j: FloatArray
    r0_per_link_loss_j: FloatArray
    balancing_r0_j: FloatArray
    balancing_v1_j: FloatArray
    load_energy_j: FloatArray
    pack_charge_ah: FloatArray
    terminal_v: FloatArray

    @classmethod
    def initial(cls, soc0: FloatArray, ocv_v: FloatArray | None = None) -> "PackState":
        n = soc0.shape[0]
        zeros = np.zeros(n)
        return cls(
            soc=soc0.copy(),
            v1_volt=np.zeros_like(soc0),
            removed_j=zeros.copy(),
            transferred_j=zeros.copy(),
            element_loss_j=zeros.copy(),
            r0_loss_j=zeros.copy(),
            r1_loss_j=zeros.copy(),
            r0_per_link_loss_j=zeros.copy(),
            balancing_r0_j=zeros.copy(),
            balancing_v1_j=zeros.copy(),
            load_energy_j=zeros.copy(),
            pack_charge_ah=zeros.copy(),
            terminal_v=np.zeros_like(soc0) if ocv_v is None else ocv_v.copy(),
        )

    def select(self, mask: BoolArray) -> "PackState":
        return _select(self, mask)

    @property
    def spread(self) -> FloatArray:
        return self.soc.max(axis=1) - self.soc.min(axis=1)

    def estimated_spread(self, soc_bias: FloatArray | None) -> FloatArray:
        if soc_bias is None:
            return self.spread
        estimate = self.soc + soc_bias
        return estimate.max(axis=1) - estimate.min(axis=1)


@dataclass(frozen=True, slots=True)
class StepControl:
    dynamics: PackDynamics
    currents: CurrentModel
    element_loss: ElementLossModel
    curve: OcvCurve
    dt_s: float
    balance_threshold: float | None = None
    track_load: bool = False
    noise_rng: np.random.Generator | None = None

    def with_dynamics(self, dynamics: PackDynamics) -> "StepControl":
        return replace(self, dynamics=dynamics)


def no_balancing_flow(
    soc: FloatArray, ocv: FloatArray, emf: FloatArray, r0: FloatArray, parameter: FloatArray
) -> StepFlow:
    n, cells = soc.shape
    zeros = np.zeros(n)
    links = np.zeros((n, cells - 1))
    return StepFlow(
        cell_current_a=np.zeros_like(soc),
        removed_w=zeros,
        delivered_w=zeros,
        link_source_a=links,
        link_sink_a=links,
        link_forward=links.astype(bool),
    )


def no_element_loss_w(
    flow: StepFlow, emf: FloatArray, r0: FloatArray, parameter: FloatArray
) -> FloatArray:
    return np.zeros(flow.cell_current_a.shape[0])


def _gate(flow: StepFlow, active: BoolArray) -> StepFlow:
    column = active[:, None]
    return StepFlow(
        cell_current_a=flow.cell_current_a * column,
        removed_w=flow.removed_w * active,
        delivered_w=flow.delivered_w * active,
        link_source_a=flow.link_source_a * column,
        link_sink_a=flow.link_sink_a * column,
        link_forward=flow.link_forward,
    )


def cell_ocv(soc: FloatArray, control: StepControl) -> FloatArray:
    ocv = open_circuit_voltage(soc, control.curve)
    tilt = control.dynamics.ocv_tilt_v_per_soc
    return ocv if tilt is None else ocv + tilt * (soc - TILT_PIVOT_SOC)


def step(state: PackState, pack_current_a: FloatArray | None, control: StepControl) -> PackState:
    dynamics, dt_s = control.dynamics, control.dt_s
    soc, v1, r0 = state.soc, state.v1_volt, dynamics.r0_ohm
    ocv = cell_ocv(soc, control)
    emf = ocv - v1
    if pack_current_a is not None:
        emf = emf - pack_current_a[:, None] * r0
    decision_soc = soc if dynamics.soc_bias is None else soc + dynamics.soc_bias
    if dynamics.noise_std is not None and control.noise_rng is not None:
        decision_soc = decision_soc + dynamics.noise_std * control.noise_rng.standard_normal(
            soc.shape
        )
    flow = control.currents(decision_soc, ocv, emf, r0, dynamics.technique_parameter)
    if control.balance_threshold is not None:
        active = state.estimated_spread(dynamics.soc_bias) >= control.balance_threshold
        if not active.all():
            flow = _gate(flow, active)
    balancing = flow.cell_current_a
    current = balancing if pack_current_a is None else balancing + pack_current_a[:, None]
    v1_target = current * dynamics.r1_ohm
    weights = RCSquareWeights(dt_s, dynamics.rc_cross, dynamics.rc_offset)
    r1_loss = state.r1_loss_j + (
        rc_branch_square_integral(v1, v1_target, weights).sum(axis=1) / dynamics.r1_ohm[:, 0]
    )
    soc_next = soc - current * dynamics.soc_per_amp_step
    v1_next = v1 + dynamics.rc_gain * (v1_target - v1)
    balancing_r0, balancing_v1 = state.balancing_r0_j, state.balancing_v1_j
    load_energy, pack_charge, terminal = state.load_energy_j, state.pack_charge_ah, state.terminal_v
    if control.track_load:
        balancing_r0 = balancing_r0 + np.einsum("ij,ij->i", current * balancing, r0) * dt_s
        v1_integral = v1_target * dt_s + (v1 - v1_target) * dynamics.rc_cross / 2.0
        balancing_v1 = balancing_v1 + np.einsum("ij,ij->i", balancing, v1_integral)
        if pack_current_a is not None:
            volt_now = ocv - current * r0 - v1
            load_energy = load_energy + pack_current_a * volt_now.sum(axis=1) * dt_s
            pack_charge = pack_charge + pack_current_a * dt_s / SECONDS_PER_HOUR
        terminal = cell_ocv(soc_next, control) - current * r0 - v1_next
    return PackState(
        soc=soc_next,
        v1_volt=v1_next,
        removed_j=state.removed_j + flow.removed_w * dt_s,
        transferred_j=state.transferred_j + flow.delivered_w * dt_s,
        element_loss_j=state.element_loss_j
        + control.element_loss(flow, emf, r0, dynamics.technique_parameter) * dt_s,
        r0_loss_j=state.r0_loss_j + np.einsum("ij,ij->i", current**2, r0) * dt_s,
        r1_loss_j=r1_loss,
        r0_per_link_loss_j=state.r0_per_link_loss_j + per_link_r0_loss_w(flow, r0) * dt_s,
        balancing_r0_j=balancing_r0,
        balancing_v1_j=balancing_v1,
        load_energy_j=load_energy,
        pack_charge_ah=pack_charge,
        terminal_v=terminal,
    )
