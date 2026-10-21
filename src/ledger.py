from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.config import FloatArray
from src.flow import StepFlow
from src.params import BuckBoostSpec, FlyingCapSpec

ElementLossModel = Callable[[StepFlow, FloatArray, FloatArray, FloatArray], FloatArray]


def passive_element_loss_w(
    flow: StepFlow, emf: FloatArray, r0: FloatArray, r_bleed_ohm: FloatArray
) -> FloatArray:
    return np.einsum("ij,i->i", flow.cell_current_a**2, r_bleed_ohm)


def buck_boost_element_loss_w(
    flow: StepFlow,
    emf: FloatArray,
    r0: FloatArray,
    inverse_lf: FloatArray,
    hardware: BuckBoostSpec,
) -> FloatArray:
    forward = flow.link_forward
    emf_source = np.where(forward, emf[:, :-1], emf[:, 1:])
    emf_sink = np.where(forward, emf[:, 1:], emf[:, :-1])
    r0_source = np.where(forward, r0[:, :-1], r0[:, 1:])
    source_terminal_v = emf_source - flow.link_source_a * r0_source
    peak_a = source_terminal_v * hardware.duty * inverse_lf[:, None]
    demagnetising_duty = peak_a / (emf_sink * inverse_lf[:, None])
    on_interval_loss = (hardware.r_inductor_ohm + hardware.r_ds_on_ohm) * peak_a**2 * hardware.duty
    off_interval_loss = (
        (hardware.r_inductor_ohm + hardware.r_ds_on_ohm) * peak_a**2 * demagnetising_duty
    )
    link_loss = (on_interval_loss + off_interval_loss) / 3.0
    return np.where(flow.link_source_a > 0.0, link_loss, 0.0).sum(axis=1)


def flying_capacitor_element_loss_w(
    flow: StepFlow,
    emf: FloatArray,
    r0: FloatArray,
    inverse_fc: FloatArray,
    hardware: FlyingCapSpec,
) -> FloatArray:
    r_external = inverse_fc[:, None] + 2.0 * hardware.r_ds_on_ohm + hardware.esr_ohm
    return (flow.link_source_a**2 * r_external).sum(axis=1)


@dataclass(frozen=True, slots=True)
class RCSquareWeights:
    target: float
    cross: float
    offset: float


def rc_square_weights(dt_s: float, tau_s: float) -> RCSquareWeights:
    decay = np.exp(-dt_s / tau_s)
    return RCSquareWeights(
        target=dt_s,
        cross=2.0 * tau_s * (1.0 - decay),
        offset=0.5 * tau_s * (1.0 - decay**2),
    )


def rc_branch_square_integral(
    v1_start: FloatArray, v1_target: FloatArray, weights: RCSquareWeights
) -> FloatArray:
    offset = v1_start - v1_target
    return v1_target * (weights.target * v1_target + weights.cross * offset) + (
        weights.offset * offset**2
    )


def per_link_r0_loss_w(flow: StepFlow, r0: FloatArray) -> FloatArray:
    if flow.link_forward.shape[1] == 0:
        return np.einsum("ij,ij->i", flow.cell_current_a**2, r0)
    forward = flow.link_forward
    r0_source = np.where(forward, r0[:, :-1], r0[:, 1:])
    r0_sink = np.where(forward, r0[:, 1:], r0[:, :-1])
    return (r0_source * flow.link_source_a**2 + r0_sink * flow.link_sink_a**2).sum(axis=1)
