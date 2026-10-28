from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.config import FloatArray
from src.params import SocEstimationSpec
from src.soc_estimation.data import Cycle
from src.soc_estimation.ekf2 import Ekf2Cell, run_ekf2
from src.soc_estimation.evaluate import Scenario, initial_soc, perturbed_inputs
from src.soc_estimation.gru import gather, window_ends

PLAIN_INPUTS = 3
HYBRID_INPUTS = 7
AUGMENTED_COPIES = 3


@dataclass(frozen=True, slots=True)
class Ekf2Params:
    process_std: float
    measurement_std_v: float
    hysteresis_rate: float


@dataclass(frozen=True, slots=True)
class Sequence:
    features: FloatArray
    base: FloatArray
    target: FloatArray


class WindowGru(nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_inputs, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(window)
        return self.head(output[:, -1]).squeeze(-1)


def parameter_count(n_inputs: int, hidden_size: int) -> int:
    return 3 * (n_inputs * hidden_size + hidden_size**2 + 2 * hidden_size) + hidden_size + 1


def macs_per_estimate(n_inputs: int, hidden_size: int, window_s: int) -> int:
    return window_s * 3 * (n_inputs * hidden_size + hidden_size**2) + hidden_size


def ekf_states(
    cell: Ekf2Cell,
    current: FloatArray,
    voltage: FloatArray,
    soc0: float,
    params: Ekf2Params,
    spec: SocEstimationSpec,
) -> FloatArray:
    return run_ekf2(
        cell,
        current,
        voltage,
        soc0,
        params.process_std,
        params.measurement_std_v,
        spec.ekf_initial_std,
        params.hysteresis_rate,
        spec.step_s,
    )


def build_sequence(
    cycle: Cycle,
    scenario: Scenario,
    rng: np.random.Generator,
    cell: Ekf2Cell,
    params: Ekf2Params,
    spec: SocEstimationSpec,
    hybrid: bool,
) -> Sequence:
    current, voltage = perturbed_inputs(cycle, scenario, rng)
    base_features = np.stack((voltage, current, cycle.case_temperature_c), axis=1)
    if not hybrid:
        return Sequence(base_features, np.zeros(cycle.n_steps), cycle.soc)
    states = ekf_states(cell, current, voltage, initial_soc(cycle, scenario, spec), params, spec)
    return Sequence(np.concatenate((base_features, states), axis=1), states[:, 0], cycle.soc)


def augmentation_scenarios(spec: SocEstimationSpec, rng: np.random.Generator) -> list[Scenario]:
    large = spec.injected_offset_fraction * spec.tester_full_scale_a
    return [
        Scenario(f"augmented_{k}", float(rng.uniform(-large, large)), spec.voltage_noise_v, False)
        for k in range(AUGMENTED_COPIES)
    ]


def stack(sequences: list[Sequence], window_s: int, stride: int) -> tuple:
    features, base, target, ends, offset = [], [], [], [], 0
    for item in sequences:
        features.append(item.features)
        base.append(item.base)
        target.append(item.target)
        ends.append(window_ends(item.target.size, window_s, stride) + offset)
        offset += item.target.size
    return (
        np.concatenate(features),
        np.concatenate(base),
        np.concatenate(target),
        np.concatenate(ends),
    )


def to_device(array: FloatArray, device: torch.device) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


@torch.no_grad()
def estimate(
    model: WindowGru, features: torch.Tensor, base: FloatArray, ends: np.ndarray, window_s: int
) -> FloatArray:
    model.eval()
    chunks = [
        model(gather(features, ends[s : s + 1024], window_s)).cpu().numpy()
        for s in range(0, ends.size, 1024)
    ]
    return np.concatenate(chunks).astype(np.float64) + base[ends]
